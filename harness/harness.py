"""π-Harness: the Eureka-style outer loop tying agents + tools + memory together.

The pipeline DAG (research §1.4 + §4.5):

    ┌──────────────────────────────────────────────────────────────┐
    │  for iteration in range(max_iterations):                     │
    │    1. π-Supervisor.decide(context) → SupervisorDecision      │
    │    2. switch decision.next_action:                           │
    │         rollout    → Rollout Analyst runs + diagnoses        │
    │         design_reward → Reward Designer proposes variant     │
    │                       → Critic gates it (single agent)       │
    │         set_curriculum → Curriculum Planner picks task order │
    │         tune_hp    → HP Tuner sets hyperparams               │
    │         train      → (direct) train_bc tool                  │
    │         eval       → (direct) eval_policy tool                │
    │         stop       → break                                   │
    │    3. Record IterationResult, update HarnessRunState         │
    │    4. Reflexion: failing agents write self-critiques         │
    │    5. Persist state + transcript to run dir                  │
    └──────────────────────────────────────────────────────────────┘

The supervisor's "Preserved Thinking" is approximated by feeding the prior
iteration's reasoning back as context (we don't have native GLM-4.7
clear_thinking=False here, but the recall + archival tiers give the same
effect: the supervisor sees its own past reasoning via memory).
"""

from __future__ import annotations

import json
import os
import time
import logging
from typing import Any, Optional

from harness.config import HarnessConfig, defaults
from harness.llm import OllamaClient
from harness.memory import get_archival_store
from harness.tools import ToolRegistry, ComputeRunner
from harness.agents import AgentTeam
from harness.schemas import (
    HarnessRunState, IterationResult, SupervisorDecision, RolloutMetrics,
    EvalResult, TrainResult, Reflection, CriticVerdict, Suite,
)

log = logging.getLogger("harness")


class Harness:
    """The top-level orchestrator. Construct, then call `.run()`.

    Example:
        from harness import Harness, HarnessConfig
        h = Harness(HarnessConfig())
        h.run(goal="Achieve >70% success on libero_spatial task 0")
    """

    def __init__(self, cfg: Optional[HarnessConfig] = None):
        self.cfg = cfg or defaults()
        os.makedirs(self.cfg.run_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.memory.archival_db) or ".", exist_ok=True)

        # Wire up the layers.
        self.archival = get_archival_store(self.cfg.memory.archival_db)
        self.compute = ComputeRunner(self.cfg.compute)
        self.tools = ToolRegistry(self.compute, self.archival, self.cfg.run_dir)
        self.team = AgentTeam(self.cfg, self.tools)
        self.llm = OllamaClient(self.cfg.llm)

        self.state: Optional[HarnessRunState] = None
        self._run_id: Optional[str] = None

    # -- public API --------------------------------------------------------

    def run(self, goal: str, starting_suite: Suite = Suite.spatial, starting_task: int = 0) -> HarnessRunState:
        """Run the Eureka-style outer loop until the supervisor says stop or
        we hit max_iterations.

        Args:
          goal: natural-language goal (e.g. "Achieve >70% success on spatial task 0").
          starting_suite / starting_task: initial focus.
        """
        self._configure_logging()
        self._run_id = f"run_{int(time.time())}"
        self.state = HarnessRunState(
            run_id=self._run_id,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            config_snapshot=self.cfg.as_dict(),
            current_suite=starting_suite,
            current_task_id=starting_task,
        )
        self._save_state()
        log.info("=== π-Harness run %s started ===", self._run_id)
        log.info("goal: %s", goal)
        log.info("model: %s (temp=%.2f top_p=%.2f top_k=%d thinking=%s)",
                 self.cfg.llm.model, self.cfg.llm.temperature, self.cfg.llm.top_p,
                 self.cfg.llm.top_k, self.cfg.llm.thinking_enabled)

        # Seed the supervisor's core memory with the goal + starting point.
        self.team.supervisor.memory.core_append(f"Goal: {goal}")
        self.team.supervisor.memory.core_append(
            f"Starting point: {starting_suite.value} task {starting_task}")
        self.team.supervisor.memory.core_append(
            f"Held-out tasks (never tune on): {', '.join(self.cfg.held_out_tasks)}")

        # Health check.
        if not self.llm.ping():
            self.state.status = "error"
            self.state.error = "Ollama/GLM-5.2 not reachable at " + self.cfg.llm.base_url
            self._save_state()
            log.error(self.state.error)
            return self.state

        for i in range(self.cfg.max_iterations):
            log.info("--- iteration %d ---", i)
            try:
                self._iteration(i, goal)
            except KeyboardInterrupt:
                log.warning("interrupted by user")
                self.state.status = "interrupted"
                break
            except Exception as e:
                log.exception("iteration %d failed", i)
                self.state.error = f"iter {i}: {e}"
                # Reflexion on the harness itself.
                self._record_failure(i, e)
                continue
            if self.state.status in ("done", "stopped"):
                break
            self._save_state()

        if self.state.status == "running":
            self.state.status = "max_iterations"
        self._save_state()
        log.info("=== run %s finished: status=%s best=%.0f%% ===",
                 self._run_id, self.state.status, self.state.best_success_rate * 100)
        return self.state

    # -- one iteration -----------------------------------------------------

    def _iteration(self, i: int, goal: str) -> None:
        ctx = self._supervisor_context(goal)
        decision = self.team.decide(ctx)
        log.info("supervisor decision: %s → %s (suite=%s task=%s)",
                 decision.next_action, decision.target_agent,
                 decision.target_suite, decision.target_task_id)

        action = decision.next_action
        suite = decision.target_suite or self.state.current_suite
        task_id = decision.target_task_id if decision.target_task_id is not None else self.state.current_task_id
        it = IterationResult(iteration=i, suite=suite, task_id=task_id)

        if action == "stop":
            self.state.status = "stopped"
            it.notes = "supervisor declared goal reached"
            self.state.add(it)
            return

        if action == "rollout":
            res = self.team.rollout.run(
                task=f"Run a rollout on {suite.value} task {task_id} and diagnose.",
                context=ctx,
            )
            it.notes = res["answer"]
            it.rollout = self._maybe_load_rollout(res)
            if it.rollout and it.rollout.success_rate < 0.5:
                self.team.rollout.reflect(
                    {"rollout": it.rollout.model_dump(), "answer": res["answer"]},
                    success=False,
                )

        elif action == "design_reward":
            res = self.team.reward.run(
                task=f"Propose a new reward/predicate variant for {suite.value} task {task_id}.",
                context=ctx,
            )
            it.notes = res["answer"]
            # Critic gate (single agent, not full debate — research §1.2).
            proposal = {"answer": res["answer"], "suite": suite.value, "task_id": task_id}
            verdict = self.team.critique(proposal)
            it.critic = verdict
            if not verdict.approved:
                it.notes += f" [CRITIC BLOCKED: {verdict.issues}]"
                log.warning("critic blocked reward proposal: %s", verdict.issues)

        elif action == "set_curriculum":
            res = self.team.curriculum.run(
                task="Propose the next curriculum (ordered list of LIBERO tasks).",
                context=ctx,
            )
            it.notes = res["answer"]

        elif action == "tune_hp":
            res = self.team.hp.run(
                task=f"Set training hyperparameters for {suite.value} task {task_id}.",
                context=ctx,
            )
            it.notes = res["answer"]

        elif action == "train":
            # Direct tool call (supervisor delegates to itself for compute).
            from harness.llm import ToolCall
            tc = ToolCall(id="direct_train", name="train_bc", arguments={
                "suite": suite.value, "task_id": task_id,
                "epochs": self.cfg.compute.default_epochs,
                "batch_size": self.cfg.compute.default_batch_size,
                "lr": self.cfg.compute.default_lr,
            })
            res = self.tools.dispatch(tc)
            if res.get("ok") and res.get("train"):
                it.train = TrainResult(**res["train"])
                self.state.current_reward_variant_id = None
            else:
                it.notes = f"train failed: {res.get('error', res)}"

        elif action == "eval":
            from harness.llm import ToolCall
            ckpt = self.state.best_ckpt or os.path.join(
                self.cfg.run_dir, f"train_{suite.value}_{task_id}_latest.pth")
            tc = ToolCall(id="direct_eval", name="eval_policy", arguments={
                "suite": suite.value, "task_id": task_id, "ckpt": ckpt,
                "n_eval": self.cfg.compute.default_n_eval,
                "max_steps": self.cfg.compute.default_max_steps,
            })
            res = self.tools.dispatch(tc)
            if res.get("ok") and res.get("eval"):
                it.eval_ = EvalResult(**res["eval"])
            else:
                it.notes = f"eval failed: {res.get('error', res)}"

        else:
            it.notes = f"unknown action: {action}"

        self.state.add(it)
        self._save_iteration(it)
        # Update current focus if the supervisor moved on.
        if decision.target_suite:
            self.state.current_suite = decision.target_suite
        if decision.target_task_id is not None:
            self.state.current_task_id = decision.target_task_id

    # -- helpers -----------------------------------------------------------

    def _supervisor_context(self, goal: str) -> str:
        """Build the context string the supervisor sees each iteration."""
        lines = [f"Goal: {goal}",
                 f"Best success rate so far: {self.state.best_success_rate:.0%}",
                 f"Current focus: {self.state.current_suite.value} task {self.state.current_task_id}",
                 f"Iteration {len(self.state.iterations)}/{self.cfg.max_iterations}"]
        if self.state.iterations:
            last = self.state.iterations[-1]
            lines.append(f"Last iteration: {last.notes[:300]}")
        # A couple of recent reflections from any agent.
        recent = self.archival.list_kind("reflection", limit=3)
        if recent:
            lines.append("Recent reflections:")
            for r in recent:
                lines.append(f"  - {r.summary}")
        return "\n".join(lines)

    def _maybe_load_rollout(self, res: dict[str, Any]) -> Optional[RolloutMetrics]:
        # The rollout agent calls run_rollout internally; its answer is prose,
        # but the tool result is in the dispatcher's return. We re-read the
        # latest rollout file from the run dir as the source of truth.
        import glob
        files = sorted(glob.glob(os.path.join(self.cfg.run_dir, "rollout_*.json")),
                        key=os.path.getmtime, reverse=True)
        if not files:
            return None
        with open(files[0]) as f:
            data = json.load(f)
        try:
            return RolloutMetrics(**data)
        except Exception as e:
            log.warning("could not parse rollout metrics: %s", e)
            return None

    def _record_failure(self, i: int, e: Exception) -> None:
        self.team.supervisor.memory.archival_add(
            kind="lesson",
            summary=f"Iteration {i} crashed: {type(e).__name__}",
            content=str(e)[:1000],
            tags=["harness_crash"], links=[],
        )

    def _save_iteration(self, it: IterationResult) -> None:
        path = os.path.join(self.cfg.run_dir, f"iter_{it.iteration:03d}.json")
        with open(path, "w") as f:
            f.write(it.model_dump_json(indent=2))
        if self.cfg.save_transcripts:
            self._save_transcript(it)

    def _save_transcript(self, it: IterationResult) -> None:
        # Dump each agent's recent recall for debugging.
        tdir = os.path.join(self.cfg.run_dir, "transcripts")
        os.makedirs(tdir, exist_ok=True)
        for name, agent in [
            ("supervisor", self.team.supervisor), ("rollout", self.team.rollout),
            ("reward", self.team.reward), ("curriculum", self.team.curriculum),
            ("hp", self.team.hp), ("critic", self.team.critic),
        ]:
            path = os.path.join(tdir, f"{name}_iter{it.iteration:03d}.json")
            with open(path, "w") as f:
                json.dump({
                    "role": name,
                    "core_memory": agent.memory.core,
                    "summary": agent.memory.summary,
                    "recall": agent.memory.recall[-8:],
                    "reflections": agent.reflections,
                }, f, indent=2, default=str)

    def _save_state(self) -> None:
        if self.state is None:
            return
        path = os.path.join(self.cfg.run_dir, "state.json")
        with open(path, "w") as f:
            f.write(self.state.model_dump_json(indent=2))

    def _configure_logging(self) -> None:
        level = {"quiet": logging.WARNING, "normal": logging.INFO,
                 "debug": logging.DEBUG}.get(self.cfg.log_level, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )