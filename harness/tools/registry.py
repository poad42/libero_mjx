"""Tool registry + compute-boundary wrappers for the LIBERO/MJX/Warp stack.

Every heavy-compute tool shells out to the repo's existing scripts via the
configured Python env (the Docker venv that has jax/warp/torch). The LLM
orchestration process NEVER imports jax/warp/torch — this is the anti-pattern
guard from research §6.5 ("running heavy JAX/Warp compute inside the LLM
tool-call handler blocks the client and the GPU").

Tools return Pydantic-typed summaries (RolloutMetrics, EvalResult, ...),
never raw arrays. The dispatcher validates args against the schemas in
harness.schemas before invoking, so a malformed LLM tool-call is caught
here rather than crashing the subprocess.

Available compute tools (wrapping existing scripts/):
  run_rollout   → scripts/eval_warp_only.py  (Warp physics + render eval)
  train_bc      → scripts/train_bc.py
  eval_policy   → scripts/eval_bc.py  (robosuite OffScreenRenderEnv)

Memory tools (in-process, no subprocess):
  search_archival → ArchivalStore.search
  read_metrics    → read a prior run's RolloutMetrics from the run dir
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import uuid
import logging
from typing import Any, Callable, Optional

from harness.config import ComputeConfig
from harness.schemas import (
    RunRolloutArgs, EvalPolicyArgs, TrainBCArgs, WriteRewardArgs,
    SetCurriculumArgs, SetHPArgs, ReadMetricsArgs, SearchArchivalArgs,
    RolloutMetrics, EvalResult, TrainResult, RewardVariant, CurriculumPlan,
    HPConfig, ArchivalHit,
)
from harness.llm import ToolCall
from harness.memory import AgentMemory, ArchivalStore

log = logging.getLogger("harness.tools")


# ---------------------------------------------------------------------------
# Subprocess runner (the compute boundary)
# ---------------------------------------------------------------------------

class ComputeRunner:
    """Runs repo scripts in a subprocess with the configured Python env.

    Captures stdout/stderr, enforces a timeout, and returns parsed JSON if the
    script emits a `HARNESS_RESULT_JSON:` line (our convention) — otherwise the
    raw stdout tail.
    """

    def __init__(self, cfg: ComputeConfig):
        self.cfg = cfg

    def run_script(self, script: str, args: list[str], env: Optional[dict[str, str]] = None) -> dict[str, Any]:
        cmd = [self.cfg.python_executable, "-u", script, *args]
        log.info("compute> %s", " ".join(shlex.quote(c) for c in cmd))
        merged = os.environ.copy()
        if env:
            merged.update(env)
        merged.setdefault("LIBERO_BASIL_PATH", self.cfg.libero_basil_path)
        try:
            proc = subprocess.run(
                cmd, cwd=self.cfg.repo_root, env=merged,
                capture_output=True, text=True,
                timeout=self.cfg.subprocess_timeout,
            )
        except subprocess.TimeoutExpired as e:
            return {"ok": False, "error": f"subprocess timeout after {self.cfg.subprocess_timeout}s",
                    "stdout": (e.stdout or "")[-2000:], "stderr": (e.stderr or "")[-2000:]}
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        result: dict[str, Any] = {"ok": proc.returncode == 0, "returncode": proc.returncode}
        # Look for our JSON marker line(s).
        marker = "HARNESS_RESULT_JSON:"
        if marker in stdout:
            line = stdout.split(marker, 1)[1].splitlines()[0]
            try:
                result["data"] = json.loads(line)
            except json.JSONDecodeError as e:
                result["data"] = None
                result["parse_error"] = str(e)
        result["stdout_tail"] = stdout[-4000:]
        result["stderr_tail"] = stderr[-4000:]
        if not result["ok"]:
            result["error"] = f"script exited {proc.returncode}"
        return result


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Holds tool specs (for the LLM) + their dispatcher callables.

    Tools are registered with (name, description, args_schema, handler).
    The handler takes a validated args-dict and returns a dict (later wrapped
    in a Pydantic return schema where applicable).
    """

    def __init__(self, compute: ComputeRunner, archival: ArchivalStore, run_dir: str):
        self.compute = compute
        self.archival = archival
        self.run_dir = run_dir
        self._tools: dict[str, dict[str, Any]] = {}
        self._register_all()

    # -- registration ------------------------------------------------------

    def register(self, name: str, description: str, schema: type, handler: Callable[[dict], Any]) -> None:
        self._tools[name] = {
            "name": name, "description": description,
            "schema": schema, "handler": handler,
        }

    def specs(self, names: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """Return OpenAI-style tool specs for the LLM."""
        out = []
        for name, t in self._tools.items():
            if names and name not in names:
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": t["description"],
                    "parameters": t["schema"].model_json_schema(),
                },
            })
        return out

    def dispatch(self, tc: ToolCall) -> dict[str, Any]:
        """Validate args + call the handler. Raises on unknown tool / bad args."""
        if tc.name not in self._tools:
            return {"error": f"unknown tool: {tc.name}"}
        t = self._tools[tc.name]
        try:
            args = t["schema"].model_validate(tc.arguments).model_dump()
        except Exception as e:
            return {"error": f"invalid args for {tc.name}: {e}", "received": tc.arguments}
        result = t["handler"](args)
        return result if isinstance(result, dict) else {"result": result}

    # -- the actual tools --------------------------------------------------

    def _register_all(self) -> None:
        c = self.compute

        # run_rollout -------------------------------------------------------
        def _run_rollout(args: dict) -> dict:
            a = RunRolloutArgs.model_validate(args)
            run_id = f"rollout_{a.suite}_{a.task_id}_{uuid.uuid4().hex[:6]}"
            script = os.path.join(c.cfg.repo_root, "scripts", "eval_warp_only.py")
            cli_args = [
                "--suite", a.suite.value, "--task-id", str(a.task_id),
                "--n-envs", str(a.n_envs), "--max-steps", str(a.max_steps),
                "--seed", str(a.seed), "--run-id", run_id,
                "--run-dir", self.run_dir,
            ]
            if a.policy_ckpt:
                cli_args += ["--ckpt", a.policy_ckpt]
            res = c.run_script(script, cli_args)
            if not res["ok"]:
                return {"ok": False, "error": res.get("error"), "stderr": res.get("stderr_tail")}
            data = res.get("data") or _parse_rollout_from_stdout(res.get("stdout_tail", ""))
            data.setdefault("run_id", run_id)
            return {"ok": True, "metrics": data}

        self.register(
            "run_rollout",
            "Run a batched GPU-parallel Warp rollout of a policy on a LIBERO task. "
            "Returns aggregate success rate + per-predicate pass rates. Use this to "
            "measure how a policy / reward variant performs. n_envs up to 4096.",
            RunRolloutArgs, _run_rollout,
        )

        # train_bc ----------------------------------------------------------
        def _train_bc(args: dict) -> dict:
            a = TrainBCArgs.model_validate(args)
            run_id = f"train_{a.suite}_{a.task_id}_{uuid.uuid4().hex[:6]}"
            save = a.save or os.path.join(self.run_dir, f"{run_id}.pth")
            script = os.path.join(c.cfg.repo_root, "scripts", "train_bc.py")
            cli_args = [
                "--suite", a.suite.value, "--task-id", str(a.task_id),
                "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
                "--lr", str(a.lr), "--seed", str(a.seed),
                "--save", save, "--run-id", run_id, "--run-dir", self.run_dir,
            ]
            res = c.run_script(script, cli_args)
            if not res["ok"]:
                return {"ok": False, "error": res.get("error"), "stderr": res.get("stderr_tail")}
            data = res.get("data") or {"ckpt": save, "final_loss": float("nan")}
            data.setdefault("run_id", run_id)
            data.setdefault("ckpt", save)
            return {"ok": True, "train": data}

        self.register(
            "train_bc",
            "Train a BC transformer policy on LIBERO demo data (PyTorch/GPU). "
            "Use after deciding hyperparameters. Returns the checkpoint path + final loss.",
            TrainBCArgs, _train_bc,
        )

        # eval_policy -------------------------------------------------------
        def _eval_policy(args: dict) -> dict:
            a = EvalPolicyArgs.model_validate(args)
            run_id = f"eval_{a.suite}_{a.task_id}_{uuid.uuid4().hex[:6]}"
            script = os.path.join(c.cfg.repo_root, "scripts", "eval_bc.py")
            cli_args = [
                "--suite", a.suite.value, "--task-id", str(a.task_id),
                "--ckpt", a.ckpt, "--n-eval", str(a.n_eval),
                "--max-steps", str(a.max_steps), "--seed", str(a.seed),
                "--run-id", run_id, "--run-dir", self.run_dir,
            ]
            res = c.run_script(script, cli_args)
            if not res["ok"]:
                return {"ok": False, "error": res.get("error"), "stderr": res.get("stderr_tail")}
            data = res.get("data") or _parse_eval_from_stdout(res.get("stdout_tail", ""))
            data.setdefault("run_id", run_id)
            return {"ok": True, "eval": data}

        self.register(
            "eval_policy",
            "Evaluate a BC checkpoint on a LIBERO task via robosuite OffScreenRenderEnv "
            "(vision eval, same distribution as training data). Returns success rate.",
            EvalPolicyArgs, _eval_policy,
        )

        # write_reward ------------------------------------------------------
        def _write_reward(args: dict) -> dict:
            a = WriteRewardArgs.model_validate(args)
            note_id = f"reward_{a.variant_id}"
            rv = RewardVariant(
                variant_id=a.variant_id, suite=a.suite, task_id=a.task_id,
                description=a.description, predicate_kind=a.predicate_kind,
                target_body=a.target_body, obj_body=a.obj_body, dist=a.dist,
                code=a.code, created_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            )
            # Persist to archival as a Zettelkasten note.
            self.archival.add(_reward_note(rv, note_id))
            return {"ok": True, "variant_id": a.variant_id, "note_id": note_id,
                    "status": "proposed"}

        self.register(
            "write_reward",
            "Propose a new reward / success-predicate variant for a LIBERO task. "
            "Stored in long-term memory with A-MEM Zettelkasten linking so the agent "
            "can reason about why past variants were abandoned. "
            "predicate_kind ∈ {distance_to, on, in_region, is_open, is_closed, is_turned_on}.",
            WriteRewardArgs, _write_reward,
        )

        # set_curriculum ----------------------------------------------------
        def _set_curriculum(args: dict) -> dict:
            a = SetCurriculumArgs.model_validate(args)
            order = [(s.value, t) for s, t in a.order]
            plan = CurriculumPlan(
                order=order, rationale=a.rationale,
                difficulty_focus=a.difficulty_focus,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            )
            note_id = f"curriculum_{int(time.time())}"
            self.archival_add_curriculum(plan, note_id)
            return {"ok": True, "note_id": note_id, "order": order}

        self.register(
            "set_curriculum",
            "Define the ordered list of (suite, task_id) LIBERO tasks to train on next. "
            "Use the suite difficulty as a natural curriculum: spatial → object → goal → "
            "scene10 → scene90. Provide a rationale.",
            SetCurriculumArgs, _set_curriculum,
        )

        # set_hp ------------------------------------------------------------
        def _set_hp(args: dict) -> dict:
            a = SetHPArgs.model_validate(args)
            hp = HPConfig(
                suite=a.suite, task_id=a.task_id, epochs=a.epochs,
                batch_size=a.batch_size, lr=a.lr, aug_strength=a.aug_strength,
                rationale=a.rationale,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            )
            note_id = f"hp_{a.suite.value}_{a.task_id}_{int(time.time())}"
            self.archival_add_hp(hp, note_id)
            return {"ok": True, "note_id": note_id, "hp": hp.model_dump()}

        self.register(
            "set_hp",
            "Set training hyperparameters for the next BC training run. "
            "Lower temperature reasoning — this is a deterministic numeric decision.",
            SetHPArgs, _set_hp,
        )

        # search_archival ---------------------------------------------------
        def _search_archival(args: dict) -> dict:
            a = SearchArchivalArgs.model_validate(args)
            hits = self.archival.search(a.query, top_k=a.top_k)
            return {"ok": True, "hits": [
                ArchivalHit(note_id=n.note_id, score=0.0, kind=n.kind,
                            summary=n.summary, content=n.content[:500]).model_dump()
                for n in hits
            ]}

        self.register(
            "search_archival",
            "Search long-term memory (reward variants, reflections, curriculum plans, "
            "lessons) by keyword. Use this to recall why a past approach was abandoned "
            "before re-proposing something similar.",
            SearchArchivalArgs, _search_archival,
        )

        # read_metrics ------------------------------------------------------
        def _read_metrics(args: dict) -> dict:
            a = ReadMetricsArgs.model_validate(args)
            path = os.path.join(self.run_dir, f"{a.run_id}.json")
            if not os.path.exists(path):
                return {"ok": False, "error": f"no metrics file for run_id={a.run_id}"}
            with open(path) as f:
                return {"ok": True, "metrics": json.load(f)}

        self.register(
            "read_metrics",
            "Read the saved metrics JSON for a prior run_id (rollout/train/eval).",
            ReadMetricsArgs, _read_metrics,
        )

    # -- archival helpers --------------------------------------------------

    def archival_add_curriculum(self, plan: CurriculumPlan, note_id: str) -> None:
        from harness.memory import ArchivalNote
        self.archival.add(ArchivalNote(
            note_id=note_id, kind="curriculum",
            summary=f"Curriculum: {len(plan.order)} tasks — {plan.difficulty_focus}",
            content=plan.model_dump_json(),
            keywords=["curriculum"] + [s for s, _ in plan.order],
            tags=["curriculum"], links=[], created_at=plan.created_at,
        ))

    def archival_add_hp(self, hp: HPConfig, note_id: str) -> None:
        from harness.memory import ArchivalNote
        self.archival.add(ArchivalNote(
            note_id=note_id, kind="hp",
            summary=f"HP: {hp.suite}/{hp.task_id} epochs={hp.epochs} bs={hp.batch_size} lr={hp.lr}",
            content=hp.model_dump_json(),
            keywords=["hp", hp.suite.value], tags=["hp"], links=[],
            created_at=hp.created_at,
        ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reward_note(rv: RewardVariant, note_id: str):
    from harness.memory import ArchivalNote
    return ArchivalNote(
        note_id=note_id, kind="reward_variant",
        summary=f"Reward {rv.variant_id} [{rv.predicate_kind}] for {rv.suite}/{rv.task_id}: {rv.description[:100]}",
        content=rv.model_dump_json(),
        keywords=[rv.predicate_kind, rv.suite.value, f"task{rv.task_id}", rv.variant_id],
        tags=["reward", rv.suite.value, f"task{rv.task_id}"],
        links=[], created_at=rv.created_at,
        metadata={"status": rv.status, "success_rate": rv.success_rate},
    )


def _parse_rollout_from_stdout(stdout: str) -> dict:
    """Best-effort parse of success rate from a script's stdout tail,
    in case the script doesn't emit HARNESS_RESULT_JSON.
    """
    import re
    m = re.search(r"success[^0-9]*([0-9]+)\s*/\s*([0-9]+)", stdout, re.IGNORECASE)
    if m:
        n, total = int(m.group(1)), int(m.group(2))
        return {"success_rate": n / max(total, 1), "n_success": n, "n_envs": total}
    return {"success_rate": 0.0, "n_success": 0, "n_envs": 0, "note": "could not parse stdout"}


def _parse_eval_from_stdout(stdout: str) -> dict:
    return _parse_rollout_from_stdout(stdout)