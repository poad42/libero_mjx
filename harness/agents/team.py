"""The five specialized agents + the π-Supervisor.

Agent roster (research §1.4 — keep ≤5 agents; Kapoor et al. show complexity
hurts cost without helping accuracy):

  π-Supervisor    — top-level orchestrator. Decides which sub-agent to
                    delegate to each outer-loop iteration. Uses Preserved
                    Thinking so its reasoning carries across turns.
  Rollout Analyst — ReAct+Reflexion. Calls run_rollout, reads metrics,
                    diagnoses failure modes (where does the policy get stuck?).
  Reward Designer — ReAct, critic-gated. Proposes reward/predicate variants
                    (Eureka-style). Higher temperature (0.8) for creative
                    code-gen. A single Critic agent gates before GPU use.
  Curriculum Planner — assembly-line. Picks the next (suite, task_id) order.
  HP Tuner        — ReAct, low-temp (0.3), thinking off. Numeric decisions.

Each agent gets its own GLMSettings override (see config.AGENT_OVERRIDES) and
its own AgentMemory instance sharing the global ArchivalStore.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from harness.agents.base import Agent
from harness.config import HarnessConfig, GLMSettings
from harness.llm import ToolCall
from harness.schemas import SupervisorDecision, CriticVerdict
from harness.tools import ToolRegistry

log = logging.getLogger("harness.agents")


# ---------------------------------------------------------------------------
# Shared prompts (concise, role-specific; GLM-5.2 follows ChatML system role)
# ---------------------------------------------------------------------------

_COMMON = """You are part of a multi-agent harness driving a GPU-parallel robotics
pipeline: LIBERO (130 manipulation tasks) ported to MuJoCo Warp (JAX) with
a BC transformer policy. Heavy GPU compute runs in a separate process; you
call tools that return aggregate metrics, never raw arrays.

Ground truth: the env exposes `state.metrics["success"]` per parallel env
(batched up to 4096 worlds on one GPU). Success predicates live in
libero_mjx/predicates/spatial.py: distance_to, on, in_region, is_open,
is_closed, is_turned_on.

Always think step by step (your thinking is logged). Call ONE tool per turn
when possible. When you have enough information, give a concise final answer.
"""


def _supervisor_prompt() -> str:
    return _COMMON + f"""
# Your role: π-Supervisor (the policy orchestrator)

You are the top-level orchestrator. Each outer-loop iteration you decide
which sub-agent to delegate to. The sub-agents are:
  - rollout    : Run/analyze a Warp rollout to measure current performance.
  - reward     : Propose a new reward/predicate variant (Eureka-style).
  - curriculum : Choose the next set of LIBERO tasks to train on.
  - hp         : Set training hyperparameters for the next BC run.
  - train      : (you handle directly) kick off BC training.
  - eval       : (you handle directly) evaluate a checkpoint.
  - stop       : declare the goal reached and stop.

Decision principles:
  1. Never train before you have a baseline rollout for the current task.
  2. If success_rate < 0.5, prefer `reward` (the predicate/reward shape is
     usually the highest-leverage change — Eureka beats human rewards 83% of
     the time).
  3. If success_rate is plateauing across 2+ reward variants, try `curriculum`
     (easier task first) or `hp` (more epochs / lower lr).
  4. Keep total agents ≤5; don't decompose further.
  5. Respect held-out tasks (never tune prompts on them).

Emit a structured SupervisorDecision.
"""


def _rollout_prompt() -> str:
    return _COMMON + """
# Your role: Rollout Analyst (ReAct + Reflexion)

Given a (suite, task_id) and optional checkpoint, run a batched Warp rollout
and diagnose what happened. Use run_rollout, then read_metrics if needed.
Report: success_rate, where the policy fails (gripper-object distance,
object-target distance), and a one-line diagnosis. If the run failed
technically (NaN, crash), reflect on why.
"""


def _reward_prompt() -> str:
    return _COMMON + """
# Your role: Reward Designer (Eureka-style, critic-gated)

Propose reward / success-predicate variants that improve the policy's
success rate. Each proposal is a `write_reward` call storing a Zettelkasten
note in long-term memory. Before proposing, call search_archival to recall
past variants for this task and WHY they were abandoned — do not re-propose
discarded ideas.

Eureka loop: see previous success rates → hypothesize what reward shape is
missing → write a new variant → (the supervisor will roll it out) → see the
new metric → refine. Be creative (temperature is high) but grounded in the
numbers you saw.

Predicate kinds available: distance_to, on, in_region, is_open, is_closed,
is_turned_on. For spatial tasks, distance_to(obj, plate, dist=0.08) is the
default; consider tightening dist, switching to `on` (adds an above+XY check),
or composing.
"""


def _curriculum_prompt() -> str:
    return _COMMON + """
# Your role: Curriculum Planner (assembly-line)

Choose the ordered list of (suite, task_id) LIBERO tasks to train on next.
Use the natural difficulty ramp: spatial (tabletop, 10 tasks) → object (floor,
10) → goal (kitchen, 10) → scene10 (multi-step) → scene90 (90 diverse). Start
easy and escalate. Provide a rationale tied to the current best success rate.
"""


def _hp_prompt() -> str:
    return _COMMON + """
# Your role: HP Tuner (ReAct, low-temperature, numeric)

Set training hyperparameters for the next BC run. This is a deterministic
numeric decision — do not be creative. Defaults: epochs=50, batch_size=32,
lr=1e-4, aug_strength=1.0. Adjust only with a reason (e.g. loss plateau →
lower lr; underfitting → more epochs; overfitting → stronger aug).
"""


def _critic_prompt() -> str:
    return _COMMON + """
# Your role: Critic (single-agent gate, NOT full debate)

You vet a proposed reward/predicate variant BEFORE it reaches the GPU. Check:
  - predicate_kind is valid; args match the schema
  - dist is physically reasonable (0.01–0.2 m for tabletop)
  - the rationale is consistent with the last rollout's failure mode
  - it is not a duplicate of an abandoned variant (call search_archival)
Emit a CriticVerdict. Be terse. Only block on real issues.
"""


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

class AgentTeam:
    """Builds the 5 specialized agents sharing one ToolRegistry + ArchivalStore.

    The supervisor is special: it doesn't call compute tools directly; it
    emits a SupervisorDecision that the harness loop interprets to delegate
    to the other agents.
    """

    def __init__(self, cfg: HarnessConfig, tools: ToolRegistry):
        self.cfg = cfg
        self.tools = tools

        # Supervisor: only the memory tools (search_archival) — it decides,
        # it doesn't compute.
        sup_tools = tools.specs(names=["search_archival", "read_metrics"])
        self.supervisor = Agent(
            role="supervisor",
            system_prompt=_supervisor_prompt(),
            tools=sup_tools,
            tool_dispatcher=tools.dispatch,
            base_llm=cfg.llm,
            mem_cfg=cfg.memory,
            max_tool_steps=4,
        )

        self.rollout = Agent(
            role="rollout",
            system_prompt=_rollout_prompt(),
            tools=tools.specs(names=["run_rollout", "read_metrics", "search_archival"]),
            tool_dispatcher=tools.dispatch,
            base_llm=cfg.llm,
            mem_cfg=cfg.memory,
            max_tool_steps=4,
        )

        self.reward = Agent(
            role="reward",
            system_prompt=_reward_prompt(),
            tools=tools.specs(names=["write_reward", "search_archival", "read_metrics"]),
            tool_dispatcher=tools.dispatch,
            base_llm=cfg.llm,
            mem_cfg=cfg.memory,
            max_tool_steps=6,
        )

        self.curriculum = Agent(
            role="curriculum",
            system_prompt=_curriculum_prompt(),
            tools=tools.specs(names=["set_curriculum", "search_archival", "read_metrics"]),
            tool_dispatcher=tools.dispatch,
            base_llm=cfg.llm,
            mem_cfg=cfg.memory,
            max_tool_steps=4,
        )

        self.hp = Agent(
            role="hp",
            system_prompt=_hp_prompt(),
            tools=tools.specs(names=["set_hp", "search_archival", "read_metrics"]),
            tool_dispatcher=tools.dispatch,
            base_llm=cfg.llm,
            mem_cfg=cfg.memory,
            max_tool_steps=4,
        )

        self.critic = Agent(
            role="critic",
            system_prompt=_critic_prompt(),
            tools=tools.specs(names=["search_archival", "read_metrics"]),
            tool_dispatcher=tools.dispatch,
            base_llm=cfg.llm,
            mem_cfg=cfg.memory,
            max_tool_steps=3,
        )

    def by_name(self, name: str) -> Agent:
        return {
            "supervisor": self.supervisor,
            "rollout": self.rollout,
            "reward": self.reward,
            "curriculum": self.curriculum,
            "hp": self.hp,
            "critic": self.critic,
        }[name]

    def decide(self, context: str) -> SupervisorDecision:
        """Ask the supervisor for the next delegation decision."""
        msgs = self.supervisor._build_initial_messages(
            task="Decide the next action for this outer-loop iteration. "
                 "Emit a SupervisorDecision JSON.",
            context=context,
        )
        data = self.supervisor.client.structured(msgs, SupervisorDecision.model_json_schema())
        return SupervisorDecision.model_validate(data)

    def critique(self, proposal: dict[str, Any]) -> CriticVerdict:
        """Single-critic gate on a reward/curriculum proposal."""
        msgs = self.critic._build_initial_messages(
            task="Vet this proposal. Emit a CriticVerdict JSON.",
            context=json.dumps(proposal, default=str),
        )
        data = self.critic.client.structured(msgs, CriticVerdict.model_json_schema())
        return CriticVerdict.model_validate(data)