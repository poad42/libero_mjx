"""Pydantic schemas — typed artifacts passed between agents and tools.

Per the MetaGPT finding (research §1.3): "naively chaining LLMs causes
cascading hallucinations; each handoff must carry a verifiable artifact
(code, metric, plot), not free text." Every tool return + every inter-agent
handoff in this harness is one of these typed models, never raw strings.

These also double as the JSON-schema we hand to GLM-5.2 for structured
output (via Ollama's `format` field) so the model is forced to emit valid
fields, not prose we regex-parse.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain enums (mirror libero_mjx/envs/libero.py SUITES)
# ---------------------------------------------------------------------------

class Suite(str, Enum):
    spatial = "spatial"
    object = "object"
    goal = "goal"
    scene10 = "scene10"
    scene90 = "scene90"


# ---------------------------------------------------------------------------
# Tool argument schemas (validated before dispatch)
# ---------------------------------------------------------------------------

class RunRolloutArgs(BaseModel):
    suite: Suite
    task_id: int = Field(ge=0, le=89)
    n_envs: int = Field(default=256, ge=1, le=4096)
    max_steps: int = Field(default=600, ge=1, le=5000)
    policy_ckpt: Optional[str] = Field(default=None, description="Path to .pth; None=zero action baseline")
    seed: int = 42
    impl: str = Field(default="warp", description="'warp' (GPU-parallel) or 'jax'")


class EvalPolicyArgs(BaseModel):
    suite: Suite
    task_id: int = Field(ge=0, le=89)
    ckpt: str
    n_eval: int = Field(default=20, ge=1, le=1000)
    max_steps: int = Field(default=600, ge=1, le=5000)
    seed: int = 42


class TrainBCArgs(BaseModel):
    suite: Suite
    task_id: int = Field(ge=0, le=89)
    epochs: int = Field(default=50, ge=1, le=2000)
    batch_size: int = Field(default=32, ge=1, le=2048)
    lr: float = Field(default=1e-4, gt=0, lt=1)
    save: Optional[str] = None
    seed: int = 42


class WriteRewardArgs(BaseModel):
    suite: Suite
    task_id: int = Field(ge=0, le=89)
    variant_id: str = Field(description="Short slug, e.g. 'v3_close_grasp'")
    description: str = Field(description="Natural-language rationale for this variant")
    predicate_kind: str = Field(default="distance_to",
                                 description="One of: distance_to, on, in_region, is_open, is_closed, is_turned_on")
    target_body: Optional[str] = None
    obj_body: Optional[str] = None
    dist: float = Field(default=0.08, gt=0, lt=1.0)
    code: Optional[str] = Field(default=None, description="Optional full Python reward fn body")


class SetCurriculumArgs(BaseModel):
    order: list[tuple[Suite, int]] = Field(description="Ordered list of (suite, task_id) to train on")
    rationale: str
    difficulty_focus: str = Field(default="",
                                  description="What skill this curriculum stage targets")


class SetHPArgs(BaseModel):
    suite: Suite
    task_id: int = Field(ge=0, le=89)
    epochs: int = Field(default=50, ge=1, le=2000)
    batch_size: int = Field(default=32, ge=1, le=2048)
    lr: float = Field(default=1e-4, gt=0, lt=1)
    aug_strength: float = Field(default=1.0, ge=0, le=2.0)
    rationale: str = Field(default="")


class ReadMetricsArgs(BaseModel):
    run_id: str


class SearchArchivalArgs(BaseModel):
    query: str = Field(description="Natural-language query over long-term memory")
    top_k: int = Field(default=5, ge=1, le=50)


# ---------------------------------------------------------------------------
# Tool return schemas (what tools hand back to the LLM)
# ---------------------------------------------------------------------------

class RolloutMetrics(BaseModel):
    """Summary of a batched Warp rollout. Never carries raw arrays."""
    run_id: str
    suite: Suite
    task_id: int
    n_envs: int
    steps: int
    success_rate: float = Field(ge=0, le=1)
    n_success: int
    mean_episode_steps: float
    # Per-component predicate pass rates (Eureka-style feedback signal)
    predicate_kind: str
    predicate_pass_rate: float = Field(ge=0, le=1)
    # Diagnostics
    gripper_obj_mean_dist: float = Field(default=0.0, description="Mean gripper↔object distance (m)")
    obj_target_mean_dist: float = Field(default=0.0, description="Mean object↔target distance (m)")
    nan_count: int = Field(default=0, description="Number of NaN actions clipped")
    elapsed_s: float
    ckpt_used: Optional[str] = None
    notes: str = ""


class EvalResult(BaseModel):
    run_id: str
    suite: Suite
    task_id: int
    ckpt: str
    n_eval: int
    success_rate: float = Field(ge=0, le=1)
    n_success: int
    mean_steps: float
    elapsed_s: float
    per_episode: list[dict[str, Any]] = Field(default_factory=list)


class TrainResult(BaseModel):
    run_id: str
    suite: Suite
    task_id: int
    ckpt: str
    epochs: int
    final_loss: float
    best_loss: float
    elapsed_s: float
    notes: str = ""


class RewardVariant(BaseModel):
    """A recorded reward/predicate variant (A-MEM Zettelkasten note)."""
    variant_id: str
    suite: Suite
    task_id: int
    description: str
    predicate_kind: str
    target_body: Optional[str] = None
    obj_body: Optional[str] = None
    dist: float
    code: Optional[str] = None
    created_at: str
    parent_variant_id: Optional[str] = None
    # Outcome once evaluated (filled after a rollout uses it)
    evaluated: bool = False
    success_rate: Optional[float] = None
    status: str = Field(default="proposed", description="proposed|active|abandoned|champion")


class CurriculumPlan(BaseModel):
    order: list[tuple[Suite, int]]
    rationale: str
    difficulty_focus: str
    created_at: str


class HPConfig(BaseModel):
    suite: Suite
    task_id: int
    epochs: int
    batch_size: int
    lr: float
    aug_strength: float
    rationale: str
    created_at: str


class ArchivalHit(BaseModel):
    note_id: str
    score: float
    kind: str
    summary: str
    content: str


# ---------------------------------------------------------------------------
# Agent decision outputs (structured output the agents emit)
# ---------------------------------------------------------------------------

class SupervisorDecision(BaseModel):
    """The π-Supervisor's decision at the top of each outer-loop iteration."""
    next_action: str = Field(description="One of: rollout|design_reward|set_curriculum|tune_hp|train|eval|stop")
    target_agent: str = Field(description="Which sub-agent to delegate to")
    rationale: str
    target_suite: Optional[Suite] = None
    target_task_id: Optional[int] = None
    expected_improvement: str = Field(default="", description="What we expect to change")


class Reflection(BaseModel):
    """A Reflexion-style verbal self-critique (Shinn et al. 2023)."""
    what_happened: str
    what_went_wrong: str = Field(default="")
    what_to_try_next: str
    confidence: float = Field(default=0.5, ge=0, le=1)


class CriticVerdict(BaseModel):
    """Single-critic gate on reward/curriculum code before it hits the GPU."""
    approved: bool
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str = Field(default="")
    severity: str = Field(default="none", description="none|low|medium|high|blocker")


# ---------------------------------------------------------------------------
# Run-state (the LangGraph-style typed state passed along the DAG)
# ---------------------------------------------------------------------------

class IterationResult(BaseModel):
    iteration: int
    suite: Suite
    task_id: int
    reward_variant_id: Optional[str] = None
    rollout: Optional[RolloutMetrics] = None
    train: Optional[TrainResult] = None
    eval_: Optional[EvalResult] = Field(default=None, alias="eval")
    reflection: Optional[Reflection] = None
    critic: Optional[CriticVerdict] = None
    notes: str = ""

    model_config = {"populate_by_name": True}


class HarnessRunState(BaseModel):
    """Full mutable state of one harness run (serialized to run dir)."""
    run_id: str
    started_at: str
    config_snapshot: dict[str, Any]
    iterations: list[IterationResult] = Field(default_factory=list)
    current_suite: Optional[Suite] = None
    current_task_id: Optional[int] = None
    current_reward_variant_id: Optional[str] = None
    best_success_rate: float = 0.0
    best_ckpt: Optional[str] = None
    status: str = "running"
    error: Optional[str] = None

    def add(self, it: IterationResult) -> None:
        self.iterations.append(it)
        if it.eval_ and it.eval_.success_rate > self.best_success_rate:
            self.best_success_rate = it.eval_.success_rate
            self.best_ckpt = it.eval_.ckpt