"""Configuration for the π-Harness.

`GLMSettings` encodes the ideal inference settings for GLM-5.2 (cloud-sourced
through Ollama) derived from the official THUDM/Z.ai recommendations for the
GLM-4.5/4.6/4.7 agentic lineage (see research/AGENT_HARNESS_RESEARCH.md §3):

  - temperature = 0.6  (official agentic/tool-use default; low enough for
    reliable tool-call JSON, high enough to explore reward variants)
  - top_p       = 0.95 (cumulative-prob threshold)
  - top_k       = 40   (filters rare tokens, keeps diversity)
  - thinking    = enabled  (GLM-5.2 supports it; essential for reward design
    and curriculum reasoning; disabled for trivial tool dispatch)
  - num_ctx     = 131072 (the cloud model advertises 1M; we cap at 128K so
    summarization stays bounded and latency predictable)
  - max_new_tokens = 8192 (thinking traces are long; raise to 30000 for deep
    reward-design episodes)
  - repetition_penalty = 1.0, presence/frequency_penalty = 0.0 (GLM does not
    need heavy penalties and they corrupt tool-call JSON)

Per-agent overrides:
  - Supervisor: thinking=enabled, preserved_thinking=True, temp=0.6
  - Reward Designer: thinking=enabled, temp=0.8 (creative code-gen)
  - HP Tuner: thinking=disabled (trivial dispatch), temp=0.3 (deterministic)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class GLMSettings:
    """Ideal Ollama inference settings for GLM-5.2 (:cloud)."""

    model: str = "glm-5.2:cloud"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 40
    num_ctx: int = 131072
    num_predict: int = 8192
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = -1
    # GLM-5.2 (via Ollama) emits a `thinking` field; we keep it on by default
    # and surface it in the agent transcript. The Ollama API does not take a
    # separate `thinking` flag for cloud models — thinking is controlled by the
    # model. We mirror the intent here for logging / agent logic.
    thinking_enabled: bool = True
    # Timeout for a single Ollama (non-streaming) chat call, in seconds.
    # Cloud models can be slow under load; keep generous.
    request_timeout: float = 600.0

    def to_ollama_options(self) -> dict[str, Any]:
        """Translate to the `options` block of Ollama's /api/chat payload."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
        }


# Per-agent temperature/thinking overrides (research §3.6).
AGENT_OVERRIDES: dict[str, dict[str, Any]] = {
    "supervisor": {"temperature": 0.6, "thinking_enabled": True, "num_predict": 8192},
    "rollout": {"temperature": 0.6, "thinking_enabled": True, "num_predict": 6144},
    "reward": {"temperature": 0.8, "thinking_enabled": True, "num_predict": 12000},
    "curriculum": {"temperature": 0.6, "thinking_enabled": True, "num_predict": 6144},
    "hp": {"temperature": 0.3, "thinking_enabled": False, "num_predict": 4096},
    "critic": {"temperature": 0.4, "thinking_enabled": True, "num_predict": 6144},
}


def settings_for(role: str, base: GLMSettings | None = None) -> GLMSettings:
    """Return a GLMSettings copy with the per-agent override applied."""
    b = base or defaults().llm
    ov = AGENT_OVERRIDES.get(role, {})
    return GLMSettings(
        model=b.model,
        base_url=b.base_url,
        temperature=ov.get("temperature", b.temperature),
        top_p=b.top_p,
        top_k=b.top_k,
        num_ctx=b.num_ctx,
        num_predict=ov.get("num_predict", b.num_predict),
        repeat_penalty=b.repeat_penalty,
        presence_penalty=b.presence_penalty,
        frequency_penalty=b.frequency_penalty,
        seed=b.seed,
        thinking_enabled=ov.get("thinking_enabled", b.thinking_enabled),
        request_timeout=b.request_timeout,
    )


@dataclass
class MemoryConfig:
    """MemGPT-tiered memory sizing.

    - core_context_tokens: hard cap on in-context working memory per agent.
      32k matches the GLM-4 OpenHands eval condenser recommendation; GLM-5.2
      has a 1M window but we keep this bounded for latency + cost.
    - recall_window: number of recent transcript turns kept verbatim before
      rolling into a running summary.
    - archival_db: path to the SQLite archival store (long-term, retrieved on
      demand via tool calls — the MemGPT "page fault" pattern).
    """
    core_context_tokens: int = 32_000
    recall_window: int = 16
    summary_max_tokens: int = 1500
    archival_db: str = "harness/runs/memory.sqlite"
    enable_zettelkasten: bool = True


@dataclass
class ComputeConfig:
    """Boundary between the LLM orchestration layer and heavy GPU compute.

    The harness never imports jax/warp/torch in-process. Every compute tool
    shells out to the repo's existing scripts (train_bc.py, eval_bc.py,
    eval_warp_only.py) inside the configured python env (typically the
    Docker venv that has the full stack). This keeps the LLM client from
    blocking on GPU work and matches the anti-pattern guidance in
    research §6.5.
    """
    python_executable: str = os.environ.get("HARNESS_PYTHON", "/opt/venv/bin/python")
    repo_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    libero_basil_path: str = os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil")
    default_n_envs: int = 256
    default_n_eval: int = 20
    default_max_steps: int = 600
    default_epochs: int = 50
    default_batch_size: int = 32
    default_lr: float = 1e-4
    device: str = "cuda:0"
    # Max wall-clock seconds for a single compute subprocess before we kill it.
    subprocess_timeout: float = 7200.0


@dataclass
class HarnessConfig:
    llm: GLMSettings = field(default_factory=GLMSettings)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    # Maximum outer-loop iterations (Eureka episodes) before the harness stops.
    max_iterations: int = 20
    # Hold out these LIBERO tasks from prompt iteration (research §1.3 anti-pattern).
    held_out_tasks: tuple[str, ...] = ("scene90",)
    # Where run transcripts / checkpoints / logs live.
    run_dir: str = "harness/runs"
    # Verbosity: "quiet" | "normal" | "debug"
    log_level: str = "normal"
    # If True, persist full Ollama responses (incl. thinking) to the run dir.
    save_transcripts: bool = True

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def defaults() -> HarnessConfig:
    """Sensible default config. Override fields after construction."""
    return HarnessConfig()