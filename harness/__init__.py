"""π-Harness: Multi-agent LLM orchestration for the LIBERO + MuJoCo Warp + JAX pipeline.

A research-backed harness (Eureka-style outer loop + supervisor-delegator +
ReAct/Reflexion sub-agents + MemGPT-tiered memory) driving the GPU-parallel
robotics stack already in this repo via GLM-5.2 (cloud-sourced through Ollama).

Architecture (see ARCHITECTURE.md for the full rationale + citations):

    π-Supervisor  (Thinking=on, Preserved Thinking across turns)
      ├─ Rollout Analyst   (ReAct+Reflexion)  → run_rollout / render
      ├─ Reward Designer   (ReAct, critic-gated) → write reward/predicate code
      ├─ Curriculum Planner (assembly-line)    → set task difficulty / order
      └─ HP Tuner          (ReAct, low-temp)   → set training hyperparams

Heavy JAX/Warp/GPU compute runs in a *separate process* (subprocess boundary);
tools return Pydantic-typed summaries, never raw arrays. Raw rollout tensors
stay in JAX/HDF5 and never enter LLM context (anti-pattern flagged in the
literature). Only aggregate metrics + verbal reflections cross the boundary.

Memory follows the MemGPT three-tier model (core / recall / archival) with
A-MEM-style Zettelkasten linking for reward/predicate evolution. Per-agent
working context is capped; long-term history is retrieved on demand.

Public API:
    from harness import Harness, HarnessConfig
    from harness.config import GLMSettings
"""

from harness.config import HarnessConfig, GLMSettings, defaults
from harness.harness import Harness

__all__ = ["Harness", "HarnessConfig", "GLMSettings", "defaults"]
__version__ = "0.1.0"