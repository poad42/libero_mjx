# π-Harness: Multi-Agent LLM Orchestration for LIBERO + MuJoCo Warp + JAX

A research-backed multi-agent harness that drives the GPU-parallel robotics
pipeline in this repo (all 130 LIBERO tasks on MuJoCo Warp) using **GLM-5.2**
sourced through **Ollama** (cloud). Built on documented patterns from the
2024–2026 multi-agent / LLM-for-robotics literature (full citations in
`research/AGENT_HARNESS_RESEARCH.md`).

## Design at a glance

```
π-Supervisor  (Thinking=on, Preserved Thinking across turns, temp=0.6)
  ├─ Rollout Analyst    (ReAct+Reflexion)  → run_rollout / read_metrics
  ├─ Reward Designer    (ReAct, critic-gated, temp=0.8) → write_reward
  ├─ Curriculum Planner (assembly-line)    → set_curriculum
  ├─ HP Tuner           (ReAct, low-temp=0.3, thinking=off) → set_hp
  └─ Critic             (single-agent gate, NOT full debate)
```

- **Pattern:** Supervisor-delegator (LangGraph-style typed-state DAG) +
  ReAct/Reflexion sub-agents + **Eureka-style** outer loop over
  reward/curriculum/HP. (MetaGPT arXiv:2308.00352; ReAct arXiv:2210.03629;
  Reflexion arXiv:2303.11366; Eureka arXiv:2310.12931.)
- **Memory:** MemGPT three-tier (core / recall / archival) with A-MEM
  Zettelkasten linking for reward-variant evolution. Raw JAX rollout tensors
  **never** enter LLM memory — only aggregate metrics + verbal reflections
  cross the boundary. (MemGPT arXiv:2310.08560; A-MEM arXiv:2502.12110.)
- **Compute boundary:** Heavy JAX/Warp/GPU work runs in a **separate
  subprocess** (the Docker venv that has the full stack). The LLM client
  never imports jax/warp/torch. Tools return Pydantic-typed summaries.
- **Anti-hallucination:** every inter-agent handoff carries a typed Pydantic
  artifact, never free text (MetaGPT §1.3). A single critic agent gates
  reward code before it reaches the GPU (research §1.2: full debate is
  too expensive; one critic is enough).

## GLM-5.2 ideal settings (Ollama, cloud)

Derived from the official THUDM/Z.ai recommendations for the GLM-4.5/4.6/4.7
agentic lineage (research §3):

| Setting | Value | Rationale |
|---|---|---|
| `model` | `glm-5.2:cloud` | Cloud-sourced through local Ollama binary |
| `temperature` | **0.6** | Official agentic/tool-use default |
| `top_p` | **0.95** | Cumulative-prob threshold |
| `top_k` | **40** | Filters rare tokens, keeps diversity |
| `num_ctx` | 131072 | Capped for bounded latency (model supports 1M) |
| `num_predict` | 8192 (default) | Thinking traces are long; 30000 for deep reward design |
| `thinking_enabled` | **True** | Essential for reward/curriculum reasoning |
| `repeat_penalty` | 1.0 | GLM doesn't need heavy penalties (they corrupt tool JSON) |
| `presence/frequency_penalty` | 0.0 | Defaults |

Per-agent overrides (in `harness/config.py::AGENT_OVERRIDES`):
- Supervisor: temp=0.6, thinking=on, num_predict=8192
- Reward Designer: temp=**0.8** (creative code-gen), thinking=on, num_predict=12000
- HP Tuner: temp=**0.3** (deterministic numeric), thinking=**off**, num_predict=4096
- Critic: temp=0.4, thinking=on

## "π agent" note

"π agent" / "Pi agent" is **not** a recognized framework in the multi-agent
LLM literature (research §5). We use **π = policy**: the π-Supervisor is the
*policy orchestrator* that decides which sub-agent acts each iteration. Do
not chase a "π agent" reference — build on the documented patterns above.

## Quick start

```bash
# 1. Health-check Ollama + GLM-5.2
python3 -m harness --check

# 2. No-GPU smoke test (LLM + tools + memory round-trips)
python3 -m harness --smoke

# 3. Show the resolved config
python3 -m harness --show

# 4. Full run (needs the GPU compute env configured; see ComputeConfig)
python3 -m harness --goal "Achieve >70% success on libero_spatial task 0" \
    --suite spatial --task-id 0 --max-iters 20
```

## Python API

```python
from harness import Harness, HarnessConfig, GLMSettings

cfg = HarnessConfig()
# cfg.compute.python_executable = "/opt/venv/bin/python"  # the GPU env
h = Harness(cfg)
state = h.run(goal="Achieve >70% success on libero_spatial task 0",
              starting_suite="spatial", starting_task=0)
print(state.best_success_rate, state.status)
```

## Package layout

```
harness/
  __init__.py        # public exports
  __main__.py        # CLI (--check / --show / --smoke / --goal)
  config.py          # GLMSettings (ideal GLM-5.2 params) + per-agent overrides
  llm.py             # OllamaClient: /api/chat, tool-calls, thinking, structured
  schemas.py         # Pydantic models (typed artifacts between agents)
  harness.py         # Harness: the Eureka-style outer loop
  memory/
    store.py         # MemGPT tiers (core/recall/archival) + A-MEM Zettelkasten
  agents/
    base.py          # ReAct+Reflexion Agent
    team.py          # 5 specialized agents + π-Supervisor + Critic
  tools/
    registry.py      # ToolRegistry + subprocess ComputeRunner (compute boundary)
  runs/              # run state, transcripts, metrics (gitignored)
```

## How the harness talks to the existing repo

The compute tools shell out to the repo's existing scripts — **no duplication
of the GPU stack**:

| Tool | Script | Notes |
|---|---|---|
| `run_rollout` | `scripts/eval_warp_only.py` | Batched Warp rollout, returns success rate |
| `train_bc` | `scripts/train_bc.py` | BC transformer training (PyTorch/GPU) |
| `eval_policy` | `scripts/eval_bc.py` | robosuite OffScreenRenderEnv vision eval |
| `write_reward` | (in-process) | Stores a Zettelkasten note in archival memory |
| `set_curriculum` | (in-process) | Stores curriculum plan in archival memory |
| `set_hp` | (in-process) | Stores HP config in archival memory |
| `search_archival` | (in-process) | MemGPT "page fault" — retrieve long-term memory |
| `read_metrics` | (in-process) | Read a prior run's saved metrics JSON |

For the subprocess tools to return structured data to the harness, scripts
emit a `HARNESS_RESULT_JSON:{...}` line on stdout (one line, JSON object).
The harness falls back to regex-parsing the stdout tail if the marker is
absent, so existing scripts work without modification.

## Research foundation

See `research/AGENT_HARNESS_RESEARCH.md` for the full literature review
(orchestration patterns, memory, GLM settings, LLMs-for-robotics, the "π
agent" investigation, and framework comparison). Key citations:

- MetaGPT arXiv:2308.00352 · ReAct arXiv:2210.03629 · Reflexion arXiv:2303.11366
- Eureka arXiv:2310.12931 · MemGPT arXiv:2310.08560 · A-MEM arXiv:2502.12110
- "AI Agents That Matter" arXiv:2407.01502 (keep ≤5 agents)
- GLM-4 report arXiv:2406.12793 · GLM-4.5 report arXiv:2508.06471