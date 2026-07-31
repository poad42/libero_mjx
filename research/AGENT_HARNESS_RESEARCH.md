# Multi-Agent LLM Harness Research for LIBERO + MuJoCo Warp + JAX Pipelines

**Research foundation for building an LLM-orchestrated harness over a GPU-parallel robotics RL/BC pipeline.**
Compiled July 2026 from arXiv papers, official framework docs, and THUDM/Z.ai model releases.

This report covers six topics: (1) multi-agent orchestration patterns, (2) memory & context
management, (3) GLM-family model settings, (4) LLM agents for robotics/RL, (5) the "π agent"
concept, and (6) practical harness frameworks. Each section ends with concrete recommendations
for the LIBERO + MuJoCo Warp + JAX pipeline in this repo.

---

## 1. Multi-Agent Orchestration Patterns

### 1.1 The foundational pattern taxonomy (2023–2024)

The literature converges on a small set of reusable interaction patterns. The 2024 survey
"Large Language Model based Multi-Agents: A Survey of Progress and Challenges"
(Guo et al., arXiv:2402.01680) organizes them as:

| Pattern | Canonical paper | Mechanism | Best for |
|---|---|---|---|
| **Conversational / role-play** | AutoGen (Wu et al., arXiv:2308.08155), CAMEL (Li et al., arXiv:2303.17760) | Two+ agents converse turn-by-turn to solve a task | Open-ended, exploratory tasks |
| **Assembly-line / SOP** | MetaGPT (Hong et al., arXiv:2308.00352), ChatDev (Qian et al., arXiv:2307.07924) | Agents in fixed roles pass artifacts along a pipeline (PM→architect→coder→tester) | Structured, multi-stage technical work |
| **Group dynamic / debate** | AgentVerse (Chen et al., arXiv:2308.10848), Multi-Agent Debate (Du et al., arXiv:2305.14325) | N agents propose solutions and critique each other over R rounds | Factuality, hard reasoning, avoiding hallucination |
| **Single-agent + tools + reflection** | ReAct (Yao et al., arXiv:2210.03629), Reflexion (Shinn et al., arXiv:2303.11366) | One agent interleaves reasoning, action, verbal self-reflection in a loop | Tool-use, sequential decision-making, RL-like trial-and-error |
| **Tree search over agent trajectories** | LATS (Zhou et al., arXiv:2310.04406) | MCTS over LLM agent states with LM value function + reflection | High-stakes single decisions, programming, web nav |
| **Tool-maker / tool-user split** | LATM (Cai et al., arXiv:2305.17126) | Strong model writes reusable tools once; cheap model calls them | Cost amortization across many similar requests |
| **Information-asymmetric / informative** | iAgents (Liu et al., arXiv:2406.14928) | Agents must actively exchange private info to solve a task | Multi-source / multi-user coordination |

### 1.2 What recent literature says works best for technical/robotics domains

Key findings, ranked by relevance to a sim+train+eval robotics pipeline:

1. **MetaGPT's SOP/assembly-line pattern is the strongest fit for staged technical
   pipelines.** MetaGPT explicitly encodes Standardized Operating Procedures as prompt
   sequences so that "agents with human-like domain expertise verify intermediate results
   and reduce errors" — the exact property you want when chaining
   `rollout → metric extraction → reward update → curriculum step`. The paper shows this
   reduces "cascading hallucinations caused by naively chaining LLMs," the dominant failure
   mode for naive multi-agent stacks. **Apply: structure the harness as a fixed DAG of
   role-specialized agents (Rollout Analyst, Reward Designer, Curriculum Planner,
   Hyperparameter Tuner) that pass typed artifacts, not free chat.**

2. **ReAct + Reflexion is the default backbone for any agent that closes a loop over a
   physical/sim environment.** ReAct (arXiv:2210.03629) defines the interleaved
   `Thought → Action → Observation` loop that all modern agent harnesses implement;
   Reflexion (arXiv:2303.11366) adds an episodic memory of verbal self-critiques that
   pushes pass@1 on HumanEval from 80% (GPT-4) to 91% without weight updates. For a
   robotics RL loop this maps directly to: *run episode → observe metrics → write a verbal
   critique of what went wrong → keep it in a buffer → retry.* **Apply: every
   environment-facing tool-call agent should be ReAct + Reflexion, not bare
   tool-calling.**

3. **Debate / critic patterns improve factuality but multiply cost ~N×R.** Du et al.
   (arXiv:2305.14325) show multi-agent debate "significantly enhances mathematical and
   strategic reasoning" and reduces hallucinations, but it's expensive (N agents × R
   rounds). AgentVerse (arXiv:2308.10848) finds unmoderated debate can also *amplify*
   errors when agents converge on a wrong answer. **Recommendation: use a single
   lightweight critic agent (1 round, not full debate) to vet reward code and curriculum
   decisions before they hit the simulator — not a full debate panel.**

4. **Hierarchical / supervisor-delegator is the production default.** All major 2025
   frameworks (LangGraph supervisor, AutoGen GroupChat with selector, CrewAI Flows+ Crews,
   OpenAI Agents SDK handoffs) implement a supervisor that routes sub-tasks to specialized
   workers. This is the pattern the OpenAI Agents SDK (March 2025) codifies with
   `handoffs=[...]`. **Apply: one top-level orchestrator routes to
   {rollout, reward, curriculum, hp} sub-agents.**

5. **LATS-style tree search is powerful but overkill per step.** LATS achieves SOTA on
   HumanEval (92.7% with GPT-4) and WebShop by running MCTS over agent trajectories with
   an LM value function. For a robotics pipeline where each "step" is a multi-minute GPU
   rollout, the branching factor must be tiny (2–3). **Apply: reserve LATS-style
   branching only for the single most consequential decision per episode
   (e.g., "which reward function variant to commit to"), not every step.**

### 1.3 Anti-patterns flagged in the literature

- **"Naively chaining LLMs" → cascading hallucinations** (MetaGPT). Each handoff between
  agents must carry a verifiable artifact (code, metric, plot), not free text.
- **Over-decomposing tasks** — "AI Agents That Matter" (Kapoor et al., arXiv:2407.01502)
  shows SOTA agents are "needlessly complex and costly"; simpler agent loops often match
  accuracy at a fraction of the cost. Don't add agents for the sake of it.
- **No holdout sets / overfitting to benchmarks** — same paper. When iterating on the
  harness, keep a fixed set of LIBERO tasks held out from any prompt engineering.
- **Unmoderated debate amplifying errors** (AgentVerse) — if using debate, add a
  termination/consensus rule.
- **Conflating model-dev and downstream-dev benchmarks** (Kapoor) — your harness eval
  should measure *pipeline outcomes* (LIBERO success rate, sample efficiency), not
  whether the LLM "sounds right."

### 1.4 Recommendation for the LIBERO + Warp + JAX pipeline

```
Supervisor (orchestrator, GLM-4.6 / 4.7, Thinking=enabled)
   ├── Rollout Analyst (ReAct+Reflexion)        → calls run_rollout() tool
   ├── Reward Designer (ReAct, critic-gated)   → calls write_reward_code() + eval_reward() tools
   ├── Curriculum Planner (assembly-line)      → calls set_task_difficulty() tool
   └── HP Tuner (ReAct, low-temp)               → calls set_hp() tool
```
- Use the **assembly-line (MetaGPT) pattern** for the outer loop because stages are
  fixed and each produces a typed artifact (metrics dict, reward .py file, curriculum
  config, HP dict).
- Use **ReAct+Reflexion** inside each sub-agent that calls the simulator.
- Use a **single critic agent** (not full debate) to gate reward-code and curriculum
  changes before they reach the GPU.
- Keep total agent count **≤ 5**; Kapoor et al. show complexity hurts cost without
  helping accuracy.

---

## 2. Memory & Context Management for Long-Running Agents

### 2.1 The memory hierarchy that the field has converged on

The OS-memory analogy introduced by **MemGPT** (Packer et al., arXiv:2310.08560) is now
the dominant mental model. MemGPT proposes *virtual context management*: a small fast
context window (RAM) backed by tiered slower stores (disk), with the LLM itself issuing
"page faults" via function calls (`core_memory_append`, `archival_memory_insert`,
`archival_memory_search`). Letta (the company/product built on MemGPT) is the production
implementation.

The **Generative Agents** paper (Park et al., arXiv:2304.03442) introduced the now-standard
three-tier structure:

| Tier | Name | Mechanism | Lifetime |
|---|---|---|---|
| 1 | **Observation / working memory** | Raw stream of events stored verbatim | Seconds–minutes |
| 2 | **Episodic / retrieval memory** | Vector-indexed observations queried by recency × importance × relevance | Hours–days |
| 3 | **Reflection / semantic memory** | Higher-level abstractions synthesized periodically from tier 2 ("Klaus Mueller is dedicated to research") | Days–forever |

The retrieval function is `score = α·recency + β·importance + γ·relevance`, with the
paper using recency decay exponent ~0.99, importance 1–10 LLM-graded, and cosine
similarity for relevance.

### 2.2 Latest memory systems (2024–2026)

- **MemGPT / Letta** (arXiv:2310.08560, letta.com) — production OS-style memory. Supports
  `core_memory` (in-context), `recall_memory` (full transcript, searchable),
  `archival_memory` (vector DB). The LLM manages its own memory via tool calls. Has a
  "memory dreaming" feature that consolidates in the background. **This is the most
  mature option for long-running agents.**

- **A-MEM** (Xu et al., arXiv:2502.12110, NeurIPS 2025) — *Agentic Memory*. Applies the
  **Zettelkasten method**: each new memory becomes a structured note (description,
  keywords, tags) that is dynamically indexed and linked to historical memories; new
  memories can *trigger updates to existing memories* ("memory evolution"). Beats
  fixed-structure graph memory baselines across 6 foundation models. Key insight:
  memory should be **agent-driven and self-organizing**, not a fixed schema.

- **Mem0** — a popular memory layer library (mem0.ai); simpler than Letta, stores
  per-user/per-session facts in a vector store with LLM-extracted "facts." Good for
  stateless API agents that need cross-session continuity.

- **iAgents mixed memory** (arXiv:2406.14928) — combines structured and unstructured
  memory to retrieve from ~70k messages in 3 min across 140-person networks; shows the
  scale at which memory architecture starts to matter.

### 2.3 Context-window packing strategies

For a robotics training loop with **thousands of steps**, you cannot keep the full
trajectory in context. The recommended layered approach:

1. **Sliding window with summaries** — keep last K turns verbatim; periodically summarize
   older turns into a running "state summary" (used by Reflexion's episodic buffer,
   OpenHands condenser, etc.).
2. **Condenser / keep_first** — GLM-4's own OpenHands eval used
   `llm_config="condenser", keep_first=1, max_size=32` (from the GLM-4 README) to cap
   context at 32k. This is the official THUDM-recommended way to run their 32B agent on
   long tasks.
3. **Hierarchical retrieval** — Park-style tier 2/3 (above): don't summarize
   everything; index raw observations and retrieve the relevant ones per turn.
4. **Tool-calling as memory offload** — MemGPT's core insight: instead of stuffing
   context, give the agent `search_archival(query)` and let it page in what it needs.

### 2.4 Recommended setup for the LIBERO pipeline

A robotics RL/BC loop generates three distinct memory streams with very different
retention needs:

| Stream | Volume | Retention | Recommended store |
|---|---|---|---|
| **Per-step obs/action/reward** | ~1000s/episode × 1000s episodes | Most discarded; keep aggregates | JAX arrays (your existing `state.metrics`), not LLM memory |
| **Per-episode summaries + rollout stats** | ~100s/loop iter | Keep recent N in context; archive rest | Letta `recall_memory` + vector index |
| **Cross-episode lessons / reward-code history / curriculum decisions** | ~10s/loop iter | Long-term, retrieved on demand | Letta `archival_memory` or A-MEM Zettelkasten |
| **Reflections / verbal critiques** (Reflexion) | ~1/failed episode | Episodic buffer, last ~5–10 | In-context working memory (capped) |

**Concrete recipe:**
- Use **Letta** as the agent runtime; it gives you MemGPT's tiered memory out of the box
  plus the "memory dreaming" consolidation.
- For the *non-LLM* high-volume data (rollouts, JAX traces), **do not** route it through
  LLM memory — keep it in JAX arrays / HDF5 / a results DB and expose *only summary
  statistics* to the agent via tool return values. This is critical: a single LIBERO
  episode at 600 steps × 256 envs would blow any context window instantly.
- Apply **A-MEM's Zettelkasten linking** for reward-function and curriculum evolution:
  each reward version is a note linked to the metrics that motivated it and the
  failure modes it was meant to fix. This lets the agent reason about *why* a past
  reward was abandoned instead of re-deriving it.
- Cap working context per agent to **32k tokens** (GLM-4 native) and enable YaRN only
  if summaries push past it (see §3).

### 2.5 Anti-patterns
- **Stuffing raw observations into context** — the #1 cause of OOM and degraded
  reasoning. Always summarize before the LLM sees it.
- **Pure sliding window with no long-term store** — loses hard-won lessons from early
  episodes; Reflexion shows keeping verbal critiques helps a lot.
- **Fixed-schema memory** (A-MEM's critique) — rigid graph schemas don't generalize
  across tasks; prefer agent-driven self-organization.

---

## 3. GLM Family Model Settings (GLM-4 → GLM-5.2)

### 3.1 Model lineage & architecture

The THUDM / Zhipu AI / Z.ai lineage (chronological):

| Model | Release | Architecture | Params | Context | License | Source |
|---|---|---|---|---|---|---|
| GLM-4-9B (Chat) | Jun 2024 | dense | 9B | 128K (1M variant) | glm-4 (custom) | github.com/zai-org/GLM-4 |
| GLM-4-9B-0414 | Apr 2025 | dense | 9B | 32K→128K (YaRN) | MIT | same repo |
| GLM-4-32B-0414 | Apr 2025 | dense | 32B | 32K→128K (YaRN) | MIT | same |
| GLM-Z1-32B-0414 (reasoning) | Apr 2025 | dense, reasoning | 32B | 32K→128K (YaRN) | MIT | same |
| GLM-4.5 / 4.5-Air | Jul 2025 | **MoE** | 355B total / **32B active** (Air: 106B/12B) | 128K | MIT | github.com/zai-org/GLM-4.5 |
| GLM-4.6 | Sep 2025 | MoE | 355B / 32B active | **200K** | MIT | same repo |
| GLM-4.7 / 4.7-Flash | early 2026 | MoE | 355B/32B (Flash: 30B/3B) | 200K | MIT | same repo |
| GLM-5 / 5.2 | Feb 2026 | MoE | **744B** (per Presenc.AI lineage) | 200K+ | mixed | Z.ai API |

Official technical report for the GLM-4 series: "ChatGLM: A Family of Large Language
Models from GLM-130B to GLM-4 All Tools" (arXiv:2406.12793). The GLM-4.5 report is
arXiv:2508.06471 ("GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models").

The open-source repos moved from `github.com/THUDM/GLM-4` to `github.com/zai-org/GLM-4`
and then `github.com/zai-org/GLM-4.5` as Zhipu rebranded to Z.ai internationally.

### 3.2 Prompt / chat format

- **All GLM-4+ models use a ChatML-style format with `system` / `user` / `assistant`
  roles**, applied via `tokenizer.apply_chat_template(messages,
  add_generation_prompt=True)`. This is OpenAI-compatible: the Z.ai API
  (`https://api.z.ai/api/paas/v4/`) and the OpenAI Python SDK work with
  `base_url="https://api.z.ai/api/paas/v4/"`.

- **Tool-calling format:** GLM-4-0414+ supports OpenAI-style function/tool calling. When
  `tools` are passed to `apply_chat_template`, the chat template prepends a `system`
  message with tool bindings (so your own `system` prompt becomes `messages[1]`). The
  Z.ai tool schema follows OpenAI's function-calling JSON. vLLM/SGLang use
  `--tool-call-parser glm47` and `--reasoning-parser glm45` for GLM-4.7.

- **GLM-Z1-Rumination-32B-0414** is special: it does **not** accept custom system prompts
  or custom tools — only 4 fixed tools (`search`, `click`, `open`, `finish`). Use this
  model only for deep-research-style tasks, not general agent harnesses.

- **Thinking-mode toggle (GLM-4.5+):** pass `"thinking": {"type": "enabled"}` (or
  `"disabled"`) in the API request. Default is *dynamic* thinking (model decides).
  In local vLLM/SGLang, disable via
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.

- **GLM-4.7 introduces Preserved Thinking & Interleaved Thinking** — thinking blocks
  persist across turns (so the model reuses prior reasoning instead of re-deriving).
  Enable with
  `"chat_template_kwargs": {"enable_thinking": true, "clear_thinking": false}`
  (SGLang only). This is materially useful for long-horizon agent loops.

### 3.3 Official recommended sampling parameters

#### GLM-4-9B-Chat (dense, non-reasoning)
From the HuggingFace model card (`zai-org/glm-4-9b-chat`):
- `temperature` = **0.95** (their vLLM example uses 0.95; the transformers example uses
  `top_k=1` which is greedy)
- `top_k` = 1 (greedy) for deterministic, or rely on temperature sampling
- `max_tokens` = 1024 (example); up to 128K context
- Stop token ids: `[151329, 151336, 151338]` (end-of-turn / user / observation markers)

#### GLM-Z1-32B-0414 (reasoning model) — official table from HF card
| Parameter | Recommended | Notes |
|---|---|---|
| `temperature` | **0.6** | Balances creativity and stability |
| `top_p` | **0.95** | Cumulative probability threshold |
| `top_k` | **40** | Filters rare tokens, keeps diversity |
| `max_new_tokens` | **30000** | Leaves room for long thinking traces |
| `do_sample` | False also supported | Greedy for deterministic reasoning |

Plus: prepend `\n\n` to force thinking; trim hidden thinking from saved history
(implemented in `chat_template.jinja`).

#### GLM-4.5 / 4.6 / 4.7 (MoE, hybrid reasoning) — from docs.z.ai
The official quick-start code consistently uses:
- `temperature` = **0.6** (GLM-4.5 examples) / **1.0** (GLM-4.6 examples) — note the
  4.6 docs literally show `temperature: 1.0` in the basic-call snippet but `0.6` in the
  streaming snippet; treat 0.6 as the reasoning-safe default and 1.0 as the
  creative-writing default.
- `max_tokens` = 4096 (examples); context 128K (4.5) / 200K (4.6/4.7), max output 96K
  (4.5) / 128K (4.6)
- `thinking.type` = `enabled` by default

GLM-4.5 series are **purpose-built agent models** ("foundational models for
agent-oriented applications") with function calling, structured (JSON) output, and
context caching as first-class features.

### 3.4 Long-context (YaRN) settings — official

From the GLM-4 README and GLM-Z1 card, when input length exceeds **8,192 tokens**, enable
YaRN rope scaling in `config.json`:
```json
"rope_scaling": {
  "type": "yarn",
  "factor": 4.0,
  "original_max_position_embeddings": 32768
}
```
GLM-4.5+ are natively 128K (no YaRN needed until you approach that). GLM-4.6 extended to
200K natively.

### 3.5 Ollama integration
Ollama ships a `glm4` model (5.5 GB, 128K context) for the 9B Chat —
`ollama run glm4`. Requires Ollama ≥ 0.2. The Ollama model card points to
`github.com/THUDM/GLM` and the `THUDM/glm-4` HF collection. For GLM-4.5/4.6/4.7 you'd
need quantized GGUF from the HF quantization tree or run via vLLM/SGLang on the BF16
or FP8 weights. For your local GLM-5.2 (the model driving this session,
`ollama-cloud/glm-5.2`), treat the sampling params as above.

### 3.6 Recommended settings for the robotics harness (agentic / tool-use)

For **agentic + tool-calling** (which is what the harness needs):
- `temperature` = **0.6** (matches official GLM-4.5 / GLM-Z1 recommendation; low enough
  for reliable tool args, high enough to explore reward-function variants)
- `top_p` = **0.95**
- `top_k` = **40**
- `max_new_tokens` = **8192–30000** depending on whether thinking mode is on (thinking
  traces are long)
- `thinking.type` = `enabled` for reward design / curriculum reasoning; `disabled` for
  trivial tool dispatch
- Enable **Preserved Thinking** (GLM-4.7) for the long-horizon supervisor agent
- `repetition_penalty` = 1.0 (default; GLM doesn't need high penalties and they can
  hurt tool-call JSON)
- `presence_penalty` / `frequency_penalty` = 0.0 (defaults)
- `num_ctx` = 128K if using GLM-4.5/4.6 locally; enable YaRN if using GLM-4-32B-0414 and
  exceeding 8k tokens

For **creative code generation** (reward shaping, novel curriculum ideas):
- `temperature` = **0.8–1.0** (GLM-4.6 creative examples use 1.0)
- otherwise same as above

### 3.7 Pitfalls
- Don't set `top_k=1` (greedy) when you want the model to explore reward variants — it
  will collapse to one style. The HF 9B example uses `top_k=1` but that's a chat demo,
  not an agent.
- Don't enable YaRN on short contexts — the GLM docs warn it "may slightly degrade
  performance on short texts."
- GLM-Z1-Rumination ignores custom tools/system prompts; don't try to use it as a
  general tool-calling agent.
- The 4.6 docs' `temperature=1.0` is for chat/writing; if you copy it into a tool-use
  harness you'll get more malformed JSON. Use 0.6 for tool-use.

---

## 4. LLM Agents for Robotics / RL Pipelines

### 4.1 The canonical paper: Eureka

**Eureka** (Ma et al., arXiv:2310.12931, ICLR 2024) is the clearest example of an LLM
wrapping a sim+train+eval loop for reward design. Key mechanics:
- GPT-4 generates reward *code* (Python functions over env state) with zero task-specific
  prompting or templates.
- **Evolutionary optimization over reward code**: each iteration, the LLM sees the
  previous rewards + their *scalar eval metrics* and writes a new candidate.
- **In-context improvement**: the reward code, the training curve stats, and the
  fitness are fed back as text. No weight updates.
- Outperforms expert human-engineered rewards on **83% of 29 tasks** across 10 robot
  morphologies, average normalized improvement **52%**.
- Enables gradient-free RLHF: human preferences injected as text to refine rewards.
- Demonstrated pen spinning (dexterous) via curriculum learning on top of Eureka rewards.

**This is the single best template for the harness you're building.** The pattern is
exactly: `LLM proposes reward code → GPU sim+train+eval → scalar metrics back → LLM
refines`. Your repo already has the GPU-parallel sim (MuJoCo Warp, 256+ envs) and the
BC eval — you're adding the Eureka-style LLM outer loop.

### 4.2 Related reward / curriculum / controller generation work

- **Language to Reward** (Yu et al.) — maps natural-language task descriptions to
  reward functions for locomotion/manipulation; trades code-gen for a learned
  reward translator.
- **Text2Reward** (Xie et al.) — generates *executable* reward code from text,
  similar spirit to Eureka but with verification against a symbolic world model.
- **DriveVLM** (Tian et al., arXiv:2402.12289) — hierarchical
  scene-description → scene-analysis → planning VLM pipeline for autonomous driving;
  the *hierarchical planning* pattern (describe → analyze → plan) is transferable to
  robotics.
- **OpenCodeInterpreter** (Zheng et al., arXiv:2402.14658) — code generation +
  execution + iterative refinement; the multi-turn execute-and-refine loop is the same
  shape as Eureka's reward-refine loop, just for general code.
- **RoboGen** (2024) — co-evolves tasks, skills, and reward functions via LLM.

### 4.3 Concrete harnesses that wrap sim+train+eval with an LLM orchestrator

From the literature, the repeating architectural shape is:
```
LLM (ReAct+Reflexion)
  ├── tool: run_simulation(params) -> metrics   # GPU step, returns scalars
  ├── tool: eval_policy(ckpt) -> success_rate    # eval harness
  ├── tool: write_reward_code(spec) -> file      # code gen
  └── tool: read_training_curve(run_id) -> stats
loop:
  LLM proposes (reward_fn | HP | curriculum)
  → run_simulation → read_training_curve → eval_policy
  → LLM reflects (Reflexion) → proposes next
```
Eureka, Text2Reward, and RoboGen all instantiate this. The OpenHands/GLM-4 eval used a
condenser to keep the loop running within 32K context.

### 4.4 LIBERO-specific LLM-agent work

LIBERO (Wang et al., 2024, "LIBERO: Lifelong Robot Learning") is primarily a BC /
lifelong-learning benchmark. Direct LLM-agent-over-LIBERO work is sparse in the
literature as of mid-2026 — most LIBERO papers focus on policy learning (BC, diffusion
policies, RL-from-demo), not on LLM orchestration of the training loop. This means:
- **There is a genuine research gap/opportunity here**: an Eureka-style LLM harness
  driving reward/curriculum/HP selection on the 130 LIBERO tasks (which your repo
  already ports to GPU-parallel Warp) would be a novel contribution.
- The closest analogues are Eureka on IsaacGym/LiberOcean-style dexterous tasks. Your
  MuJoCo Warp port is actually a *better* substrate than Eureka's original IsaacGym
  because Warp is JAX-native and batched, so the LLM-facing `run_simulation` tool can
  return in seconds rather than the minutes IsaacGym CPU rollout takes.

### 4.5 Recommendation for the LIBERO + Warp + JAX pipeline
1. **Implement Eureka's outer loop verbatim** as the first version:
   - LLM generates a reward function (or success-predicate weights, since your
     `predicates/spatial.py` already has the building blocks).
   - Tool calls your batched Warp env (`LiberoEnv(suite, task_id, impl="warp",
     n_envs=256)`) to run rollouts.
   - Tool reads `state.metrics["success"]` (you already have this).
   - LLM sees the scalar success rate + per-component predicate pass rates and
     refines.
2. **Add a Reflexion buffer** so the agent remembers which reward shapes failed on
   which LIBERO tasks and why.
3. **Use the per-suite structure** (spatial / object / goal / scene10 / scene90) as a
   natural curriculum — start the LLM on `spatial` (10 easy tabletop tasks), then
   escalate to `scene90`.
4. **Hyperparameter selection** can be a separate low-temperature ReAct sub-agent
   (lr, batch size, epochs for the BC transformer in `scripts/train_bc.py`).
5. **DAgger/BC orchestration**: the LLM can decide *when* to collect more demos and
   *which failed states* to target — your repo already supports robosuite
   `OffScreenRenderEnv` for eval.

---

## 5. The "π agent" / "Pi agent" Concept

### 5.1 What the search found

A DuckDuckGo search for `"pi agent" OR "π agent" LLM agent framework OR "policy
iteration" agent orchestration` returned **zero results** — there is no established,
widely-cited framework or paper that uses the literal name "pi agent" or "π agent" in
the multi-agent LLM literature as of mid-2026.

### 5.2 The three plausible referents

1. **Inflection AI's "Pi" assistant.** Inflection AI (founded 2022 by Mustafa Suleyman,
   Reid Hoffman, Karén Simonyan) released the **Pi** personal AI assistant in May 2023
   — a *personal-intelligence* chatbot ("π" = "personal intelligence"). Pi is a single
   conversational assistant, **not** a multi-agent orchestration framework. Inflection
   was largely absorbed by Microsoft in March 2024 (Suleyman → Microsoft AI). So
   "Pi agent" most likely = Inflection's assistant, which is not relevant to a robotics
   harness.

2. **"PI" = Policy Iteration in RL.** In classical RL, "PI" denotes *policy iteration*,
   the iterate of policy evaluation + policy improvement. A "policy-iteration agent"
   would just be a standard RL agent — but this term is not used in the *LLM-agent*
   orchestration literature. It's possible someone is using "π agent" as shorthand for
   a policy-learning agent inside an LLM-orchestrated RL pipeline, but no named
   framework uses this label.

3. **"PI" = Proximal Policy (as in PPO/Proximal Policy Optimization).** Same situation:
   "proximal policy" appears in PPO but no agent framework is branded "π agent."

### 5.3 Conclusion
"π agent" is **not** a recognized named framework, paper, or pattern in the multi-agent
LLM harness space. If you encountered the term, it most likely refers either to
Inflection AI's Pi (a personal assistant, not an orchestration framework) or to an
informal shorthand for a policy-learning (policy-iteration / PPO) sub-agent inside an
RL pipeline. Treat it as a colloquialism, not a citable architecture. Build your
harness on the documented patterns in §1 (supervisor-delegator + ReAct/Reflexion +
Eureka-style outer loop) rather than chasing a "π agent" reference.

---

## 6. Practical Harness Frameworks (Python, 2026 state of the art)

### 6.1 Comparison table

| Framework | Multi-agent memory | State handoff | Tool-calling | Streaming | Ollama integration | Structured output | Heavy compute tools (JAX/GPU) | Notes |
|---|---|---|---|---|---|---|---|---|
| **LangGraph** | Yes (checkpointers, persisted state) | Yes (graph nodes pass typed state) | Yes | Yes | Via LangChain LLM wrapper / Ollama | Yes (Pydantic / with_structured_output) | Yes (tools are plain Python functions; you wrap JAX) | Graph-first; the most explicit state-machine model; best for fixed DAG pipelines |
| **AutoGen v0.4+** | Yes (agent chat history; BringUp persistent via extensions) | Yes (GroupChat, handoffs) | Yes | Yes | Via `autogen-ext` OpenAI-compatible client (point at Ollama) | Yes | Yes (code exec via Docker; custom tools) | Event-driven core; scalable/distributed; Python 3.10+ |
| **CrewAI** | Crew-level shared memory; Flows manage state | Yes (Flows → Crews → tasks) | Yes | Yes | Via LiteLLM/Ollama provider | Yes | Yes (any Python callable as tool) | Crews=autonomous teams, Flows=structured pipelines; production-focused |
| **Agno / Phidata** | Session + vector DB memory | Yes (team handoffs) | Yes | Yes | Native Ollama support | Yes (Pydantic) | Yes | Lightweight, model-agnostic; docs site was intermittently unreachable in research |
| **Letta** (MemGPT successor) | **Strongest**: core/recall/archival tiers + dreaming | Yes (agent state + memory) | Yes | Yes | Via OpenAI-compatible endpoint (Ollama) | Yes | Yes (custom tools) | Best for *long-running stateful* agents; the memory-first design |
| **smolagents** (HF) | Basic (agent scratchpad + Hub-shared) | Limited (CodeAgent/ToolCallingAgent) | Yes | Yes | Native (Ollama + Transformers) | Yes | Yes (sandboxed code exec: Modal/E2B/Docker) | ~1000 LOC, minimal abstractions; best for quick prototypes |
| **OpenAI Agents SDK** (Swarm successor) | Minimal (per-run) | Yes (`handoffs=[...]`) | Yes (built-in + custom) | Yes | Via Chat-Completions-compatible endpoint (Ollama works) | Yes (Pydantic) | Yes | Production guardrails + tracing; tied to OpenAI Responses API but works with any CC-style endpoint |
| **PydanticAI** | Message history + graph state | Yes (graph + dependencies) | Yes | Yes (structured streaming) | Native Ollama provider | **Best** (Pydantic-native, typed outputs) | Yes | Type-safe, FastAPI-style ergonomics; great for structured tool schemas; supports durable execution |
| **LlamaIndex Agents** | Via memory modules | Yes (agent orchestration) | Yes | Yes | Via Ollama integration | Yes | Yes | Strongest when the agent's job is RAG over a doc corpus |

### 6.2 Recommendation for the LIBERO + Warp + JAX pipeline

The harness must orchestrate **both** LLM calls and heavy non-LLM compute (JAX physics
steps, BC training on GPU). Two frameworks stand out:

1. **LangGraph** for the outer orchestration graph.
   - Its explicit state-machine model fits the fixed DAG
     (rollout → analyze → reward-design → curriculum → hp-tune → loop).
   - Typed state passed between nodes prevents the "cascading hallucination" anti-pattern.
   - Checkpointing lets you resume a long training loop after a crash.
   - Tools are plain Python functions → trivial to wrap your existing
     `LiberoEnv.step`, `train_bc.py`, `eval_bc.py`.

2. **Letta** for any agent that must be *long-running and stateful* across many
   episodes (e.g., the supervisor or the reward-designer).
   - MemGPT-tiered memory is exactly what you need so the agent remembers reward
     history across thousands of loop iterations without blowing context.
   - "Memory dreaming" consolidates while the GPU is busy training.

A practical hybrid: **LangGraph as the pipeline backbone + Letta-backed agents at nodes
that need long-term memory.** This is a common production pattern in 2026.

If you want a single-framework simpler stack, **PydanticAI** is the strongest
type-safe option — its structured-output + dependency-injection design maps cleanly
to "tool returns a `RolloutMetrics` Pydantic model," and it has native Ollama support
and durable execution for long loops. It's lighter than LangGraph+Letta.

### 6.3 Ollama integration (since you're running GLM-5.2 via Ollama)
All frameworks above can point at Ollama's OpenAI-compatible endpoint
(`http://localhost:11434/v1`, model `glm4` or a custom GLM-5.2 tag). For GLM-4.5+ with
thinking mode, you'll need to pass the `thinking` parameter through; verify your
framework's Ollama client forwards `extra_body` (PydanticAI and LangChain do; raw
OpenAI SDK does). For local GLM-4.5/4.6/4.7 weights, **vLLM or SGLang** is recommended
over Ollama for production (MTP speculative decoding, `--tool-call-parser glm47`).

### 6.4 Structured outputs
Use **Pydantic models** for every tool return value (rollout metrics, reward-fn
metadata, curriculum config). This is enforced natively by PydanticAI and available
in LangGraph (`with_structured_output`), AutoGen, CrewAI, and OpenAI Agents SDK. GLM
supports JSON-mode structured output (docs.z.ai lists it as a capability). This
prevents the LLM from emitting free-text that you then have to regex-parse.

### 6.5 Anti-patterns
- **Using a chat-only framework (raw OpenAI SDK) for a stateful pipeline** — you'll
  re-implement checkpoints, memory, and handoffs poorly. Use a framework.
- **Running heavy JAX/Warp compute *inside* the LLM tool-call handler** — keep GPU
  work in a separate process/queue and let the tool `await` a future, so the LLM
  client doesn't time out and the GPU isn't blocked on LLM tokens.
- **Per-step LLM calls in the inner training loop** — the LLM should act at
  *episode/iteration* granularity, never per physics step (would be 1000× too slow).
  Eureka calls the LLM ~once per reward revision (every few GPU training runs).

---

## TL;DR — Concrete configuration for this repo

- **Pattern:** Supervisor-delegator (LangGraph) + ReAct+Reflexion sub-agents + Eureka-style
  outer loop over reward/curriculum/HP.
- **Agents (≤5):** Supervisor (GLM-4.7, Thinking+Preserved), Rollout Analyst,
  Reward Designer, Curriculum Planner, HP Tuner.
- **Memory:** Letta (MemGPT tiers) for long-term reward/curriculum history; keep raw
  JAX rollout data out of LLM memory, expose only summary stats via tools.
- **Model:** GLM-4.6 or GLM-4.7 via vLLM/SGLang (BF16 or FP8), or GLM-5.2 via Ollama.
  - `temperature=0.6`, `top_p=0.95`, `top_k=40`, `max_new_tokens=8192–30000`,
    `thinking=enabled` for reasoning agents, `disabled` for trivial tool dispatch.
  - YaRN factor 4.0 if using GLM-4-32B-0414 past 8k tokens; not needed for 4.5+.
- **Tool layer:** Wrap existing `LiberoEnv` (Warp), `train_bc.py`, `eval_bc.py`,
  `predicates/spatial.py` as Pydantic-typed Python callables.
- **Compute boundary:** JAX/Warp GPU work runs in a separate process; tools return
  Pydantic `RolloutMetrics` summaries, never raw arrays.
- **Eval:** Hold out a fixed set of LIBERO tasks from prompt iteration; measure
  pipeline outcomes (success rate, sample efficiency), not LLM fluency.
- **"π agent":** not a real framework — ignore; build on the patterns above.

Key citations:
- AutoGen arXiv:2308.08155; MetaGPT arXiv:2308.00352; CAMEL arXiv:2303.17760;
  ChatDev arXiv:2307.07924; AgentVerse arXiv:2308.10848; MAD arXiv:2305.14325;
  ReAct arXiv:2210.03629; Reflexion arXiv:2303.11366; LATS arXiv:2310.04406;
  LATM arXiv:2305.17126; iAgents arXiv:2406.14928; MAS survey arXiv:2402.01680;
  AgentScope arXiv:2402.14034; "AI Agents That Matter" arXiv:2407.01502.
- MemGPT arXiv:2310.08560; Generative Agents arXiv:2304.03442;
  A-MEM arXiv:2502.12110.
- GLM-4 technical report arXiv:2406.12793; GLM-4.5 report arXiv:2508.06471;
  GLM-4 repo github.com/zai-org/GLM-4; GLM-4.5+ repo github.com/zai-org/GLM-4.5;
  docs.z.ai/guides/llm/glm-4.5 and /glm-4.6.
- Eureka arXiv:2310.12931; DriveVLM arXiv:2402.12289; OpenCodeInterpreter arXiv:2402.14658.
- OpenAI Agents SDK: openai.com/index/new-tools-for-building-agents (Mar 2025).
- Framework docs: langchain-ai.github.io/langgraph, docs.crewai.com, docs.letta.com,
  ai.pydantic.dev, huggingface.co/docs/smolagents.