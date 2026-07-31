# AGENTS.md — guide for opencode (and any AI agent) working in this repo

## Project

`libero-mjx-port`: all 130 LIBERO manipulation tasks ported to MuJoCo Warp
(GPU-parallel MuJoCo via JAX/Warp) + a BC transformer training/eval pipeline.

## Two stacks

1. **GPU compute stack** (jax / warp / mujoco / torch / libero) — runs inside
   the Docker venv at `/opt/venv/bin/python` (set via `HARNESS_PYTHON`). The
   system `python3` (3.14) does NOT have jax/warp/torch; do not try to import
   them in-process from the harness.
2. **Harness orchestration stack** (pydantic + httpx + requests, stdlib) —
   runs on the system `python3`. Talks to GLM-5.2 via Ollama
   (`http://localhost:11434`, model `glm-5.2:cloud`) and shells out to the
   GPU scripts via subprocess.

## Two harness implementations

There are two ways to run the multi-agent harness over this pipeline:

### A) Python π-Harness (`harness/`) — standalone, no Node needed
```bash
python3 -m harness --check            # health-check Ollama + GLM-5.2
python3 -m harness --smoke            # no-GPU smoke test (LLM+tools+memory)
python3 -m harness --show             # print resolved config
python3 -m harness --goal "..." --suite spatial --task-id 0 --max-iters 20
```
Built on documented patterns (Eureka + MetaGPT + ReAct/Reflexion + MemGPT).
See `harness/ARCHITECTURE.md`.

### B) Pi (pi.dev) — the real "π agent" harness with extensions/addons
Installed in an **isolated Node env** at `~/.pi-env` (does not pollute system):
```bash
# Activate the isolated env (add to ~/.bashrc or source on demand):
export PATH="$HOME/.pi-env/node/bin:$HOME/.pi-env/npm-prefix/bin:$PATH"

# Run Pi with GLM-5.2 via Ollama, loading the pi-libero skill + extensions:
pi --provider ollama --model "glm-5.2:cloud"
# Or in print mode (one-shot, no TUI):
pi -p --provider ollama --model "glm-5.2:cloud" "list the LIBERO suites"
# Multi-agent: the subagent extension + .pi/agents/*.md give you
# supervisor / rollout / reward / curriculum / hp sub-agents.
```
- Config: `~/.pi/agent/models.json` (Ollama + GLM-5.2), `.pi/settings.json`
  (project extensions + skills).
- Skill: `.pi/skills/pi-libero/SKILL.md` (tool reference, loaded on-demand).
- Extensions: `.pi/extensions/pi-libero.ts` (GPU compute tools),
  `.pi/extensions/libero-compaction.ts` (structured run-state compaction),
  `.pi/extensions/subagent/` (multi-agent delegation, from Pi examples).
- Agents: `.pi/agents/{supervisor,rollout,reward,curriculum,hp}.md`.
- System prompt: `SYSTEM.md` (loaded by Pi automatically).
- Reusable template: `/home/adhitya/workspace/pi-template` —
  `scripts/pi-project-init.sh <dir>` bootstraps any project with the same
  setup (isolated Node+Pi env, GLM-5.2 via Ollama, .pi/ scaffold, trust).

## Commands (repo + GPU scripts)

```bash
# Repo tests (need GPU env):
#   pytest tests/   (run inside the Docker venv)

# Existing GPU scripts (both harnesses wrap these):
#   python scripts/eval_warp_only.py --suite spatial --task-id 0 --ckpt ... --run-id X --run-dir harness/runs
#   python scripts/train_bc.py --suite spatial --task-id 0 --epochs 50 --save ckpt.pth
#   python scripts/eval_bc.py --suite spatial --task-id 0 --ckpt ckpt.pth --n-eval 20
```

## Lint / typecheck

No configured linter. Validate Python syntax with:
```bash
python3 -m py_compile $(find harness -name '*.py')
```

## π-Harness architecture (short version)

Multi-agent LLM orchestration over the GPU pipeline. GLM-5.2 (cloud via
Ollama) with ideal agentic settings: temp=0.6, top_p=0.95, top_k=40,
thinking=on. 5 agents (π-Supervisor + Rollout/Reward/Curriculum/HP) +
single Critic gate. MemGPT-tiered memory (core/recall/archival SQLite +
A-MEM Zettelkasten). Eureka-style outer loop. Heavy compute in subprocess.
See `harness/ARCHITECTURE.md` + `research/AGENT_HARNESS_RESEARCH.md`.

## Conventions

- Typed artifacts (Pydantic models in `harness/schemas.py`) between agents —
  never free text (anti-hallucination, MetaGPT).
- Raw JAX rollout tensors never enter LLM memory; only aggregate metrics do.
- GPU scripts emit `HARNESS_RESULT_JSON:{...}` (see
  `libero_mjx/harness_bridge.py`) so the harness gets structured data.
- No comments in code unless asked. No emojis.
- Do not commit `harness/runs/` or `*.sqlite` or `.pi/` runtime state.