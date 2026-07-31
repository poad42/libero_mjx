#!/usr/bin/env python3
"""π-Harness CLI.

Usage:
  python -m harness --goal "Achieve >70% success on libero_spatial task 0"
  python -m harness --goal "..." --suite spatial --task-id 0 --max-iters 20
  python -m harness --check        # health-check Ollama + GLM-5.2
  python -m harness --show         # print the default config
  python -m harness --smoke       # one-tool-call smoke test (no GPU)
"""

from __future__ import annotations

import argparse
import json
import sys
import logging

from harness.config import HarnessConfig, GLMSettings, defaults
from harness.schemas import Suite


def main() -> int:
    p = argparse.ArgumentParser(prog="harness", description="π-Harness CLI")
    p.add_argument("--goal", default="Achieve >70% success on libero_spatial task 0",
                   help="Natural-language goal for the supervisor")
    p.add_argument("--suite", default="spatial", choices=[s.value for s in Suite])
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--max-iters", type=int, default=20)
    p.add_argument("--model", default="glm-5.2:cloud")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--run-dir", default="harness/runs")
    p.add_argument("--log-level", default="normal", choices=["quiet", "normal", "debug"])
    p.add_argument("--check", action="store_true", help="Health-check Ollama + GLM-5.2 and exit")
    p.add_argument("--show", action="store_true", help="Print the resolved config and exit")
    p.add_argument("--smoke", action="store_true", help="Run a no-GPU smoke test and exit")
    args = p.parse_args()

    cfg = defaults()
    cfg.llm = GLMSettings(model=args.model, base_url=args.base_url)
    cfg.run_dir = args.run_dir
    cfg.log_level = args.log_level
    cfg.max_iterations = args.max_iters

    if args.show:
        print(json.dumps(cfg.as_dict(), indent=2, default=str))
        return 0

    # Configure logging early for --check / --smoke.
    logging.basicConfig(
        level={"quiet": logging.WARNING, "normal": logging.INFO,
               "debug": logging.DEBUG}[args.log_level],
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.check:
        from harness.llm import OllamaClient
        c = OllamaClient(cfg.llm)
        ok = c.ping()
        print(f"Ollama @ {cfg.llm.base_url} model={cfg.llm.model}: {'OK' if ok else 'FAIL'}")
        return 0 if ok else 1

    if args.smoke:
        return _smoke(cfg)

    # Full run.
    from harness.harness import Harness
    h = Harness(cfg)
    state = h.run(goal=args.goal, starting_suite=Suite(args.suite), starting_task=args.task_id)
    print(f"\n=== run {state.run_id}: status={state.status} "
          f"best={state.best_success_rate:.0%} iters={len(state.iterations)} ===")
    return 0 if state.status in ("stopped", "done", "max_iterations") else 1


def _smoke(cfg: HarnessConfig) -> int:
    """No-GPU smoke test: LLM round-trip + tool dispatch + memory round-trip."""
    print("[smoke] 1. LLM round-trip...")
    from harness.llm import OllamaClient
    c = OllamaClient(cfg.llm)
    if not c.ping():
        print("[smoke] FAIL: Ollama/GLM-5.2 not reachable")
        return 1
    r = c.chat(messages=[{"role": "user", "content": "Reply with the single word: PONG"}],
               options_override={**cfg.llm.to_ollama_options(), "num_predict": 256})
    print(f"  → content={r.content!r} thinking_len={len(r.thinking)}")
    if "PONG" not in r.content.upper():
        print("[smoke] FAIL: expected PONG")
        return 1

    print("[smoke] 2. Tool-calling round-trip...")
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo back the message",
            "parameters": {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
        },
    }]
    r = c.chat(
        messages=[{"role": "user", "content": "Use the echo tool with msg='hello harness'."}],
        tools=tools,
    )
    print(f"  → tool_calls={len(r.tool_calls)}")
    if not r.tool_calls:
        print("[smoke] FAIL: model did not call the tool")
        return 1
    tc = r.tool_calls[0]
    print(f"  → name={tc.name} args={tc.arguments}")
    if tc.name != "echo" or tc.arguments.get("msg") != "hello harness":
        print("[smoke] FAIL: wrong tool/args")
        return 1

    print("[smoke] 3. Structured output...")
    from harness.schemas import Reflection
    data = c.structured(
        messages=[{"role": "user", "content":
            "Produce a Reflection about a failed robot rollout where the gripper "
            "never reached the object. what_to_try_next should be 'move closer first'."}],
        schema=Reflection.model_json_schema(),
    )
    refl = Reflection.model_validate(data)
    print(f"  → what_went_wrong={refl.what_went_wrong[:60]!r} next={refl.what_to_try_next[:60]!r}")

    print("[smoke] 4. Memory round-trip (MemGPT tiers)...")
    from harness.memory import new_agent_memory
    mem = new_agent_memory("smoke", cfg.memory)
    nid = mem.archival_add(kind="lesson", summary="Always clip OSC actions to [-1,1]",
                           content="NaN actions crash the Warp sim", tags=["bug"])
    hits = mem.archival_search("OSC action clip NaN")
    print(f"  → stored note {nid}, search returned {len(hits)} hit(s)")
    if not hits:
        print("[smoke] FAIL: archival search returned nothing")
        return 1
    mem.recall_append("user", "hello")
    mem.recall_append("assistant", "hi")
    print(f"  → recall={len(mem.recall)} core={len(mem.core)} summary_len={len(mem.summary)}")

    print("[smoke] 5. Tool registry (no GPU)...")
    from harness.tools import ToolRegistry, ComputeRunner
    from harness.memory import get_archival_store
    from harness.llm import ToolCall
    arch = get_archival_store(cfg.memory.archival_db)
    cr = ComputeRunner(cfg.compute)
    reg = ToolRegistry(cr, arch, cfg.run_dir)
    # search_archival is in-process — safe to call without GPU.
    res = reg.dispatch(ToolCall(id="t1", name="search_archival",
                                arguments={"query": "OSC action clip", "top_k": 3}))
    print(f"  → search_archival ok={res.get('ok')} hits={len(res.get('hits', []))}")
    if not res.get("ok"):
        print("[smoke] FAIL: search_archival failed")
        return 1

    print("[smoke] 6. Agent construction (no LLM call)...")
    from harness.agents import AgentTeam
    cfg2 = cfg
    team = AgentTeam(cfg2, reg)
    print(f"  → team has agents: supervisor, rollout, reward, curriculum, hp, critic")
    print(f"  → supervisor settings: temp={team.supervisor.settings.temperature} "
          f"thinking={team.supervisor.settings.thinking_enabled} "
          f"num_predict={team.supervisor.settings.num_predict}")
    print(f"  → hp settings: temp={team.hp.settings.temperature} "
          f"thinking={team.hp.settings.thinking_enabled}")
    print(f"  → reward settings: temp={team.reward.settings.temperature} "
          f"thinking={team.reward.settings.thinking_enabled}")

    print("\n[smoke] ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())