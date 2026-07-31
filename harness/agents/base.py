"""ReAct + Reflexion base agent.

ReAct (Yao et al., arXiv:2210.03629): interleaved Thought → Action →
Observation loop. Reflexion (Shinn et al., arXiv:2303.11366): after a failed
episode, the agent writes a verbal self-critique that stays in an episodic
buffer and improves the next attempt (HumanEval 80%→91% without weight
updates).

This base class implements the tool-calling loop against Ollama GLM-5.2,
keeps MemGPT-tiered memory, and records reflections into archival memory so
the agent remembers *why* past approaches failed (the A-MEM Zettelkasten
linking means a new reward variant can be linked to the reflection that
motivated it).

Subclasses (supervisor, rollout, reward, curriculum, hp) set their own
system prompt + tool set + role-specific GLMSettings override.
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any, Callable, Optional

from harness.config import GLMSettings, settings_for
from harness.llm import OllamaClient, ChatResponse, ToolCall
from harness.memory import AgentMemory, new_agent_memory
from harness.config import MemoryConfig

log = logging.getLogger("harness.agent")


class Agent:
    """ReAct+Reflexion agent backed by Ollama GLM-5.2.

    A *run* is one outer-loop delegation. Inside a run the agent may issue
    multiple tool calls (the ReAct loop) until it produces a final answer.
    """

    def __init__(
        self,
        role: str,
        system_prompt: str,
        tools: list[dict[str, Any]],
        tool_dispatcher: Callable[[ToolCall], dict[str, Any]],
        base_llm: GLMSettings,
        mem_cfg: MemoryConfig,
        max_tool_steps: int = 8,
    ):
        self.role = role
        self.system_prompt = system_prompt.strip()
        self.tools = tools
        self.dispatch = tool_dispatcher
        self.settings = settings_for(role, base_llm)
        self.client = OllamaClient(self.settings)
        self.memory = new_agent_memory(f"agent:{role}", mem_cfg)
        self.max_tool_steps = max_tool_steps
        # Episodic Reflexion buffer (last N reflections, in-context)
        self.reflections: list[str] = []
        self.max_reflections = 5
        self._step = 0

    # -- public API --------------------------------------------------------

    def run(self, task: str, context: Optional[str] = None) -> dict[str, Any]:
        """Execute one delegation. Returns {"answer": str, "tool_calls": int,
        "thinking": str, "elapsed_s": float}.

        The agent loops Thought→Action→Observation until it emits a final
        answer with no tool call, or hits max_tool_steps.
        """
        t0 = time.time()
        self._step = 0
        messages = self._build_initial_messages(task, context)
        all_thinking: list[str] = []
        n_tool_calls = 0

        for _ in range(self.max_tool_steps):
            self._step += 1
            resp = self.client.chat(messages=messages, tools=self.tools)
            # Record the assistant turn in recall (for future-window summary)
            self.memory.recall_append("assistant", resp.content, resp.thinking)
            all_thinking.append(resp.thinking)
            messages.append(self._assistant_message(resp))

            if not resp.wants_tool:
                # Final answer.
                return {
                    "answer": resp.content,
                    "tool_calls": n_tool_calls,
                    "thinking": "\n---\n".join(t for t in all_thinking if t),
                    "elapsed_s": time.time() - t0,
                    "steps": self._step,
                }

            # Execute each requested tool and feed results back (ReAct).
            for tc in resp.tool_calls:
                n_tool_calls += 1
                result = self._safe_dispatch(tc)
                tool_msg = {
                    "role": "tool",
                    "content": json.dumps(result),
                    "name": tc.name,
                }
                messages.append(tool_msg)
                self.memory.recall_append("tool", json.dumps(result))

            log.info("[%s] step %d: %d tool calls", self.role, self._step, len(resp.tool_calls))

        # Hit the step budget: ask the model for a final synthesis.
        messages.append({
            "role": "system",
            "content": "Tool-step budget reached. Synthesize your final answer now from the observations above.",
        })
        resp = self.client.chat(messages=messages, tools=None)
        return {
            "answer": resp.content,
            "tool_calls": n_tool_calls,
            "thinking": "\n---\n".join(t for t in all_thinking + [resp.thinking] if t),
            "elapsed_s": time.time() - t0,
            "steps": self._step,
        }

    def reflect(self, outcome: dict[str, Any], success: bool) -> Optional[str]:
        """Reflexion: produce a verbal self-critique after a run.

        On failure, the agent writes a structured Reflection (what happened,
        what went wrong, what to try next) and stores it both in the episodic
        buffer (in-context for the next run) and in archival memory (long-term,
        Zettelkasten-linked to the relevant reward variant / curriculum note).
        """
        if success:
            return None
        from harness.schemas import Reflection
        schema = Reflection.model_json_schema()
        msgs = self._build_initial_messages(
            task="Reflect on the failed run and produce a structured self-critique.",
            context=json.dumps(outcome, default=str),
        )
        try:
            data = self.client.structured(msgs, schema)
            refl = Reflection.model_validate(data)
        except Exception as e:
            log.warning("[%s] reflection failed: %s", self.role, e)
            return None
        entry = (
            f"REFLECTION ({self.role}): what_happened={refl.what_happened}; "
            f"what_went_wrong={refl.what_went_wrong}; next={refl.what_to_try_next} "
            f"(confidence={refl.confidence:.2f})"
        )
        self.reflections.append(entry)
        if len(self.reflections) > self.max_reflections:
            self.reflections = self.reflections[-self.max_reflections :]
        # Persist to archival with links to the relevant notes (A-MEM linking).
        links = outcome.get("archival_links", [])
        self.memory.archival_add(
            kind="reflection",
            summary=f"Failed {self.role} run: {refl.what_went_wrong[:80]}",
            content=entry,
            tags=[self.role, "failure"],
            links=links,
        )
        return entry

    # -- internals ---------------------------------------------------------

    def _build_initial_messages(self, task: str, context: Optional[str]) -> list[dict[str, Any]]:
        sys = self.system_prompt
        sys += f"\n\n# Core memory (always true):\n{self.memory.core_text()}"
        if self.reflections:
            sys += "\n\n# Recent reflections (lessons from past failures):\n- " + "\n- ".join(self.reflections)
        if context:
            sys += f"\n\n# Provided context for this task:\n{context}"
        msgs: list[dict[str, Any]] = [{"role": "system", "content": sys}]
        # Include condensed recall (sliding window + summary) for continuity.
        msgs.extend(self.memory.recall_messages())
        msgs.append({"role": "user", "content": task})
        # The user turn also goes into recall so it survives the window.
        self.memory.recall_append("user", task)
        return msgs

    def _assistant_message(self, resp: ChatResponse) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": resp.content}
        if resp.tool_calls:
            msg["tool_calls"] = [tc.to_message() for tc in resp.tool_calls]
        return msg

    def _safe_dispatch(self, tc: ToolCall) -> dict[str, Any]:
        """Call the tool; never let an exception kill the agent loop."""
        try:
            result = self.dispatch(tc)
            if not isinstance(result, dict):
                result = {"result": str(result)}
            return result
        except Exception as e:
            log.error("[%s] tool %s raised: %s", self.role, tc.name, e)
            return {"error": f"{type(e).__name__}: {e}", "tool": tc.name}