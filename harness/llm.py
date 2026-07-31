"""Ollama chat client for GLM-5.2 (:cloud) with tool-calling + thinking support.

Talks to Ollama's native /api/chat endpoint (http://localhost:11434) rather
than the OpenAI-compatible shim, because:
  1. The native endpoint surfaces the `thinking` field that GLM-5.2 emits
     (essential for logging agent reasoning and for Reflexion).
  2. Tool-calling is verified working on this endpoint for glm-5.2:cloud
     (see the smoke test in the research phase: it returns `tool_calls` with
     correct `function.name` + `function.arguments` JSON).
  3. We control options (temperature/top_p/top_k/num_ctx) directly.

The client is synchronous and uses `requests` (stdlib-available) so the
harness has zero hard dependencies beyond pydantic. httpx is used for
streaming if available, otherwise we fall back to requests streaming.

Reference: Ollama /api/chat payload:
  {
    "model": "glm-5.2:cloud",
    "messages": [{"role","content","tool_calls"?,"images"?}],
    "tools": [{"type":"function","function":{"name","description","parameters"}}],
    "stream": false,
    "options": {"temperature":0.6,"top_p":0.95,"top_k":40,"num_ctx":131072,...},
    "format": "json" | <json_schema>   # optional structured output
  }
Response:
  {
    "message": {"role":"assistant","content":"...","thinking":"...","tool_calls":[...]},
    "done": true, "done_reason":"stop"|"tool_calls",
    "total_duration":..., "prompt_eval_count":..., "eval_count":...
  }
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

from harness.config import GLMSettings

log = logging.getLogger("harness.llm")


@dataclass
class ToolCall:
    """A single tool call requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]

    def to_message(self) -> dict[str, Any]:
        """Serialize to the Ollama assistant-message `tool_calls` entry."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


@dataclass
class ChatResponse:
    """Parsed Ollama chat response."""
    content: str
    thinking: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = True
    done_reason: str = "stop"
    prompt_eval_count: int = 0
    eval_count: int = 0
    total_duration_ns: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)

    @property
    def elapsed_s(self) -> float:
        return self.total_duration_ns / 1e9


class OllamaClient:
    """Thin synchronous client for Ollama /api/chat targeting GLM-5.2.

    Stateless beyond the settings — safe to share across agents. Each agent
    gets its own `settings_for(role)` copy so temperature/thinking differ.
    """

    def __init__(self, settings: GLMSettings):
        self.s = settings
        self._session = requests.Session()

    @property
    def model(self) -> str:
        return self.s.model

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        format: Optional[dict[str, Any] | str] = None,
        options_override: Optional[dict[str, Any]] = None,
        stream: bool = False,
    ) -> ChatResponse:
        """One (non-streaming) chat turn. Returns a parsed ChatResponse.

        Args:
          messages: OpenAI-style message list. Tool-call assistant turns must
            include `tool_calls`; tool-result turns must have role="tool" and
            `content` = the JSON string result.
          tools: OpenAI-style function specs.
          format: "json" or a JSON-schema dict for structured output.
          options_override: per-call override of GLMSettings options.
        """
        payload: dict[str, Any] = {
            "model": self.s.model,
            "messages": messages,
            "stream": False,
            "options": options_override or self.s.to_ollama_options(),
        }
        if tools:
            payload["tools"] = tools
        if format is not None:
            payload["format"] = format

        url = f"{self.s.base_url.rstrip('/')}/api/chat"
        t0 = time.time()
        resp = self._session.post(
            url,
            json=payload,
            timeout=self.s.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            tool_calls.append(ToolCall(
                id=tc.get("id", f"call_{len(tool_calls)}"),
                name=fn.get("name", ""),
                arguments=args,
            ))

        out = ChatResponse(
            content=msg.get("content", "") or "",
            thinking=msg.get("thinking", "") or "",
            tool_calls=tool_calls,
            done=data.get("done", True),
            done_reason=data.get("done_reason", "stop"),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            eval_count=data.get("eval_count", 0),
            total_duration_ns=data.get("total_duration", 0),
            raw=data,
        )
        log.debug(
            "chat %s: %d in / %d out tokens, %.2fs, %d tool_calls, done=%s",
            self.s.model, out.prompt_eval_count, out.eval_count,
            time.time() - t0, len(tool_calls), out.done_reason,
        )
        return out

    def structured(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        options_override: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Force structured (JSON-schema) output and parse it.

        GLM-5.2 supports JSON-mode structured output (docs.z.ai lists it as a
        capability). We pass the schema via Ollama's `format` field AND inject
        a JSON-only system instruction (the cloud model sometimes emits prose
        + markdown fences even with `format` set, especially with thinking on,
        so we also strip fences and extract the first {...} block as a
        fallback).
        """
        # Inject a JSON-only instruction as the first system message.
        # Name the required fields explicitly — GLM-5.2 (cloud, thinking on)
        # sometimes invents its own field names unless we pin them.
        required = schema.get("required", [])
        props = list(schema.get("properties", {}).keys())
        field_hint = (
            f" Required fields: {', '.join(required)}. "
            f"Allowed fields: {', '.join(props)}."
        ) if props else ""
        instr = (
            "You MUST respond with ONLY a single valid JSON object matching the "
            "given schema. Do NOT wrap it in markdown code fences. Do NOT add "
            "any prose before or after the JSON. Output the JSON object and "
            "nothing else." + field_hint
        )
        msgs = list(messages)
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + instr}
        else:
            msgs.insert(0, {"role": "system", "content": instr})

        r = self.chat(messages=msgs, format=schema, options_override=options_override)
        if not r.content.strip():
            raise ValueError("structured call returned empty content")
        txt = r.content.strip()
        # Strip markdown code fences if present.
        if txt.startswith("```"):
            lines = txt.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            txt = "\n".join(lines).strip()
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # Fallback: extract the first balanced {...} block.
            start = txt.find("{")
            end = txt.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(txt[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                f"structured output not valid JSON. raw content:\n{r.content!r}"
            )

    def ping(self) -> bool:
        """Health check. Allow enough tokens for thinking + the answer."""
        try:
            r = self.chat(
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                options_override={**self.s.to_ollama_options(), "num_predict": 256},
            )
            # GLM-5.2 emits a `thinking` block first; with a tiny num_predict
            # the whole budget is spent thinking and content is empty. Accept
            # either a non-empty content OR a non-empty thinking as "alive".
            return bool(r.content.strip() or r.thinking.strip())
        except Exception as e:
            log.warning("ping failed: %s", e)
            return False