"""MemGPT-tiered memory + A-MEM Zettelkasten archival store.

Three tiers (Packer et al., MemGPT arXiv:2310.08560; Park et al., Generative
Agents arXiv:2304.03442):

  core   — in-context working memory (bounded token budget; trimmed via
           rolling summary so the agent never blows its context window).
  recall — recent verbatim transcript turns (sliding window).
  archival — long-term store retrieved on demand via tool calls
           (`search_archival`), the MemGPT "page fault" pattern.

The archival tier uses SQLite (stdlib, zero deps) with a simple TF-IDF +
keyword index. Each archival note is an A-MEM-style Zettelkasten note
(Xu et al., arXiv:2502.12110): it has a description, keywords, tags, and
*links* to related notes — so the agent can reason about why a past reward
variant was abandoned instead of re-deriving it.

Critical rule (research §2.4): raw JAX rollout tensors NEVER enter LLM
memory. Only summary statistics + verbal reflections are stored here.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from harness.config import MemoryConfig

log = logging.getLogger("harness.memory")

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _tok(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) > 1]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


# ---------------------------------------------------------------------------
# Archival store (SQLite + TF-IDF-ish keyword index + Zettelkasten links)
# ---------------------------------------------------------------------------

@dataclass
class ArchivalNote:
    note_id: str
    kind: str  # "reward_variant" | "reflection" | "curriculum" | "hp" | "lesson"
    summary: str
    content: str
    keywords: list[str]
    tags: list[str]
    links: list[str]  # ids of related notes
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ArchivalStore:
    """SQLite-backed long-term memory with keyword search + Zettelkasten links.

    Schema:
      notes(note_id PK, kind, summary, content, keywords JSON, tags JSON,
            links JSON, created_at, metadata JSON, tsv)
      links(src, dst)  -- denormalized for reverse lookup
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS notes (
                note_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '[]',
                tags TEXT NOT NULL DEFAULT '[]',
                links TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                tsv TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_notes_kind ON notes(kind);
            CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at);
            CREATE TABLE IF NOT EXISTS links (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                PRIMARY KEY(src, dst)
            );
            CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst);
            """
        )
        self._conn.commit()

    def add(self, note: ArchivalNote) -> None:
        tsv = " ".join(_tok(note.summary + " " + note.content))
        self._conn.execute(
            """INSERT OR REPLACE INTO notes
               (note_id, kind, summary, content, keywords, tags, links,
                created_at, metadata, tsv)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                note.note_id, note.kind, note.summary, note.content,
                json.dumps(note.keywords), json.dumps(note.tags),
                json.dumps(note.links), note.created_at,
                json.dumps(note.metadata), tsv,
            ),
        )
        for dst in note.links:
            self._conn.execute(
                "INSERT OR IGNORE INTO links(src,dst) VALUES(?,?)",
                (note.note_id, dst),
            )
        self._conn.commit()

    def get(self, note_id: str) -> Optional[ArchivalNote]:
        row = self._conn.execute(
            "SELECT * FROM notes WHERE note_id=?", (note_id,)
        ).fetchone()
        return self._row_to_note(row) if row else None

    def search(self, query: str, top_k: int = 5, kind: Optional[str] = None) -> list[ArchivalNote]:
        """Simple TF-IDF-ish keyword search. Good enough for agent memory;
        swap in a real vector index (e.g. sqlite-vec / faiss) if scale demands.
        """
        q_tokens = _tok(query)
        if not q_tokens:
            return []
        like = " AND ".join(["tsv LIKE ?"] * len(q_tokens))
        params = [f"%{t}%" for t in q_tokens]
        sql = f"SELECT * FROM notes WHERE {like}"
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(top_k * 4)
        rows = self._conn.execute(sql, params).fetchall()
        scored = []
        for row in rows:
            tsv_tokens = _tok(row["tsv"])
            overlap = len(set(q_tokens) & set(tsv_tokens))
            score = overlap / (1 + math.log(1 + len(tsv_tokens)))
            scored.append((score, row))
        scored.sort(key=lambda x: -x[0])
        return [self._row_to_note(r) for _, r in scored[:top_k] if _ > 0]

    def linked_from(self, note_id: str) -> list[ArchivalNote]:
        rows = self._conn.execute(
            """SELECT n.* FROM notes n JOIN links l ON l.dst=n.note_id
               WHERE l.src=? ORDER BY n.created_at DESC""",
            (note_id,),
        ).fetchall()
        return [self._row_to_note(r) for r in rows]

    def links_to(self, note_id: str) -> list[ArchivalNote]:
        rows = self._conn.execute(
            """SELECT n.* FROM notes n JOIN links l ON l.src=n.note_id
               WHERE l.dst=? ORDER BY n.created_at DESC""",
            (note_id,),
        ).fetchall()
        return [self._row_to_note(r) for r in rows]

    def list_kind(self, kind: str, limit: int = 50) -> list[ArchivalNote]:
        rows = self._conn.execute(
            "SELECT * FROM notes WHERE kind=? ORDER BY created_at DESC LIMIT ?",
            (kind, limit),
        ).fetchall()
        return [self._row_to_note(r) for r in rows]

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> ArchivalNote:
        return ArchivalNote(
            note_id=row["note_id"],
            kind=row["kind"],
            summary=row["summary"],
            content=row["content"],
            keywords=json.loads(row["keywords"]),
            tags=json.loads(row["tags"]),
            links=json.loads(row["links"]),
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]),
        )

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Core (in-context) + Recall (sliding window) per agent
# ---------------------------------------------------------------------------

@dataclass
class AgentMemory:
    """Per-agent memory tiers. The LLM client owns `core` + `recall` tokens;
    `archival` is shared across agents via the ArchivalStore.
    """
    agent_id: str
    cfg: MemoryConfig
    archival: ArchivalStore
    # core: bounded list of "memory blocks" (facts the agent always sees)
    core: list[str] = field(default_factory=list)
    # recall: recent transcript turns (role, content, thinking) verbatim
    recall: list[dict[str, str]] = field(default_factory=list)
    # running summary of rolled-out recall turns
    summary: str = ""

    # -- core memory (always in context) -----------------------------------

    def core_append(self, fact: str) -> None:
        self.core.append(fact)
        self._trim_core()

    def core_replace(self, idx: int, fact: str) -> None:
        if 0 <= idx < len(self.core):
            self.core[idx] = fact
        else:
            self.core.append(fact)
        self._trim_core()

    def core_text(self) -> str:
        return "\n".join(f"- {f}" for f in self.core) if self.core else "(none yet)"

    def _trim_core(self) -> None:
        # Keep core memory under ~2k tokens worth of text (rough 4 chars/token).
        budget = 8000
        kept: list[str] = []
        total = 0
        for f in reversed(self.core):
            if total + len(f) > budget:
                break
            kept.append(f)
            total += len(f)
        self.core = list(reversed(kept))

    # -- recall (sliding window) -------------------------------------------

    def recall_append(self, role: str, content: str, thinking: str = "") -> None:
        self.recall.append({"role": role, "content": content, "thinking": thinking})
        if len(self.recall) > self.cfg.recall_window:
            # Roll the oldest turns into the running summary.
            overflow = self.recall[: len(self.recall) - self.cfg.recall_window]
            self.recall = self.recall[len(self.recall) - self.cfg.recall_window :]
            self._roll_into_summary(overflow)

    def _roll_into_summary(self, turns: list[dict[str, str]]) -> None:
        # Lightweight extractive summary (no LLM call — keeps memory cheap).
        # Keep role + first 200 chars of each content; drop thinking.
        bits = []
        for t in turns:
            c = t["content"][:200].replace("\n", " ")
            bits.append(f"[{t['role']}] {c}")
        chunk = " || ".join(bits)
        if self.summary:
            self.summary = (self.summary + "\n" + chunk)[-self.cfg.summary_max_tokens * 4 :]
        else:
            self.summary = chunk

    def recall_messages(self) -> list[dict[str, Any]]:
        """Return recall as OpenAI-style messages for the next LLM call.

        Older turns are represented by a single 'system' summary message; the
        most recent `recall_window` turns are verbatim. This is the standard
        sliding-window-with-summary condenser (OpenHands/GLM-4 style).
        """
        msgs: list[dict[str, Any]] = []
        if self.summary:
            msgs.append({
                "role": "system",
                "content": f"[Prior context summary]\n{self.summary}",
            })
        for t in self.recall:
            m: dict[str, Any] = {"role": t["role"], "content": t["content"]}
            # Preserve tool_calls on assistant turns for API continuity.
            if "tool_calls" in t:
                m["tool_calls"] = t["tool_calls"]
            if t.get("name"):
                m["name"] = t["name"]
            msgs.append(m)
        return msgs

    # -- archival (long-term, tool-retrieved) ------------------------------

    def archival_add(
        self,
        kind: str,
        summary: str,
        content: str,
        keywords: list[str] | None = None,
        tags: list[str] | None = None,
        links: list[str] | None = None,
        note_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        nid = note_id or f"{kind}_{int(time.time()*1000)}"
        note = ArchivalNote(
            note_id=nid, kind=kind, summary=summary, content=content,
            keywords=keywords or _tok(summary)[:12],
            tags=tags or [], links=links or [],
            created_at=_now(), metadata=metadata or {},
        )
        self.archival.add(note)
        log.debug("archival+ %s [%s] %s", nid, kind, summary[:80])
        return nid

    def archival_search(self, query: str, top_k: int = 5, kind: Optional[str] = None) -> list[ArchivalNote]:
        return self.archival.search(query, top_k=top_k, kind=kind)

    def archival_get(self, note_id: str) -> Optional[ArchivalNote]:
        return self.archival.get(note_id)

    def archival_linked(self, note_id: str) -> list[ArchivalNote]:
        return self.archival.links_to(note_id)


# ---------------------------------------------------------------------------
# Shared store factory
# ---------------------------------------------------------------------------

_global_store: Optional[ArchivalStore] = None


def get_archival_store(db_path: str) -> ArchivalStore:
    global _global_store
    if _global_store is None or _global_store.db_path != db_path:
        _global_store = ArchivalStore(db_path)
    return _global_store


def new_agent_memory(agent_id: str, cfg: MemoryConfig) -> AgentMemory:
    return AgentMemory(agent_id=agent_id, cfg=cfg, archival=get_archival_store(cfg.archival_db))