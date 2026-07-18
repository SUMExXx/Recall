"""MCP Server (stdio) — §4 GW. Exposes the memory engine to Claude Desktop /
Claude Code as tools, built on the official `mcp` SDK (FastMCP):

  recall_search_memory   retrieval only (citations + snippets)
  recall_ask             route + retrieve + LLM answer
  recall_bookmark_moment boost importance of the most recent memory
  recall_forget_range    the forget button over a meeting time range
  recall_dump            list stored memories

Register with:
  claude mcp add recall -- <venv-python> -m hub.mcp_server --db <path>

Run: python -m hub.mcp_server [--db recall.db] [--backend ollama|npu|hash]
"""
from __future__ import annotations

import argparse
import time

from mcp.server.fastmcp import FastMCP

from recall_memory.backends import get_backend
from recall_memory.config import RecallConfig
from recall_memory.retrieval import Retriever
from recall_memory.store import MemoryStore


class RecallTools:
    """The engine behind the MCP tools. Kept separate from the FastMCP wiring
    so the tool logic is directly unit-testable without a transport."""

    def __init__(self, db_path: str, cfg: RecallConfig | None = None):
        self.cfg = cfg or RecallConfig(db_path=db_path)
        self.store = MemoryStore(db_path, check_same_thread=False)
        self.backend = get_backend(self.cfg)
        self.retriever = Retriever(self.store, self.cfg, self.backend)

    def search_memory(self, query: str, k: int = 6) -> str:
        ctxs = self.retriever.retrieve(query, top_k=k)
        if not ctxs:
            return "No matching memories."
        return "\n\n".join(
            f"{c.citation()} ({c.source_type}: {c.chunk_title or c.title}, score "
            f"{c.score:.3f})\n{c.text[:400]}" for c in ctxs)

    def ask(self, query: str, k: int = 6) -> str:
        out = self.retriever.ask(query, top_k=k)
        srcs = "\n".join(
            f"  {c.citation()} {c.source_type}: {c.chunk_title or c.title}"
            for c in out.get("contexts", []))
        return f"{out.get('answer')}\n\nSources:\n{srcs or '  (none)'}"

    def bookmark_moment(self) -> str:
        row = self.store.db.execute(
            """SELECT memory_id, title FROM memories WHERE archived=0
               ORDER BY created_at DESC LIMIT 1""").fetchone()
        if not row:
            return "No memories to bookmark."
        self.store.db.execute(
            "UPDATE memories SET importance=1.0, "
            "tags=json_insert(tags,'$[#]','bookmarked') WHERE memory_id=?",
            (row["memory_id"],))
        self.store.db.execute(
            "UPDATE chunks SET importance=1.0 WHERE memory_id=?",
            (row["memory_id"],))
        self.store.commit()
        return f"Bookmarked: {row['title']} ({row['memory_id'][:8]})"

    def forget_range(self, meeting_id: str, last_minutes: float = 5) -> str:
        t_to = time.time()
        out = self.store.forget_time_range(
            meeting_id, t_to - float(last_minutes) * 60.0, t_to)
        return (f"Forgotten from {meeting_id}: "
                f"{out['episodes']} episodes, {out['chunks']} chunks.")

    def dump(self) -> str:
        rows = self.store.db.execute(
            """SELECT memory_id, source_type, title, importance, archived
               FROM memories ORDER BY created_at DESC""").fetchall()
        return "\n".join(
            f"{r['memory_id'][:8]} [{r['source_type']:12s}] "
            f"imp={r['importance']:.2f}{' (archived)' if r['archived'] else ''} "
            f"{r['title']}" for r in rows) or "No memories stored."


def build_server(db_path: str, cfg: RecallConfig | None = None):
    """Wire the five tools onto a FastMCP server. Returns (mcp, tools)."""
    tools = RecallTools(db_path, cfg)
    mcp = FastMCP("recall")

    @mcp.tool(description="Search Recall's on-device memory (meetings, code, "
                          "PDFs, whiteboards). Returns cited snippets.")
    def recall_search_memory(query: str, k: int = 6) -> str:
        return tools.search_memory(query, k)

    @mcp.tool(description="Ask Recall a question; answers from stored memories "
                          "with citations (local LLM synthesis).")
    def recall_ask(query: str, k: int = 6) -> str:
        return tools.ask(query, k)

    @mcp.tool(description="Bookmark the most recent memory (importance -> 1.0).")
    def recall_bookmark_moment() -> str:
        return tools.bookmark_moment()

    @mcp.tool(description="Forget: erase all episodes and chunks of a meeting "
                          "in the last N minutes. Irreversible.")
    def recall_forget_range(meeting_id: str, last_minutes: float = 5) -> str:
        return tools.forget_range(meeting_id, last_minutes)

    @mcp.tool(description="List stored memories (id, type, title, importance).")
    def recall_dump() -> str:
        return tools.dump()

    return mcp, tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="recall.db")
    ap.add_argument("--backend", default=None, choices=["npu", "ollama", "hash"])
    args = ap.parse_args()
    overrides = {"db_path": args.db}
    if args.backend:
        overrides["backend"] = args.backend
    mcp, _ = build_server(args.db, RecallConfig(**overrides))
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
