"""OKF — Organized Knowledge Files (plan §7).

An OKF is NOT storage; it's a per-source-root *manifest / table of contents*
(one per repo, per meeting, per document), generated in the Dream tier and
regenerated when the source changes meaningfully. At query time the retriever
hands the OKF to the LLM *first* — the way a human skims a README before opening
files — so we avoid dumping a whole repo's chunks into context.

Extractive by default (works on the offline `hash` backend); if the active
backend exposes an LLM, the free-text `summary` field is polished by it.
"""
from __future__ import annotations

import hashlib
import json

from .config import RecallConfig
from .store import MemoryStore

_TOP_ENTITIES = 25


def _hash(*parts: str) -> str:
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


class OKFGenerator:
    def __init__(self, store: MemoryStore, cfg: RecallConfig | None = None,
                 llm=None):
        self.store = store
        self.cfg = cfg or RecallConfig()
        self.llm = llm

    # --------------------------------------------------------------- driver

    def generate_all(self) -> dict:
        """(Re)build every OKF whose underlying content changed. Returns counts."""
        counts = {"github_repo": 0, "meeting": 0, "pdf": 0}
        for repo in self._distinct_chunk_col("github_repo", "repo"):
            if self._gen_repo(repo):
                counts["github_repo"] += 1
        for meeting_id in self._distinct_meeting_ids():
            if self._gen_meeting(meeting_id):
                counts["meeting"] += 1
        for doc in self._distinct_chunk_col("pdf", "document_title"):
            if self._gen_pdf(doc):
                counts["pdf"] += 1
        self.store.commit()
        return counts

    # ---------------------------------------------------------- discovery

    def _distinct_chunk_col(self, source_type: str, col: str) -> list[str]:
        rows = self.store.db.execute(
            f"SELECT DISTINCT {col} AS v FROM chunks WHERE source_type=? "
            f"AND {col} IS NOT NULL", (source_type,)).fetchall()
        return [r["v"] for r in rows if r["v"]]

    def _distinct_meeting_ids(self) -> list[str]:
        rows = self.store.db.execute(
            """SELECT DISTINCT json_extract(source_meta, '$.meeting_id') AS mid
               FROM memories WHERE source_type='meeting' AND archived=0""").fetchall()
        return [r["mid"] for r in rows if r["mid"]]

    def _maybe_summarize(self, fallback: str, facts: str) -> str:
        """LLM polish when available; else the extractive fallback."""
        if self.llm is not None and getattr(self.llm, "available", False):
            try:
                return self.llm.generate(
                    "Summarize this source in 2-3 sentences for a table of "
                    f"contents. Facts:\n{facts}", timeout=60)
            except Exception:
                pass
        return fallback

    def _save(self, source_type: str, source_key: str, manifest: dict,
              content_hash: str) -> bool:
        existing = self.store.get_okf(source_type, source_key)
        if existing and existing["content_hash"] == content_hash:
            return False  # unchanged — skip regeneration
        self.store.upsert_okf(source_type, source_key, manifest, content_hash)
        return True

    # ------------------------------------------------------------ builders

    def _gen_repo(self, repo: str) -> bool:
        files = sorted({r["file_path"] for r in self.store.db.execute(
            "SELECT DISTINCT file_path FROM chunks WHERE repo=?", (repo,))
            if r["file_path"]})
        ents = [r["entity_text"] for r in self.store.db.execute(
            """SELECT em.entity_text, count(*) c FROM entity_mentions em
               JOIN chunks c ON c.chunk_id = em.chunk_id
               WHERE c.repo=? GROUP BY em.entity_id
               ORDER BY c DESC LIMIT ?""", (repo, _TOP_ENTITIES))]
        mems = self.store.db.execute(
            """SELECT DISTINCT m.memory_id, m.summary, m.source_meta FROM memories m
               JOIN chunks c ON c.memory_id = m.memory_id
               WHERE c.repo=? AND m.archived=0""", (repo,)).fetchall()
        imports, functions = set(), set()
        for m in mems:
            meta = json.loads(m["source_meta"] or "{}")
            imports.update(meta.get("imports", []))
            functions.update(meta.get("functions", []))
        facts = (f"repo={repo}; files={files}; key_identifiers={ents}; "
                 f"imports={sorted(imports)}; functions={sorted(functions)}")
        manifest = {
            "kind": "repo", "repo": repo,
            "files": files, "key_modules": files[:10],
            "dependencies": sorted(imports), "functions": sorted(functions),
            "entities": ents,
            "summary": self._maybe_summarize(
                f"Repository {repo} with {len(files)} indexed file(s).", facts),
        }
        return self._save("github_repo", repo, manifest,
                          _hash(repo, *files, *ents))

    def _gen_meeting(self, meeting_id: str) -> bool:
        mem = self.store.db.execute(
            """SELECT * FROM memories WHERE source_type='meeting' AND archived=0
               AND json_extract(source_meta, '$.meeting_id')=? LIMIT 1""",
            (meeting_id,)).fetchone()
        if mem is None:
            return False
        meta = json.loads(mem["source_meta"] or "{}")
        decisions = meta.get("decisions", [])
        actions = meta.get("action_items", [])
        facts = (f"title={mem['title']}; participants={meta.get('participants')}; "
                 f"decisions={decisions}; action_items={actions}")
        manifest = {
            "kind": "meeting", "meeting_id": meeting_id, "title": mem["title"],
            "participants": meta.get("participants", []),
            "agenda": (mem["summary"] or "")[:200],
            "decisions": decisions, "action_items": actions,
            "questions": meta.get("questions", []),
            "summary": self._maybe_summarize(mem["summary"] or mem["title"], facts),
        }
        return self._save("meeting", meeting_id, manifest,
                          _hash(meeting_id, *decisions, *actions))

    def _gen_pdf(self, document_title: str) -> bool:
        rows = self.store.db.execute(
            """SELECT DISTINCT page FROM chunks WHERE document_title=?
               AND page IS NOT NULL ORDER BY page""", (document_title,)).fetchall()
        pages = [r["page"] for r in rows]
        mem = self.store.db.execute(
            """SELECT * FROM memories m JOIN chunks c ON c.memory_id=m.memory_id
               WHERE c.document_title=? AND m.archived=0 LIMIT 1""",
            (document_title,)).fetchone()
        meta = json.loads(mem["source_meta"] or "{}") if mem else {}
        headings = [meta[k] for k in ("heading", "chapter") if meta.get(k)]
        facts = (f"title={document_title}; author={meta.get('author')}; "
                 f"pages={pages}; headings={headings}")
        manifest = {
            "kind": "pdf", "document_title": document_title,
            "author": meta.get("author", ""), "pages": pages,
            "headings": headings,
            "summary": self._maybe_summarize(
                (mem["summary"] if mem else "") or document_title, facts),
        }
        return self._save("pdf", document_title, manifest,
                          _hash(document_title, *map(str, pages), *headings))
