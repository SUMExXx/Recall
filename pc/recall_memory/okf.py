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
from .llm_validate import is_valid_llm_summary
from .store import MemoryStore
from .tracing import ensure_trace, step

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
        with ensure_trace("okf_generate_all"):
            counts = {"github_repo": 0, "meeting": 0, "pdf": 0}
            for repo in self._distinct_chunk_col("github_repo", "repo"):
                with step("okf:repo", repo=repo) as s:
                    built = self._gen_repo(repo)
                    s.detail(regenerated=built)
                if built:
                    counts["github_repo"] += 1
            for meeting_id in self._distinct_meeting_ids():
                with step("okf:meeting", meeting_id=meeting_id) as s:
                    built = self._gen_meeting(meeting_id)
                    s.detail(regenerated=built)
                if built:
                    counts["meeting"] += 1
            for doc in self._distinct_chunk_col("pdf", "document_title"):
                with step("okf:pdf", document_title=doc) as s:
                    built = self._gen_pdf(doc)
                    s.detail(regenerated=built)
                if built:
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
            prompt = ("Summarize this source in 2-3 sentences for a table of "
                      f"contents. Facts:\n{facts}")
            with step("okf:llm_polish", model=getattr(self.llm, "model",
                                                       self.llm.name)) as s:
                s.detail(prompt=prompt)
                try:
                    text = self.llm.generate(prompt, timeout=60)
                    s.detail(response=text)
                    if is_valid_llm_summary(text, facts):
                        return text
                    s.detail(rejected="failed output validation "
                                     "(refusal/off-topic/too short)")
                except Exception as e:
                    s.detail(fallback=f"LLM unavailable: {e}")
        return fallback

    def _unchanged(self, source_type: str, source_key: str,
                   content_hash: str) -> bool:
        """Checked BEFORE building a manifest, so unchanged sources never cost
        an LLM call on the periodic Dream pass."""
        existing = self.store.get_okf(source_type, source_key)
        return bool(existing and existing["content_hash"] == content_hash)

    def _save(self, source_type: str, source_key: str, manifest: dict,
              content_hash: str) -> bool:
        if self._unchanged(source_type, source_key, content_hash):
            return False
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
        sorted_imports = sorted(imports)
        sorted_functions = sorted(functions)
        content_hash = _hash(repo, *files, *ents, *sorted_imports, *sorted_functions)
        if self._unchanged("github_repo", repo, content_hash):
            return False
        facts = (f"repo={repo}; files={files}; key_identifiers={ents}; "
                 f"imports={sorted_imports}; functions={sorted_functions}")
        manifest = {
            "kind": "repo", "repo": repo,
            "files": files, "key_modules": files[:10],
            "dependencies": sorted_imports, "functions": sorted_functions,
            "entities": ents,
            "summary": self._maybe_summarize(
                f"Repository {repo} with {len(files)} indexed file(s).", facts),
        }
        return self._save("github_repo", repo, manifest, content_hash)

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
        participants = meta.get("participants", [])
        title = mem["title"] or ""
        summary = mem["summary"] or ""
        content_hash = _hash(meeting_id, title, summary, *participants, *decisions, *actions)
        if self._unchanged("meeting", meeting_id, content_hash):
            return False
        facts = (f"title={title}; participants={participants}; "
                 f"decisions={decisions}; action_items={actions}")
        manifest = {
            "kind": "meeting", "meeting_id": meeting_id, "title": title,
            "participants": participants,
            "agenda": summary[:200],
            "decisions": decisions, "action_items": actions,
            "questions": meta.get("questions", []),
            "summary": self._maybe_summarize(summary or title, facts),
        }
        return self._save("meeting", meeting_id, manifest, content_hash)

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
        author = meta.get("author", "")
        summary = (mem["summary"] if mem else "") or ""
        content_hash = _hash(document_title, author, summary, *map(str, pages), *headings)
        if self._unchanged("pdf", document_title, content_hash):
            return False
        facts = (f"title={document_title}; author={author}; "
                 f"pages={pages}; headings={headings}")
        manifest = {
            "kind": "pdf", "document_title": document_title,
            "author": author, "pages": pages,
            "headings": headings,
            "summary": self._maybe_summarize(
                summary or document_title, facts),
        }
        return self._save("pdf", document_title, manifest, content_hash)
