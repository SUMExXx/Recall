"""Dream-tier consolidation agent — plan §12, MVP jobs 1, 3, 4, 10.

job 1  duplicate detection      hash exact + embedding near-dup (cosine > 0.95)
job 3  contradiction detection  same (subject, predicate), different object
job 4  importance scoring       recency + mentions + decisions + entity degree
job 10 dual-mic reconciliation  time-overlap merge keyed on meeting_id

Runs as a background/idle job — never inline with ingestion.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import time

import numpy as np

from .config import DIM_FULL, RecallConfig
from .extractors import llm_extract_relation
from .llm_validate import is_valid_llm_summary
from .nmo import now_epoch
from .okf import OKFGenerator
from .store import MemoryStore
from .tracing import ensure_trace, step


def _h(*parts: str) -> str:
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


class Consolidator:
    def __init__(self, store: MemoryStore, cfg: RecallConfig | None = None,
                 ingestor=None):
        self.store = store
        self.cfg = cfg or RecallConfig()
        self.ingestor = ingestor  # needed by job 10 to re-ingest merged meetings
        # the LLM (if any) rides along on the ingestor's backend — used by the
        # summary ladder and the §6 relation-extraction fallback; None on hash.
        self.llm = getattr(getattr(ingestor, "backend", None), "llm", None)
        self._llm_attempted: set[int] = set()   # chunks already escalated once

    # ------------------------------------------------- job 1: duplicates

    def _mean_embedding(self, memory_id: str) -> np.ndarray | None:
        rows = self.store.db.execute(
            "SELECT emb_full FROM chunks WHERE memory_id=?", (memory_id,)).fetchall()
        vecs = [np.frombuffer(r["emb_full"], dtype=np.float32)
                for r in rows if r["emb_full"]]
        vecs = [v for v in vecs if v.shape[0] == DIM_FULL]
        if not vecs:
            return None
        mean = np.mean(vecs, axis=0)
        n = np.linalg.norm(mean)
        return mean / n if n else mean

    def job_dedup(self) -> int:
        """Merge exact-hash and near-duplicate memories; keep the more complete one."""
        merged = 0
        rows = self.store.db.execute(
            """SELECT memory_id, hash, source_type, length(content) AS len,
                      created_at FROM memories WHERE archived=0
               ORDER BY created_at""").fetchall()
        # exact hash groups
        by_hash: dict[str, list] = {}
        for r in rows:
            by_hash.setdefault(r["hash"], []).append(r)
        survivors = []
        for group in by_hash.values():
            keep = max(group, key=lambda r: r["len"])
            survivors.append(keep)
            for r in group:
                if r["memory_id"] != keep["memory_id"]:
                    self._merge_into(keep["memory_id"], r["memory_id"], "exact-dup")
                    merged += 1
        # near-dup within same source_type via mean chunk embedding
        by_type: dict[str, list] = {}
        for r in survivors:
            by_type.setdefault(r["source_type"], []).append(r)
        for group in by_type.values():
            embs = {r["memory_id"]: self._mean_embedding(r["memory_id"])
                    for r in group}
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    ea, eb = embs.get(a["memory_id"]), embs.get(b["memory_id"])
                    if ea is None or eb is None:
                        continue
                    if float(np.dot(ea, eb)) > self.cfg.near_dup_cosine:
                        keep, drop = (a, b) if a["len"] >= b["len"] else (b, a)
                        self._merge_into(keep["memory_id"], drop["memory_id"],
                                         "near-dup")
                        merged += 1
        self.store.commit()
        return merged

    def _merge_into(self, keep_id: str, drop_id: str, reason: str):
        """Archive `drop`, cascade its index entries away, log the merge."""
        with self.store.lock:
            self.store.append_history(keep_id, "merge", drop_id, f"absorbed ({reason})")
            chunk_ids = [r["chunk_id"] for r in self.store.db.execute(
                "SELECT chunk_id FROM chunks WHERE memory_id=?", (drop_id,))]
            self.store._delete_chunks(chunk_ids)
            self.store.db.execute("DELETE FROM episodes WHERE memory_id=?", (drop_id,))
            self.store.db.execute("UPDATE relations SET memory_id=? WHERE memory_id=?", (keep_id, drop_id))
            self.store.db.execute(
                "UPDATE memories SET archived=1, updated_at=? WHERE memory_id=?",
                (now_epoch(), drop_id))

    # --------------------------------------------- job 3: contradictions

    def job_contradictions(self) -> int:
        """Same (subject, predicate) with a different object -> close the old
        edge (invalid_at) instead of deleting it; log to history. Bi-temporal."""
        updated = 0
        rows = self.store.db.execute(
            """SELECT * FROM relations WHERE invalid_at IS NULL
               ORDER BY subject, predicate, COALESCE(valid_at, transaction_at)"""
        ).fetchall()
        by_key: dict[tuple, list] = {}
        for r in rows:
            by_key.setdefault(
                (r["subject"].lower(), r["predicate"].lower()), []).append(r)
        for group in by_key.values():
            for older, newer in zip(group, group[1:]):
                if older["object"].strip().lower() == newer["object"].strip().lower():
                    continue
                cutoff = newer["valid_at"] or newer["transaction_at"]
                self.store.db.execute(
                    "UPDATE relations SET invalid_at=? WHERE id=?",
                    (cutoff, older["id"]))
                self.store.append_history(
                    older["memory_id"],
                    f"relation:{older['subject']}:{older['predicate']}",
                    older["object"], newer["object"])
                updated += 1
        self.store.commit()
        return updated

    # ------------------------------------------------ job 4: importance

    def job_importance(self) -> int:
        """importance = 0.4*recency + 0.3*mention_density + 0.2*has_decision
        + 0.1*entity_degree — recomputed on memories and mirrored onto chunks."""
        now = time.time()
        n = 0
        for m in self.store.db.execute(
                "SELECT memory_id, created_at FROM memories WHERE archived=0"):
            mid = m["memory_id"]
            age_days = max(0.0, (now - m["created_at"]) / 86400.0)
            recency = math.pow(0.5, age_days / 30.0)
            mentions = self.store.db.execute(
                "SELECT count(*) c FROM entity_mentions WHERE memory_id=?",
                (mid,)).fetchone()["c"]
            n_chunks = self.store.db.execute(
                "SELECT count(*) c FROM chunks WHERE memory_id=?",
                (mid,)).fetchone()["c"] or 1
            density = min(1.0, (mentions / n_chunks) / 10.0)
            has_decision = 1.0 if self.store.db.execute(
                "SELECT 1 FROM relations WHERE memory_id=? AND predicate='decided' "
                "LIMIT 1", (mid,)).fetchone() else 0.0
            degree = min(1.0, self.store.db.execute(
                "SELECT count(DISTINCT entity_id) c FROM entity_mentions "
                "WHERE memory_id=?", (mid,)).fetchone()["c"] / 20.0)
            importance = round(0.4 * recency + 0.3 * density
                               + 0.2 * has_decision + 0.1 * degree, 4)
            self.store.db.execute(
                "UPDATE memories SET importance=? WHERE memory_id=?",
                (importance, mid))
            self.store.db.execute(
                "UPDATE chunks SET importance=? WHERE memory_id=?",
                (importance, mid))
            n += 1
        self.store.commit()
        return n

    # -------------------------------------- job 10: dual-mic reconciliation

    def job_dualmic(self) -> int:
        """Merge multi-device captures of the same meeting_id.

        For overlapping time windows keep the higher-asr_confidence episode
        (real score comparison, not a hardcoded device preference); stitch
        non-overlapping gaps from the secondary device; re-ingest the merged
        transcript as the canonical memory and archive the sources.
        """
        if self.ingestor is None:
            raise RuntimeError("job_dualmic needs an Ingestor for re-ingestion")
        merged_meetings = 0
        groups = self.store.db.execute(
            """SELECT m.json_meeting AS meeting_id FROM (
                 SELECT json_extract(source_meta, '$.meeting_id') AS json_meeting,
                        count(*) AS n
                 FROM memories
                 WHERE source_type='meeting' AND archived=0
                 GROUP BY json_meeting HAVING n >= 2) m""").fetchall()
        for g in groups:
            meeting_id = g["meeting_id"]
            mems = self.store.db.execute(
                """SELECT memory_id, device, title, source_meta FROM memories
                   WHERE source_type='meeting' AND archived=0
                     AND json_extract(source_meta, '$.meeting_id') = ?
                   ORDER BY created_at""", (meeting_id,)).fetchall()
            if len(mems) < 2:
                continue
            eps = self.store.db.execute(
                """SELECT * FROM episodes WHERE meeting_id=? ORDER BY t_start""",
                (meeting_id,)).fetchall()
            chosen = self._reconcile_episodes(eps)
            meta0 = json.loads(mems[0]["source_meta"])
            new_meeting = {
                "meeting_id": meeting_id + "-merged",
                "title": mems[0]["title"] + " (reconciled)",
                "start_time": min(e["t_start"] for e in chosen),
                "capture_device": "merged",
                "device_id": "+".join(sorted({m["device"] for m in mems})),
                "participants": meta0.get("participants", []),
                "utterances": [
                    {"speaker": e["speaker"], "t_start": e["t_start"],
                     "t_end": e["t_end"], "text": e["text"], "lang": e["lang"],
                     "asr_provider": e["asr_provider"],
                     "asr_confidence": e["asr_confidence"]}
                    for e in chosen],
            }
            new_id_ = self.ingestor.ingest_meeting(new_meeting)
            for m in mems:
                self._merge_into(new_id_, m["memory_id"], "dual-mic reconcile")
                self.store.db.execute(
                    "DELETE FROM episodes WHERE memory_id=?", (m["memory_id"],))
            merged_meetings += 1
        self.store.commit()
        return merged_meetings

    def _reconcile_episodes(self, eps: list) -> list:
        """Time-overlap dedup: within an overlap, keep higher asr_confidence.
        Near-identical text (fuzzy > threshold) counts as the same utterance."""
        chosen: list = []
        for ep in eps:
            dup_of = None
            for i, kept in enumerate(chosen):
                overlap = (min(ep["t_end"], kept["t_end"])
                           - max(ep["t_start"], kept["t_start"]))
                if overlap <= 0:
                    continue
                sim = difflib.SequenceMatcher(
                    None, ep["text"].lower(), kept["text"].lower()).ratio()
                if sim >= self.cfg.dualmic_fuzzy_threshold:
                    dup_of = i
                    break
            if dup_of is None:
                chosen.append(ep)
            elif ep["asr_confidence"] > chosen[dup_of]["asr_confidence"]:
                chosen[dup_of] = ep
        return sorted(chosen, key=lambda e: e["t_start"])

    # ------------------------------------------- job 2: entity resolution

    @staticmethod
    def _same_entity(a: str, b: str) -> bool:
        a, b = a.lower(), b.lower()
        if a == b:
            return False
        if a in b or b in a:                       # postgres ⊂ postgresql, jwt ⊂ jwts
            return abs(len(a) - len(b)) <= 4
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.9

    def job_entity_resolution(self) -> int:
        """Cluster entity ids that denote the same thing and rewrite mentions to
        one canonical id (the most-frequent variant). This is what keeps the
        cross-source entity/reverse index from fragmenting ("GPT"/"ChatGPT")."""
        rows = self.store.db.execute(
            """SELECT entity_id, entity_text, count(*) c FROM entity_mentions
               GROUP BY entity_id ORDER BY c DESC, entity_id""").fetchall()
        ids = [(r["entity_id"], r["entity_text"]) for r in rows]
        parent = {eid: eid for eid, _ in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merges = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if find(ids[i][0]) == find(ids[j][0]):
                    continue
                if self._same_entity(ids[i][1], ids[j][1]):
                    parent[find(ids[j][0])] = find(ids[i][0])  # into higher-count
                    merges += 1
        if not merges:
            return 0
        rep: dict[str, tuple[str, str]] = {}
        for eid, txt in ids:                       # first per root = highest count
            rep.setdefault(find(eid), (eid, txt))
        for eid, _ in ids:
            canon_id, canon_txt = rep[find(eid)]
            if eid != canon_id:
                self.store.db.execute(
                    "UPDATE entity_mentions SET entity_id=?, entity_text=? "
                    "WHERE entity_id=?", (canon_id, canon_txt, eid))
        self.store.commit()
        return merges

    # -------------------------------------------- job 5: community detection

    def job_communities(self) -> int:
        """Group memories that co-mention entities into clusters (union-find),
        labelled by their most-shared entity. Lets retrieval pull a whole
        cluster ("everything touching Redis + caching")."""
        rows = self.store.db.execute(
            """SELECT DISTINCT em.entity_id, em.memory_id FROM entity_mentions em
               JOIN memories m ON m.memory_id = em.memory_id
               WHERE m.archived = 0""").fetchall()
        ent_to_mems: dict[str, set] = {}
        for r in rows:
            ent_to_mems.setdefault(r["entity_id"], set()).add(r["memory_id"])
        parent: dict[str, str] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for mems in ent_to_mems.values():
            members = list(mems)
            root = find(members[0])
            for m in members[1:]:
                parent[find(m)] = root
        comps: dict[str, set] = {}
        for m in list(parent):
            comps.setdefault(find(m), set()).add(m)
        communities: list[tuple[str, list[str]]] = []
        for members in comps.values():
            if len(members) < 2:
                continue
            label = max(ent_to_mems,
                        key=lambda e: len(ent_to_mems[e] & members))
            communities.append((label, sorted(members)))
        self.store.replace_communities(communities)
        return len(communities)

    # --------------------------------------------- job 6: summary ladder

    def _summarize(self, title: str, facts: str) -> str:
        if self.llm is not None and getattr(self.llm, "available", False):
            prompt = (
                "You are a summarization assistant. Summarize the following meeting details "
                "in 2-3 sentences. Do not explain concepts or hallucinate details not present.\n\n"
                f"Meeting Title: {title}\nFacts: {facts}"
            )
            with step("summarize:llm_polish", model=getattr(
                    self.llm, "model", self.llm.name)) as s:
                s.detail(prompt=prompt)
                try:
                    text = self.llm.generate(prompt, timeout=60)
                    s.detail(response=text)
                    if is_valid_llm_summary(text, facts or title):
                        return text
                    s.detail(rejected="failed output validation "
                                     "(refusal/off-topic/too short)")
                except Exception as e:
                    s.detail(fallback=f"LLM unavailable: {e}")
        return facts or title

    def job_summaries(self) -> int:
        """Chunk -> meeting -> daily ladder (plan §12 job 6). Regenerates only
        changed levels — the content hash is checked BEFORE any LLM call, so an
        idle Dream pass costs zero generations. Returns summaries (re)written."""
        n = 0
        for mem in self.store.db.execute(
                "SELECT * FROM memories WHERE source_type='meeting' AND archived=0"):
            meta = json.loads(mem["source_meta"] or "{}")
            parts = []
            if meta.get("decisions"):
                parts.append("Decisions: " + "; ".join(meta["decisions"]))
            if meta.get("action_items"):
                parts.append("Actions: " + "; ".join(meta["action_items"]))
            extractive = " ".join(parts) or mem["title"]
            key = meta.get("meeting_id") or mem["memory_id"]
            h = _h(key, extractive)
            existing = self.store.get_summary("meeting", key)
            if existing and existing["content_hash"] == h:
                continue                        # unchanged — skip the LLM
            text = self._summarize(mem["title"], extractive)
            self.store.upsert_summary("meeting", key, text, h)
            if not (mem["summary"] or "").strip():
                self.store.db.execute(
                    "UPDATE memories SET summary=? WHERE memory_id=?",
                    (text[:400], mem["memory_id"]))
            n += 1
        days: dict[str, list] = {}
        for mem in self.store.db.execute(
                "SELECT title, created_at FROM memories WHERE archived=0"):
            day = time.strftime("%Y-%m-%d", time.localtime(mem["created_at"]))
            days.setdefault(day, []).append(mem["title"])
        for day, titles in days.items():
            text = " · ".join(titles[:20])
            h = _h(day, text)
            existing = self.store.get_summary("daily", day)
            if existing and existing["content_hash"] == h:
                continue
            self.store.upsert_summary("daily", day, text, h)
            n += 1
        self.store.commit()
        return n

    # ---------------------------------- job 7: re-embed on model upgrade

    def job_reembed(self, batch_size: int = 200) -> int:
        """Batched re-embedding when the model/pipeline version changes (plan
        §12 job 7). Chunks tagged with an older `processing_version` get
        fresh float[768]+int8[256] vectors from the CURRENT backend embedder;
        a no-op once every chunk is already current. Low priority — bounded
        to `batch_size` chunks per Dream pass so a large backlog (after a
        model swap) doesn't turn one consolidation cycle into a multi-minute
        re-embed of the whole store."""
        from .config import PROCESSING_VERSION
        from .embeddings import matryoshka_coarse
        embedder = getattr(getattr(self.ingestor, "backend", None), "embedder", None)
        if embedder is None:
            return 0
        rows = self.store.db.execute(
            """SELECT c.chunk_id, c.text, c.created_at, c.source_type,
                      m.rowid AS mem_rowid
               FROM chunks c JOIN memories m ON m.memory_id = c.memory_id
               WHERE c.processing_version != ? LIMIT ?""",
            (PROCESSING_VERSION, batch_size)).fetchall()
        if not rows:
            return 0
        texts = [r["text"] for r in rows]
        sub_batch = 32
        embeddings = []
        for i in range(0, len(texts), sub_batch):
            batch = texts[i:i + sub_batch]
            embeddings.append(embedder.embed_documents(batch))
        full = np.vstack(embeddings)
        coarse = matryoshka_coarse(full)
        for i, r in enumerate(rows):
            self.store.db.execute(
                "UPDATE chunks SET emb_full=?, processing_version=? WHERE chunk_id=?",
                (full[i].astype(np.float32).tobytes(), PROCESSING_VERSION,
                 r["chunk_id"]))
            self.store.db.execute(
                "UPDATE vec_chunks SET emb_coarse=? WHERE chunk_id=?",
                (coarse[i].astype(np.int8).tobytes(), r["chunk_id"]))
        self.store.commit()
        return len(rows)

    # ---------------------------------------- job 8: dead-memory archival

    def job_archival(self, max_age_days: float = 90.0,
                     importance_floor: float = 0.15) -> int:
        """Low-importance + old + never-referenced -> archive out of the hot
        indexes. Never hard-deletes (plan §12 job 8)."""
        now = time.time()
        n = 0
        for m in self.store.db.execute(
                "SELECT memory_id, created_at, importance, tags FROM memories "
                "WHERE archived=0"):
            age_days = (now - m["created_at"]) / 86400.0
            refs = self.store.db.execute(
                "SELECT count(*) c FROM entity_mentions WHERE memory_id=?",
                (m["memory_id"],)).fetchone()["c"]
            bookmarked = "bookmarked" in (m["tags"] or "")
            if (not bookmarked and age_days > max_age_days
                    and (m["importance"] or 0.0) < importance_floor and refs == 0):
                self.store.db.execute(
                    "UPDATE memories SET archived=1, updated_at=? WHERE memory_id=?",
                    (now_epoch(), m["memory_id"]))
                n += 1
        self.store.commit()
        return n

    # ------------------------------------------------ job 9: graph repair

    def job_graph_repair(self) -> int:
        """Sweep orphaned chunks / entity mentions / relations left by merges
        or partial deletes (plan §12 job 9)."""
        repaired = 0
        orphans = [r["chunk_id"] for r in self.store.db.execute(
            "SELECT chunk_id FROM chunks WHERE memory_id NOT IN "
            "(SELECT memory_id FROM memories)")]
        self.store._delete_chunks(orphans)
        repaired += len(orphans)
        repaired += self.store.db.execute(
            "DELETE FROM entity_mentions WHERE chunk_id NOT IN "
            "(SELECT chunk_id FROM chunks)").rowcount
        repaired += self.store.db.execute(
            "DELETE FROM relations WHERE memory_id NOT IN "
            "(SELECT memory_id FROM memories)").rowcount
        self.store.commit()
        return repaired

    # ---------------------------- §6 relation extraction (LLM fallback)

    def job_llm_relations(self, limit: int = 20) -> int:
        """Escalate meeting chunks with no extracted relations to the LLM for a
        (subject, predicate, object) triple. No-op without an available LLM."""
        if self.llm is None or not getattr(self.llm, "available", False):
            return 0
        added = 0
        rows = self.store.db.execute(
            """SELECT chunk_id, memory_id, text FROM chunks
               WHERE source_type='meeting'
                 AND chunk_id NOT IN (SELECT DISTINCT chunk_id FROM relations WHERE chunk_id IS NOT NULL)
               LIMIT ?""", (limit,)).fetchall()
        for r in rows:
            if r["chunk_id"] in self._llm_attempted:
                continue   # once per chunk — don't re-pay LLM cost every pass
            self._llm_attempted.add(r["chunk_id"])
            rel = llm_extract_relation(r["text"], self.llm)
            if rel:
                self.store.add_relation(r["memory_id"], rel["subject"],
                                        rel["predicate"], rel["object"],
                                        chunk_id=r["chunk_id"])
                added += 1
        self.store.commit()
        return added

    # ------------------------------------------------- OKF (plan §7)

    def generate_okf(self) -> dict:
        return OKFGenerator(self.store, self.cfg, self.llm).generate_all()

    # ---------------------------------------------------------------- all

    def _run_job(self, name: str, fn) -> int | dict:
        """One consolidation job, timed as its own trace step (background
        Dream-tier work — see plan §12)."""
        with step(f"consolidate:{name}") as s:
            result = fn()
            s.detail(result=result)
            return result

    def run_mvp(self, should_abort=None) -> dict:
        """Hackathon MVP: jobs 1, 3, 4, 10.

        `should_abort`, when given, is polled BETWEEN jobs: the scheduler passes
        an "is a capture happening right now?" check so a background Dream pass
        yields the store to a fresh ingestion the instant the user speaks,
        instead of making a just-spoken memory's dot wait for the whole pass
        (LLM summaries/OKF included) to finish. A yielded pass just resumes on
        the next idle cycle — no job is skipped permanently."""
        jobs = []
        if self.ingestor:
            jobs.append(("dualmic_merged", "job10_dualmic", self.job_dualmic))
        jobs += [
            ("duplicates_merged", "job1_dedup", self.job_dedup),
            ("contradictions_closed", "job3_contradictions", self.job_contradictions),
            ("importance_updated", "job4_importance", self.job_importance),
        ]
        out: dict = {"dualmic_merged": 0}
        with ensure_trace("consolidate", scope="mvp"):
            for key, name, fn in jobs:
                if should_abort and should_abort():
                    out["aborted"] = True
                    break
                out[key] = self._run_job(name, fn)
            return out

    def run_full(self, should_abort=None) -> dict:
        """The full Dream tier: MVP jobs + entity resolution, communities, the
        summary ladder, archival, graph repair, LLM relations, and OKF regen.

        See run_mvp for `should_abort` — the same check gates every job here so
        the whole (potentially multi-second, LLM-heavy) pass steps aside for a
        fresh capture."""
        with ensure_trace("consolidate", scope="full"):
            out = self.run_mvp(should_abort)
            jobs = [
                ("entities_resolved", "job2_entity_resolution", self.job_entity_resolution),
                ("communities", "job5_communities", self.job_communities),
                ("summaries", "job6_summaries", self.job_summaries),
                ("reembedded", "job7_reembed", self.job_reembed),
                ("archived", "job8_archival", self.job_archival),
                ("graph_repaired", "job9_graph_repair", self.job_graph_repair),
                ("llm_relations", "job_llm_relations", self.job_llm_relations),
                ("okf", "okf_generate", self.generate_okf),
            ]
            for key, name, fn in jobs:
                if out.get("aborted") or (should_abort and should_abort()):
                    out["aborted"] = True
                    break
                out[key] = self._run_job(name, fn)
            return out
