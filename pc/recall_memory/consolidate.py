"""Dream-tier consolidation agent — plan §12, MVP jobs 1, 3, 4, 10.

job 1  duplicate detection      hash exact + embedding near-dup (cosine > 0.95)
job 3  contradiction detection  same (subject, predicate), different object
job 4  importance scoring       recency + mentions + decisions + entity degree
job 10 dual-mic reconciliation  time-overlap merge keyed on meeting_id

Runs as a background/idle job — never inline with ingestion.
"""
from __future__ import annotations

import difflib
import json
import math
import time

import numpy as np

from .config import DIM_FULL, RecallConfig
from .nmo import now_epoch
from .store import MemoryStore


class Consolidator:
    def __init__(self, store: MemoryStore, cfg: RecallConfig | None = None,
                 ingestor=None):
        self.store = store
        self.cfg = cfg or RecallConfig()
        self.ingestor = ingestor  # needed by job 10 to re-ingest merged meetings

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
        self.store.append_history(keep_id, "merge", drop_id, f"absorbed ({reason})")
        chunk_ids = [r["chunk_id"] for r in self.store.db.execute(
            "SELECT chunk_id FROM chunks WHERE memory_id=?", (drop_id,))]
        self.store._delete_chunks(chunk_ids)
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

    # ---------------------------------------------------------------- all

    def run_mvp(self) -> dict:
        out = {"dualmic_merged": self.job_dualmic() if self.ingestor else 0}
        out["duplicates_merged"] = self.job_dedup()
        out["contradictions_closed"] = self.job_contradictions()
        out["importance_updated"] = self.job_importance()
        return out
