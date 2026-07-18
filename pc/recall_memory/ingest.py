"""Ingestion pipeline — plan §1/§14a.

Raw input -> Normalizer (NMO + metadata) -> Enrichment (cheap extractors) ->
Fixed-size chunking (+ episodes side-channel for meetings) -> Embedding
(float[768] + Matryoshka int8[256]) -> Index writes (FTS5, vec_chunks,
entity_mentions, relations, metadata columns).
"""
from __future__ import annotations

import logging
import time
import numpy as np

from .backends import get_backend
from .chunker import build_transcript, chunk_document, chunk_meeting
from .config import RecallConfig
from .embeddings import matryoshka_coarse
from .extractors import extract_decisions, extract_entities
from .nmo import NMO, Chunk, Episode, new_id
from .store import MemoryStore
from .tracing import ensure_trace, step

log = logging.getLogger("recall.ingest")


class Ingestor:
    def __init__(self, store: MemoryStore, cfg: RecallConfig | None = None,
                 backend=None):
        self.store = store
        self.cfg = cfg or RecallConfig()
        self.backend = backend or get_backend(self.cfg)
        self.embedder = self.backend.embedder
        self._tokenize = self.backend.tokenizer.tokenize

    # ------------------------------------------------------------ meetings

    def ingest_meeting(self, meeting: dict) -> str:
        """meeting = {meeting_id?, title, device?, capture_device?, device_id?,
        capture_confidence?, participants?, start_time?, utterances: [
          {speaker, t_start, t_end, text, lang?, asr_provider?, asr_confidence?}]}

        Utterance times may be absolute epoch seconds or relative offsets from
        start_time (epoch). Returns the memory_id.
        """
        meeting_id = meeting.get("meeting_id") or new_id()
        with ensure_trace("ingest", source_type="meeting",
                          title=meeting.get("title"), meeting_id=meeting_id):
            return self._ingest_meeting_impl(meeting, meeting_id)

    def _ingest_meeting_impl(self, meeting: dict, meeting_id: str) -> str:
        start_time = float(meeting.get("start_time") or time.time())
        memory_id_placeholder = ""  # set after NMO creation

        episodes: list[Episode] = []
        for u in meeting["utterances"]:
            t0, t1 = float(u["t_start"]), float(u["t_end"])
            if t0 < 1e9:  # relative offsets -> absolute epoch
                t0, t1 = start_time + t0, start_time + t1
            episodes.append(Episode(
                episode_id=new_id(), meeting_id=meeting_id,
                memory_id=memory_id_placeholder, speaker=u.get("speaker", ""),
                t_start=t0, t_end=t1, text=u["text"], lang=u.get("lang", "en"),
                asr_provider=u.get("asr_provider", "whisper_local"),
                asr_confidence=float(u.get("asr_confidence", 1.0))))

        transcript, _ = build_transcript(episodes)
        participants = sorted({e.speaker for e in episodes if e.speaker})
        nmo = NMO.create(
            source_type="meeting",
            source=meeting.get("source", "arduino"),
            title=meeting.get("title", f"Meeting {meeting_id[:8]}"),
            content=transcript,
            created_at=int(start_time),
            device=meeting.get("capture_device", meeting.get("device", "arduino")),
            source_meta={
                "meeting_id": meeting_id,
                "participants": meeting.get("participants", participants),
                "start_time": start_time,
                "end_time": max((e.t_end for e in episodes), default=start_time),
                "capture_device": meeting.get("capture_device", "arduino"),
                "device_id": meeting.get("device_id", ""),
                "capture_confidence": float(meeting.get("capture_confidence", 1.0)),
                "asr_provider": meeting.get("asr_provider", "whisper_local"),
                "action_items": [], "decisions": [], "questions": [],
            })
        for ep in episodes:
            ep.memory_id = nmo.memory_id

        # Enrichment: decisions/action items per utterance (speaker attribution
        # is free at episode granularity); entities over the whole transcript.
        with step("enrich_decisions_actions", episodes=len(episodes)) as s:
            n_decisions = 0
            for ep in episodes:
                for d in extract_decisions(ep.text, speaker=ep.speaker):
                    bucket = "decisions" if d["kind"] == "decision" else "action_items"
                    nmo.source_meta[bucket].append(d["sentence"])
                    nmo.relations.append({
                        "subject": d["subject"], "predicate": d["predicate"],
                        "object": d["object"], "valid_at": int(ep.t_start)})
                    n_decisions += 1
            s.detail(extracted=n_decisions)

        with step("chunk_meeting", tokenizer=self.backend.tokenizer.name) as s:
            chunks = chunk_meeting(nmo, episodes, self._tokenize)
            s.detail(chunks=len(chunks))
        self._finish(nmo, chunks, episodes)
        return nmo.memory_id

    # ----------------------------------------------------------- documents

    def ingest_document(self, source_type: str, title: str, content: str,
                        source_meta: dict | None = None, source: str = "desktop",
                        **kw) -> str:
        """pdf / text / note / image (image = pre-extracted OCR block)."""
        with ensure_trace("ingest", source_type=source_type, title=title):
            nmo = NMO.create(source_type=source_type, source=source, title=title,
                             content=content, source_meta=source_meta or {}, **kw)
            with step("chunk_document", tokenizer=self.backend.tokenizer.name) as s:
                chunks = chunk_document(nmo, self._tokenize)
                s.detail(chunks=len(chunks))
            self._finish(nmo, chunks, [])
            return nmo.memory_id

    def ingest_github_file(self, repo: str, file_path: str, content: str,
                           source_meta: dict | None = None, **kw) -> str:
        with ensure_trace("ingest", source_type="github_repo",
                          title=f"{repo}/{file_path}"):
            meta = {"repo": repo, "file_path": file_path, **(source_meta or {})}
            nmo = NMO.create(source_type="github_repo", source="chrome_extension",
                             title=f"{repo}/{file_path}", content=content,
                             source_meta=meta, **kw)
            with step("chunk_document", tokenizer=self.backend.tokenizer.name) as s:
                chunks = chunk_document(nmo, self._tokenize)
                s.detail(chunks=len(chunks))
            self._finish(nmo, chunks, [])
            return nmo.memory_id

    # --------------------------------------------------------------- core

    def _finish(self, nmo: NMO, chunks: list[Chunk], episodes: list[Episode]):
        """Embed + write NMO, episodes, chunks, and every index in one txn."""
        t0 = time.perf_counter()
        with step("extract_entities", chunks=len(chunks)) as s:
            entity_sets = [extract_entities(c.text) for c in chunks]
            for ents in entity_sets:
                for e in ents:
                    if e["name"] not in [x["name"] for x in nmo.entities]:
                        nmo.entities.append({"name": e["name"], "type": e["type"],
                                             "entity_id": ""})
            s.detail(mentions=sum(len(e) for e in entity_sets))

        if chunks:
            with step("embed_documents", embedder=self.embedder.name,
                      chunks=len(chunks)) as s:
                texts = [c.text for c in chunks]
                batch_size = 32
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    embeddings.append(self.embedder.embed_documents(batch))
                full = np.vstack(embeddings)
                coarse = matryoshka_coarse(full)
                s.detail(dim_full=int(full.shape[-1]), dim_coarse=int(coarse.shape[-1]))

        with step("write_indexes") as s:
            mem_rowid = self.store.add_memory(nmo)
            for ep in episodes:
                self.store.add_episode(ep)
            for i, chunk in enumerate(chunks):
                cid = self.store.add_chunk(chunk, mem_rowid, full[i], coarse[i])
                for e in entity_sets[i]:
                    self.store.add_entity_mention(
                        e["name"], e["type"], cid, nmo.memory_id,
                        (e["start"], e["end"]))
            for rel in nmo.relations:
                r_chunk_id = None
                if rel.get("valid_at") is not None:
                    for chunk in chunks:
                        if chunk.t_start is not None and chunk.t_end is not None:
                            if int(chunk.t_start) <= rel["valid_at"] <= int(chunk.t_end + 0.999):
                                r_chunk_id = chunk.chunk_id
                                break
                self.store.add_relation(nmo.memory_id, rel["subject"],
                                        rel["predicate"], rel["object"],
                                        rel.get("valid_at"), chunk_id=r_chunk_id)
            self.store.commit()
            s.detail(memory=1, episodes=len(episodes), chunks=len(chunks),
                     relations=len(nmo.relations))
        log.info("ingested %s %r [%s]: %d chunks, %d episodes, %d relations "
                 "in %.0f ms", nmo.source_type, nmo.title[:50],
                 nmo.memory_id[:8], len(chunks), len(episodes),
                 len(nmo.relations), (time.perf_counter() - t0) * 1000)
