"""SQLite storage — all tables and indexes from plan §4d.

memories        NMOs: universal metadata as real columns + source_meta JSON blob
episodes        immutable utterance atoms (meetings only)
chunks          universal retrieval unit, 3 metadata groups + emb_full float32 BLOB
entity_mentions cross-source bridge (indexed columns = the "hash index")
relations       graph edges with bi-temporal fields (valid_at / invalid_at / transaction_at)
fts_chunks      FTS5 BM25 over chunk text
vec_chunks      int8[256] Matryoshka coarse vectors, brute-force KNN via numpy
                (not sqlite-vec/vec0 — that package has no win_arm64 wheel)
"""
import json
import re
import sqlite3
import threading

import numpy as np

from .config import DIM_COARSE, DIM_FULL
from .nmo import NMO, Chunk, Episode, now_epoch

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS memories (
  memory_id   TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  source      TEXT NOT NULL,
  project     TEXT,
  title       TEXT NOT NULL,
  content     TEXT NOT NULL,
  summary     TEXT DEFAULT '',
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL,
  language    TEXT DEFAULT 'en',
  importance  REAL DEFAULT 0.0,
  confidence  REAL DEFAULT 1.0,
  user        TEXT DEFAULT 'default',
  device      TEXT DEFAULT '',
  hash        TEXT NOT NULL,
  processing_version TEXT NOT NULL,
  source_meta TEXT DEFAULT '{{}}',
  entities    TEXT DEFAULT '[]',
  tags        TEXT DEFAULT '[]',
  history     TEXT DEFAULT '[]',
  parent      TEXT,
  attachments TEXT DEFAULT '[]',
  children    TEXT DEFAULT '[]',
  archived    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mem_type_time ON memories(source_type, created_at);
CREATE INDEX IF NOT EXISTS idx_mem_hash ON memories(hash);
CREATE INDEX IF NOT EXISTS idx_mem_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_mem_meeting_id ON memories(json_extract(source_meta, '$.meeting_id')) WHERE source_type = 'meeting';

CREATE TABLE IF NOT EXISTS episodes (
  episode_id TEXT PRIMARY KEY,
  meeting_id TEXT NOT NULL,
  memory_id  TEXT NOT NULL,
  speaker    TEXT DEFAULT '',
  t_start    REAL NOT NULL,
  t_end      REAL NOT NULL,
  text       TEXT NOT NULL,
  lang       TEXT DEFAULT 'en',
  asr_provider   TEXT DEFAULT 'whisper_local',
  asr_confidence REAL DEFAULT 1.0,
  source_meta TEXT DEFAULT '{{}}'
);
CREATE INDEX IF NOT EXISTS idx_epi_meeting_time ON episodes(meeting_id, t_start);
CREATE INDEX IF NOT EXISTS idx_epi_memory ON episodes(memory_id);
CREATE INDEX IF NOT EXISTS idx_epi_speaker ON episodes(speaker);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id   TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  token_count INTEGER NOT NULL,
  text        TEXT NOT NULL,
  char_start  INTEGER NOT NULL,
  char_end    INTEGER NOT NULL,
  title       TEXT DEFAULT '',
  episode_ids TEXT DEFAULT '[]',
  t_start     REAL,
  t_end       REAL,
  speaker_span TEXT DEFAULT '[]',
  source_type TEXT NOT NULL,
  created_at  INTEGER NOT NULL,
  language    TEXT DEFAULT 'en',
  importance  REAL DEFAULT 0.0,
  confidence  REAL DEFAULT 1.0,
  user        TEXT DEFAULT 'default',
  device      TEXT DEFAULT '',
  processing_version TEXT NOT NULL,
  meeting_id  TEXT,
  repo        TEXT,
  file_path   TEXT,
  page        INTEGER,
  document_title TEXT,
  ocr_confidence REAL,
  emb_full    BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunk_memory ON chunks(memory_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunk_type_time ON chunks(source_type, created_at);
CREATE INDEX IF NOT EXISTS idx_chunk_meeting ON chunks(meeting_id);
CREATE INDEX IF NOT EXISTS idx_chunk_repo ON chunks(repo, file_path);
CREATE INDEX IF NOT EXISTS idx_chunk_created_at ON chunks(created_at);
CREATE INDEX IF NOT EXISTS idx_chunk_proc_ver ON chunks(processing_version);

CREATE TABLE IF NOT EXISTS entity_mentions (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  entity_id   TEXT NOT NULL,
  entity_text TEXT NOT NULL,
  entity_type TEXT DEFAULT '',
  chunk_id    INTEGER NOT NULL,
  memory_id   TEXT NOT NULL,
  char_span   TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_ent_id ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_ent_chunk ON entity_mentions(chunk_id);
CREATE INDEX IF NOT EXISTS idx_ent_memory ON entity_mentions(memory_id);

CREATE TABLE IF NOT EXISTS relations (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id  TEXT NOT NULL,
  subject    TEXT NOT NULL,
  predicate  TEXT NOT NULL,
  object     TEXT NOT NULL,
  valid_at   INTEGER,
  invalid_at INTEGER,
  chunk_id   INTEGER,
  transaction_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rel_subject ON relations(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_rel_object ON relations(object);
CREATE INDEX IF NOT EXISTS idx_rel_memory ON relations(memory_id);
CREATE INDEX IF NOT EXISTS idx_rel_subject_lower ON relations(lower(subject));
CREATE INDEX IF NOT EXISTS idx_rel_object_lower ON relations(lower(object));

CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
  text, memory_id UNINDEXED
);

CREATE TABLE IF NOT EXISTS okf (
  okf_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  source_type  TEXT NOT NULL,          -- meeting | github_repo | pdf
  source_key   TEXT NOT NULL,          -- meeting_id | repo | document_title
  manifest     TEXT NOT NULL,          -- JSON table-of-contents (plan §7)
  content_hash TEXT NOT NULL,          -- staleness check for regeneration
  generated_at INTEGER NOT NULL,
  UNIQUE(source_type, source_key)
);

CREATE TABLE IF NOT EXISTS summaries (
  summary_id  INTEGER PRIMARY KEY AUTOINCREMENT,
  level       TEXT NOT NULL,           -- meeting | daily | project (ladder, §12 job 6)
  scope_key   TEXT NOT NULL,           -- meeting_id | yyyy-mm-dd | project name
  text        TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  generated_at INTEGER NOT NULL,
  UNIQUE(level, scope_key)
);

CREATE TABLE IF NOT EXISTS communities (
  community_id INTEGER PRIMARY KEY AUTOINCREMENT,
  label        TEXT NOT NULL,          -- dominant shared entity/topic (§12 job 5)
  member_ids   TEXT NOT NULL,          -- JSON list of memory_id
  generated_at INTEGER NOT NULL
);
"""

_VEC_SCHEMA = """
CREATE TABLE IF NOT EXISTS vec_chunks (
  chunk_id    INTEGER PRIMARY KEY,
  mem_rowid   INTEGER,
  created_at  INTEGER,
  source_type TEXT,
  emb_coarse  BLOB
);
CREATE INDEX IF NOT EXISTS idx_vec_source_type ON vec_chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_vec_created_at ON vec_chunks(created_at);
"""

_ENTITY_NORM_RE = re.compile(r"[^a-z0-9]+")


def entity_id_for(text: str) -> str:
    """Canonical entity id: lowercase, alnum-collapsed ("Postgre SQL" == "postgresql")."""
    return _ENTITY_NORM_RE.sub("", text.lower())


class MemoryStore:
    def __init__(self, db_path: str = ":memory:", check_same_thread: bool = True):
        # check_same_thread=False is used by the hub, which serializes access
        # through a lock while handlers hop between event-loop and threadpool.
        self.lock = threading.RLock()
        self.db = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")   # wait, don't throw 'locked'
        self.db.executescript(_SCHEMA)
        self.db.executescript(_VEC_SCHEMA)
        self._migrate()
        self.db.commit()

        # Dynamically wrap public database methods to be thread-safe
        for name in dir(self):
            if not name.startswith("__") and name not in ("lock", "db", "_filter_sql", "_ensure_column", "_migrate"):
                attr = getattr(self, name)
                if callable(attr):
                    def make_wrapper(fn):
                        def wrapper(*args, **kw):
                            with self.lock:
                                return fn(*args, **kw)
                        return wrapper
                    setattr(self, name, make_wrapper(attr))

    def _ensure_column(self, table: str, column: str, decl: str):
        """`CREATE TABLE IF NOT EXISTS` only creates a table that doesn't
        exist yet — it is a no-op on a database file from before a column was
        added, so that column is simply missing on every existing row and
        every read of it raises IndexError. Any new column added to _SCHEMA
        for an EXISTING table needs a matching line here."""
        cols = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _migrate(self):
        self._ensure_column("chunks", "title", "TEXT DEFAULT ''")
        self._ensure_column("relations", "chunk_id", "INTEGER")

    def close(self):
        self.db.close()

    # ------------------------------------------------------------- writes

    def add_memory(self, nmo: NMO) -> int:
        """Insert an NMO; returns the memories rowid (vec partition key)."""
        cur = self.db.execute(
            """INSERT INTO memories(memory_id, source_type, source, project, title,
                 content, summary, created_at, updated_at, language, importance,
                 confidence, user, device, hash, processing_version, source_meta,
                 entities, tags, history, parent, attachments, children)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (nmo.memory_id, nmo.source_type, nmo.source, nmo.project, nmo.title,
             nmo.content, nmo.summary, nmo.created_at, nmo.updated_at, nmo.language,
             nmo.importance, nmo.confidence, nmo.user, nmo.device, nmo.hash,
             nmo.processing_version, json.dumps(nmo.source_meta),
             json.dumps(nmo.entities), json.dumps(nmo.tags), json.dumps(nmo.history),
             nmo.parent, json.dumps(nmo.attachments), json.dumps(nmo.children)))
        return cur.lastrowid

    def add_episode(self, ep: Episode):
        self.db.execute(
            """INSERT INTO episodes(episode_id, meeting_id, memory_id, speaker,
                 t_start, t_end, text, lang, asr_provider, asr_confidence, source_meta)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (ep.episode_id, ep.meeting_id, ep.memory_id, ep.speaker, ep.t_start,
             ep.t_end, ep.text, ep.lang, ep.asr_provider, ep.asr_confidence,
             json.dumps(ep.source_meta)))

    def add_chunk(self, chunk: Chunk, mem_rowid: int,
                  emb_full: np.ndarray, emb_coarse: np.ndarray) -> int:
        if emb_full.shape != (DIM_FULL,):
            raise ValueError(
                f"emb_full has shape {emb_full.shape}; expected ({DIM_FULL},). "
                "Set RECALL_EMBEDDING_DIM to match your embedder's output dimension."
            )
        if emb_coarse.shape != (DIM_COARSE,):
            raise ValueError(
                f"emb_coarse has shape {emb_coarse.shape}; expected ({DIM_COARSE},)."
            )
        cur = self.db.execute(
            """INSERT INTO chunks(memory_id, chunk_index, token_count, text,
                 char_start, char_end, title, episode_ids, t_start, t_end,
                 speaker_span, source_type, created_at, language, importance,
                 confidence, user, device, processing_version, meeting_id, repo,
                 file_path, page, document_title, ocr_confidence, emb_full)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (chunk.memory_id, chunk.chunk_index, chunk.token_count, chunk.text,
             chunk.char_start, chunk.char_end, chunk.title,
             json.dumps(chunk.episode_ids), chunk.t_start, chunk.t_end,
             json.dumps(chunk.speaker_span), chunk.source_type, chunk.created_at,
             chunk.language, chunk.importance, chunk.confidence, chunk.user,
             chunk.device, chunk.processing_version, chunk.meeting_id, chunk.repo,
             chunk.file_path, chunk.page, chunk.document_title,
             chunk.ocr_confidence, emb_full.astype(np.float32).tobytes()))
        chunk_id = cur.lastrowid
        chunk.chunk_id = chunk_id
        self.db.execute(
            "INSERT INTO fts_chunks(rowid, text, memory_id) VALUES (?,?,?)",
            (chunk_id, chunk.text, chunk.memory_id))
        self.db.execute(
            """INSERT INTO vec_chunks(chunk_id, mem_rowid, created_at, source_type, emb_coarse)
               VALUES (?,?,?,?,?)""",
            (chunk_id, mem_rowid, chunk.created_at, chunk.source_type,
             emb_coarse.astype(np.int8).tobytes()))
        return chunk_id

    def add_entity_mention(self, entity_text: str, entity_type: str,
                           chunk_id: int, memory_id: str, char_span=()):
        self.db.execute(
            """INSERT INTO entity_mentions(entity_id, entity_text, entity_type,
                 chunk_id, memory_id, char_span) VALUES (?,?,?,?,?,?)""",
            (entity_id_for(entity_text), entity_text, entity_type,
             chunk_id, memory_id, json.dumps(list(char_span))))

    def add_relation(self, memory_id: str, subject: str, predicate: str,
                     object_: str, valid_at: int | None = None, chunk_id: int | None = None) -> int:
        cur = self.db.execute(
            """INSERT INTO relations(memory_id, subject, predicate, object,
                 valid_at, chunk_id, transaction_at) VALUES (?,?,?,?,?,?,?)""",
            (memory_id, subject, predicate, object_, valid_at, chunk_id, now_epoch()))
        return cur.lastrowid

    def update_chunk_title(self, chunk_id: int, title: str):
        self.db.execute("UPDATE chunks SET title=? WHERE chunk_id=?", (title, chunk_id))

    def append_history(self, memory_id: str, field: str, old_value, new_value):
        row = self.db.execute(
            "SELECT history FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        if row is None:
            return
        hist = json.loads(row["history"])
        hist.append({"field": field, "old_value": old_value,
                     "new_value": new_value, "changed_at": now_epoch()})
        self.db.execute(
            "UPDATE memories SET history=?, updated_at=? WHERE memory_id=?",
            (json.dumps(hist), now_epoch(), memory_id))

    def commit(self):
        self.db.commit()

    # -------------------------------------------------------------- reads

    def get_memory(self, memory_id: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()

    def get_chunk(self, chunk_id: int) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()

    def get_chunks_batch(self, chunk_ids: list[int]) -> dict[int, sqlite3.Row]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.db.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", chunk_ids).fetchall()
        return {r["chunk_id"]: r for r in rows}

    def neighbor_chunks(self, chunk: sqlite3.Row) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT * FROM chunks WHERE memory_id=? AND chunk_index IN (?,?)
               ORDER BY chunk_index""",
            (chunk["memory_id"], chunk["chunk_index"] - 1,
             chunk["chunk_index"] + 1)).fetchall()

    def episodes_for_chunk(self, chunk: sqlite3.Row) -> list[sqlite3.Row]:
        ids = json.loads(chunk["episode_ids"] or "[]")
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        return self.db.execute(
            f"SELECT * FROM episodes WHERE episode_id IN ({marks}) ORDER BY t_start",
            ids).fetchall()

    def known_speakers(self) -> set[str]:
        return {r["speaker"] for r in
                self.db.execute("SELECT DISTINCT speaker FROM episodes")
                if r["speaker"]}

    def entity_exists(self, term: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM entity_mentions WHERE entity_id=? LIMIT 1",
            (entity_id_for(term),)).fetchone() is not None

    # ------------------------------------------------------ retrieval routes

    @staticmethod
    def _filter_sql(filters: dict, alias: str = "c") -> tuple[str, list]:
        """Shared WHERE fragment over the denormalized chunk columns."""
        conds, params = [], []
        if filters.get("source_type"):
            conds.append(f"{alias}.source_type = ?")
            params.append(filters["source_type"])
        if filters.get("meeting_id"):
            conds.append(f"{alias}.meeting_id = ?")
            params.append(filters["meeting_id"])
        if filters.get("repo"):
            conds.append(f"{alias}.repo = ?")
            params.append(filters["repo"])
        dr = filters.get("date_range")
        if dr:
            conds.append(f"{alias}.created_at BETWEEN ? AND ?")
            params.extend([int(dr[0]), int(dr[1])])
        if filters.get("speaker"):
            conds.append(
                f"EXISTS (SELECT 1 FROM json_each({alias}.speaker_span) je"
                f"        WHERE lower(je.value) = lower(?))")
            params.append(filters["speaker"])
        return (" AND ".join(conds), params)

    def bm25_search(self, query: str, filters: dict, topn: int = 50) -> list[tuple[int, float]]:
        """FTS5 BM25. Returns [(chunk_id, rank_score)] best-first."""
        terms = re.findall(r"\w+", query.lower())
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        where, params = self._filter_sql(filters, "c")
        sql = ("SELECT f.rowid AS chunk_id, bm25(fts_chunks) AS score "
               "FROM fts_chunks f JOIN chunks c ON c.chunk_id = f.rowid "
               "WHERE fts_chunks MATCH ?")
        if where:
            sql += " AND " + where
        sql += " ORDER BY score LIMIT ?"
        try:
            rows = self.db.execute(sql, [match, *params, topn]).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["chunk_id"], -r["score"]) for r in rows]  # bm25() is lower-better

    def vector_search(self, emb_coarse_query: np.ndarray, filters: dict,
                      k: int = 200) -> list[tuple[int, float]]:
        """Brute-force int8[256] KNN via numpy, with metadata pre-filters.

        Score only has to preserve rank order — `_rrf_fuse` (retrieval.py)
        fuses routes by rank position, not by this value, so an exact match
        with sqlite-vec's internal distance formula isn't required.
        """
        conds, params = [], []
        if filters.get("source_type"):
            conds.append("source_type = ?")
            params.append(filters["source_type"])
        dr = filters.get("date_range")
        if dr:
            conds.append("created_at BETWEEN ? AND ?")
            params.extend([int(dr[0]), int(dr[1])])
        sql = "SELECT chunk_id, emb_coarse FROM vec_chunks"
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        rows = self.db.execute(sql, params).fetchall()
        if not rows:
            return []
        chunk_ids = [r["chunk_id"] for r in rows]
        mat = (np.frombuffer(b"".join(r["emb_coarse"] for r in rows), dtype=np.int8)
                 .reshape(len(rows), DIM_COARSE).astype(np.int32))
        q = emb_coarse_query.astype(np.int32)
        dist = np.sum((mat - q) ** 2, axis=1)   # squared L2 over int8 vectors
        order = np.argsort(dist, kind="stable")[:k]
        out = [(chunk_ids[i], 1.0 / (1.0 + float(dist[i]))) for i in order]
        # Non-vec metadata filters (speaker, meeting) applied post-KNN on the chunk rows.
        post = {k_: v for k_, v in filters.items()
                if k_ in ("speaker", "meeting_id", "repo") and v}
        if post and out:
            where, p2 = self._filter_sql(post, "c")
            marks = ",".join("?" * len(out))
            keep = {r["chunk_id"] for r in self.db.execute(
                f"SELECT c.chunk_id FROM chunks c WHERE c.chunk_id IN ({marks})"
                f" AND {where}", [cid for cid, _ in out] + p2)}
            out = [(cid, s) for cid, s in out if cid in keep]
        return out

    def entity_search(self, entities: list[str], filters: dict,
                      topn: int = 50) -> list[tuple[int, float]]:
        """Cross-source bridge: chunks mentioning the query entities."""
        if not entities:
            return []
        ids = [entity_id_for(e) for e in entities]
        marks = ",".join("?" * len(ids))
        where, params = self._filter_sql(filters, "c")
        sql = (f"SELECT em.chunk_id, count(*) AS hits FROM entity_mentions em "
               f"JOIN chunks c ON c.chunk_id = em.chunk_id "
               f"WHERE em.entity_id IN ({marks})")
        if where:
            sql += " AND " + where
        sql += " GROUP BY em.chunk_id ORDER BY hits DESC, c.created_at DESC LIMIT ?"
        rows = self.db.execute(sql, [*ids, *params, topn]).fetchall()
        return [(r["chunk_id"], float(r["hits"])) for r in rows]

    def metadata_search(self, filters: dict, topn: int = 50) -> list[tuple[int, float]]:
        """Pure structured filter, recency-ranked (the 'metadata only' route)."""
        where, params = self._filter_sql(filters, "c")
        sql = "SELECT c.chunk_id, c.created_at FROM chunks c"
        if where:
            sql += " WHERE " + where
        sql += " ORDER BY c.created_at DESC LIMIT ?"
        rows = self.db.execute(sql, [*params, topn]).fetchall()
        return [(r["chunk_id"], 1.0) for r in rows]

    def graph_search(self, entities: list[str], filters: dict,
                     topn: int = 50) -> list[tuple[int, float]]:
        """Relation traversal: entities as subject/object -> owning memories' chunks."""
        if not entities:
            return []
        conds, params = [], []
        for e in entities:
            conds.append("(lower(r.subject) = ? OR lower(r.object) = ?)")
            params.extend([e.lower(), e.lower()])
        rows = self.db.execute(
            f"""SELECT r.memory_id, count(*) AS hits FROM relations r
                WHERE ({' OR '.join(conds)}) AND r.invalid_at IS NULL
                GROUP BY r.memory_id ORDER BY hits DESC LIMIT 10""",
            params).fetchall()
        out: list[tuple[int, float]] = []
        where, fparams = self._filter_sql(filters, "c")
        for r in rows:
            sql = "SELECT c.chunk_id FROM chunks c WHERE c.memory_id = ?"
            if where:
                sql += " AND " + where
            sql += " ORDER BY c.chunk_index LIMIT ?"
            for cr in self.db.execute(sql, [r["memory_id"], *fparams,
                                            max(1, topn // len(rows))]):
                out.append((cr["chunk_id"], float(r["hits"])))
        return out

    # -------------------------------------------------------------- forget

    def _delete_chunks(self, chunk_ids: list[int]):
        for cid in chunk_ids:
            self.db.execute("DELETE FROM fts_chunks WHERE rowid=?", (cid,))
            self.db.execute("DELETE FROM vec_chunks WHERE chunk_id=?", (cid,))
            self.db.execute("DELETE FROM entity_mentions WHERE chunk_id=?", (cid,))
            self.db.execute("DELETE FROM chunks WHERE chunk_id=?", (cid,))

    def forget_time_range(self, meeting_id: str, t_from: float, t_to: float) -> dict:
        """The forget button: erase episodes in [t_from, t_to] + every chunk touching them."""
        eps = self.db.execute(
            """SELECT episode_id, memory_id FROM episodes
               WHERE meeting_id=? AND t_end >= ? AND t_start <= ?""",
            (meeting_id, t_from, t_to)).fetchall()
        ep_ids = {r["episode_id"] for r in eps}
        mem_ids = {r["memory_id"] for r in eps}
        if not ep_ids:
            return {"episodes": 0, "chunks": 0}
        chunk_ids = [
            r["chunk_id"] for r in self.db.execute(
                "SELECT chunk_id, episode_ids FROM chunks WHERE meeting_id=?",
                (meeting_id,))
            if ep_ids & set(json.loads(r["episode_ids"] or "[]"))
        ]
        self._delete_chunks(chunk_ids)
        marks = ",".join("?" * len(ep_ids))
        self.db.execute(f"DELETE FROM episodes WHERE episode_id IN ({marks})",
                        list(ep_ids))
        for mid in mem_ids:
            self.db.execute(
                "DELETE FROM relations WHERE memory_id=? AND valid_at BETWEEN ? AND ?",
                (mid, t_from, t_to))
            self.append_history(mid, "forget",
                                f"episodes {t_from}-{t_to}", "erased")
        self.db.execute("DELETE FROM okf WHERE source_type='meeting' AND source_key=?", (meeting_id,))
        self.db.execute("DELETE FROM summaries WHERE level='meeting' AND scope_key=?", (meeting_id,))
        self.commit()
        return {"episodes": len(ep_ids), "chunks": len(chunk_ids)}

    def forget_memory(self, memory_id: str) -> dict:
        mem = self.get_memory(memory_id)
        if mem:
            source_type = mem["source_type"]
            meta = json.loads(mem["source_meta"] or "{}")
            if source_type == "meeting":
                mid = meta.get("meeting_id") or memory_id
                self.db.execute("DELETE FROM okf WHERE source_type='meeting' AND source_key=?", (mid,))
                self.db.execute("DELETE FROM summaries WHERE level='meeting' AND scope_key=?", (mid,))
            elif source_type == "github_repo":
                repo = meta.get("repo")
                if repo:
                    self.db.execute("DELETE FROM okf WHERE source_type='github_repo' AND source_key=?", (repo,))
            elif source_type == "pdf":
                title = mem["title"]
                self.db.execute("DELETE FROM okf WHERE source_type='pdf' AND source_key=?", (title,))

        chunk_ids = [r["chunk_id"] for r in self.db.execute(
            "SELECT chunk_id FROM chunks WHERE memory_id=?", (memory_id,))]
        self._delete_chunks(chunk_ids)
        n_ep = self.db.execute(
            "DELETE FROM episodes WHERE memory_id=?", (memory_id,)).rowcount
        self.db.execute("DELETE FROM relations WHERE memory_id=?", (memory_id,))
        self.db.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
        self.commit()
        return {"episodes": n_ep, "chunks": len(chunk_ids)}

    # ----------------------------------------------------- OKF / summaries

    def upsert_okf(self, source_type: str, source_key: str, manifest: dict,
                   content_hash: str):
        self.db.execute(
            """INSERT INTO okf(source_type, source_key, manifest, content_hash,
                 generated_at) VALUES (?,?,?,?,?)
               ON CONFLICT(source_type, source_key) DO UPDATE SET
                 manifest=excluded.manifest, content_hash=excluded.content_hash,
                 generated_at=excluded.generated_at""",
            (source_type, source_key, json.dumps(manifest), content_hash, now_epoch()))

    def get_okf(self, source_type: str, source_key: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM okf WHERE source_type=? AND source_key=?",
            (source_type, source_key)).fetchone()

    def upsert_summary(self, level: str, scope_key: str, text: str,
                       content_hash: str):
        self.db.execute(
            """INSERT INTO summaries(level, scope_key, text, content_hash,
                 generated_at) VALUES (?,?,?,?,?)
               ON CONFLICT(level, scope_key) DO UPDATE SET
                 text=excluded.text, content_hash=excluded.content_hash,
                 generated_at=excluded.generated_at""",
            (level, scope_key, text, content_hash, now_epoch()))

    def get_summary(self, level: str, scope_key: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM summaries WHERE level=? AND scope_key=?",
            (level, scope_key)).fetchone()

    def replace_communities(self, communities: list[tuple[str, list[str]]]):
        self.db.execute("DELETE FROM communities")
        for label, members in communities:
            self.db.execute(
                "INSERT INTO communities(label, member_ids, generated_at) "
                "VALUES (?,?,?)", (label, json.dumps(members), now_epoch()))

    # --------------------------------------------------------------- misc

    def stats(self) -> dict:
        q = lambda t: self.db.execute(f"SELECT count(*) c FROM {t}").fetchone()["c"]
        return {t: q(t) for t in
                ("memories", "episodes", "chunks", "entity_mentions",
                 "relations", "vec_chunks")}
