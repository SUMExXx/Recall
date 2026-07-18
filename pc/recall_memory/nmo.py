"""Normalized Memory Object (NMO) and Chunk/Episode shapes — plan §3-§4.

Every source maps into the NMO before anything downstream sees it. Universal
metadata becomes real SQLite columns; source-specific metadata lives in the
`source_meta` JSON blob and keys exist only when relevant.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field

from .config import PROCESSING_VERSION

SOURCE_TYPES = {"meeting", "github_repo", "pdf", "image", "text", "note"}

# §4b — the allowed source-specific key sets (soft-validated: unknown keys warn).
SOURCE_META_KEYS: dict[str, set[str]] = {
    "meeting": {
        "meeting_id", "speaker", "speaker_confidence", "participants",
        "start_time", "end_time", "action_items", "decisions", "questions",
        "capture_device", "device_id", "capture_confidence", "asr_provider",
    },
    "github_repo": {
        "repo", "branch", "commit", "file_path", "language", "framework",
        "imports", "functions", "classes", "api_endpoints",
    },
    "pdf": {"page", "chapter", "heading", "document_title", "author"},
    "image": {
        "ocr_confidence", "image_width", "image_height",
        "detected_objects", "location",
    },
    "text": set(),
    "note": set(),
}


def new_id() -> str:
    return uuid.uuid4().hex


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_epoch() -> int:
    return int(time.time())


@dataclass
class Episode:
    """Immutable utterance atom (meetings only) — citations / forget / dual-mic."""
    episode_id: str
    meeting_id: str
    memory_id: str
    speaker: str
    t_start: float           # absolute epoch seconds
    t_end: float
    text: str
    lang: str = "en"
    asr_provider: str = "whisper_local"
    asr_confidence: float = 1.0
    source_meta: dict = field(default_factory=dict)


@dataclass
class NMO:
    memory_id: str
    source_type: str                  # == NMO "type"
    source: str                       # mobile | chrome_extension | desktop | arduino
    title: str
    content: str                      # immutable — corrections go to history[]
    project: str | None = None
    summary: str = ""
    created_at: int = 0               # epoch seconds
    updated_at: int = 0
    language: str = "en"
    importance: float = 0.0
    confidence: float = 1.0
    user: str = "default"
    device: str = ""
    hash: str = ""
    processing_version: str = PROCESSING_VERSION
    source_meta: dict = field(default_factory=dict)
    entities: list = field(default_factory=list)     # {name, type, entity_id}
    relations: list = field(default_factory=list)    # {subject, predicate, object, valid_at}
    tags: list = field(default_factory=list)
    history: list = field(default_factory=list)      # {field, old_value, new_value, changed_at}
    parent: str | None = None
    attachments: list = field(default_factory=list)
    children: list = field(default_factory=list)

    @classmethod
    def create(cls, source_type: str, source: str, title: str, content: str,
               source_meta: dict | None = None, **kw) -> "NMO":
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"unknown source_type {source_type!r}")
        meta = dict(source_meta or {})
        allowed = SOURCE_META_KEYS[source_type]
        unknown = set(meta) - allowed
        if unknown and allowed:
            # Soft validation: keep the keys (schemaless blob) but make it visible.
            import warnings
            warnings.warn(f"{source_type} source_meta has non-spec keys: {sorted(unknown)}")
        ts = now_epoch()
        return cls(
            memory_id=new_id(), source_type=source_type, source=source,
            title=title, content=content, source_meta=meta,
            created_at=kw.pop("created_at", ts), updated_at=kw.pop("updated_at", ts),
            hash=content_hash(content), **kw,
        )


@dataclass
class Chunk:
    """The universal retrieval unit — fixed-size token window (plan §4c/§5)."""
    # Group 1 — chunk-own
    memory_id: str
    chunk_index: int
    token_count: int
    text: str
    char_start: int
    char_end: int
    episode_ids: list = field(default_factory=list)   # meetings only
    t_start: float | None = None                      # meetings only
    t_end: float | None = None
    speaker_span: list = field(default_factory=list)
    # Group 2 — inherited universal (denormalized for join-free filtering)
    source_type: str = ""
    created_at: int = 0
    language: str = "en"
    importance: float = 0.0
    confidence: float = 1.0
    user: str = "default"
    device: str = ""
    processing_version: str = PROCESSING_VERSION
    # Group 3 — inherited source-specific filter keys (queryable subset only)
    meeting_id: str | None = None
    repo: str | None = None
    file_path: str | None = None
    page: int | None = None
    document_title: str | None = None
    ocr_confidence: float | None = None
    # set by the store on insert
    chunk_id: int | None = None

    def inherit(self, nmo: NMO) -> "Chunk":
        self.source_type = nmo.source_type
        self.created_at = nmo.created_at
        self.language = nmo.language
        self.importance = nmo.importance
        self.confidence = nmo.confidence
        self.user = nmo.user
        self.device = nmo.device
        self.processing_version = nmo.processing_version
        m = nmo.source_meta
        if nmo.source_type == "meeting":
            self.meeting_id = m.get("meeting_id")
        elif nmo.source_type == "github_repo":
            self.repo, self.file_path = m.get("repo"), m.get("file_path")
        elif nmo.source_type == "pdf":
            self.page, self.document_title = m.get("page"), m.get("document_title")
        elif nmo.source_type == "image":
            self.ocr_confidence = m.get("ocr_confidence")
        return self
