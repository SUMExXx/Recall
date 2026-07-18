"""Central configuration for the Recall memory engine.

Values mirror plans/memory_engineering_v2.md. On the dev laptop the embedder and
LLM are Ollama substitutes (nomic-embed-text / llama3.2:3b); on the X Elite event
PC they map to the AI Hub Nomic v1.5 NPU asset and Qwen3-4B via GenieX.
"""
from __future__ import annotations

from dataclasses import dataclass, field


PROCESSING_VERSION = "v2.0"

DIM_FULL = 768     # Nomic-Embed-Text v1.5 output
DIM_COARSE = 256   # Matryoshka truncation for the hot int8 scan


@dataclass(frozen=True)
class ChunkSpec:
    size: int      # tokens
    overlap: int   # tokens


# §5 of the plan: fixed-size token windows per source. image = whole OCR block.
CHUNK_SPECS: dict[str, ChunkSpec] = {
    "meeting": ChunkSpec(256, 38),       # ~15%
    "github_repo": ChunkSpec(256, 26),   # ~10%
    "pdf": ChunkSpec(400, 50),           # ~12-15%
    "text": ChunkSpec(256, 26),
    "note": ChunkSpec(256, 26),
}

# §4c: source-specific keys denormalized onto every chunk (the queryable subset).
INHERITED_SOURCE_KEYS: dict[str, list[str]] = {
    "meeting": ["meeting_id"],
    "github_repo": ["repo", "file_path"],
    "pdf": ["page", "document_title"],
    "image": ["ocr_confidence"],
}

RRF_K = 60

# §10: per-query_type RRF weight profiles + post-fusion boost weights.
WEIGHT_PROFILES: dict[str, dict] = {
    "code": {
        "routes": {"bm25": 1.0, "vector": 0.6, "graph": 0.3, "entity": 0.9, "metadata": 0.3},
        "boost": {"recency": 0.1, "importance": 0.1},
    },
    "decision": {
        "routes": {"bm25": 0.5, "vector": 0.5, "graph": 1.0, "entity": 0.7, "metadata": 0.9},
        "boost": {"recency": 0.3, "importance": 0.2},
    },
    "timeline": {
        "routes": {"bm25": 0.3, "vector": 0.3, "graph": 0.8, "entity": 0.5, "metadata": 1.0},
        "boost": {"recency": 0.4, "importance": 0.1},
    },
    "general": {
        "routes": {"bm25": 0.6, "vector": 1.0, "graph": 0.6, "entity": 0.5, "metadata": 0.4},
        "boost": {"recency": 0.15, "importance": 0.25},
    },
}

RECENCY_HALF_LIFE_DAYS = 30.0

COARSE_TOPK = 200   # coarse-scan hits rescored with float[768]
RERANK_TOPK = 50    # slice handed to the (stub) cross-encoder reranker
ANSWER_TOPK = 6     # contexts handed to the LLM


@dataclass
class RecallConfig:
    db_path: str = "recall.db"
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2:3b"
    use_tier3_planner: bool = False   # off by default, per plan recommendation #3
    embedder: str = "auto"            # auto | ollama | hash
    near_dup_cosine: float = 0.95
    dualmic_fuzzy_threshold: float = 0.85
    extra: dict = field(default_factory=dict)
