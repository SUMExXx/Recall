"""Central configuration for the Recall memory engine (pydantic-settings).

The whole dev-laptop-vs-event-PC story collapses into ONE switch: `backend`.

    RECALL_BACKEND=npu      Snapdragon X Elite event PC   (DEFAULT — the real target)
    RECALL_BACKEND=ollama   your laptop, via Ollama       (local development)
    RECALL_BACKEND=hash     deterministic offline stub    (tests / CI, no models)

Everything else (model names, service URLs, thresholds) is a field on
`RecallConfig` and overridable from the environment with the `RECALL_` prefix,
e.g. `RECALL_OLLAMA_URL`, `RECALL_NPU_LLM_ENDPOINT`, `RECALL_DB_PATH`.

The domain constants below (dimensions, chunk specs, RRF weight profiles) are
fixed by plans/memory_engineering_v2.md and are NOT environment config.
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROCESSING_VERSION = "v2.0"

import os
DIM_FULL = int(os.environ.get("RECALL_EMBEDDING_DIM", "768"))
DIM_COARSE = 256   # Matryoshka truncation for the hot int8 scan

Backend = Literal["npu", "ollama", "hash"]


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
RERANK_TOPK = 50    # slice handed to the cross-encoder reranker
ANSWER_TOPK = 6     # contexts handed to the LLM


class RecallConfig(BaseSettings):
    """Runtime configuration. Reads `RECALL_*` env vars; kwargs override env."""

    model_config = SettingsConfigDict(
        env_prefix="RECALL_", extra="ignore", populate_by_name=True,
        env_file=".env", env_file_encoding="utf-8")

    # --- the one switch --------------------------------------------------
    backend: Backend = "npu"

    db_path: str = "recall.db"
    log_level: str = "INFO"              # RECALL_LOG_LEVEL: DEBUG for stage detail

    # Per-request step trace (chunking/embedding/retrieval/rerank/planner/LLM
    # prompt+response/background jobs) written as one readable block per
    # operation — see recall_memory/tracing.py. Off disables the file write;
    # steps are still visible via the `logging` module at DEBUG.
    trace_enabled: bool = True
    trace_file: str = "logs/recall_trace.log"

    # Dream tier cadence: the hub runs the consolidation agent this often when
    # idle (never inline with capture). 0 disables the automatic runs.
    consolidate_every_s: float = 120.0

    # --- router / retrieval / consolidation knobs ------------------------
    use_tier3_planner: bool = True          # off by default (plan rec. #3)
    near_dup_cosine: float = 0.95
    dualmic_fuzzy_threshold: float = 0.85

    # --- ollama backend (local dev) --------------------------------------
    ollama_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"   # 768-dim, same family as Nomic v1.5
    ollama_llm_model: str = "llama3.2:3b"

    # --- npu backend (Snapdragon X Elite) --------------------------------
    # Embedder: Nomic-Embed-Text v1.5 re-exported at seqlen 256/512, ORT-QNN.
    npu_embed_onnx_256: str = "models/nomic-v1.5-seq256.onnx"
    npu_embed_onnx_512: str = "models/nomic-v1.5-seq512.onnx"
    # LLM: Qwen3-4B-Instruct-2507 (w4a16) served by GenieX/QAIRT over HTTP.
    npu_llm_endpoint: str = "http://localhost:8090"
    npu_llm_model: str = "qwen3-4b-instruct-2507"
    # Reranker: Qwen3-Reranker-0.6B via ORT-QNN (bge-reranker CPU fallback).
    npu_reranker_onnx: str = "models/qwen3-reranker-0.6b.onnx"
    # HF tokenizer id used for exact on-device token budgeting.
    npu_tokenizer_id: str = "Qwen/Qwen3-4B"

    # --- ASR ---------------------------------------------------------------
    # auto (default): transcribe in-process via faster-whisper when installed,
    #   else fall back to the HTTP contract below. embedded: require in-process.
    #   http: always use the external server (the event PC's ORT-QNN Whisper).
    asr_mode: Literal["auto", "embedded", "http"] = "auto"
    whisper_model: str = "base.en"                 # faster-whisper model (in-process)
    whisper_url: str = "http://localhost:8080"     # http mode: whisper.cpp contract
    hinglish_url: str = "http://localhost:8081"    # Oriserve Hinglish BYOM slot
    # Decoding knobs (faster-whisper). beam_size was hardcoded to 1 (greedy)
    # for latency, which is the direct cause of a lot of mangled domain-term
    # transcription — 5 is the faster-whisper library default and materially
    # more accurate at a modest CPU cost.
    whisper_beam_size: int = 5
    # Vocabulary bias for domain jargon Whisper otherwise mangles (e.g.
    # "FAISS", "bi-temporal", "Matryoshka", "OKF", "RRF") — passed as
    # initial_prompt. Empty by default; set per deployment.
    whisper_initial_prompt: str = ""

    # --- Sarvam (cloud ASR, policy-gated opt-in — plan §2) ------------------
    # NEVER called unless PolicyEngine.is_cloud_allowed() says so. Contract:
    # docs.sarvam.ai/api-reference/speech-to-text/transcribe (POST
    # https://api.sarvam.ai/speech-to-text, multipart/form-data, header
    # api-subscription-key). Get a key at https://dashboard.sarvam.ai.
    sarvam_api_key: str = "sk_kobw09xo_JqDdUSFxsT6AmRQIV38qQglc"
    sarvam_endpoint: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_model: str = "saaras:v3"          # saaras:v3 (recommended) | saarika:v2.5
    sarvam_language_code: str = "unknown"    # BCP-47, or "unknown" to auto-detect
    sarvam_mode: str = "transcribe"          # transcribe|translate|verbatim|translit|codemix
    sarvam_timeout_s: float = 3.0            # plan §2: 3s timeout -> local fallback
    cloud_optin: bool = False

