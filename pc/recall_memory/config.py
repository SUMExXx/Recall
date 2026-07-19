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

    # --- retrieval / consolidation knobs ----------------------------------
    near_dup_cosine: float = 0.95
    dualmic_fuzzy_threshold: float = 0.85
    # Reranking. The npu backend has no cross-encoder ONNX asset, so it reranks
    # with the on-device LLM (a real relevance judgment, one extra generation
    # per query). Kill-switch: set RECALL_RERANK_ENABLED=false to skip reranking
    # entirely (fused+cosine order only) if the NPU LLM is under load.
    rerank_enabled: bool = True

    # --- ollama backend (local dev) --------------------------------------
    ollama_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"   # 768-dim, same family as Nomic v1.5
    ollama_llm_model: str = "llama3.2:3b"
    # Real cross-encoder reranker (ollama backend only — Ollama itself has no
    # rerank endpoint). Empty = passthrough (fused order kept, no reordering).
    # Loaded via sentence-transformers, NOT through Ollama; needs
    # `pip install -e ".[rerank]"`. e.g. "BAAI/bge-reranker-v2-m3".
    reranker_model: str = ""

    # --- npu backend (Snapdragon X Elite) --------------------------------
    # Embedder: Nomic-Embed-Text v1.5 re-exported at seqlen 256/512, ORT-QNN.
    npu_embed_onnx_256: str = "models\\nomic_embed_text-onnx-float\\nomic_embed_text.onnx"
    npu_embed_onnx_512: str = "models\\nomic_embed_text-onnx-float\\nomic_embed_text.onnx"
    # LLM serving mode:
    #   auto     — use npu_llm_endpoint if a server is already up there, else
    #              load the model in-process via the `geniex` package (which
    #              auto-downloads the precompiled AI Hub bundle on first use)
    #   geniex   — always in-process via geniex (no sidecar to start)
    #   endpoint — only the external OpenAI-compatible server (never in-process)
    npu_llm_mode: Literal["auto", "geniex", "endpoint"] = "geniex"
    # Model id: a Qualcomm AI Hub bundle id (auto-pulled by geniex, NPU-ready)
    # — also sent as the OpenAI `model` field in endpoint mode, matching how
    # `geniex serve` names its models.
    npu_llm_model: str = "ai-hub-models/Qwen3-4B-Instruct-2507"
    # geniex device_map: npu (qairt) | cpu | gpu | hybrid (llama_cpp) | auto.
    npu_llm_device_map: str = "npu"
    npu_llm_max_new_tokens: int = 1024
    npu_llm_endpoint: str = "http://localhost:8090"
    # Reranker cross-encoder, ORT-QNN. Must be a real .onnx to load — ORT
    # cannot read .gguf (llama.cpp). Empty (default) => rerank via the on-device
    # LLM instead (see NpuBackend.reranker / LlmReranker). Point this at a
    # Qwen3-Reranker .onnx export to run a dedicated cross-encoder on the NPU.
    npu_reranker_onnx: str = ""
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

    # --- Sarvam STT (cloud ASR, policy-gated opt-in — plan §2) --------------
    # NEVER called unless PolicyEngine.is_cloud_allowed() says so. Primary STT
    # whenever cloud is opted in (falls back to local Whisper on any failure
    # — network, timeout, not opted in). Contract:
    # docs.sarvam.ai/api-reference/speech-to-text/transcribe (POST
    # https://api.sarvam.ai/speech-to-text, multipart/form-data, header
    # api-subscription-key). Get a key at https://dashboard.sarvam.ai.
    sarvam_api_key: str = "sk_kobw09xo_JqDdUSFxsT6AmRQIV38qQglc"
    sarvam_endpoint: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_model: str = "saaras:v3"          # saaras:v3 (recommended) | saarika:v2.5
    sarvam_language_code: str = "en-IN"    # BCP-47, or "unknown" to auto-detect
    sarvam_mode: str = "transcribe"          # transcribe|translate|verbatim|translit|codemix
    sarvam_timeout_s: float = 10.0           # primary path now — generous before local fallback

    # --- Sarvam TTS (Bulbul — docs.sarvam.ai/api-reference/text-to-speech) -
    # POST https://api.sarvam.ai/text-to-speech, same api-subscription-key
    # header as STT. Response is JSON {"audios": ["<base64 wav>", ...]}.
    sarvam_tts_endpoint: str = "https://api.sarvam.ai/text-to-speech"
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_tts_speaker: str = "manan"          # bulbul:v3 default voice
    sarvam_tts_language_code: str = "en-IN"   # BCP-47; one of the 11 supported
    sarvam_tts_sample_rate: int = 24000       # Hz — 8000..48000, see docs
    sarvam_tts_codec: str = "wav"             # wav|mp3|linear16|mulaw|alaw|opus|flac|aac
    sarvam_tts_pace: float = 1.0              # 0.5-2.0
    sarvam_tts_timeout_s: float = 15.0
    # Streaming TTS (wss://.../text-to-speech/ws) — a separate endpoint from
    # the REST one above, used for the full-pipeline streaming /ask/stream.
    # Confirmed directly against the real endpoint: it currently serves
    # bulbul:v2 regardless of what "model" is sent in the config message, so
    # it needs its own, v2-compatible speaker (the REST default "shubh" is
    # v3-only and gets rejected there).
    sarvam_tts_stream_endpoint: str = "wss://api.sarvam.ai/text-to-speech/ws"
    sarvam_tts_stream_speaker: str = "manan"
    cloud_optin: bool = False

