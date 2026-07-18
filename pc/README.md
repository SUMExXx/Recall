# Recall PC Hub — Memory Engine + Hub Layers

Implements **`plans/memory_engineering_v2.md`** (memory engine) and the PC-hub
backend layers of **`docs/system_architectures_v1.md` §4**: gateway (WS hub /
REST / MCP), ASR workers, proactive recall, policy engine, device registry,
supervisor, metrics, and the judge dashboard.

Targets the **Snapdragon X Elite / NPU** event PC. One switch runs the exact
same code on your laptop via Ollama.

## The one switch — `RECALL_BACKEND`

| Value | Where | Embedder · LLM · Reranker · Tokenizer |
|---|---|---|
| `npu` *(default)* | Snapdragon X Elite | Nomic v1.5 (ORT-QNN) · Qwen3-4B (GenieX) · Qwen3-Reranker (ORT-QNN) · HF tokenizer |
| `ollama` | your laptop | `nomic-embed-text` · `llama3.2:3b` · passthrough · HF tokenizer |
| `hash` | tests / CI | deterministic hash vectors · none · passthrough · regex |

Everything else (model names, service URLs, thresholds) is a `RECALL_*` env var
bound to `RecallConfig` (pydantic-settings). The backend is the only seam the
engine turns — see [recall_memory/backends/](recall_memory/backends/).

```bash
# laptop: develop against Ollama
RECALL_BACKEND=ollama python -m recall_memory --db demo.db demo
# event PC: the default — no flag needed (needs the [npu] extra + model assets)
python -m recall_memory --db demo.db demo
```

## Layout

Memory engine (`recall_memory/`):

| Module | Plan § | What it does |
|---|---|---|
| `config.py` | — | `RecallConfig` (pydantic-settings); the `RECALL_BACKEND` switch + all env-bound settings |
| `backends/` | §2 | `get_backend()` → Embedder / LLM / Reranker / Tokenizer; `npu` (ORT-QNN + GenieX, guarded), `ollama`, `hash` |
| `nmo.py` | §3-4 | NMO + Episode + Chunk shapes, universal/source metadata, 3-group chunk metadata |
| `store.py` | §4d | SQLite: memories, episodes, chunks, entity_mentions, relations (bi-temporal), FTS5, `vec_chunks` vec0 int8[256], `okf`, `summaries`, `communities` |
| `tokenizer.py` | §5 | `RegexTokenizer` (offline) + `ModelTokenizer` (HF, char-offset aware) behind the backend |
| `chunker.py` | §5 | Fixed token windows (meeting 256/15%, code 256/10%, pdf 400/12%, OCR whole-block), speaker-attribution rule |
| `embeddings.py` | §2, §9 | Matryoshka float[768] → int8[256] math + offline `HashEmbedder` |
| `extractors.py` | §6 | Cheap-first entity/decision/action-item extraction; LLM relation-extraction fallback |
| `ingest.py` | §14a | Normalize → NMO → enrich → chunk → embed → write every index |
| `router.py` | §8 | Tier 0 commands, tier 1 rules+filters, tier 2 embedding prototypes, tier 3 LLM plan JSON (off by default) |
| `retrieval.py` | §9-11 | needs{}-gated parallel indices, weighted RRF k=60, recency/importance boost, float[768] rescore, reranker, small-to-big, **OKF skim**, cited answers |
| `okf.py` | §7 | Organized Knowledge Files — per-repo/meeting/pdf manifests, regenerated when content changes |
| `consolidate.py` | §12 | Dream tier — MVP jobs 1/3/4/10 **and** full jobs 2 (entity resolution), 5 (communities), 6 (summary ladder), 8 (archival), 9 (graph repair), OKF regen |
| `cli.py` | — | `demo / ask / search / ingest-* / forget / consolidate [--full] / stats` |

Hub layers (`docs/system_architectures_v1.md` §4):

| Module | Arch box | What it does |
|---|---|---|
| `hub/app.py` | GW | **FastAPI** gateway: WS hub (topics, seq, resume tokens, binary PCM16 ingest), REST with Pydantic models (`/ask /search /forget /consolidate /dump /links /digest /policy /health /stats /metrics /devices`), dashboard + capture pages |
| `hub/mcp_server.py` | GW | MCP server on the **official `mcp` SDK** (FastMCP): `recall_search_memory, recall_ask, recall_bookmark_moment, recall_forget_range, recall_dump` |
| `hub/asr.py` | NPU | ASRProvider workers (plan §2): whisper.cpp-contract local provider, Hinglish BYOM slot, policy-gated Sarvam stub, selection policy + fallback chain |
| `hub/sessions.py` | CORE | Live meeting sessions: utterance buffering, bookmark, forget-last-N-min, ingest on `meeting_end` |
| `hub/proactive.py` | CORE | Proactive recall: rolling 45 s window, threshold + 60 s cooldown → recall push + LED |
| `hub/policy.py` | CORE | Privacy tags, cloud opt-in (default OFF), outbox for batch cloud jobs |
| `hub/metrics.py` | REL | Bench logger: p50/p95 per stage (asr, proactive, ask_total, ingest…) |
| `hub/registry.py` | REL | Device registry: heartbeats, 30 s leases, resume tokens |
| `hub/dashboard/` | UI | Judge dashboard + `/capture` web-mic fallback |

## Quick start (laptop)

Tasks run through [`dev.py`](dev.py) — plain Python, no `make` needed, same on
Windows / macOS / Linux:

```bash
python dev.py setup                       # venv (py 3.12) + pip install -e ".[dev]"
python dev.py test                        # 44 tests, offline (hash backend)
python dev.py demo                        # seed sample cross-source data
python dev.py ask "Who decided to use JWT authentication?"
python dev.py consolidate                 # MVP: dual-mic merge + dedup + contradictions + importance
python dev.py consolidate --full          # full Dream tier: + OKF, communities, summaries, entity resolution
```

Every command takes `--backend ollama|npu|hash` (default `$RECALL_BACKEND` or
`ollama`). The engine also installs a `recall` console script — e.g. the forget
button:

```bash
recall --db demo.db --backend ollama forget --meeting mtg-auth-sync --last-minutes 5
```

## Running the hub

```bash
python dev.py hub --port 8000            # seeds demo_hub.db, serves :8000
```

- Dashboard: http://localhost:8000 · web-mic capture: http://localhost:8000/capture
- Capture devices connect to `ws://<hub>:8000/ws`: `hello` → `meeting_start` →
  `utterance` (text) or binary PCM16 frames (transcribed in ~4 s windows,
  in-process via faster-whisper by default; `RECALL_ASR_MODE=http` for an
  external server) → `button` (bookmark / forget_last) → `meeting_end`.
- Untitled captures are auto-titled from the first thing said.
- MCP (Claude Desktop / Claude Code):
  `claude mcp add recall -- <pc>/.venv/Scripts/python.exe -m hub.mcp_server --db <pc>/demo_hub.db --backend ollama`

## Event PC (Snapdragon X Elite)

```bash
pip install -e ".[npu]"          # onnxruntime-qnn (ARM64), transformers, onnx
# drop the model assets into models/ (see models/README.md), then:
python -m hub                    # RECALL_BACKEND defaults to npu
```

No code path changes between laptop and event PC — only the backend switch and
the model assets. The NPU providers in `backends/npu.py` are written against the
ORT-QNN / GenieX-QAIRT contracts with guarded imports, so the package still
imports and the `ollama`/`hash` backends still run without the QNN runtime.
