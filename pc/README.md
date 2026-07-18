# Recall PC Hub — Memory Engine

Implements **`plans/memory_engineering_v2.md`**: NMO-normalized ingestion, fixed-size
chunking, two-tier Matryoshka vectors, 4-tier query router, weighted-RRF retrieval,
the forget button, and the Dream-tier consolidation MVP jobs.

## Layout

| Module | Plan § | What it does |
|---|---|---|
| `recall_memory/nmo.py` | §3-4 | NMO + Episode + Chunk shapes, universal/source metadata, 3-group chunk metadata |
| `recall_memory/store.py` | §4d | SQLite: memories, episodes, chunks, entity_mentions, relations (bi-temporal), FTS5, `vec_chunks` vec0 int8[256] |
| `recall_memory/chunker.py` | §5 | Fixed token windows (meeting 256/15%, code 256/10%, pdf 400/12%, OCR whole-block), speaker-attribution rule |
| `recall_memory/embeddings.py` | §2, §9 | Ollama nomic-embed-text (dev sub for AI Hub Nomic v1.5) + Matryoshka float[768] → int8[256]; deterministic hash fallback |
| `recall_memory/extractors.py` | §6 | Cheap-first entity/decision/action-item extraction; LLM escalation hook |
| `recall_memory/ingest.py` | §14a | Normalize → NMO → enrich → chunk → embed → write every index |
| `recall_memory/router.py` | §8 | Tier 0 commands, tier 1 rules+filters, tier 2 embedding prototypes, tier 3 LLM plan JSON (off by default) |
| `recall_memory/retrieval.py` | §9-11 | needs{}-gated parallel indices, weighted RRF k=60, recency/importance boost, float[768] rescore, reranker slot, small-to-big, cited answers |
| `recall_memory/consolidate.py` | §12 | MVP jobs 1 (dedup), 3 (contradictions, bi-temporal), 4 (importance), 10 (dual-mic reconciliation) |
| `recall_memory/cli.py` | — | `demo / ask / search / ingest-* / forget / consolidate / stats` |

## Quick start

```
make setup          # venv + deps (numpy, sqlite-vec, requests, pytest)
make test           # 23 tests, no Ollama needed (hash embedder)
make demo           # seed sample cross-source data (uses Ollama if running)
make ask Q="Who decided to use JWT authentication?"
make consolidate    # dual-mic merge + dedup + contradictions + importance
```

Forget button:

```
.venv/Scripts/python -m recall_memory --db demo.db forget --meeting mtg-auth-sync --last-minutes 5
```

## Dev substitutes vs event PC (X Elite)

| Role | Here (x64 laptop) | Event PC |
|---|---|---|
| Embeddings | Ollama `nomic-embed-text` (768-dim, same model family) | AI Hub Nomic v1.5, ORT-QNN, seqlen 256/512 re-export |
| LLM (tier-3 planner + synthesis) | Ollama `llama3.2:3b` | Qwen3-4B-Instruct-2507 w4a16, GenieX |
| Reranker | `PassthroughReranker` slot | Qwen3-Reranker-0.6B (bge-reranker CPU fallback) |
| ASR | upstream (server/ from Recall 1.0) | Whisper-Base-En / Small-En, ORT-QNN |

No code path changes — swap the provider classes in `embeddings.py` / `retrieval.py`.
