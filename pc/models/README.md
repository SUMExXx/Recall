# On-device model assets (npu backend)

The `npu` backend (`RECALL_BACKEND=npu`, the default) loads its model assets
from this directory. Paths are configurable via `RECALL_*` env vars — see
`recall_memory/config.py`. Drop the following here on the Snapdragon X Elite
event PC (binaries are git-ignored):

| File (default path)                | Env override                | Source |
|------------------------------------|-----------------------------|--------|
| `nomic-v1.5-seq256.onnx`           | `RECALL_NPU_EMBED_ONNX_256` | Nomic-Embed-Text v1.5 re-exported at seqlen 256 (ORT-QNN) |
| `nomic-v1.5-seq512.onnx`           | `RECALL_NPU_EMBED_ONNX_512` | …re-exported at seqlen 512 (for PDF 400-token chunks) |
| `qwen3-reranker-0.6b.onnx`         | `RECALL_NPU_RERANKER_ONNX`  | Qwen3-Reranker-0.6B (ORT-QNN); bge-reranker-v2-m3 INT8 CPU fallback |

The LLM (Qwen3-4B-Instruct-2507, w4a16) is **not** a file here — it is served by
GenieX/QAIRT behind an OpenAI-compatible HTTP endpoint at
`RECALL_NPU_LLM_ENDPOINT` (default `http://localhost:8090`).

Tokenizers are pulled by id (`RECALL_NPU_TOKENIZER_ID`, default `Qwen/Qwen3-4B`)
via HuggingFace `tokenizers`, cached on first use; ship the `tokenizer.json`
here and point the id at the local path to run fully offline.

Re-export note (plan §17 rec. #2): the AI Hub Nomic catalog default is seqlen
128 — you MUST re-export at 256 and 512 before Phase 3 or chunks get truncated
at the NPU boundary.
