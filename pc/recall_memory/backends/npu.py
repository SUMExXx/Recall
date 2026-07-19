"""NpuBackend — the real target: Snapdragon X Elite, on-device.

Written against the documented runtime contracts from
plans/memory_engineering_v2.md §2. Every heavy import (`onnxruntime`,
`tokenizers`, GenieX) is lazy and guarded, so this module imports fine on a
plain x64 laptop; failures only surface if you actually select `npu` without
the runtime present. On the laptop, use `RECALL_BACKEND=ollama` instead.

  Embedder  Nomic-Embed-Text v1.5, ORT-QNN, re-exported seqlen 256/512 graphs
            (float[768], mean-pooled + L2-normed; task prefixes)
  LLM       Qwen3-4B-Instruct-2507 via GenieX — either in-process through the
            `geniex` package (auto-downloads the precompiled AI Hub bundle,
            runs on the Hexagon NPU via the qairt plugin) or an external
            OpenAI-compatible HTTP endpoint (`geniex serve`). `npu_llm_mode`
            picks; `auto` prefers a running endpoint, else loads in-process.
  Reranker  Qwen3-Reranker-0.6B via ORT-QNN (falls back to passthrough order
            if the asset/runtime is missing)
  Tokenizer the model tokenizer (HF `tokenizers`)
"""
from __future__ import annotations

import importlib.util
import json
import logging
import threading
from functools import cached_property

import numpy as np

from ..config import DIM_FULL, RecallConfig
from ..embeddings import DOC_PREFIX, QUERY_PREFIX, _normalize
from .base import Backend, PassthroughReranker

log = logging.getLogger("recall.backends.npu")

_QNN_PROVIDERS = ["QNNExecutionProvider", "CPUExecutionProvider"]
_QNN_PROVIDER_OPTIONS = [{"backend_path": "QnnHtp.dll"}, {}]
_NOMIC_TOKENIZER = "nomic-ai/nomic-embed-text-v1.5"

_NO_RUNTIME_MSG = (
    "RECALL_BACKEND=npu needs the on-device runtime, which isn't installed here.\n"
    "  - laptop, semantic quality : --backend ollama  (or RECALL_BACKEND=ollama; needs Ollama running)\n"
    "  - laptop, zero-setup check : --backend hash     (offline, deterministic)\n"
    "  - Snapdragon X Elite PC    : pip install -e \".[npu]\"  + drop assets in models/"
)


def _require_onnxruntime():
    try:
        import onnxruntime as ort
        return ort
    except ModuleNotFoundError as e:
        raise RuntimeError(_NO_RUNTIME_MSG) from e


class NomicQnnEmbedder:
    """Nomic v1.5 on the NPU. Two static graphs (seqlen 256 / 512); the shorter
    one that fits the batch is used so nothing is silently truncated at the NPU
    boundary (plan §5). Output is mean-pooled over tokens then L2-normalized."""

    name = "nomic-qnn"

    def __init__(self, cfg: RecallConfig):
        self.cfg = cfg
        self._sessions: dict[int, object] = {}
        self._tok = None

    def _session(self, seqlen: int):
        ort = _require_onnxruntime()
        if seqlen not in self._sessions:
            path = (self.cfg.npu_embed_onnx_256 if seqlen <= 256
                    else self.cfg.npu_embed_onnx_512)
            self._sessions[seqlen] = ort.InferenceSession(
                path, providers=_QNN_PROVIDERS,
                provider_options=_QNN_PROVIDER_OPTIONS)
        return self._sessions[seqlen]

    def _tokenizer(self):
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_pretrained(_NOMIC_TOKENIZER)
        return self._tok

    def _embed(self, texts: list[str]) -> np.ndarray:
        tok = self._tokenizer()
        encs = [tok.encode(t) for t in texts]
        max_len = max((len(e.ids) for e in encs), default=1)
        seqlen = 256 if max_len <= 256 else 512
        n = len(texts)
        input_ids = np.zeros((n, seqlen), dtype=np.int64)
        attn = np.zeros((n, seqlen), dtype=np.int64)
        for i, e in enumerate(encs):
            ids = e.ids[:seqlen]
            input_ids[i, :len(ids)] = ids
            attn[i, :len(ids)] = 1
        sess = self._session(seqlen)
        feeds = {"input_ids": input_ids, "attention_mask": attn}
        # some exports also want token_type_ids
        wanted = {i.name for i in sess.get_inputs()}
        if "token_type_ids" in wanted:
            feeds["token_type_ids"] = np.zeros((n, seqlen), dtype=np.int64)
        out = sess.run(None, {k: v for k, v in feeds.items() if k in wanted})[0]
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 3:  # (n, seq, 768) token embeddings -> masked mean pool
            mask = attn[:, :, None].astype(np.float32)
            out = (out * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return _normalize(out.astype(np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        mat = self._embed([DOC_PREFIX + t for t in texts])
        if mat.shape[-1] != DIM_FULL:
            raise ValueError(f"expected {DIM_FULL}-dim embeddings, got {mat.shape[-1]}")
        return mat

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([QUERY_PREFIX + text])[0]


class QwenGenieLLM:
    """Qwen3-4B-Instruct-2507 via GenieX/QAIRT.

    Genie is deprecating in favor of GenieX (build against GenieX/QAIRT >= 2.42).
    We front the on-device model with an OpenAI-compatible HTTP server (the
    genie sample app / a thin QAIRT wrapper) at `npu_llm_endpoint`, so the
    engine talks the same chat contract regardless of backend.
    """

    name = "qwen3-genie"

    def __init__(self, cfg: RecallConfig):
        self.endpoint = cfg.npu_llm_endpoint.rstrip("/")
        self.model = cfg.npu_llm_model

    @property
    def available(self) -> bool:
        import requests
        try:
            requests.get(f"{self.endpoint}/v1/models", timeout=2).raise_for_status()
            return True
        except Exception:
            return False

    def generate(self, prompt: str, *, json: bool = False,
                 schema: dict | None = None, timeout: float = 180.0) -> str:
        import requests
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "user", "content": prompt}]}
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "plan", "schema": schema}}
        elif json:
            body["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(f"{self.endpoint}/v1/chat/completions",
                              json=body, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as e:
            if schema is not None and e.response is not None and e.response.status_code == 400:
                body["response_format"] = {"type": "json_object"}
                r = requests.post(f"{self.endpoint}/v1/chat/completions",
                                  json=body, timeout=timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            raise e

    def generate_stream(self, prompt: str, *, timeout: float = 180.0):
        """Same OpenAI-compatible endpoint, SSE streaming mode: lines of
        `data: {...}` with incremental `choices[0].delta.content`, terminated
        by a literal `data: [DONE]`."""
        import requests
        body = {"model": self.model, "stream": True,
                "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(f"{self.endpoint}/v1/chat/completions",
                          json=body, timeout=timeout, stream=True)
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            text = line.decode() if isinstance(line, bytes) else line
            if not text.startswith("data: "):
                continue
            payload = text[len("data: "):].strip()
            if payload == "[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta", {}).get("content", "")
            if delta:
                yield delta


class GenieXInProcessLLM:
    """Qwen3-4B-Instruct-2507 loaded in-process via the `geniex` package
    (Qualcomm's on-device GenAI runtime, github.com/qualcomm/GenieX).

    `AutoModelForCausalLM.from_pretrained("ai-hub-models/...")` pulls the
    precompiled, chipset-matched AI Hub bundle into geniex's local cache on
    first use and loads it on the Hexagon NPU (qairt plugin) — no sidecar to
    start, no assets to drop in models/. The hub's startup warmup generate is
    what triggers the download+load, so the first user query doesn't pay it.

    Thread safety: the engine calls the LLM from threadpool workers, and one
    NPU model handle must not run two generations at once — every call
    serializes on `_lock` (streams hold it for the whole iteration).

    JSON: `json_mode=True` grammar-constrains decoding to valid JSON. Schema
    *conformance* is prompt-enforced only (geniex has no json_schema mode) —
    callers on this contract already detect-and-retry malformed shapes.
    """

    name = "geniex-inprocess"

    def __init__(self, cfg: RecallConfig):
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()
        self._failed: Exception | None = None

    @property
    def available(self) -> bool:
        return self._failed is None and importlib.util.find_spec("geniex") is not None

    def _load(self):
        """Load (downloading first if uncached) under the lock. Raises with a
        pointed message on failure and remembers it so `available` flips off
        instead of re-attempting a doomed multi-GB pull every call."""
        if self._model is None:
            if self._failed is not None:
                raise RuntimeError(
                    f"geniex model load already failed: {self._failed}") from self._failed
            try:
                from geniex import AutoModelForCausalLM
                log.info("geniex: loading %s (device_map=%s) — downloads the "
                         "AI Hub bundle on first run", self.cfg.npu_llm_model,
                         self.cfg.npu_llm_device_map)
                # progress left at None (geniex's own default progress UI) —
                # this geniex version rejects progress=True ("must be
                # callable, False, or None").
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.npu_llm_model,
                    device_map=self.cfg.npu_llm_device_map)
            except Exception as e:
                self._failed = e
                raise RuntimeError(
                    f"geniex failed to load {self.cfg.npu_llm_model!r}: {e}\n"
                    "  - is this a Snapdragon device? geniex runs only on Qualcomm SoCs\n"
                    "  - first run needs network access to pull the AI Hub bundle\n"
                    "  - or run `geniex serve` separately and set RECALL_NPU_LLM_MODE=endpoint"
                ) from e
        return self._model

    def _prompt(self, model, prompt: str) -> str:
        # enable_thinking=False skips Qwen3's thinking turn — planner/extractor
        # callers need the answer, not reasoning tokens, and streamed thinking
        # would leak into spoken answers. geniex forces it back to True (with
        # a warning) on models with no thinking mode, so this is always safe.
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, enable_thinking=False)

    def generate(self, prompt: str, *, json: bool = False,
                 schema: dict | None = None, timeout: float = 180.0) -> str:
        import json as _json
        if schema is not None:
            prompt += ("\n\nRespond with ONLY a JSON object that matches this "
                       "JSON Schema — no prose, no code fences:\n"
                       + _json.dumps(schema))
        with self._lock:
            model = self._load()
            out = model.generate(
                self._prompt(model, prompt),
                max_new_tokens=self.cfg.npu_llm_max_new_tokens,
                json_mode=bool(json or schema is not None))
        return out.text.strip()

    def generate_stream(self, prompt: str, *, timeout: float = 180.0):
        """Yields text deltas. GenerateOutput's <think>-stripping only exists
        on the blocking path, but the template already suppresses the thinking
        turn, so deltas stream clean."""
        with self._lock:
            model = self._load()
            streamer = model.generate(
                self._prompt(model, prompt),
                max_new_tokens=self.cfg.npu_llm_max_new_tokens,
                stream=True)
            for chunk in streamer:
                if chunk:
                    yield chunk


class QwenQnnReranker:
    """Qwen3-Reranker-0.6B cross-encoder on the NPU via ORT-QNN.

    Scores each (query, passage) pair and re-sorts. If the asset or QNN runtime
    is missing it degrades to fused order (passthrough) rather than failing the
    query — matches the plan's bge-reranker-CPU-fallback intent.
    """

    name = "qwen3-reranker-qnn"

    def __init__(self, cfg: RecallConfig):
        self.cfg = cfg
        self._sess = None
        self._tok = None
        self._ok = True

    def _load(self):
        if self._sess is None and self._ok:
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer
                self._sess = ort.InferenceSession(
                    self.cfg.npu_reranker_onnx, providers=_QNN_PROVIDERS,
                    provider_options=_QNN_PROVIDER_OPTIONS)
                self._tok = Tokenizer.from_pretrained(self.cfg.npu_tokenizer_id)
            except Exception:
                self._ok = False  # fall back to passthrough for the rest of the run
        return self._sess is not None

    def rerank(self, query: str,
               candidates: list[tuple[int, float, str]]) -> list[tuple[int, float]]:
        if not candidates or not self._load():
            return [(cid, s) for cid, s, _ in candidates]
        scored = []
        for cid, _prev, text in candidates:
            enc = self._tok.encode(f"query: {query}", f"passage: {text}")
            ids = np.array([enc.ids], dtype=np.int64)
            attn = np.array([enc.attention_mask], dtype=np.int64)
            logit = self._sess.run(None, {"input_ids": ids, "attention_mask": attn})[0]
            scored.append((cid, float(np.ravel(logit)[0])))
        scored.sort(key=lambda t: -t[1])
        return scored


class NpuBackend(Backend):
    name = "npu"

    @cached_property
    def embedder(self):
        return NomicQnnEmbedder(self.cfg)

    @cached_property
    def llm(self):
        """npu_llm_mode picks the serving path (decided once, at first use):
        endpoint — the external OpenAI-compatible server, always
        geniex   — in-process via the geniex package, always
        auto     — a server already listening on npu_llm_endpoint wins (so an
                   operator-managed `geniex serve` is respected); otherwise
                   load in-process, which auto-downloads the AI Hub bundle.
                   With neither, the HTTP client's available=False keeps the
                   engine's existing no-LLM degradation."""
        mode = self.cfg.npu_llm_mode
        if mode == "endpoint":
            return QwenGenieLLM(self.cfg)
        if mode == "geniex":
            return GenieXInProcessLLM(self.cfg)
        http = QwenGenieLLM(self.cfg)
        if http.available:
            log.info("npu llm: using running endpoint %s", self.cfg.npu_llm_endpoint)
            return http
        inproc = GenieXInProcessLLM(self.cfg)
        if inproc.available:
            log.info("npu llm: no endpoint up — in-process geniex (%s)",
                     self.cfg.npu_llm_model)
            return inproc
        log.warning("npu llm: no endpoint at %s and geniex not installed — "
                    "LLM unavailable", self.cfg.npu_llm_endpoint)
        return http

    @cached_property
    def reranker(self):
        return QwenQnnReranker(self.cfg)

    @cached_property
    def tokenizer(self):
        from ..tokenizer import ModelTokenizer
        return ModelTokenizer(self.cfg.npu_tokenizer_id)
