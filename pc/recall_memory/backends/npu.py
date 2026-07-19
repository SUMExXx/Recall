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


def _qnn_accelerators():
    """Yield (accelerator_label, backend_dll_path) candidates in priority
    order — Hexagon NPU (HTP) first, then the Adreno GPU, both via QNN.

    A single QNN EP instance targets exactly ONE physical accelerator
    (`backend_path` picks it) — there is no "try HTP, else GPU" within one
    session/provider list, so trying multiple accelerators means attempting a
    FRESH session per candidate and keeping the first one that actually lands
    on QNN (not silently falling back to CPU internally, which ORT does
    per-node without raising). `_qnn_session` below does that.

    Returns nothing if `onnxruntime-qnn` isn't installed (e.g. x64 dev laptop)."""
    try:
        import onnxruntime_qnn as oq
    except ModuleNotFoundError:
        return
    yield "NPU/HTP", oq.get_qnn_htp_path()
    yield "GPU/Adreno", oq.get_qnn_gpu_path()


def _register_qnn(ort) -> bool:
    """The QNN Execution Provider ships in the separate `onnxruntime-qnn`
    package as a *plugin* — it is NOT compiled into the stock `onnxruntime`
    wheel, so `QNNExecutionProvider` is absent from `get_available_providers()`
    until we register the plugin library. Without this call every ORT session
    silently runs on CPU (the bug this fixes)."""
    try:
        import onnxruntime_qnn as oq
    except ModuleNotFoundError:
        return False
    if "QNNExecutionProvider" not in ort.get_available_providers():
        try:
            ort.register_execution_provider_library(oq.get_ep_name(), oq.get_library_path())
        except Exception as e:   # already registered, or unsupported build
            log.warning("QNN EP registration failed (%s) — ORT will use CPU", e)
    return "QNNExecutionProvider" in ort.get_available_providers()


def _qnn_session(path: str, label: str):
    """Build an InferenceSession, trying each Qualcomm accelerator in turn
    (NPU, then GPU) before falling back to plain CPU, and log where it
    actually landed. `sess.get_providers()` drops QNN when the target backend
    claimed zero nodes (e.g. a float graph a given accelerator can't run), so
    its first entry is a reliable "did this really land on QNN?" signal —
    ORT does that fallback silently, per node, without raising, so checking
    the exception path alone would miss it."""
    ort = _require_onnxruntime()
    if _register_qnn(ort):
        for accel_label, backend_path in _qnn_accelerators():
            sess = ort.InferenceSession(
                path, providers=["QNNExecutionProvider", "CPUExecutionProvider"],
                provider_options=[{"backend_path": backend_path}, {}])
            eff = sess.get_providers()
            if eff and eff[0] == "QNNExecutionProvider":
                log.info("%s: running on %s", label, accel_label)
                return sess
            log.warning("%s: %s declined the graph (needs a quantized or "
                       "QNN-context model variant, not the float export) — "
                       "trying the next accelerator", label, accel_label)
    log.warning("%s: no Qualcomm accelerator took the graph — running on CPU", label)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


class NomicQnnEmbedder:
    """Nomic-Embed-Text v1.5 on the NPU via ORT-QNN.

    The graph's I/O contract is READ from the loaded model, not assumed, so the
    same code drives both shapes we ship:

      * the Qualcomm AI Hub bundle (models/nomic_embed_text-onnx-float): fixed
        batch=1, fixed seqlen=128, int32 inputs named ``input_tokens`` /
        ``attention_masks``, and an already mean-pooled ``embeddings`` output
        (float[512] — the Matryoshka-512 level, so set RECALL_EMBEDDING_DIM=512);
      * a generic re-export: dynamic batch, seqlen 256/512, int64 ``input_ids``
        / ``attention_mask``, and a token-level output we mean-pool here.

    NOTE the AI Hub bundle's fixed 128-token window truncates longer chunks at
    the NPU boundary — inherent to that export, not a bug here. Output is
    L2-normalized (idempotent when the graph already normalized)."""

    name = "nomic-qnn"

    def __init__(self, cfg: RecallConfig):
        self.cfg = cfg
        self._sess = None
        self._spec: dict | None = None
        self._tok = None

    def _session(self):
        if self._sess is None:
            # AI Hub ships one fixed-shape graph (both onnx_256/512 point at it);
            # a two-graph re-export can differ, in which case the longer-seqlen
            # 512 path is the safe default.
            path = self.cfg.npu_embed_onnx_512 or self.cfg.npu_embed_onnx_256
            self._sess = _qnn_session(path, "embedder")
            self._spec = self._introspect(self._sess)
        return self._sess

    @staticmethod
    def _np_dtype(ort_type: str):
        return {"tensor(int32)": np.int32,
                "tensor(int64)": np.int64}.get(ort_type, np.int64)

    def _introspect(self, sess) -> dict:
        """Derive feed keys, dtypes, seqlen and batching from the real graph."""
        ins = sess.get_inputs()
        mask_in = next((i for i in ins if "mask" in i.name.lower()), None)
        tok_in = next((i for i in ins if i is not mask_in), ins[0])

        def fixed(dim):
            return dim if isinstance(dim, int) and dim > 0 else None

        seqlen = fixed(tok_in.shape[1]) if len(tok_in.shape) > 1 else None
        names = {i.name for i in ins}
        return {
            "tok_name": tok_in.name,
            "mask_name": mask_in.name if mask_in is not None else None,
            "tok_dtype": self._np_dtype(tok_in.type),
            "mask_dtype": self._np_dtype(mask_in.type) if mask_in is not None else np.int64,
            "type_ids": "token_type_ids" in names,
            "seqlen": seqlen or 512,                # dynamic graph -> generous cap
            "batch1": fixed(tok_in.shape[0]) == 1,  # fixed batch=1 -> feed one at a time
        }

    def _tokenizer(self):
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_pretrained(_NOMIC_TOKENIZER)
        return self._tok

    def _embed(self, texts: list[str]) -> np.ndarray:
        tok = self._tokenizer()
        sess = self._session()
        sp = self._spec
        seqlen, n = sp["seqlen"], len(texts)
        ids = np.zeros((n, seqlen), dtype=sp["tok_dtype"])
        attn = np.zeros((n, seqlen), dtype=sp["mask_dtype"])
        for i, t in enumerate(texts):
            enc = tok.encode(t).ids[:seqlen]
            ids[i, :len(enc)] = enc
            attn[i, :len(enc)] = 1

        def _run(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            feeds = {sp["tok_name"]: a}
            if sp["mask_name"]:
                feeds[sp["mask_name"]] = b
            if sp["type_ids"]:
                feeds["token_type_ids"] = np.zeros_like(a)
            return np.asarray(sess.run(None, feeds)[0], dtype=np.float32)

        if sp["batch1"]:  # fixed batch=1 graph: run each row, stack results
            out = np.concatenate([_run(ids[i:i + 1], attn[i:i + 1])
                                  for i in range(n)], axis=0)
        else:
            out = _run(ids, attn)

        if out.ndim == 3:  # (n, seq, dim) token embeddings -> masked mean pool
            mask = attn[:, :, None].astype(np.float32)
            out = (out * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return _normalize(out.astype(np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        mat = self._embed([DOC_PREFIX + t for t in texts])
        if mat.shape[-1] != DIM_FULL:
            raise ValueError(
                f"expected {DIM_FULL}-dim embeddings, got {mat.shape[-1]} — set "
                f"RECALL_EMBEDDING_DIM={mat.shape[-1]} to match this model")
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

    def _recover(self, exc: Exception) -> None:
        """A transient NPU/QAIRT graph-execution fault ("Graph execute failed",
        raised as GenieXError) can leave the model handle permanently wedged —
        confirmed empirically: every subsequent call on that SAME handle fails
        immediately (even at n_past=0, right after reset()), for the rest of
        the process's life. reset() cannot repair it, only a fresh handle can.

        A bare reload without closing the old handle first ALSO fails —
        confirmed: "Could not create context from binary ... Device Free
        failure" — the wedged handle never released its device-side QAIRT
        context, so the new handle can't allocate a replacement. close() is
        the documented release path; call it best-effort (it may itself raise
        on an already-corrupted handle — that's fine, we're discarding it
        either way) before dropping the reference so the NEXT call reloads
        clean instead of staying wedged until the whole hub process restarts."""
        log.warning("geniex: generate failed (%s) — releasing and reloading "
                   "the model handle on the next call", exc)
        model, self._model = self._model, None
        if model is not None:
            try:
                model.close()
            except Exception:
                pass

    def _prompt(self, model, prompt: str) -> str:
        # enable_thinking=False skips Qwen3's thinking turn — planner/extractor
        # callers need the answer, not reasoning tokens, and streamed thinking
        # would leak into spoken answers. geniex forces it back to True (with
        # a warning) on models with no thinking mode, so this is always safe.
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, enable_thinking=False)

    def _check_context_length(self, profile) -> None:
        """geniex absorbs a context-length overflow into the C layer rather
        than raising — `.text` comes back EMPTY and `profile.stop_reason ==
        'context_length'` with no exception, at all. Every caller in this
        codebase (ask, rerank, title/entity/relation extraction) treats an
        empty string as "the model said nothing", not "the model failed" — so
        without this check, a context overflow silently became a blank answer
        that still logged `total_ms` in the low single digits and `OK` in the
        trace (observed live: `llm_synthesize_stream 2.3 ms ... response=''`).
        Raising surfaces it as a real failure so existing except/fallback
        paths (ask_stream's non-streaming _synthesize retry, _no_context_answer,
        the reranker's fused-order fallback) actually engage."""
        if not profile or profile.stop_reason != "context_length":
            return
        raise RuntimeError(
            f"geniex: prompt exceeded the model's context window "
            f"({profile.prompt_tokens} prompt tokens) — reset() was applied "
            "but this single prompt is still too long on its own")

    def generate(self, prompt: str, *, json: bool = False,
                 schema: dict | None = None, timeout: float = 180.0) -> str:
        import json as _json
        if schema is not None:
            prompt += ("\n\nRespond with ONLY a JSON object that matches this "
                       "JSON Schema — no prose, no code fences:\n"
                       + _json.dumps(schema))
        with self._lock:
            model = self._load()
            # Every call here is an independent one-shot prompt (title refine,
            # entity/relation extraction, reranking, synthesis — none of these
            # are a multi-turn conversation), but geniex's KV cache is
            # conversational by default and never clears itself. Without this
            # reset, n_past keeps climbing across EVERY call for the life of
            # the process until it overflows the model's context window —
            # observed as a silent empty response (see _check_context_length).
            model.reset()
            try:
                out = model.generate(
                    self._prompt(model, prompt),
                    max_new_tokens=self.cfg.npu_llm_max_new_tokens,
                    json_mode=bool(json or schema is not None))
            except Exception as e:
                self._recover(e)
                raise
            self._check_context_length(out.profile)
        return out.text.strip()

    def generate_stream(self, prompt: str, *, timeout: float = 180.0):
        """Yields text deltas. GenerateOutput's <think>-stripping only exists
        on the blocking path, but the template already suppresses the thinking
        turn, so deltas stream clean."""
        with self._lock:
            model = self._load()
            model.reset()   # see generate() — same stateful-KV-cache issue
            streamer = model.generate(
                self._prompt(model, prompt),
                max_new_tokens=self.cfg.npu_llm_max_new_tokens,
                stream=True)
            try:
                for chunk in streamer:
                    if chunk:
                        yield chunk
            except Exception as e:
                self._recover(e)
                raise
            self._check_context_length(streamer.output.profile if streamer.output else None)


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
                from tokenizers import Tokenizer
                self._sess = _qnn_session(self.cfg.npu_reranker_onnx, "reranker")
                self._tok = Tokenizer.from_pretrained(self.cfg.npu_tokenizer_id)
            except Exception as e:
                # No valid ONNX asset (the default config points at a .gguf,
                # which ORT can't load) or QNN runtime missing -> passthrough
                # order for the rest of the run.
                log.warning("reranker unavailable (%s) — using fused order", e)
                self._ok = False
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
        """Prefer a real cross-encoder ONNX on the NPU when the asset is
        actually present; otherwise rerank with the on-device LLM (a genuine
        relevance judgment) instead of silently keeping fused order. The stock
        config points npu_reranker_onnx at a .gguf that (a) doesn't ship and
        (b) ORT can't load — so in practice the LLM reranker is what runs."""
        import os
        path = self.cfg.npu_reranker_onnx
        if path and path.lower().endswith(".onnx") and os.path.exists(path):
            return QwenQnnReranker(self.cfg)
        from .base import LlmReranker
        log.info("npu reranker: no cross-encoder ONNX asset — reranking via the "
                 "on-device LLM (%s)", self.cfg.npu_llm_model)
        return LlmReranker(self.llm, self.cfg)

    @cached_property
    def tokenizer(self):
        from ..tokenizer import ModelTokenizer
        return ModelTokenizer(self.cfg.npu_tokenizer_id)
