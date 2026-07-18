"""Structured per-operation step tracing — a full step-by-step record of ONE
end-to-end operation, written to a text file for after-the-fact debugging.

This is deliberately separate from the `logging` module loggers
(`recall.hub`, `recall.ingest`, `recall.retrieval`, `recall.asr`, ...), which
each emit ONE summary line per operation for live monitoring. A `Trace`
instead captures EVERY named stage of one operation — chunking, embedding,
each retrieval route, RRF fusion, reranking, the query-planner tier that
fired, the exact LLM prompt and response — with wall-clock timing, then
writes the whole story as one readable block to `RECALL_TRACE_FILE`
(default `logs/recall_trace.log`) when the operation finishes.

Usage — wrap exactly one top-level operation per request:

    with ensure_trace("ask", query=query, k=top_k):
        with step("embed_query") as s:
            vec = embedder.embed_query(query)
            s.detail(embedder=embedder.name, dim=len(vec))
        ...

`ensure_trace` only opens a NEW trace if none is already active on this
thread — so a lower-level function (e.g. `Retriever.retrieve`) can open its
own top-level trace when called standalone (`/search`), but nest cleanly
under an outer one when called from `Retriever.ask()`. Nested code anywhere
on the call stack — including inside `anyio.to_thread.run_sync` worker
threads, since contextvars propagate across that boundary — can call
`step(...)` without the Trace object being threaded through every function
signature; if no trace is active, `step()` is a harmless no-op.

Call `configure(cfg)` once (done automatically by `backends.get_backend`)
to pick up `RECALL_TRACE_ENABLED` / `RECALL_TRACE_FILE` from RecallConfig.
File writing can be disabled (steps are still visible via `logging` at
DEBUG) with `RECALL_TRACE_ENABLED=false`.
"""
from __future__ import annotations

import contextvars
import itertools
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime

log = logging.getLogger("recall.trace")

TRACE_ENABLED = True
TRACE_FILE = "logs/recall_trace.log"

_MAX_BYTES = 10 * 1024 * 1024   # rotate past 10 MB so the file never grows unbounded
_DETAIL_CHARS = 1200            # cap on any single detail value written to the file

_current: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar(
    "recall_trace", default=None)
_seq = itertools.count(1)
_file_lock = threading.Lock()


def configure(cfg) -> None:
    """Pick up trace_enabled / trace_file from a RecallConfig. Idempotent —
    safe to call once per backend construction (it always is, via get_backend)."""
    global TRACE_ENABLED, TRACE_FILE
    TRACE_ENABLED = bool(getattr(cfg, "trace_enabled", TRACE_ENABLED))
    TRACE_FILE = getattr(cfg, "trace_file", TRACE_FILE)


class Step:
    """One named stage within a Trace. `with step("name") as s: ... s.detail(k=v)`."""

    __slots__ = ("trace", "name", "data", "t0")

    def __init__(self, trace: "Trace | None", name: str, data: dict):
        self.trace = trace
        self.name = name
        self.data = data
        self.t0 = 0.0

    def detail(self, **kw) -> None:
        self.data.update(kw)

    def __enter__(self) -> "Step":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        ms = (time.perf_counter() - self.t0) * 1000.0
        if exc is not None:
            self.data["error"] = f"{exc_type.__name__}: {exc}"
        if self.trace is not None:
            self.trace._steps.append({"step": self.name, "ms": round(ms, 1),
                                      **self.data})
        return False   # never suppress — tracing must not change control flow


class Trace:
    """One end-to-end operation (an ask / ingest / consolidate / asr run)."""

    def __init__(self, kind: str, meta: dict):
        self.kind = kind
        self.meta = meta
        self.trace_id = f"{kind}-{next(_seq)}"
        self._steps: list[dict] = []
        self._t0 = time.perf_counter()
        self._token = None

    def step(self, name: str, **detail) -> Step:
        return Step(self, name, detail)

    def __enter__(self) -> "Trace":
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        total_ms = (time.perf_counter() - self._t0) * 1000.0
        _current.reset(self._token)
        _emit(self, total_ms, exc)
        return False


@contextmanager
def start_trace(kind: str, **meta):
    """Always begin a NEW top-level trace, even if one is already active
    (rarely what you want directly — prefer `ensure_trace`)."""
    with Trace(kind, meta) as t:
        yield t


@contextmanager
def ensure_trace(kind: str, **meta):
    """Begin a trace only if none is active on this thread; otherwise attach
    to the one already running. Use this at every public entry point so
    standalone calls (CLI, /search, MCP) still get a top-level trace, while
    calls nested under another (retrieve() inside ask()) don't fork a second,
    truncated trace."""
    existing = _current.get()
    if existing is not None:
        yield existing
    else:
        with Trace(kind, meta) as t:
            yield t


class _NoopStep:
    """Returned by step() when no Trace is active — one attribute lookup, no I/O."""
    __slots__ = ()

    def detail(self, **kw) -> None:
        pass

    def __enter__(self) -> "_NoopStep":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


_NOOP = _NoopStep()


def step(name: str, **detail):
    """Record one stage on whichever Trace is active in this thread (or its
    parent, across an anyio worker-thread boundary); a no-op outside a trace."""
    t = _current.get()
    return t.step(name, **detail) if t is not None else _NOOP


def current_trace_id() -> str | None:
    t = _current.get()
    return t.trace_id if t is not None else None


def _truncate(v):
    if isinstance(v, str) and len(v) > _DETAIL_CHARS:
        return v[:_DETAIL_CHARS] + f"…[{len(v)} chars total]"
    return v


def _format_block(trace: Trace, total_ms: float, exc: BaseException | None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "OK" if exc is None else f"FAILED: {type(exc).__name__}: {exc}"
    meta_str = ", ".join(f"{k}={_truncate(v)!r}" for k, v in trace.meta.items())

    lines = ["=" * 100,
             f"[{ts}] TRACE {trace.kind}  {trace.trace_id}  "
             f"total={total_ms:.0f}ms  {status}"]
    if meta_str:
        lines.append(f"  meta: {meta_str}")
    lines.append("-" * 100)

    for i, s in enumerate(trace._steps, 1):
        extra = {k: _truncate(v) for k, v in s.items()
                 if k not in ("step", "ms", "error")}
        multiline = {k: v for k, v in extra.items()
                     if isinstance(v, str) and "\n" in v}
        inline = {k: v for k, v in extra.items() if k not in multiline}
        inline_str = " ".join(f"{k}={v!r}" for k, v in inline.items())
        err_str = f"   ERROR: {s['error']}" if "error" in s else ""
        lines.append(f"  {i:>3}. {s['step']:<28} {s['ms']:>9.1f} ms   "
                     f"{inline_str}{err_str}")
        for k, v in multiline.items():
            for j, ln in enumerate(v.split("\n")):
                lines.append((f"       {k}: " if j == 0 else "          ") + ln)

    lines.append("=" * 100)
    lines.append("")
    return "\n".join(lines)


def _emit(trace: Trace, total_ms: float, exc: BaseException | None) -> None:
    block = _format_block(trace, total_ms, exc)
    log.debug("%s", block)
    if not TRACE_ENABLED:
        return
    try:
        _write(block)
    except Exception:
        log.exception("failed to write trace file %s", TRACE_FILE)


def _write(block: str) -> None:
    with _file_lock:
        d = os.path.dirname(TRACE_FILE)
        if d:
            os.makedirs(d, exist_ok=True)
        if os.path.exists(TRACE_FILE) and os.path.getsize(TRACE_FILE) > _MAX_BYTES:
            try:
                os.replace(TRACE_FILE, TRACE_FILE + ".1")
            except OSError:
                pass
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(block + "\n")
