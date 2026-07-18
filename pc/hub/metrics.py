"""Metrics + bench logger (§4 REL) — the numbers the hackathon rubric rewards:
per-stage latency (asr / embed / retrieve / llm), query counts, p50/p95."""
from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager


class Metrics:
    def __init__(self, window: int = 200):
        self.samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.counters: dict[str, int] = defaultdict(int)
        self.started_at = time.time()

    def record(self, stage: str, ms: float):
        self.samples[stage].append(ms)

    def incr(self, name: str, n: int = 1):
        self.counters[name] += n

    @contextmanager
    def timer(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - t0) * 1000.0)

    def summary(self) -> dict:
        stages = {}
        for stage, vals in self.samples.items():
            v = list(vals)
            if not v:
                continue
            stages[stage] = {
                "n": len(v),
                "p50_ms": round(statistics.median(v), 1),
                "p95_ms": round(sorted(v)[max(0, int(len(v) * 0.95) - 1)], 1),
                "last_ms": round(v[-1], 1),
            }
        return {"uptime_s": round(time.time() - self.started_at, 1),
                "stages": stages, "counters": dict(self.counters)}
