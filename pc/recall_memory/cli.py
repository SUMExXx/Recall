"""Recall memory engine CLI.

  python -m recall_memory demo                     seed sample data
  python -m recall_memory ask "who decided ..."    route + retrieve + answer
  python -m recall_memory search "jwt"             retrieval only (no LLM)
  python -m recall_memory ingest-meeting file.json
  python -m recall_memory ingest-text file.txt --type pdf --title "..."
  python -m recall_memory forget --meeting mtg-x --last-minutes 5
  python -m recall_memory consolidate
  python -m recall_memory stats
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .backends import get_backend
from .config import RecallConfig
from .consolidate import Consolidator
from .ingest import Ingestor
from .retrieval import Retriever
from .store import MemoryStore


def _open(args) -> tuple[MemoryStore, RecallConfig]:
    overrides = {"db_path": args.db}
    if args.backend:                       # else fall through to RECALL_BACKEND/env
        overrides["backend"] = args.backend
    if getattr(args, "tier3", False):
        overrides["use_tier3_planner"] = True
    cfg = RecallConfig(**overrides)
    return MemoryStore(cfg.db_path), cfg


def main(argv=None) -> int:
    """Entry point — turns backend/runtime errors into a clean message."""
    try:
        return _main(argv)
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="recall_memory", description=__doc__)
    p.add_argument("--db", default="recall.db")
    p.add_argument("--backend", default=None, choices=["npu", "ollama", "hash"],
                   help="override RECALL_BACKEND (default: npu)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo")
    sub.add_parser("stats")
    cons = sub.add_parser("consolidate")
    cons.add_argument("--full", action="store_true",
                      help="run the full Dream tier (OKF, communities, "
                           "summaries, entity resolution, archival, repair)")

    ask = sub.add_parser("ask")
    ask.add_argument("query")
    ask.add_argument("--tier3", action="store_true")
    ask.add_argument("-k", type=int, default=6)

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=6)

    im = sub.add_parser("ingest-meeting")
    im.add_argument("file")

    it = sub.add_parser("ingest-text")
    it.add_argument("file")
    it.add_argument("--type", default="text",
                    choices=["text", "note", "pdf", "image"])
    it.add_argument("--title", default=None)

    fg = sub.add_parser("forget")
    fg.add_argument("--meeting", required=False)
    fg.add_argument("--memory", required=False)
    fg.add_argument("--last-minutes", type=float, default=None)

    args = p.parse_args(argv)
    store, cfg = _open(args)
    backend = get_backend(cfg)

    if args.cmd == "demo":
        from .demo_data import seed
        ids = seed(Ingestor(store, cfg, backend))
        print(json.dumps({"seeded": ids, "stats": store.stats(),
                          "backend": backend.name,
                          "embedder": backend.embedder.name}, indent=2))

    elif args.cmd == "stats":
        print(json.dumps({**store.stats(), "backend": backend.name}, indent=2))

    elif args.cmd == "ask":
        r = Retriever(store, cfg, backend)
        t0 = time.perf_counter()
        out = r.ask(args.query, top_k=args.k)
        ms = (time.perf_counter() - t0) * 1000
        plan = out["plan"]
        print(f"[tier {plan.tier} | {plan.query_type} | profile "
              f"{plan.weight_profile} | path {plan.path} | {ms:.0f} ms]")
        if out.get("command"):
            print("COMMAND:", out["command"])
        else:
            print(out["answer"])
            print("\nSources:")
            for c in out["contexts"]:
                print(f"  {c.citation()} {c.source_type:12s} {c.title}")

    elif args.cmd == "search":
        r = Retriever(store, cfg, backend)
        for c in r.retrieve(args.query, top_k=args.k):
            print(f"{c.score:6.3f} {c.citation()} {c.source_type:12s} "
                  f"{c.title[:40]:40s} {c.text[:80]!r}")

    elif args.cmd == "ingest-meeting":
        with open(args.file, encoding="utf-8") as f:
            meeting = json.load(f)
        mid = Ingestor(store, cfg, backend).ingest_meeting(meeting)
        print(f"ingested meeting -> memory_id {mid}")

    elif args.cmd == "ingest-text":
        with open(args.file, encoding="utf-8") as f:
            content = f.read()
        mid = Ingestor(store, cfg, backend).ingest_document(
            args.type, args.title or args.file, content)
        print(f"ingested {args.type} -> memory_id {mid}")

    elif args.cmd == "forget":
        if args.memory:
            print(json.dumps(store.forget_memory(args.memory)))
        elif args.meeting and args.last_minutes:
            t_to = time.time()
            out = store.forget_time_range(
                args.meeting, t_to - args.last_minutes * 60, t_to)
            print(json.dumps(out))
        else:
            print("need --memory or (--meeting and --last-minutes)")
            return 2

    elif args.cmd == "consolidate":
        c = Consolidator(store, cfg, Ingestor(store, cfg, backend))
        out = c.run_full() if getattr(args, "full", False) else c.run_mvp()
        print(json.dumps(out, indent=2))

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
