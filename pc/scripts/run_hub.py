"""Hub launcher.

Usage:
  python scripts/run_hub.py [--db demo_hub.db] [--port 8000] [--seed]
                            [--backend ollama|npu|hash]

Starts EMPTY by default — pass --seed to load the sample cross-source demo
data. Backend defaults to RECALL_BACKEND (pc/.env).
"""
import argparse
import logging
import os
import sys

PC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PC_DIR)
os.chdir(PC_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="demo_hub.db")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", action="store_true",
                    help="seed the sample demo data (default: start empty)")
    ap.add_argument("--backend", default=None, choices=["npu", "ollama", "hash"],
                    help="override RECALL_BACKEND for this run")
    args = ap.parse_args()

    os.environ["RECALL_DB_PATH"] = args.db
    if args.backend:
        os.environ["RECALL_BACKEND"] = args.backend

    from hub.asr import make_local_whisper
    from recall_memory.backends import get_backend
    from recall_memory.config import RecallConfig
    cfg = RecallConfig(db_path=args.db)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname).1s %(name)s  %(message)s",
        datefmt="%H:%M:%S")

    if args.seed:
        from recall_memory.demo_data import seed
        from recall_memory.ingest import Ingestor
        from recall_memory.store import MemoryStore
        store = MemoryStore(args.db)
        if store.stats()["memories"] == 0:
            seed(Ingestor(store, cfg, get_backend(cfg)))
            print(f"seeded demo data into {args.db} (backend={cfg.backend})")
        store.close()

    asr = make_local_whisper(cfg)
    asr_how = (f"in-process faster-whisper ({cfg.whisper_model})"
               if type(asr).__name__ == "FasterWhisperProvider"
               else f"http server at {cfg.whisper_url}")
    print(f"""
Recall PC Hub
  backend    {cfg.backend}   (RECALL_BACKEND / pc/.env)
  db         {args.db}{'' if args.seed else '   (empty start — use --seed for demo data)'}
  asr        {asr_how}
  dream      consolidation every {cfg.consolidate_every_s:.0f}s (idle Dream tier)
  dashboard  http://localhost:{args.port}
  mic        http://localhost:{args.port}/capture
""")

    import uvicorn
    # SINGLE process — do not set workers=N. This was set back to 3 and is the
    # confirmed root cause of the duplicate/overlapping dream_tier runs and
    # trace-id resets seen in the trace log (each worker is a separate process
    # with its OWN HubState: its own `_last_dream`/`_dream_stats`, its own
    # in-memory trace sequence counter, its own WS connection set). Concretely,
    # with workers=3 you get three independent supervisor loops, each
    # deciding on its own schedule to run the Dream tier against the SAME
    # sqlite file — hence two consolidations of the same meeting ~1s apart.
    # It also reproduces the earlier "database is locked" crash (two writers)
    # and silently drops events (a WS client's socket lives on one worker;
    # captures on another worker never reach it without a reload).
    # Heavy work (ASR, embed, ingest, Dream tier) is already offloaded to
    # background threads inside this ONE process, so the event loop never
    # blocks — there is no concurrency upside to more worker processes here,
    # only shared-state bugs. If you need to scale request handling, that has
    # to go through a shared store (Postgres) + a shared pub/sub for the WS
    # fabric first; don't set workers>1 on this app before that exists.
    uvicorn.run("hub.app:app", host="127.0.0.1", port=args.port,
                log_level=cfg.log_level.lower())


if __name__ == "__main__":
    main()
