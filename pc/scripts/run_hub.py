"""Dev launcher: seed a demo DB if empty, then serve the hub.

Usage:
  python scripts/run_hub.py [--db demo_hub.db] [--port 8000] [--no-seed]
                            [--backend ollama|npu|hash]

The backend defaults to RECALL_BACKEND (npu). On your laptop run with
`--backend ollama` (Ollama running) or `--backend hash` (fully offline).
"""
import argparse
import os
import sys

PC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PC_DIR)
os.chdir(PC_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="demo_hub.db")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-seed", action="store_true")
    ap.add_argument("--backend", default=None, choices=["npu", "ollama", "hash"],
                    help="override RECALL_BACKEND for this run")
    args = ap.parse_args()

    os.environ["RECALL_DB_PATH"] = args.db
    if args.backend:
        os.environ["RECALL_BACKEND"] = args.backend

    if not args.no_seed:
        from recall_memory.backends import get_backend
        from recall_memory.config import RecallConfig
        from recall_memory.demo_data import seed
        from recall_memory.ingest import Ingestor
        from recall_memory.store import MemoryStore
        cfg = RecallConfig(db_path=args.db)
        store = MemoryStore(args.db)
        if store.stats()["memories"] == 0:
            seed(Ingestor(store, cfg, get_backend(cfg)))
            print(f"seeded demo data into {args.db} (backend={cfg.backend})")
        store.close()

    import uvicorn
    uvicorn.run("hub.app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
