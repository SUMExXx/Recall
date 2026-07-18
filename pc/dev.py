#!/usr/bin/env python3
"""Cross-platform task runner (replaces the Makefile — no `make` needed).

    python dev.py setup                       # venv (3.12) + pip install -e ".[dev]"
    python dev.py test
    python dev.py demo         [--backend ollama|npu|hash]
    python dev.py ask "Who decided to use JWT?"
    python dev.py consolidate  [--full]
    python dev.py stats
    python dev.py hub          [--port 8000]

Backend defaults to $RECALL_BACKEND (or `ollama` for local dev); override with
--backend on any command. Runs the same on Windows, macOS, and Linux.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = "demo.db"


def venv_python() -> str:
    """The project venv's python if it exists, else whatever's running us."""
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    p = ROOT / ".venv" / sub / exe
    return str(p) if p.exists() else sys.executable


def run(cmd: list[str], **env) -> int:
    e = {**os.environ, **{k: str(v) for k, v in env.items()}}
    return subprocess.run(cmd, cwd=ROOT, env=e).returncode


def _py312() -> list[str]:
    """Find a Python 3.12 launcher for `setup` (see memory: 3.14 lacks wheels)."""
    for cand in ([["py", "-3.12"]] if os.name == "nt" else []) + \
                [["python3.12"], ["python3"], ["python"]]:
        try:
            out = subprocess.run(cand + ["--version"], capture_output=True,
                                 text=True)
            if out.returncode == 0 and "3.12" in (out.stdout + out.stderr):
                return cand
        except FileNotFoundError:
            continue
    sys.exit("Python 3.12 not found. Install it (Windows: `py -3.12`) and retry.")


def cmd_setup(_args) -> int:
    py = _py312()
    if run([*py, "-m", "venv", str(ROOT / ".venv")]):
        return 1
    vpy = venv_python()
    return (run([vpy, "-m", "pip", "install", "--upgrade", "pip"])
            or run([vpy, "-m", "pip", "install", "-e", ".[dev,asr]"]))


def cmd_test(_args) -> int:
    return run([venv_python(), "-m", "pytest", "tests/", "-q"])


def _engine(args, *extra: str) -> int:
    return run([venv_python(), "-m", "recall_memory", "--db", DB,
                "--backend", args.backend, *extra])


def cmd_demo(args) -> int:
    return _engine(args, "demo")


def cmd_ask(args) -> int:
    return _engine(args, "ask", args.query)


def cmd_consolidate(args) -> int:
    return _engine(args, "consolidate", *(["--full"] if args.full else []))


def cmd_stats(args) -> int:
    return _engine(args, "stats")


def cmd_hub(args) -> int:
    return run([venv_python(), "scripts/run_hub.py",
                "--backend", args.backend, "--port", str(args.port)])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dev.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default=os.environ.get("RECALL_BACKEND", "ollama"),
                   choices=["npu", "ollama", "hash"],
                   help="model backend (default: $RECALL_BACKEND or ollama)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    sub.add_parser("test").set_defaults(fn=cmd_test)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    ask = sub.add_parser("ask"); ask.add_argument("query"); ask.set_defaults(fn=cmd_ask)
    con = sub.add_parser("consolidate")
    con.add_argument("--full", action="store_true")
    con.set_defaults(fn=cmd_consolidate)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    hub = sub.add_parser("hub"); hub.add_argument("--port", type=int, default=8000)
    hub.set_defaults(fn=cmd_hub)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
