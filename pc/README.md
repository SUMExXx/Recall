# Recall — PC Hub

This is the brain of Recall. It's the program that runs on your computer,
listens to everything you feed it — meetings, notes, PDFs, code, whiteboard
photos — and turns it into one searchable, private memory you can ask
questions of later. Everything else (the phone app, the browser extension)
just sends things here.

Nothing leaves your device unless you turn on cloud features yourself. By
default, Recall is entirely offline.

**The core idea, in one paragraph:** everything you capture gets broken into
small searchable pieces, turned into both a keyword index and a "meaning"
index, and linked up with any people/decisions/facts mentioned in it. When
you ask a question, Recall searches all of that at once, picks the most
relevant pieces, and asks a local AI model to write you an answer — with the
original sources attached, so you can always check where it came from.

## Getting started (Snapdragon X Elite / NPU event PC)

**1. Use ARM64 Python**

The event PC runs Windows on ARM. Install a native **ARM64 build of Python
3.12+** — not the regular x64 installer running under emulation — otherwise
the NPU packages below won't install as real, fast, native wheels.

**2. Install the base requirements**

```bash
cd pc
pip install -r requirements.txt
```

This installs everything the hub needs to run at all (the server, the
database layer, the MCP server). It does **not** yet include the NPU-specific
pieces — that's the next step.

**3. Get the two models**

Recall's on-device thinking uses two separate models:

| Model | What it's for | How it gets here |
|---|---|---|
| **Nomic-Embed-Text v1.5** | Turns text into a "fingerprint" of numbers so similar meanings can be found later — this is what powers search. | Download the ONNX export from **Qualcomm AI Hub** and drop it into `models/` (see `models/README.md` for exact filenames). |
| **Qwen3-4B-Instruct** | The model that actually reads your memories and writes the answer. | Nothing to download by hand — **GenieX** (Qualcomm's on-device model runtime) pulls and caches it automatically the first time the hub runs. |

**4. Turn on NPU processing**

```bash
pip install -e ".[npu]"
```

This adds the Qualcomm-specific packages (`onnxruntime-qnn`, `geniex`) that
let the models above actually run on the NPU chip instead of the CPU.

**5. Start the hub**

```bash
python scripts/run_hub.py
```

That's it — the dashboard is at `http://localhost:8000`, and your phone or
browser extension can now point at this PC to save and search memories.

> **Don't have the Snapdragon hardware?** You can run all of this on a normal
> laptop instead, using [Ollama](https://ollama.com) in place of steps 3–4:
> `RECALL_BACKEND=ollama python scripts/run_hub.py`. Same code, same
> features, just a different (slower, non-NPU) engine underneath.

## Directory structure, briefly

```
pc/
├── recall_memory/     the memory engine: chunking, embeddings, storage,
│                       search/retrieval, and the background "tidy up" job
├── hub/                the server: the API, the live dashboard, and the
│                       MCP server Claude connects to
├── models/             NPU model files go here (see step 3 above)
├── scripts/run_hub.py  starts everything
└── tests/              the test suite — runs fully offline, no models needed
```

## How search actually works (briefly)

When you ask a question, Recall doesn't just do one search — it runs a
keyword search, a "meaning" (vector) search, and a search over how people and
topics are connected, all at once. It merges those results, re-ranks them for
genuine relevance, and only then hands the best handful of passages to the
language model to write a cited answer. This is why Recall can answer things
like "what did we decide" and not just "find me the word decide."

## The database, briefly

Everything lives in **one SQLite file**. The tables that matter most:

- **memories** — one row per thing you saved (a meeting, a PDF, a note…)
- **chunks** — the smaller, searchable pieces each memory gets split into
- **episodes** — individual spoken lines, for meetings specifically
- **entity_mentions / relations** — the "who/what is connected to what" graph
- **fts_chunks** / **vec_chunks** — the keyword index and the meaning index that make search fast

## Using Recall from Claude (or any MCP-compatible tool)

Recall ships its own MCP server, so Claude Desktop, Claude Code, or any other
MCP client can search and ask your memory directly, like any other tool it
has access to.

**What it can do:**

| Tool | What it does |
|---|---|
| `recall_search_memory` | Search memory, get back cited snippets |
| `recall_ask` | Ask a full question, get a synthesized, cited answer |
| `recall_bookmark_moment` | Mark the most recent memory as important |
| `recall_forget_range` | Delete a window of memory (e.g. the last 5 minutes of a meeting) |
| `recall_dump` | List everything currently stored |

**To connect it:**

```bash
claude mcp add recall -- <path-to-pc>/.venv/Scripts/python.exe -m hub.mcp_server --db <path-to-pc>/demo_hub.db --backend ollama
```

Swap `--backend ollama` for `npu` on the event PC. Once connected, you can
just ask Claude things like "search my memory for the JWT decision" and it
will call the tool for you.
