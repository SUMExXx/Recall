# Recall

Ambient, privacy-first, **on-device AI memory companion** — Qualcomm Snapdragon
Multiverse Hackathon. Meetings, code, PDFs, and whiteboards become one searchable,
citable memory that never leaves the device.

- **Engineering plan:** `plans/memory_engineering_v2.md`
- **System architecture:** `docs/system_architectures_v1.md`
- **Implementation:** [`pc/`](pc/) — memory engine + PC hub (start here: [`pc/README.md`](pc/README.md))

The build targets the **Snapdragon X Elite / NPU**. A single switch,
`RECALL_BACKEND`, runs the identical code on a laptop via Ollama
(`RECALL_BACKEND=ollama`) or fully offline for tests (`hash`).

```bash
cd pc
python dev.py setup
python dev.py test                    # 44 tests, offline
python dev.py demo --backend ollama   # try it on your laptop
```
