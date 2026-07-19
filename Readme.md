# Recall

Recall is a memory companion that never forgets what you tell it — and never
sends it anywhere without your say-so. Say something out loud, highlight
something on a webpage, drop in a PDF or a GitHub repo, and it all becomes one
private, searchable memory you can later ask questions of, out loud or in
text, with the original source always attached so you can check the answer.

Everything runs on-device. There's no account, no cloud service you're
required to use — cloud features exist, but they're off unless you turn them
on.

## The four pieces

Recall is made of four things that can each capture memories, plus one of
them — the PC hub — that does the actual thinking for all the others.

| Piece | What it is | Where |
|---|---|---|
| **PC Hub** | The brain. Stores everything, runs the search and the AI model, answers questions. Everything below sends memories *here*. | [`pc/`](pc/) — [readme](pc/README.md) |
| **Phone app** | An always-listening companion — say "Hey Recall" and it saves what you say next. Can also work fully on its own, no PC required. | [`app/`](app/) — [readme](app/README.md) |
| **Browser extension** | Save highlighted text, whole PDFs, GitHub repos, or screenshots straight from your browser. | [`extension/`](extension/) — [readme](extension/README.md) |
| **Arduino UNO Q** | A tiny ambient microphone in the room — say "Hey Arduino!" and it records and sends it to the hub. Shows what it's doing with a little face on its LED display. | [`unoq/`](unoq/) — [readme](unoq/README.md) |

## How they fit together

Think of it as capture → memory → recall:

1. **Capture** — the phone, the browser extension, and the Arduino board are
   all just different ways of getting something *into* Recall: your voice,
   a webpage, a PDF, a repo, a photo of a whiteboard.
2. **Memory** — everything funnels into the **PC hub**, which is the only
   piece that actually understands what it stored: it figures out who said
   what, pulls out names and decisions, and files it away so it can be found
   again by keyword, by meaning, or by how things connect to each other.
3. **Recall** — ask a question from any of the four pieces, and the hub
   searches everything it knows and answers you in plain language, with the
   source cited so you can double-check it.

The phone app is the one exception — it can also capture *and* answer
entirely on its own, using its own on-device model, if the PC hub isn't
running. Everything else needs the hub.

## Getting started

Start with the **PC hub** — it's the only piece the others actually depend
on:

```bash
cd pc
python scripts/run_hub.py
```

Full setup (Python version, models, the NPU vs. laptop-only path) is in
[`pc/README.md`](pc/README.md). Once the hub is running, it prints the
address to point the phone app, browser extension, or Arduino board at.

Then set up whichever capture devices you actually have:
- Phone: [`app/README.md`](app/README.md)
- Browser: [`extension/README.md`](extension/README.md)
- Arduino UNO Q: [`unoq/README.md`](unoq/README.md)

## Privacy, in short

Every piece here talks only to your own PC hub over your own local network —
never to an outside server, unless you deliberately turn on a cloud feature
in the hub's settings. By default, nothing you say or save ever leaves your
devices.
