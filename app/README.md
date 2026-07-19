# Recall

Recall is a little memory companion that lives on your phone. It listens in the
background, and when you say "Hey Recall" it saves whatever you say next. Later
you can search through everything you've kept, or just ask a question and it
answers you out loud from your own memories.

It all runs on the phone, offline. There's an optional companion hub you can run
on a PC for better answers, but you don't need it.

## The two halves

The phone does the whole job by itself — capture your speech, store it, answer
questions about it. If you've got the hub running on a PC on the same network,
the phone quietly mirrors its memories over there and can hand questions off to
it for a smarter reply. When the PC isn't around, it just answers on its own.

## Capturing a memory

On first launch it loads its on-device models, asks you to record a few seconds
of your voice (so it can tell who's talking), and shows a quick intro to the
wake word.

After that it's always half-listening. What happens to a bit of speech, roughly:

- the mic audio goes through a voice-activity detector (Silero VAD) that splits
  it into separate utterances
- each utterance gets transcribed by Whisper, right on the phone
- a small filter tosses out the junk — background noise, `[Music]`, anything
  that isn't really language
- unless it started with "Hey Recall" it's ignored — that's the gate that stops
  it from saving everything it hears
- it works out who spoke, and saves it

The glowing orb is just the "I'm listening" cue. It rises when the wake word
fires and drops back once it's caught one message.

## How memories are stored

Each memory gets turned into a list of 384 numbers (an embedding, from a small
bge model) that captures what it means, and goes into a local SQLite database.
Searching is the same idea in reverse — your search text gets embedded too, and
the closest memories come back by cosine similarity. The Memories tab lays them
out as cards you can search, or long-press to select and delete.

## Asking questions

Three ways to ask: type it, tap the mic and talk (your voice goes to Sarvam's
speech-to-text), or say "Hey Recall" on the Ask tab and let the orb catch it.

Where the answer comes from depends on the PC hub. If it's connected, the
question goes there for a fuller answer; if not, it's answered on the phone by
the Snapdragon NPU. Either way the reply is read back to you with Sarvam's
text-to-speech, and the memory it came from shows up in a small box under the
answer — only the answer is spoken, the source just stays on screen.

## Talking to the PC hub

Two separate connections, both plain over your local network.

**Sync.** The phone keeps an outbox of memories it hasn't sent yet and streams
each one to the hub over a WebSocket (`{id, timestamp, speaker, text}`). The hub
stores it and sends back an ack, and only then does the phone cross it off — so
if the connection drops, nothing is lost. A little chip up top shows whether
it's connected.

**Ask.** Questions go over a normal HTTP request to the hub's `/ask`. The hub
finds the relevant memories, runs its own LLM, and returns the answer with a
real source tag attached (added in code, so it's never made up). If the hub
can't be reached it quietly falls back to answering on the phone.

The hub itself is a small FastAPI server. It listens on the whole network,
prints the exact `ws://…:8000/ws` address to paste into the app when it starts,
and can run a few different backends (Ollama on a laptop, the NPU on the event
PC, or a stub for testing).

## How it's wired

Every stage is an interface in [`lib/pipeline/`](lib/pipeline) with a default
implementation in [`lib/impl/`](lib/impl). They're all assembled in one place —
[`lib/memory_pipeline.dart`](lib/memory_pipeline.dart) — so swapping any piece
(a different transcriber, store, or answer engine) is a one-line change and
nothing else has to know.

## Running it

The models aren't checked in (they're big, ~157 MB), so grab them first:

```sh
powershell -File get-models.ps1
flutter pub get
flutter run
```

You'll need a physical arm64 phone — the build targets arm64 only. First launch
loads the models, walks you through voice setup, and you're going. To use the
hub, start it on your PC and paste the `ws://` address it prints into the app's
PC-sync settings.
