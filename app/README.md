# Recall (phone app)

On-device AI memory. Capture speech → detect the speaker → transcribe → store as a
searchable memory. Fully offline; every model runs on the phone.

## Pipeline, as swappable modules

Each stage is a Dart `abstract interface class` in [`lib/pipeline/`](lib/pipeline).
The default provider for each lives in [`lib/impl/`](lib/impl). Swap a provider by
changing the one `create()` call in [`lib/memory_pipeline.dart`](lib/memory_pipeline.dart)
(the composition root) — nothing else depends on the concrete classes.

| Interface | Default impl | Backed by |
|-----------|-------------|-----------|
| `SpeechCapture` | `MicVadCapture` | `mic_stream` mic + sherpa-onnx Silero VAD |
| `SpeakerRecognizer` | `SherpaSpeakerRecognizer` | sherpa-onnx speaker embeddings + sqflite profiles |
| `Transcriber` | `SherpaWhisperTranscriber` | sherpa-onnx Whisper tiny.en (INT8) |
| `VectorStore` | `LocalMemory` | sqflite + bge-small embeddings, cosine search |
| `InferenceEngine` | `QualInferenceEngine` | ONNX Runtime on the Snapdragon NPU (NNAPI), RAG over the memory store |

`VectorStore` has a second impl, `SyncedMemory` (remote/synced backend), stubbed and
not yet wired in. Audio contract everywhere: 16 kHz mono float PCM in [-1, 1].

## Setup

Models are not committed (~157 MB). Download them into `assets/models/` first:

```
powershell -File get-models.ps1
flutter pub get
flutter run          # needs a physical arm64 device (see notes)
```

## Notes

- **Build targets arm64 only** ([`android/app/build.gradle.kts`](android/app/build.gradle.kts)).
  Add `armeabi-v7a` / `x86_64` there for older phones or an emulator.
- Whisper `decode()` is a blocking native call — a few seconds per utterance will
  jank the UI. Move the recognizer into a worker isolate if smoothness matters
  (see the `ponytail:` note in `sherpa_whisper_transcriber.dart`).
- No wake word yet: capture is a manual Start/Stop toggle; VAD segments speech into
  utterances. A `WakeWordDetector` stage can be added the same way (sherpa-onnx has a
  `KeywordSpotter`).
