import 'dart:async';
import 'dart:typed_data';

import 'impl/local_memory.dart';
import 'impl/mic_vad_capture.dart';
import 'impl/qual_inference_engine.dart';
import 'impl/sherpa_speaker_recognizer.dart';
import 'impl/sherpa_whisper_transcriber.dart';
import 'impl/synced_memory.dart';
import 'pipeline/inference_engine.dart';
import 'pipeline/speaker_recognizer.dart';
import 'pipeline/speech_capture.dart';
import 'pipeline/transcriber.dart';
import 'pipeline/vector_store.dart';

/// Composition root + facade. This is the ONE place that picks concrete
/// providers for each interface — swap a `create()` call here to change the
/// implementation of any stage. Everything else depends only on the interfaces.
class MemoryPipeline {
  final SpeechCapture _capture;
  final SpeakerRecognizer _speaker;
  final Transcriber _transcriber;
  final VectorStore _store;
  final SyncedMemory _synced; // same object as _store; kept for sync controls
  final InferenceEngine _inference;

  final _onMemory = StreamController<Memory>.broadcast();
  StreamSubscription<Float32List>? _sub;
  Future<void> _tail = Future.value(); // serializes utterance processing

  MemoryPipeline._(this._capture, this._speaker, this._transcriber, this._store,
      this._synced, this._inference);

  static Future<MemoryPipeline> create() async {
    // Sequential loads — ~150 MB of models; avoid loading all at once.
    // Local storage, wrapped by the synced store which mirrors to the PC hub.
    final local = await LocalMemory.create();
    final store = await SyncedMemory.create(local: local);
    final transcriber = await SherpaWhisperTranscriber.create();
    final speaker = await SherpaSpeakerRecognizer.create();
    final capture = await MicVadCapture.create();
    // Inference engine runs on the Snapdragon NPU and reads from the memory store.
    final inference = await QualInferenceEngine.create(memory: store);

    final pipe =
        MemoryPipeline._(capture, speaker, transcriber, store, store, inference);
    pipe._sub = capture.utterances.listen(pipe._enqueue);
    return pipe;
  }

  Stream<String> get status => _capture.status;

  /// Fires when a new memory is captured and stored.
  Stream<Memory> get onMemory => _onMemory.stream;

  Future<bool> start() => _capture.start();

  Future<void> stop() => _capture.stop();

  void _enqueue(Float32List samples) {
    _tail = _tail.then((_) => _process(samples));
  }

  Future<void> _process(Float32List samples) async {
    final text = await _transcriber.transcribe(samples);
    if (text.isEmpty) return;
    final who = await _speaker.identify(samples) ?? 'unknown';
    final id = await _store.add(Memory(
      timestamp: DateTime.now(),
      speaker: who,
      text: text,
      embedding: Float32List(0),
    ));
    _onMemory.add(Memory(
      id: id,
      timestamp: DateTime.now(),
      speaker: who,
      text: text,
      embedding: Float32List(0),
    ));
  }

  Future<List<Memory>> recent(int limit) => _store.recent(limit);

  Future<List<Memory>> search(String query) => _store.search(query);

  /// Answers a natural-language question using the on-device inference engine.
  Future<String> ask(String question) => _inference.ask(question);

  /// Whether the on-device LLM is downloaded/ready (else Ask answers extractively).
  Future<bool> modelReady() => _inference.isModelReady();

  /// Downloads a GenieX model (by Qualcomm AI Hub name) onto the device.
  Future<void> downloadModel(String name) => _inference.downloadModel(name);

  /// Registers a local GenieX bundle folder (e.g. `/sdcard/models/my-bundle`).
  Future<void> registerLocalModel(String path) => _inference.registerLocalModel(path);

  /// Whether all-files storage access (to read local bundles) is granted.
  Future<bool> hasStorageAccess() => _inference.hasStorageAccess();

  /// Prompts the user to grant all-files storage access.
  Future<void> requestStorageAccess() => _inference.requestStorageAccess();

  /// Live PC-sync status ("connected", "3 pending", "disconnected", …).
  Stream<String> get syncStatus => _synced.status;

  /// The configured PC hub WebSocket URL, or null/empty if sync is off.
  String? get syncServerUrl => _synced.serverUrl;

  /// Sets the PC hub URL (e.g. `ws://192.168.1.20:8765`) and (re)connects.
  /// Empty string disables sync. Unsent memories flush automatically on connect.
  Future<void> setSyncServer(String url) => _synced.setServerUrl(url);

  Future<void> enroll(String name, Float32List samples) => _speaker.enroll(name, samples);

  Future<List<String>> enrolledSpeakers() => _speaker.enrolledSpeakers();

  Future<void> removeSpeaker(String name) => _speaker.remove(name);

  Future<void> dispose() async {
    await _sub?.cancel();
    await _capture.dispose();
    await _speaker.dispose();
    await _transcriber.dispose();
    await _inference.dispose();
    await _store.dispose();
    await _onMemory.close();
  }
}
