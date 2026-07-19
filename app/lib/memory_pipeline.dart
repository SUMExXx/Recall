import 'dart:async';
import 'dart:typed_data';

import 'impl/local_memory.dart';
import 'impl/mic_vad_capture.dart';
import 'impl/qual_inference_engine.dart';
import 'impl/routed_inference_engine.dart';
import 'impl/sarvam_stt.dart';
import 'impl/sarvam_tts.dart';
import 'impl/sherpa_speaker_recognizer.dart';
import 'impl/sherpa_whisper_transcriber.dart';
import 'impl/synced_memory.dart';
import 'pipeline/inference_engine.dart';
import 'pipeline/speaker_recognizer.dart';
import 'pipeline/speech_capture.dart';
import 'pipeline/speech_to_text.dart';
import 'pipeline/text_to_speech.dart';
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

  // ---- Ask feature: Sarvam speech-to-text + text-to-speech -------------
  final TextToSpeech _tts = SarvamTts();
  final SpeechToText _stt = SarvamStt();
  final _questions = StreamController<String>.broadcast();
  bool _askMode = false; // true while the Ask tab is active

  // ---- Wake word ("Hey Recall") ----------------------------------------
  // Matches "hey recall" and the ways Whisper commonly mishears it
  // ("hey, recall", "hey rekall", "hey record"). Case/punctuation-insensitive.
  static final RegExp _wakeRe =
      RegExp(r'hey[\s,]+re[ck]a?o?l?l|hey[\s,]+record', caseSensitive: false);

  /// When true, utterances are ignored until the wake word is heard; only then
  /// is speech saved to memory. When false, every utterance is saved (the
  /// original always-on behaviour).
  bool _wakeMode = true;
  DateTime? _awakeUntil;
  Timer? _wakeTimer;
  final _listening = StreamController<bool>.broadcast();

  /// Safety timeout: if the wake word opens the orb but no message follows, the
  /// orb auto-hides after this. A captured message closes it sooner.
  static const Duration _awakeWindow = Duration(seconds: 12);

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
    // Route answering to the PC hub while connected (it answers from the synced
    // memories); fall back to on-device when offline.
    final routed = RoutedInferenceEngine(
      inference,
      serverUrl: () => store.serverUrl,
      connected: () => store.isConnected,
    );

    final pipe =
        MemoryPipeline._(capture, speaker, transcriber, store, store, routed);
    pipe._sub = capture.utterances.listen(pipe._enqueue);
    return pipe;
  }

  Stream<String> get status => _capture.status;

  /// Fires when a new memory is captured and stored.
  Stream<Memory> get onMemory => _onMemory.stream;

  Future<bool> start() => _capture.start();

  Future<void> stop() => _capture.stop();

  /// Fires `true` when the wake word opens a listening window, `false` when it
  /// closes. The UI shows the glowing orb while this is `true`.
  Stream<bool> get listening => _listening.stream;

  /// Whether wake-word gating is on.
  bool get wakeWordEnabled => _wakeMode;

  /// Turns wake-word gating on/off. Off = save every utterance (original mode).
  void setWakeWord(bool on) {
    _wakeMode = on;
    if (!on) _closeWindow(); // stop gating; any open orb hides
  }

  bool get _isAwake =>
      _awakeUntil != null && DateTime.now().isBefore(_awakeUntil!);

  void _openWindow() {
    final wasClosed = !_isAwake;
    _awakeUntil = DateTime.now().add(_awakeWindow);
    _wakeTimer?.cancel();
    _wakeTimer = Timer(_awakeWindow, _closeWindow);
    if (wasClosed) _listening.add(true);
  }

  void _closeWindow() {
    _wakeTimer?.cancel();
    _wakeTimer = null;
    if (_awakeUntil != null) {
      _awakeUntil = null;
      _listening.add(false);
    }
  }

  /// If [text] contains the wake word, returns whatever was said after it
  /// (possibly empty); otherwise null.
  static String? _afterWakeWord(String text) {
    final m = _wakeRe.firstMatch(text);
    if (m == null) return null;
    // Drop leading punctuation Whisper leaves after the wake phrase.
    return text.substring(m.end).replaceFirst(RegExp(r'^[\s,.!?:;-]+'), '').trim();
  }

  void _enqueue(Float32List samples) {
    _tail = _tail.then((_) => _process(samples));
  }

  /// Whisper emits junk for silence/background noise — bare punctuation,
  /// non-Latin gibberish, or bracketed sound tags like "[Music]". Keep only
  /// text that is mostly real (Latin) words so noise never reaches memory.
  static bool isLikelyLanguage(String text) {
    // Drop non-speech annotations Whisper wraps in brackets/parens.
    final stripped = text.replaceAll(RegExp(r'[\[(][^\])]*[\])]'), ' ');
    final letters = RegExp(r'[a-zA-Z]').allMatches(stripped).length;
    final dense = stripped.replaceAll(RegExp(r'\s'), '').length;
    if (letters < 2 || dense == 0) return false;
    return letters / dense >= 0.5; // majority actual letters, not symbols
  }

  Future<void> _process(Float32List samples) async {
    final text = await _transcriber.transcribe(samples);
    if (text.isEmpty) return;
    if (!isLikelyLanguage(text)) return; // discard mis-transcribed noise

    if (_wakeMode && !_isAwake) {
      // Asleep: only the wake word matters. Anything else is ignored.
      final after = _afterWakeWord(text);
      if (after == null) return;
      _openWindow(); // orb rises
      if (after.isNotEmpty) {
        // Whole message came in one breath → capture it and close the orb now.
        await _consume(after, samples);
        _closeWindow();
      }
      // Otherwise wait for the follow-up utterance (or the safety timeout).
      return;
    }

    if (_wakeMode) {
      // Awake: this utterance is the message we opened the orb for. Capture it,
      // then close — one wake word captures one message.
      await _consume(text, samples);
      _closeWindow();
      return;
    }

    // Wake mode off: save every utterance, no orb.
    await _consume(text, samples);
  }

  /// Terminal action for a captured utterance: in ask mode it becomes a spoken
  /// question (answered + voiced by the UI); otherwise it's saved to memory.
  Future<void> _consume(String text, Float32List samples) async {
    if (_askMode) {
      if (text.trim().isNotEmpty) _questions.add(text.trim());
    } else {
      await _save(text, samples);
    }
  }

  Future<void> _save(String text, Float32List samples) async {
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

  /// Permanently deletes the memories with these row ids.
  Future<void> deleteMemories(List<int> ids) => _store.delete(ids);

  /// Answers a natural-language question using the on-device inference engine.
  Future<String> ask(String question) => _inference.ask(question);

  // ---- Ask feature -----------------------------------------------------

  /// In ask mode, wake-word captures become spoken questions (emitted on
  /// [spokenQuestions] to be answered + voiced) instead of saved memories.
  void setAskMode(bool on) => _askMode = on;

  /// Questions captured by the wake word while in ask mode.
  Stream<String> get spokenQuestions => _questions.stream;

  /// Speaks [text] via Sarvam TTS (used to voice answers in the Ask tab).
  Future<void> speak(String text) => _tts.speak(text);

  /// Stops any in-progress speech.
  Future<void> stopSpeaking() => _tts.stop();

  /// Transcribes recorded mic [samples] via Sarvam STT (the Ask mic button).
  Future<String> transcribeQuestion(Float32List samples) =>
      _stt.transcribe(samples);

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

  /// Current sync status right now, to seed the UI on load (the stream's early
  /// events fire while models are still loading, before the UI subscribes).
  String get syncStatusNow => _synced.statusNow;

  /// The configured PC hub WebSocket URL, or null/empty if sync is off.
  String? get syncServerUrl => _synced.serverUrl;

  /// Sets the PC hub URL (e.g. `ws://192.168.1.20:8765`) and (re)connects.
  /// Empty string disables sync. Unsent memories flush automatically on connect.
  Future<void> setSyncServer(String url) => _synced.setServerUrl(url);

  Future<void> enroll(String name, Float32List samples) => _speaker.enroll(name, samples);

  Future<List<String>> enrolledSpeakers() => _speaker.enrolledSpeakers();

  Future<void> removeSpeaker(String name) => _speaker.remove(name);

  Future<void> dispose() async {
    _wakeTimer?.cancel();
    await _listening.close();
    await _questions.close();
    await _sub?.cancel();
    await _capture.dispose();
    await _speaker.dispose();
    await _transcriber.dispose();
    await _inference.dispose();
    await _tts.dispose();
    await _store.dispose();
    await _onMemory.close();
  }
}
