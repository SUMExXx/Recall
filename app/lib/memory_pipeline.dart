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
  // On-device Whisper (tiny.en) mishears the wake phrase constantly, so an
  // exact/near-exact regex misses most of the time. We instead scan the first
  // few words for a "recall"-like token (edit-distance tolerant + a curated
  // mishear set), optionally led by a "hey"-like filler. Deliberately biased
  // toward recall over precision: a stray trigger just opens the orb for a few
  // seconds, whereas a missed one silently loses the whole memory.
  static const Set<String> _recallLike = {
    'recall', 'recalls', 'rekall', 'rekal', 'recal', 'recoll', 'regall',
    'ricall', 'recalled', 'record', 'records', 'recording', 'reccall',
  };
  static const Set<String> _recallDeny = {
    'recent', 'recently', 'receive', 'received', 'recipe', 'reception',
    'recommend', 'recover', 'require', 'research',
  };
  static const Set<String> _heyLike = {
    'hey', 'hay', 'ey', 'ay', 'hi', 'ok', 'okay', 'a', 'eh', 'yo',
  };

  /// When true, utterances are ignored until the wake word is heard; only then
  /// is speech saved to memory. When false, every utterance is saved (the
  /// original always-on behaviour).
  bool _wakeMode = true;
  bool _awake = false;
  Timer? _wakeTimer;
  final _listening = StreamController<bool>.broadcast();
  // Accumulated message for the current wake window: on-device VAD splits on
  // ~0.5 s of silence, so one spoken message arrives as several segments. We
  // gather them all and save the concatenation when speech stops, instead of
  // capturing only the first segment and dropping the rest.
  final StringBuffer _msg = StringBuffer();
  Float32List? _msgSamples; // first content segment, used for speaker id

  /// Max wait, after the wake word, for the message to START before the orb
  /// auto-hides with nothing captured.
  static const Duration _awakeWindow = Duration(seconds: 12);

  /// Once the message is under way, this much trailing silence marks its end —
  /// the accumulated text is saved and the orb closes.
  static const Duration _trailingSilence = Duration(seconds: 4);

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
    if (!on) _flushAndClose(); // stop gating; save anything pending, hide orb
  }

  bool get _isAwake => _awake;

  /// (Re)arm the wake timer for [d]; when it fires, whatever was accumulated is
  /// saved and the window closes.
  void _armTimer(Duration d) {
    _wakeTimer?.cancel();
    _wakeTimer = Timer(d, _flushAndClose);
  }

  void _openWindow() {
    if (!_awake) {
      _awake = true;
      _listening.add(true);
    }
    _armTimer(_awakeWindow); // wait for the message to start
  }

  void _closeWindow() {
    _wakeTimer?.cancel();
    _wakeTimer = null;
    if (_awake) {
      _awake = false;
      _listening.add(false);
    }
  }

  /// Append a message segment and reset the trailing-silence timer so a message
  /// spoken over several VAD segments is gathered into one memory.
  void _appendMessage(String text, Float32List samples) {
    final t = text.trim();
    if (t.isEmpty) return;
    if (_msg.isNotEmpty) _msg.write(' ');
    _msg.write(t);
    _msgSamples ??= samples;
    _armTimer(_trailingSilence);
  }

  /// Save the accumulated message (if any) and close the window. Runs from the
  /// timer; the save is serialized onto the same tail as live utterances.
  void _flushAndClose() {
    final text = _msg.toString().trim();
    final samples = _msgSamples;
    _msg.clear();
    _msgSamples = null;
    if (text.isNotEmpty && samples != null) {
      _tail = _tail.then((_) => _consume(text, samples));
    }
    _closeWindow();
  }

  /// If [text] begins with the wake word (fuzzily — see [_recallLike]), returns
  /// whatever was said after it (possibly empty); otherwise null.
  static String? _afterWakeWord(String text) {
    // Tokenize with end offsets so we can return the original-cased remainder.
    final words = RegExp(r'\S+').allMatches(text).toList();
    final scan = words.length < 5 ? words.length : 5; // wake word is up front
    for (var i = 0; i < scan; i++) {
      if (!_isRecallLike(words[i][0]!)) continue;
      // Accept if it's at the very start, or a "hey"-like filler precedes it —
      // this keeps a mid-sentence "record" from falsely waking.
      final led = i == 0 ||
          _heyLike.contains(_norm(words[i - 1][0]!));
      if (!led) continue;
      return text
          .substring(words[i].end)
          .replaceFirst(RegExp(r'^[\s,.!?:;-]+'), '')
          .trim();
    }
    return null;
  }

  static String _norm(String w) => w.toLowerCase().replaceAll(RegExp(r'[^a-z]'), '');

  /// Whether a token is a plausible (mis)hearing of "recall"/"record".
  static bool _isRecallLike(String word) {
    final w = _norm(word);
    if (w.length < 4 || w.length > 9) return false;
    if (_recallDeny.contains(w)) return false;
    if (_recallLike.contains(w)) return true;
    // Distance 1 only: distance 2 would pull in "call" and wake mid-sentence.
    return _lev(w, 'recall') <= 1 || _lev(w, 'record') <= 1;
  }

  /// Levenshtein edit distance (small strings, so the simple DP is fine).
  static int _lev(String a, String b) {
    final prev = List<int>.generate(b.length + 1, (i) => i);
    final cur = List<int>.filled(b.length + 1, 0);
    for (var i = 1; i <= a.length; i++) {
      cur[0] = i;
      for (var j = 1; j <= b.length; j++) {
        final cost = a[i - 1] == b[j - 1] ? 0 : 1;
        cur[j] = [cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost]
            .reduce((x, y) => x < y ? x : y);
      }
      for (var j = 0; j <= b.length; j++) {
        prev[j] = cur[j];
      }
    }
    return prev[b.length];
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

    if (!_wakeMode) {
      // Wake mode off: save every utterance immediately, no orb.
      await _consume(text, samples);
      return;
    }

    if (!_isAwake) {
      // Asleep: only the wake word matters. Anything else is ignored.
      final after = _afterWakeWord(text);
      if (after == null) return;
      _openWindow(); // orb rises; wait for (the rest of) the message
      // Whatever followed the wake word in this same breath starts the message;
      // further VAD segments are appended until _trailingSilence of quiet, so a
      // message spoken with pauses isn't truncated to its first segment.
      if (after.isNotEmpty) _appendMessage(after, samples);
      return;
    }

    // Awake: this segment is (part of) the message. Accumulate; the trailing-
    // silence timer saves the whole thing and closes the orb once speech stops.
    _appendMessage(text, samples);
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
