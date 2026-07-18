import 'dart:async';
import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;

import '../pipeline/speech_capture.dart';
import '../util/asset_utils.dart';
import '../util/audio_util.dart';

/// Default [SpeechCapture]: microphone via `mic_stream`, utterance segmentation
/// via sherpa-onnx Silero VAD. Manual start/stop; VAD gates which audio becomes
/// an utterance so only speech leaves this stage.
class MicVadCapture implements SpeechCapture {
  static const int _window = 512; // Silero window at 16 kHz

  final sherpa.VoiceActivityDetector _vad;
  final _utterances = StreamController<Float32List>.broadcast();
  final _status = StreamController<String>.broadcast();
  final _pending = <double>[];

  StreamSubscription<Uint8List>? _sub;
  String _lastStatus = '';

  MicVadCapture._(this._vad);

  static Future<MicVadCapture> create() async {
    final model = await copyAssetFile('assets/models/vad/silero_vad.onnx');
    final config = sherpa.VadModelConfig(
      sileroVad: sherpa.SileroVadModelConfig(
        model: model,
        minSilenceDuration: 0.5,
        minSpeechDuration: 0.25,
      ),
      numThreads: 1,
      debug: false,
    );
    final vad = sherpa.VoiceActivityDetector(
      config: config,
      bufferSizeInSeconds: 30,
    );
    return MicVadCapture._(vad);
  }

  @override
  Stream<Float32List> get utterances => _utterances.stream;

  @override
  Stream<String> get status => _status.stream;

  @override
  Future<bool> start() async {
    if (!await ensureMicPermission()) return false;
    _emitStatus('Listening…');
    _sub = micStream().listen(_onBytes, onError: (_) => _emitStatus('Mic error'));
    return true;
  }

  void _onBytes(Uint8List data) {
    _pending.addAll(pcm16ToFloat32(data));
    while (_pending.length >= _window) {
      final window = Float32List.fromList(_pending.sublist(0, _window));
      _pending.removeRange(0, _window);
      _vad.acceptWaveform(window);
      _emitStatus(_vad.isDetected() ? 'Recording…' : 'Listening…');
      _drain();
    }
  }

  void _drain() {
    while (!_vad.isEmpty()) {
      _utterances.add(_vad.front().samples);
      _vad.pop();
    }
  }

  void _emitStatus(String s) {
    if (s != _lastStatus) {
      _lastStatus = s;
      _status.add(s);
    }
  }

  @override
  Future<void> stop() async {
    await _sub?.cancel();
    _sub = null;
    _vad.flush();
    _drain();
    _vad.clear();
    _pending.clear();
    _emitStatus('Idle');
  }

  @override
  Future<void> dispose() async {
    await stop();
    await _utterances.close();
    await _status.close();
    _vad.free();
  }
}
