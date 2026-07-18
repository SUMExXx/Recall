import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;

import '../pipeline/transcriber.dart';
import '../util/asset_utils.dart';

/// Default [Transcriber]: sherpa-onnx OfflineRecognizer with Whisper tiny.en (INT8).
///
// ponytail: decode() is a blocking native call on the calling isolate — a few
// seconds per utterance will jank the UI. Move the recognizer into a dedicated
// worker isolate (create + decode there, pass Float32List in / String out) if
// smoothness matters.
class SherpaWhisperTranscriber implements Transcriber {
  final sherpa.OfflineRecognizer _recognizer;

  SherpaWhisperTranscriber._(this._recognizer);

  static Future<SherpaWhisperTranscriber> create() async {
    final encoder = await copyAssetFile('assets/models/whisper/encoder.onnx');
    final decoder = await copyAssetFile('assets/models/whisper/decoder.onnx');
    final tokens = await copyAssetFile('assets/models/whisper/tokens.txt');

    final config = sherpa.OfflineRecognizerConfig(
      model: sherpa.OfflineModelConfig(
        whisper: sherpa.OfflineWhisperModelConfig(
          encoder: encoder,
          decoder: decoder,
        ),
        tokens: tokens,
        modelType: 'whisper',
        numThreads: 2,
        debug: false,
      ),
    );
    return SherpaWhisperTranscriber._(sherpa.OfflineRecognizer(config));
  }

  @override
  Future<String> transcribe(Float32List samples) async {
    final stream = _recognizer.createStream();
    try {
      stream.acceptWaveform(samples: samples, sampleRate: 16000);
      _recognizer.decode(stream);
      return _recognizer.getResult(stream).text.trim();
    } finally {
      stream.free();
    }
  }

  @override
  Future<void> dispose() async => _recognizer.free();
}
