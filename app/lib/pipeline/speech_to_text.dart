import 'dart:typed_data';

/// Transcribes a recorded audio clip to text. [samples] are 16 kHz mono float
/// PCM in [-1, 1] (what the mic produces). Swappable like every other stage.
abstract interface class SpeechToText {
  /// Returns the transcript of [samples], or '' if nothing was recognized.
  Future<String> transcribe(Float32List samples);
}
