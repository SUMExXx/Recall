import 'dart:typed_data';

/// Speech-to-text over a whole utterance.
///
/// Audio contract: 16 kHz mono float PCM in [-1, 1].
abstract interface class Transcriber {
  /// Returns the transcript, or an empty string if nothing was recognized.
  Future<String> transcribe(Float32List samples);

  Future<void> dispose();
}
