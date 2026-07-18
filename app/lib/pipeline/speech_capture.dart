import 'dart:typed_data';

/// Captures microphone audio and emits complete speech utterances.
///
/// Audio contract: each emitted utterance is 16 kHz mono float PCM in [-1, 1].
/// How utterances are detected (VAD, wake word, push-to-talk) is up to the
/// implementation — callers only see finished utterances.
abstract interface class SpeechCapture {
  /// One event per detected utterance, in order.
  Stream<Float32List> get utterances;

  /// Coarse status for the UI (e.g. "listening", "recording"), or null.
  Stream<String> get status;

  /// Begins capturing. Requests microphone permission if needed.
  /// Returns false if permission was denied.
  Future<bool> start();

  /// Stops capturing. Safe to call when already stopped.
  Future<void> stop();

  Future<void> dispose();
}
