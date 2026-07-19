/// Converts answer text to speech and plays it. The implementation picks the
/// engine (cloud, on-device, …) and owns playback.
abstract interface class TextToSpeech {
  /// Synthesizes [text] and plays it, interrupting any current playback.
  Future<void> speak(String text);

  /// Stops any current playback.
  Future<void> stop();

  Future<void> dispose();
}
