import 'dart:typed_data';

/// Identifies enrolled speakers from utterance audio.
///
/// Audio contract: whole utterance, 16 kHz mono float PCM in [-1, 1].
abstract interface class SpeakerRecognizer {
  /// Extracts a voiceprint from [samples] and persists it under [name].
  Future<void> enroll(String name, Float32List samples);

  /// Returns the enrolled speaker's name, or null if none matches.
  Future<String?> identify(Float32List samples);

  Future<List<String>> enrolledSpeakers();

  Future<void> remove(String name);

  Future<void> dispose();
}
