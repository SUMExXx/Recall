import 'dart:convert';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:http/http.dart' as http;

import '../pipeline/text_to_speech.dart';
import 'sarvam_stt.dart' show sarvamApiKey;

/// Text-to-speech via Sarvam's `bulbul:v3` cloud model. Synthesizes to WAV and
/// plays it, interrupting any current playback.
class SarvamTts implements TextToSpeech {
  static const _endpoint = 'https://api.sarvam.ai/text-to-speech';
  static const _maxChars = 2500; // bulbul:v3 hard limit

  final String apiKey;
  final AudioPlayer _player = AudioPlayer();

  SarvamTts({this.apiKey = sarvamApiKey});

  @override
  Future<void> speak(String text) async {
    final t = text.trim();
    if (t.isEmpty) return;
    final resp = await http
        .post(
          Uri.parse(_endpoint),
          headers: {
            'api-subscription-key': apiKey,
            'Content-Type': 'application/json',
          },
          body: jsonEncode({
            'text': t.length > _maxChars ? t.substring(0, _maxChars) : t,
            'target_language_code': 'en-IN',
            'model': 'bulbul:v3',
            'speaker': 'priya', // valid bulbul:v3 speaker
            'speech_sample_rate': 24000,
            'output_audio_codec': 'wav',
            'pace': 1.0,
          }),
        )
        .timeout(const Duration(seconds: 20));
    if (resp.statusCode != 200) {
      throw Exception('Sarvam TTS ${resp.statusCode}: ${resp.body}');
    }
    final audios = (jsonDecode(resp.body) as Map<String, dynamic>)['audios'] as List?;
    if (audios == null || audios.isEmpty) return;
    final bytes = base64Decode(audios.first as String);
    await _player.stop();
    await _player.play(BytesSource(Uint8List.fromList(bytes), mimeType: 'audio/wav'));
  }

  @override
  Future<void> stop() => _player.stop();

  @override
  Future<void> dispose() => _player.dispose();
}
