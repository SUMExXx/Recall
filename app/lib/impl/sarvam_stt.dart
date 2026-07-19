import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart' show MediaType;

import '../pipeline/speech_to_text.dart';
import '../util/audio_util.dart';

/// Sarvam API key — mirrors the PC hub's default (pc/recall_memory/config.py).
const sarvamApiKey = 'sk_kobw09xo_JqDdUSFxsT6AmRQIV38qQglc';

/// Speech-to-text via Sarvam's `saaras:v3` cloud model. Uploads the recorded
/// clip as a WAV and returns the transcript (auto-detects language).
class SarvamStt implements SpeechToText {
  static const _endpoint = 'https://api.sarvam.ai/speech-to-text';
  final String apiKey;

  SarvamStt({this.apiKey = sarvamApiKey});

  @override
  Future<String> transcribe(Float32List samples) async {
    if (samples.isEmpty) return '';
    final req = http.MultipartRequest('POST', Uri.parse(_endpoint))
      ..headers['api-subscription-key'] = apiKey
      ..fields['model'] = 'saaras:v3'
      ..fields['language_code'] = 'unknown' // auto-detect
      ..fields['mode'] = 'transcribe'
      ..files.add(http.MultipartFile.fromBytes(
        'file', floatSamplesToWav(samples),
        filename: 'audio.wav', contentType: MediaType('audio', 'wav'),
      ));
    final streamed = await req.send().timeout(const Duration(seconds: 20));
    final resp = await http.Response.fromStream(streamed);
    if (resp.statusCode != 200) {
      throw Exception('Sarvam STT ${resp.statusCode}: ${resp.body}');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    return ((data['transcript'] as String?) ?? '').trim();
  }
}
