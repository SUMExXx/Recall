import 'dart:async';
import 'dart:typed_data';

import 'package:mic_stream/mic_stream.dart';

/// Requests mic permission (if needed) and reports whether it is granted.
Future<bool> ensureMicPermission() => MicStream.permissionStatus;

/// A 16 kHz mono PCM16 microphone stream. Each event is a chunk of raw bytes.
Stream<Uint8List> micStream() => MicStream.microphone(
      sampleRate: 16000,
      channelConfig: ChannelConfig.CHANNEL_IN_MONO,
      audioFormat: AudioFormat.ENCODING_PCM_16BIT,
    );

/// Little-endian 16-bit PCM bytes → float in [-1, 1].
Float32List pcm16ToFloat32(Uint8List bytes) {
  final data = ByteData.view(bytes.buffer, bytes.offsetInBytes, bytes.length);
  final out = Float32List(bytes.length ~/ 2);
  for (var i = 0; i < out.length; i++) {
    out[i] = data.getInt16(i * 2, Endian.little) / 32768.0;
  }
  return out;
}

/// Encodes 16 kHz mono float samples [-1, 1] as a PCM16 WAV file — the upload
/// format for cloud STT (Sarvam).
Uint8List floatSamplesToWav(Float32List samples, {int sampleRate = 16000}) {
  final dataSize = samples.length * 2;
  final pcm = Uint8List(dataSize);
  final view = ByteData.view(pcm.buffer);
  for (var i = 0; i < samples.length; i++) {
    final s = (samples[i] * 32767).round().clamp(-32768, 32767);
    view.setInt16(i * 2, s, Endian.little);
  }
  final out = BytesBuilder();
  void str(String s) => out.add(s.codeUnits);
  void u32(int v) => out.add([v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff]);
  void u16(int v) => out.add([v & 0xff, (v >> 8) & 0xff]);
  str('RIFF'); u32(36 + dataSize); str('WAVE');
  str('fmt '); u32(16); u16(1); u16(1);          // PCM, mono
  u32(sampleRate); u32(sampleRate * 2); u16(2); u16(16); // byte rate, block align, bits
  str('data'); u32(dataSize);
  out.add(pcm);
  return out.toBytes();
}

/// Records a fixed window from the mic and returns the samples (16 kHz mono).
/// Returns null if microphone permission is denied.
Future<Float32List?> recordSamples(Duration duration) async {
  if (!await ensureMicPermission()) return null;
  final out = <double>[];
  final sub = micStream().listen((b) => out.addAll(pcm16ToFloat32(b)));
  await Future.delayed(duration);
  await sub.cancel();
  return Float32List.fromList(out);
}
