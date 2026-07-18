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
