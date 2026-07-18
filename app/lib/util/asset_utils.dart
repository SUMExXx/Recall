import 'dart:io';

import 'package:flutter/services.dart' show rootBundle;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Copies a bundled asset to a real filesystem path (sherpa-onnx configs take
/// paths, not bytes). Idempotent: skips the copy if a same-size file exists.
Future<String> copyAssetFile(String assetPath, [String? dstName]) async {
  final dir = await getApplicationSupportDirectory();
  final target = p.join(dir.path, dstName ?? p.basename(assetPath));
  final data = await rootBundle.load(assetPath);

  final file = File(target);
  if (!file.existsSync() || file.lengthSync() != data.lengthInBytes) {
    await file.writeAsBytes(
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
      flush: true,
    );
  }
  return target;
}
