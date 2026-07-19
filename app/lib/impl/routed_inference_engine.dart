import 'dart:async';
import 'dart:convert';
import 'dart:io';

import '../pipeline/inference_engine.dart';

/// Routes `ask` to the PC hub's `/ask` HTTP API while the phone is connected to
/// it, falling back to the on-device engine when disconnected or on any PC
/// error. Everything else (model download/registration, storage access) stays
/// local — the PC only answers questions, using the memories synced to it.
class RoutedInferenceEngine implements InferenceEngine {
  final InferenceEngine _local;
  final String? Function() _serverUrl; // sync WS url, or null/empty when off
  final bool Function() _connected;

  RoutedInferenceEngine(
    this._local, {
    required String? Function() serverUrl,
    required bool Function() connected,
  })  : _serverUrl = serverUrl,
        _connected = connected;

  @override
  Future<String> ask(String question) async {
    final url = _serverUrl();
    if (_connected() && url != null && url.isNotEmpty) {
      try {
        final answer = await _askPc(url, question);
        if (answer != null) return answer;
      } catch (_) {
        // PC unreachable/slow/erroring → fall back to on-device.
      }
    }
    return _local.ask(question);
  }

  /// `ws://host:port/ws` (the sync URL) → `http://host:port/ask`.
  static Uri askUri(String wsUrl) {
    final ws = Uri.parse(wsUrl);
    return ws.replace(scheme: ws.scheme == 'wss' ? 'https' : 'http', path: '/ask');
  }

  /// Returns the PC's answer, or null (→ caller falls back to local) when the
  /// hub gives no usable answer.
  Future<String?> _askPc(String wsUrl, String question) async {
    final client = HttpClient()..connectionTimeout = const Duration(seconds: 3);
    try {
      final req = await client.postUrl(askUri(wsUrl));
      req.headers.contentType = ContentType.json;
      req.add(utf8.encode(jsonEncode({'query': question, 'k': 6})));
      final resp = await req.close().timeout(const Duration(seconds: 45));
      if (resp.statusCode != 200) return null;
      final body = await resp.transform(utf8.decoder).join();
      final answer = (jsonDecode(body) as Map<String, dynamic>)['answer'] as String?;
      return (answer != null && answer.trim().isNotEmpty) ? answer : null;
    } finally {
      client.close(force: true);
    }
  }

  @override
  Future<bool> isModelReady() => _local.isModelReady();

  @override
  Future<void> downloadModel(String modelName) => _local.downloadModel(modelName);

  @override
  Future<void> registerLocalModel(String path) => _local.registerLocalModel(path);

  @override
  Future<bool> hasStorageAccess() => _local.hasStorageAccess();

  @override
  Future<void> requestStorageAccess() => _local.requestStorageAccess();

  @override
  Future<void> dispose() => _local.dispose();
}
