import 'package:flutter/services.dart';

import '../pipeline/inference_engine.dart';
import '../pipeline/vector_store.dart';

/// Qualcomm/Snapdragon [InferenceEngine]. Runs a generative LLM on the device
/// **NPU** through Qualcomm's GenieX runtime (native `com.qualcomm.qti:geniex-android`),
/// reached over the `geniex` platform channel. Answers are grounded in memories
/// retrieved through the [VectorStore] interface (retrieval-augmented generation).
///
/// The native side loads a GenieX model (downloaded from Qualcomm AI Hub) and
/// runs it on the NPU. If no model is present, `generate` reports
/// `GENIE_UNAVAILABLE` and this falls back to returning the retrieved memories.
class QualInferenceEngine implements InferenceEngine {
  static const MethodChannel _channel = MethodChannel('geniex');
  // static const int _contextMemories = 5; // RAG retrieval size (disabled)
  // ponytail: dumps EVERY memory into the prompt — fine while the store is
  // small; restore the search() retrieval below before this outgrows the LLM
  // context window.
  static const int _allMemories = 1 << 30; // effectively unbounded

  final VectorStore _memory;

  QualInferenceEngine(this._memory);

  static Future<QualInferenceEngine> create({required VectorStore memory}) async {
    return QualInferenceEngine(memory);
  }

  @override
  Future<String> ask(String question) async {
    // All memories go into the prompt for now — retrieval disabled.
    // final context = await _memory.search(question, topK: _contextMemories);
    final context = await _memory.recent(_allMemories);
    final prompt = _buildPrompt(question, context);
    try {
      final answer = await _channel.invokeMethod<String>('generate', {'prompt': prompt});
      final text = answer?.trim() ?? '';
      return text.isNotEmpty ? text : _fallback(context);
    } on MissingPluginException {
      return _fallback(context); // channel not registered on this platform
    } on PlatformException catch (e) {
      if (e.code == 'GENIE_UNAVAILABLE') return _fallback(context);
      return 'Inference error: ${e.message}';
    }
  }

  @override
  Future<bool> isModelReady() async {
    try {
      return await _channel.invokeMethod<bool>('isModelReady') ?? false;
    } on PlatformException {
      return false;
    } on MissingPluginException {
      return false;
    }
  }

  @override
  Future<void> downloadModel(String modelName) async {
    await _channel.invokeMethod<void>('downloadModel', {'model': modelName});
  }

  @override
  Future<void> registerLocalModel(String path) async {
    await _channel.invokeMethod<void>('registerLocalModel', {'path': path});
  }

  @override
  Future<bool> hasStorageAccess() async {
    try {
      return await _channel.invokeMethod<bool>('hasStorageAccess') ?? false;
    } on PlatformException {
      return false;
    } on MissingPluginException {
      return false;
    }
  }

  @override
  Future<void> requestStorageAccess() async {
    await _channel.invokeMethod<void>('requestStorageAccess');
  }

  String _buildPrompt(String question, List<Memory> context) {
    final memories = context.isEmpty
        ? '(no relevant memories found)'
        : context.map((m) => '- ${m.speaker}: ${m.text}').join('\n');
    return 'You are a personal memory assistant. Answer the question using only '
        'the memories below. If they do not contain the answer, say so.\n\n'
        'Memories:\n$memories\n\nQuestion: $question\nAnswer:';
  }

  /// Used until a GenieX model is installed: return the grounding memories.
  String _fallback(List<Memory> context) {
    if (context.isEmpty) return "I don't have any memories about that yet.";
    return context.map((m) => '• ${m.speaker}: ${m.text}').join('\n');
  }

  @override
  Future<void> dispose() async {}
}
