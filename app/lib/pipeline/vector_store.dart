import 'dart:typed_data';

/// One captured memory.
class Memory {
  final int id;
  final DateTime timestamp;
  final String speaker;
  final String text;

  /// Semantic embedding of [text]; empty until the store computes it.
  final Float32List embedding;

  const Memory({
    this.id = 0,
    required this.timestamp,
    required this.speaker,
    required this.text,
    required this.embedding,
  });
}

/// Stores memories and retrieves them by semantic similarity.
///
/// The implementation owns how text becomes a vector (the embedding model),
/// so swapping the store swaps the embedding strategy with it.
abstract interface class VectorStore {
  /// Embeds [text] (if [Memory.embedding] is empty), persists, returns row id.
  Future<int> add(Memory memory);

  /// Most recent memories, newest first.
  Future<List<Memory>> recent(int limit);

  /// Memories most similar to [query], best first.
  Future<List<Memory>> search(String query, {int topK = 20});

  /// Permanently removes the memories with these row ids.
  Future<void> delete(List<int> ids);

  Future<void> dispose();
}
