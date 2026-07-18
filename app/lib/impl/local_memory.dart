import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

import '../pipeline/vector_store.dart';
import '../util/float_bytes.dart';
import 'bge_embedder.dart';

/// On-device [VectorStore]: sqflite rows + bge embeddings, brute-force cosine
/// search. Embeddings are L2-normalized, so cosine similarity == dot product.
/// The "local" memory backend (as opposed to a future synced/remote one).
class LocalMemory implements VectorStore {
  final Database _db;
  final BgeEmbedder _embedder;

  LocalMemory._(this._db, this._embedder);

  static Future<LocalMemory> create() async {
    final embedder = await BgeEmbedder.create();
    final db = await openDatabase(
      p.join(await getDatabasesPath(), 'memories.db'),
      version: 1,
      onCreate: (db, _) => db.execute(
        'CREATE TABLE memories ('
        'id INTEGER PRIMARY KEY AUTOINCREMENT, '
        'timestamp INTEGER NOT NULL, '
        'speaker TEXT NOT NULL, '
        'text TEXT NOT NULL, '
        'embedding BLOB NOT NULL)',
      ),
    );
    return LocalMemory._(db, embedder);
  }

  @override
  Future<int> add(Memory memory) async {
    final embedding = memory.embedding.isNotEmpty
        ? memory.embedding
        : await _embedder.embed(memory.text);
    return _db.insert('memories', {
      'timestamp': memory.timestamp.millisecondsSinceEpoch,
      'speaker': memory.speaker,
      'text': memory.text,
      'embedding': floatsToBytes(embedding),
    });
  }

  @override
  Future<List<Memory>> recent(int limit) async {
    final rows = await _db.query('memories', orderBy: 'timestamp DESC', limit: limit);
    return rows.map(_toMemory).toList();
  }

  @override
  Future<List<Memory>> search(String query, {int topK = 20}) async {
    final q = await _embedder.embedQuery(query);
    final rows = await _db.query('memories');
    // ponytail: O(n) scan per query; fine on-device. Move to sqlite-vec past ~50k rows.
    final scored = rows
        .map((r) => (_dot(q, bytesToFloats(r['embedding'] as Uint8List)), r))
        .toList()
      ..sort((a, b) => b.$1.compareTo(a.$1));
    return scored.take(topK).map((e) => _toMemory(e.$2)).toList();
  }

  static Memory _toMemory(Map<String, Object?> r) => Memory(
        id: r['id'] as int,
        timestamp: DateTime.fromMillisecondsSinceEpoch(r['timestamp'] as int),
        speaker: r['speaker'] as String,
        text: r['text'] as String,
        embedding: bytesToFloats(r['embedding'] as Uint8List),
      );

  static double _dot(Float32List a, Float32List b) {
    final n = a.length < b.length ? a.length : b.length;
    var sum = 0.0;
    for (var i = 0; i < n; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  @override
  Future<void> dispose() async {
    _embedder.dispose();
    await _db.close();
  }
}
