import 'dart:typed_data';

import 'package:path/path.dart' as p;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;
import 'package:sqflite/sqflite.dart';

import '../pipeline/speaker_recognizer.dart';
import '../util/asset_utils.dart';
import '../util/float_bytes.dart';

/// Default [SpeakerRecognizer]: sherpa-onnx speaker embeddings. The extractor
/// produces voiceprints and the manager is the in-memory match index (it has no
/// persistence), so profiles are mirrored in a small self-owned sqflite table.
class SherpaSpeakerRecognizer implements SpeakerRecognizer {
  /// Below this cosine score a voice is "unknown". Tune on-device.
  static const double _threshold = 0.5;

  /// Embeddings from very short clips are unreliable.
  static const int _minSamples = 16000 * 3 ~/ 2; // 1.5 s

  final sherpa.SpeakerEmbeddingExtractor _extractor;
  final sherpa.SpeakerEmbeddingManager _manager;
  final Database _db;

  SherpaSpeakerRecognizer._(this._extractor, this._manager, this._db);

  static Future<SherpaSpeakerRecognizer> create() async {
    final model = await copyAssetFile('assets/models/speaker/model.onnx');
    final extractor = sherpa.SpeakerEmbeddingExtractor(
      config: sherpa.SpeakerEmbeddingExtractorConfig(model: model),
    );
    final manager = sherpa.SpeakerEmbeddingManager(extractor.dim);

    final db = await openDatabase(
      p.join(await getDatabasesPath(), 'speakers.db'),
      version: 1,
      onCreate: (db, _) => db.execute(
        'CREATE TABLE speakers (name TEXT PRIMARY KEY, embedding BLOB NOT NULL)',
      ),
    );
    for (final row in await db.query('speakers')) {
      manager.add(
        name: row['name'] as String,
        embedding: bytesToFloats(row['embedding'] as Uint8List),
      );
    }
    return SherpaSpeakerRecognizer._(extractor, manager, db);
  }

  Float32List _embed(Float32List samples) {
    final stream = _extractor.createStream();
    try {
      stream.acceptWaveform(samples: samples, sampleRate: 16000);
      stream.inputFinished();
      return _extractor.compute(stream);
    } finally {
      stream.free();
    }
  }

  @override
  Future<void> enroll(String name, Float32List samples) async {
    final embedding = _embed(samples);
    if (_manager.contains(name)) _manager.remove(name);
    _manager.add(name: name, embedding: embedding);
    await _db.insert(
      'speakers',
      {'name': name, 'embedding': floatsToBytes(embedding)},
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
  }

  @override
  Future<String?> identify(Float32List samples) async {
    if (samples.length < _minSamples || _manager.numSpeakers == 0) return null;
    final name = _manager.search(embedding: _embed(samples), threshold: _threshold);
    return name.isEmpty ? null : name;
  }

  @override
  Future<List<String>> enrolledSpeakers() async => _manager.allSpeakerNames;

  @override
  Future<void> remove(String name) async {
    _manager.remove(name);
    await _db.delete('speakers', where: 'name = ?', whereArgs: [name]);
  }

  @override
  Future<void> dispose() async {
    _manager.free();
    _extractor.free();
    await _db.close();
  }
}
