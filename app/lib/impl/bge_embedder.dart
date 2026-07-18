import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';

import 'wordpiece_tokenizer.dart';

/// bge-small-en-v1.5 text embeddings via onnxruntime. CLS pooling + L2 norm, 384-dim.
/// Not a public pipeline interface — an internal detail of [SqliteVectorStore].
///
/// Requires `OrtEnv.instance.init()` to have been called once at app startup.
class BgeEmbedder {
  static const String _queryPrefix =
      'Represent this sentence for searching relevant passages: ';

  final OrtSession _session;
  final WordPieceTokenizer _tokenizer;

  BgeEmbedder._(this._session, this._tokenizer);

  static Future<BgeEmbedder> create() async {
    final vocab = await rootBundle.loadString('assets/models/embed/vocab.txt');
    final raw = await rootBundle.load('assets/models/embed/model.onnx');
    final options = OrtSessionOptions()..setIntraOpNumThreads(1);
    final session = OrtSession.fromBuffer(raw.buffer.asUint8List(), options);
    return BgeEmbedder._(session, WordPieceTokenizer(vocab));
  }

  Future<Float32List> embed(String text) => _run(text);

  Future<Float32List> embedQuery(String query) => _run(_queryPrefix + query);

  Future<Float32List> _run(String text) async {
    final ids = tokenIds(text);
    final shape = [1, ids.length];
    final ones = List<int>.filled(ids.length, 1);
    final zeros = List<int>.filled(ids.length, 0);

    final idsOrt = OrtValueTensor.createTensorWithDataList(Int64List.fromList(ids), shape);
    final maskOrt = OrtValueTensor.createTensorWithDataList(Int64List.fromList(ones), shape);
    final typeOrt = OrtValueTensor.createTensorWithDataList(Int64List.fromList(zeros), shape);
    final runOptions = OrtRunOptions();

    List<OrtValue?>? outputs;
    try {
      outputs = await _session.runAsync(runOptions, {
        'input_ids': idsOrt,
        'attention_mask': maskOrt,
        'token_type_ids': typeOrt,
      });
      var i = _session.outputNames.indexOf('last_hidden_state');
      if (i < 0) i = 0;
      // last_hidden_state: [1, seq, 384] — take the [CLS] vector at position 0.
      final lhs = outputs![i]!.value as List<List<List<double>>>;
      return _normalize(lhs[0][0]);
    } finally {
      idsOrt.release();
      maskOrt.release();
      typeOrt.release();
      runOptions.release();
      outputs?.forEach((o) => o?.release());
    }
  }

  /// Exposed for tests: the tokenizer's [CLS]…[SEP] id sequence.
  List<int> tokenIds(String text) => _tokenizer.encode(text);

  static Float32List _normalize(List<double> v) {
    var sum = 0.0;
    for (final x in v) {
      sum += x * x;
    }
    final norm = sum > 0 ? 1.0 / math.sqrt(sum) : 0.0;
    final out = Float32List(v.length);
    for (var i = 0; i < v.length; i++) {
      out[i] = v[i] * norm;
    }
    return out;
  }

  void dispose() => _session.release();
}
