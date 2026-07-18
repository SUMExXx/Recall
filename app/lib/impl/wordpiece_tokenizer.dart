/// BERT-uncased WordPiece tokenizer (bge-small-en-v1.5 uses the bert-base-uncased vocab).
/// Lowercase, split on whitespace + punctuation, greedy longest-match against vocab.txt.
class WordPieceTokenizer {
  static const int _maxTokens = 512;
  final Map<String, int> _vocab;
  final int _cls, _sep, _unk;

  WordPieceTokenizer._(this._vocab, this._cls, this._sep, this._unk);

  /// Builds from the newline-separated vocab.txt contents (line number = id).
  factory WordPieceTokenizer(String vocabTxt) {
    final vocab = <String, int>{};
    final lines = vocabTxt.split('\n');
    for (var i = 0; i < lines.length; i++) {
      final token = lines[i].trim();
      if (token.isNotEmpty) vocab[token] = i;
    }
    return WordPieceTokenizer._(
      vocab,
      vocab['[CLS]']!,
      vocab['[SEP]']!,
      vocab['[UNK]']!,
    );
  }

  /// Token ids including [CLS] … [SEP], truncated to the model limit.
  List<int> encode(String text) {
    final ids = <int>[_cls];
    for (final word in _basicTokenize(text.toLowerCase())) {
      _wordPiece(word, ids);
      if (ids.length >= _maxTokens - 1) break;
    }
    ids.add(_sep);
    return ids;
  }

  /// Split on whitespace, and peel each punctuation char into its own token.
  static List<String> _basicTokenize(String text) {
    final words = <String>[];
    final cur = StringBuffer();
    void flush() {
      if (cur.isNotEmpty) {
        words.add(cur.toString());
        cur.clear();
      }
    }

    for (final rune in text.runes) {
      final c = String.fromCharCode(rune);
      if (_isWhitespace(rune)) {
        flush();
      } else if (_isPunct(rune)) {
        flush();
        words.add(c);
      } else {
        cur.write(c);
      }
    }
    flush();
    return words;
  }

  static bool _isWhitespace(int c) =>
      c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D;

  static bool _isPunct(int c) {
    final isAsciiPunct = (c >= 33 && c <= 47) ||
        (c >= 58 && c <= 64) ||
        (c >= 91 && c <= 96) ||
        (c >= 123 && c <= 126);
    if (isAsciiPunct) return true;
    // Non-alphanumeric, non-whitespace symbols count as punctuation too.
    final ch = String.fromCharCode(c);
    return !_isAlphaNumeric(ch) && !_isWhitespace(c);
  }

  static bool _isAlphaNumeric(String ch) =>
      RegExp(r'[\p{L}\p{N}]', unicode: true).hasMatch(ch);

  /// Greedy longest-match WordPiece; unknown word → single [UNK].
  void _wordPiece(String word, List<int> ids) {
    var start = 0;
    final pieces = <int>[];
    while (start < word.length) {
      var end = word.length;
      int? match;
      while (start < end) {
        final sub = (start > 0 ? '##' : '') + word.substring(start, end);
        final id = _vocab[sub];
        if (id != null) {
          match = id;
          break;
        }
        end--;
      }
      if (match == null) {
        ids.add(_unk);
        return;
      }
      pieces.add(match);
      start = end;
    }
    ids.addAll(pieces);
  }
}
