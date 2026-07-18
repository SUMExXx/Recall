import 'package:flutter_test/flutter_test.dart';
import 'package:recall/impl/wordpiece_tokenizer.dart';

void main() {
  // ids: [PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 hello=4 world=5 test=6 ##ing=7 ,=8
  const vocab = '[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\ntest\n##ing\n,';
  final tok = WordPieceTokenizer(vocab);

  test('wraps with CLS and SEP', () {
    expect(tok.encode('hello world'), [2, 4, 5, 3]);
  });

  test('greedy wordpiece splits a known suffix', () {
    expect(tok.encode('testing'), [2, 6, 7, 3]); // test + ##ing
  });

  test('unknown word maps to UNK', () {
    expect(tok.encode('xyzzy'), [2, 1, 3]);
  });

  test('punctuation becomes its own token', () {
    expect(tok.encode('hello, world'), [2, 4, 8, 5, 3]);
  });
}
