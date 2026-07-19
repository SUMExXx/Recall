import 'package:flutter_test/flutter_test.dart';
import 'package:recall/memory_pipeline.dart';

void main() {
  test('keeps real language', () {
    expect(MemoryPipeline.isLikelyLanguage('Remember to buy milk'), isTrue);
    expect(MemoryPipeline.isLikelyLanguage('OK'), isTrue);
    expect(MemoryPipeline.isLikelyLanguage('I have 3 dogs!'), isTrue);
  });

  test('discards noise and non-language', () {
    expect(MemoryPipeline.isLikelyLanguage('...'), isFalse);
    expect(MemoryPipeline.isLikelyLanguage('♪♪♪'), isFalse);
    expect(MemoryPipeline.isLikelyLanguage('[Music]'), isFalse);
    expect(MemoryPipeline.isLikelyLanguage('(upbeat music)'), isFalse);
    expect(MemoryPipeline.isLikelyLanguage('你好世界'), isFalse);
    expect(MemoryPipeline.isLikelyLanguage('12345'), isFalse);
  });
}
