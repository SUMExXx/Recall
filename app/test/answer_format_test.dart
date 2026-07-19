import 'package:flutter_test/flutter_test.dart';
import 'package:recall/util/answer_format.dart';

void main() {
  test('splits "From memory excerpt" citation', () {
    final (a, r) = splitAnswer('Green. From memory excerpt [b104abcd] your bike.');
    expect(a, 'Green.');
    expect(r, 'From memory excerpt [b104abcd] your bike.');
  });

  test('splits "Source:" citation', () {
    final (a, r) = splitAnswer(
        'You parked on level 3 of the garage. Source: [b7d2d578:1] (note: Parking)');
    expect(a, 'You parked on level 3 of the garage.');
    expect(r, 'Source: [b7d2d578:1] (note: Parking)');
  });

  test('splits a bare bracketed id', () {
    final (a, r) = splitAnswer('Blue [a1b2c3]');
    expect(a, 'Blue');
    expect(r, '[a1b2c3]');
  });

  test('no reference → whole answer, null', () {
    final (a, r) = splitAnswer('I have no memory about that yet.');
    expect(a, 'I have no memory about that yet.');
    expect(r, isNull);
  });
}
