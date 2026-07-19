import 'package:flutter_test/flutter_test.dart';
import 'package:recall/impl/routed_inference_engine.dart';

void main() {
  test('derives the http /ask endpoint from the sync ws url', () {
    expect(RoutedInferenceEngine.askUri('ws://10.251.182.196:8000/ws').toString(),
        'http://10.251.182.196:8000/ask');
    expect(RoutedInferenceEngine.askUri('wss://host:8443/ws').toString(),
        'https://host:8443/ask');
  });
}
