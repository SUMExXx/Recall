import 'package:flutter/material.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;

import 'home_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  sherpa.initBindings(); // wire sherpa-onnx FFI
  OrtEnv.instance.init(); // wire onnxruntime (bge embeddings)
  runApp(const RecallApp());
}

class RecallApp extends StatelessWidget {
  const RecallApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Recall',
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF6750A4),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}
