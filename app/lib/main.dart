import 'package:flutter/material.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;

import 'home_page.dart';
import 'theme.dart';

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
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light(),
      darkTheme: AppTheme.dark(),
      themeMode: ThemeMode.system, // dark-first design, follows the phone
      home: const HomePage(),
    );
  }
}
