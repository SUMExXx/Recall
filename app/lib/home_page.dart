import 'dart:async';

import 'package:flutter/material.dart';

import 'util/audio_util.dart';
import 'chat_store.dart';
import 'memory_pipeline.dart';
import 'pipeline/vector_store.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  MemoryPipeline? _pipe;
  String? _initError;

  final _queryController = TextEditingController();
  final _nameController = TextEditingController(text: 'Me');
  final _askController = TextEditingController();
  final List<Memory> _memories = [];
  StreamSubscription<String>? _statusSub;
  StreamSubscription<Memory>? _memorySub;

  String _status = 'Loading models…';
  bool _capturing = false;
  bool _searching = false;
  bool _needsSetup = false; // first run: no speaker enrolled yet
  bool _recording = false;
  bool _asking = false;
  StreamSubscription<String>? _syncSub;
  String _syncStatus = '';

  ChatStore? _chat;
  final List<ChatMessage> _messages = [];
  final _chatScroll = ScrollController();

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final pipe = await MemoryPipeline.create();
      _statusSub = pipe.status.listen((s) => setState(() => _status = s));
      _memorySub = pipe.onMemory.listen((_) => _refresh());
      _syncSub = pipe.syncStatus.listen((s) => setState(() => _syncStatus = s));
      _pipe = pipe;
      _chat = await ChatStore.create();
      final history = await _chat!.history();
      setState(() => _messages.addAll(history));
      final speakers = await pipe.enrolledSpeakers();
      setState(() {
        _status = 'Idle';
        _needsSetup = speakers.isEmpty; // first run → recognize + add the user
      });
      await _refresh();
    } catch (e) {
      setState(() => _initError = '$e');
    }
  }

  /// First-run step: record the user's voice and add them as a speaker.
  Future<void> _setupVoice() async {
    final name = _nameController.text.trim();
    if (name.isEmpty) {
      _toast('Enter your name first');
      return;
    }
    setState(() => _recording = true);
    final samples = await recordSamples(const Duration(seconds: 8));
    if (samples == null) {
      setState(() => _recording = false);
      _toast('Microphone permission denied');
      return;
    }
    await _pipe!.enroll(name, samples);
    setState(() {
      _recording = false;
      _needsSetup = false;
    });
    _toast('Added "$name"');
  }

  Future<void> _refresh() async {
    final items = await _pipe!.recent(200);
    setState(() {
      _memories
        ..clear()
        ..addAll(items);
    });
  }

  Future<void> _toggleCapture() async {
    final pipe = _pipe!;
    if (_capturing) {
      await pipe.stop();
      setState(() => _capturing = false);
    } else {
      final ok = await pipe.start();
      if (!ok) {
        _toast('Microphone permission denied');
        return;
      }
      setState(() => _capturing = true);
    }
  }

  Future<void> _search() async {
    final q = _queryController.text.trim();
    if (q.isEmpty) {
      await _refresh();
      return;
    }
    setState(() => _searching = true);
    final results = await _pipe!.search(q);
    setState(() {
      _searching = false;
      _memories
        ..clear()
        ..addAll(results);
    });
  }

  Future<void> _askQuestion() async {
    final q = _askController.text.trim();
    if (q.isEmpty || _asking) return;
    FocusScope.of(context).unfocus();
    _askController.clear();

    final userMsg = await _chat!.add(ChatMessage(
      timestamp: DateTime.now(),
      fromUser: true,
      text: q,
    ));
    setState(() {
      _messages.add(userMsg);
      _asking = true;
    });
    _scrollToBottom();

    String answer;
    try {
      answer = await _pipe!.ask(q);
    } catch (e) {
      answer = 'Error: $e';
    }
    if (!mounted) return;

    final botMsg = await _chat!.add(ChatMessage(
      timestamp: DateTime.now(),
      fromUser: false,
      text: answer,
    ));
    setState(() {
      _messages.add(botMsg);
      _asking = false;
    });
    _scrollToBottom();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_chatScroll.hasClients) return;
      _chatScroll.animateTo(
        _chatScroll.position.maxScrollExtent,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> _clearChat() async {
    await _chat!.clear();
    setState(() => _messages.clear());
  }

  Future<void> _syncSettings() async {
    final controller = TextEditingController(text: _pipe!.syncServerUrl ?? '');
    final url = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('PC sync'),
        content: TextField(
          controller: controller,
          autofocus: true,
          keyboardType: TextInputType.url,
          decoration: const InputDecoration(
            labelText: 'PC WebSocket URL',
            hintText: 'ws://192.168.1.20:8765',
            helperText: 'Empty disables sync. Unsent memories flush on connect.',
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, controller.text.trim()),
            child: const Text('Save'),
          ),
        ],
      ),
    );
    if (url == null) return;
    await _pipe!.setSyncServer(url);
    _toast(url.isEmpty ? 'PC sync disabled' : 'Syncing to $url');
  }

  Future<void> _loadLocalModel() async {
    if (!await _pipe!.hasStorageAccess()) {
      await _pipe!.requestStorageAccess();
      _toast('Grant "All files access", then tap Load local model again');
      return;
    }
    if (!mounted) return;
    final controller = TextEditingController(
      text: '/sdcard/models/qwen2_5_vl_7b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite_gen5',
    );
    final path = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Load local model'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(
            labelText: 'GenieX bundle folder',
            helperText: 'Folder under /sdcard/models containing genie_config.json',
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, controller.text.trim()),
            child: const Text('Load'),
          ),
        ],
      ),
    );
    if (path == null || path.isEmpty) return;
    _toast('Registering model… this can take a while');
    try {
      await _pipe!.registerLocalModel(path);
      _toast('Model registered');
    } catch (e) {
      _toast('Failed: $e');
    }
  }

  Future<void> _downloadModel() async {
    final controller = TextEditingController();
    final name = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Download LLM model'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(
            labelText: 'Qualcomm AI Hub model name',
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, controller.text.trim()),
            child: const Text('Download'),
          ),
        ],
      ),
    );
    if (name == null || name.isEmpty) return;
    _toast('Downloading "$name"… this can take a while');
    try {
      await _pipe!.downloadModel(name);
      _toast('Model "$name" ready');
    } catch (e) {
      _toast('Download failed: $e');
    }
  }

  Future<void> _enroll() async {
    final name = await _promptName();
    if (name == null || name.isEmpty) return;

    if (_capturing) await _toggleCapture(); // free the mic
    _toast('Recording 8s — keep talking…');
    final samples = await recordSamples(const Duration(seconds: 8));
    if (samples == null) {
      _toast('Microphone permission denied');
      return;
    }
    await _pipe!.enroll(name, samples);
    _toast('Enrolled "$name"');
  }

  Future<String?> _promptName() {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Enroll speaker'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: const InputDecoration(labelText: 'Name'),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, controller.text.trim()),
            child: const Text('Record'),
          ),
        ],
      ),
    );
  }

  void _toast(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  @override
  void dispose() {
    _statusSub?.cancel();
    _memorySub?.cancel();
    _syncSub?.cancel();
    _queryController.dispose();
    _nameController.dispose();
    _askController.dispose();
    _chatScroll.dispose();
    _chat?.dispose();
    _pipe?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_pipe != null && !_needsSetup) return _buildReady();

    return Scaffold(
      appBar: AppBar(title: const Text('Recall')),
      body: _initError != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text('Failed to start:\n$_initError'),
              ),
            )
          : _pipe == null
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const CircularProgressIndicator(),
                      const SizedBox(height: 16),
                      Text(_status),
                    ],
                  ),
                )
              : _buildSetup(),
    );
  }

  /// Main UI once models are loaded and a speaker is enrolled: two tabs
  /// (Memories, Ask) plus the global capture Start/Stop button.
  Widget _buildReady() {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Recall'),
          actions: [
            IconButton(
              tooltip: 'PC sync',
              onPressed: _syncSettings,
              icon: const Icon(Icons.sync),
            ),
            IconButton(
              tooltip: 'Download LLM model',
              onPressed: _downloadModel,
              icon: const Icon(Icons.download),
            ),
            IconButton(
              tooltip: 'Load local model',
              onPressed: _loadLocalModel,
              icon: const Icon(Icons.folder_open),
            ),
            IconButton(
              tooltip: 'Enroll speaker',
              onPressed: _enroll,
              icon: const Icon(Icons.person_add),
            ),
            IconButton(
              tooltip: 'Clear chat',
              onPressed: _messages.isEmpty ? null : _clearChat,
              icon: const Icon(Icons.delete_sweep),
            ),
          ],
          bottom: const TabBar(
            tabs: [
              Tab(text: 'Memories', icon: Icon(Icons.list)),
              Tab(text: 'Ask', icon: Icon(Icons.question_answer)),
            ],
          ),
        ),
        body: TabBarView(
          children: [_buildBody(), _buildAsk()],
        ),
        floatingActionButton: FloatingActionButton.extended(
          onPressed: _toggleCapture,
          icon: Icon(_capturing ? Icons.stop : Icons.mic),
          label: Text(_capturing ? 'Stop' : 'Start'),
        ),
      ),
    );
  }

  Widget _buildAsk() {
    return Column(
      children: [
        Expanded(
          child: _messages.isEmpty && !_asking
              ? const Center(
                  child: Padding(
                    padding: EdgeInsets.all(24),
                    child: Text(
                      'Ask a question and Recall will answer from your memories.',
                      textAlign: TextAlign.center,
                    ),
                  ),
                )
              : ListView.builder(
                  controller: _chatScroll,
                  padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
                  itemCount: _messages.length + (_asking ? 1 : 0),
                  itemBuilder: (context, i) {
                    if (i == _messages.length) return _buildTyping();
                    return _buildBubble(_messages[i]);
                  },
                ),
        ),
        const Divider(height: 1),
        SafeArea(
          top: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _askController,
                    textInputAction: TextInputAction.send,
                    onSubmitted: (_) => _askQuestion(),
                    minLines: 1,
                    maxLines: 4,
                    decoration: const InputDecoration(
                      hintText: 'Ask about your memories…',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filled(
                  onPressed: _asking ? null : _askQuestion,
                  icon: const Icon(Icons.send),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildBubble(ChatMessage m) {
    final scheme = Theme.of(context).colorScheme;
    final bg = m.fromUser ? scheme.primary : scheme.surfaceContainerHighest;
    final fg = m.fromUser ? scheme.onPrimary : scheme.onSurface;
    return Align(
      alignment: m.fromUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.78,
        ),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(16),
        ),
        child: SelectableText(m.text, style: TextStyle(color: fg)),
      ),
    );
  }

  Widget _buildTyping() {
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(16),
        ),
        child: const SizedBox(
          width: 20,
          height: 20,
          child: CircularProgressIndicator(strokeWidth: 2),
        ),
      ),
    );
  }

  Widget _buildSetup() {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Icon(Icons.record_voice_over, size: 64),
            const SizedBox(height: 16),
            Text(
              'Set up your voice',
              style: Theme.of(context).textTheme.headlineSmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            const Text(
              'Record a few seconds of speech so Recall can recognize you '
              'and tag your memories with your name.',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            TextField(
              controller: _nameController,
              enabled: !_recording,
              decoration: const InputDecoration(
                labelText: 'Your name',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _recording ? null : _setupVoice,
              icon: _recording
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.mic),
              label: Text(_recording ? 'Recording… keep talking' : 'Record my voice (8s)'),
            ),
            TextButton(
              onPressed: _recording ? null : () => setState(() => _needsSetup = false),
              child: const Text('Skip for now'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBody() {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Row(
            children: [
              const Icon(Icons.circle, size: 12, color: Colors.grey),
              const SizedBox(width: 8),
              Text(_status),
              const Spacer(),
              if (_syncStatus.isNotEmpty) ...[
                const Icon(Icons.sync, size: 14, color: Colors.grey),
                const SizedBox(width: 4),
                Text(_syncStatus, style: const TextStyle(color: Colors.grey)),
              ],
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _queryController,
                  textInputAction: TextInputAction.search,
                  onSubmitted: (_) => _search(),
                  decoration: const InputDecoration(
                    hintText: 'Search memories…',
                    isDense: true,
                  ),
                ),
              ),
              IconButton(
                onPressed: _searching ? null : _search,
                icon: const Icon(Icons.search),
              ),
            ],
          ),
        ),
        const Divider(height: 1),
        Expanded(
          child: _memories.isEmpty
              ? const Center(child: Text('No memories yet'))
              : ListView.separated(
                  itemCount: _memories.length,
                  separatorBuilder: (_, __) => const Divider(height: 1),
                  itemBuilder: (_, i) {
                    final m = _memories[i];
                    return ListTile(
                      title: Text(m.text),
                      subtitle: Text('${m.speaker} · ${_time(m.timestamp)}'),
                    );
                  },
                ),
        ),
      ],
    );
  }

  static String _time(DateTime t) {
    String two(int n) => n.toString().padLeft(2, '0');
    return '${two(t.hour)}:${two(t.minute)}';
  }
}
