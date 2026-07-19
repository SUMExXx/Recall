import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'util/answer_format.dart';
import 'util/audio_util.dart';
import 'chat_store.dart';
import 'memory_pipeline.dart';
import 'pipeline/vector_store.dart';
import 'theme.dart';
import 'widgets/listening_orb.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  MemoryPipeline? _pipe;
  String? _initError;

  int _tab = 0; // 0 = Memories, 1 = Ask

  final _queryController = TextEditingController();
  final _nameController = TextEditingController(text: 'Me');
  final _askController = TextEditingController();
  final List<Memory> _memories = [];
  final Set<int> _selected = {}; // memory ids selected for deletion
  StreamSubscription<String>? _statusSub;
  StreamSubscription<Memory>? _memorySub;

  String _status = 'Loading models…';
  bool _capturing = false; // mic subscription is live
  bool _manualCapture = false; // FAB: actively saving every utterance
  bool _searching = false;
  bool _needsSetup = false; // first run: no speaker enrolled yet
  bool _recording = false;
  bool _asking = false;
  StreamSubscription<String>? _syncSub;
  String _syncStatus = '';

  StreamSubscription<bool>? _wakeSub;
  bool _listening = false; // orb visible: wake window open
  bool _wakeEnabled = true;
  bool _showWakeIntro = false; // first-run wake-word onboarding

  ChatStore? _chat;
  final List<ChatMessage> _messages = [];
  final _chatScroll = ScrollController();

  // Voice-ask (mic button in the Ask tab): push-to-talk recording → Sarvam STT.
  StreamSubscription<String>? _questionSub; // wake-word questions in ask mode
  StreamSubscription<Uint8List>? _voiceSub;
  final List<double> _voiceBuf = [];
  bool _voiceAsking = false;
  bool _voiceWasCapturing = false;

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
      _wakeSub = pipe.listening.listen((v) => setState(() => _listening = v));
      // Wake-word question while on the Ask tab → answer it (chat + speech).
      _questionSub = pipe.spokenQuestions
          .listen((q) => _askQuestion(prefilled: q));
      _pipe = pipe;
      _chat = await ChatStore.create();
      final history = await _chat!.history();
      setState(() => _messages.addAll(history));
      final speakers = await pipe.enrolledSpeakers();
      setState(() {
        _status = 'Idle';
        _syncStatus = pipe.syncStatusNow; // seed: early sync events already fired
        _wakeEnabled = pipe.wakeWordEnabled;
        _needsSetup = speakers.isEmpty; // first run → recognize + add the user
        _showWakeIntro = speakers.isEmpty; // first run → explain the wake word
      });
      await _refresh();
      // Always-on wake word: start background listening as soon as the app
      // opens so "Hey Recall" works without tapping Start (like Google
      // Assistant). First-run users start from the wake-word intro instead.
      if (!_needsSetup && !_showWakeIntro) {
        _applyWakeMode();
        await _syncMic();
      }
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

  void _toggleSelect(int id) {
    setState(() {
      if (!_selected.remove(id)) _selected.add(id);
    });
  }

  Future<void> _deleteSelected() async {
    final n = _selected.length;
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Delete $n ${n == 1 ? 'memory' : 'memories'}?'),
        content: const Text('This cannot be undone.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Delete')),
        ],
      ),
    );
    if (ok != true) return;
    final ids = _selected.toList();
    await _pipe!.deleteMemories(ids);
    setState(() {
      _memories.removeWhere((m) => _selected.contains(m.id));
      _selected.clear();
    });
    _toast('Deleted $n ${n == 1 ? 'memory' : 'memories'}');
  }

  /// The FAB toggles a manual "save everything" session. Ending it leaves the
  /// mic running for the wake word (if enabled) — only turning the wake word
  /// off actually stops background listening.
  Future<void> _toggleCapture() async {
    setState(() => _manualCapture = !_manualCapture);
    _applyWakeMode();
    if (!await _syncMic()) {
      // Permission denied — roll back the toggle.
      setState(() => _manualCapture = !_manualCapture);
      _applyWakeMode();
    }
  }

  /// Tells the pipeline whether to gate on the wake word. We save every
  /// utterance during a manual capture session; otherwise we gate whenever the
  /// wake word is enabled.
  void _applyWakeMode() {
    _pipe!.setWakeWord(_wakeEnabled && !_manualCapture);
  }

  /// Brings the mic subscription in line with what's needed: it must run while
  /// the wake word is enabled or a manual capture session is active, and can be
  /// released otherwise. Returns false only if mic permission was denied.
  Future<bool> _syncMic() async {
    final shouldRun = _wakeEnabled || _manualCapture;
    if (shouldRun && !_capturing) {
      final ok = await _pipe!.start();
      if (!ok) {
        _toast('Microphone permission denied');
        return false;
      }
      setState(() => _capturing = true);
    } else if (!shouldRun && _capturing) {
      await _pipe!.stop();
      setState(() => _capturing = false);
    }
    return true;
  }

  Future<void> _toggleWake() async {
    setState(() => _wakeEnabled = !_wakeEnabled);
    _applyWakeMode();
    if (!await _syncMic()) {
      setState(() => _wakeEnabled = !_wakeEnabled);
      _applyWakeMode();
      return;
    }
    _toast(_wakeEnabled
        ? 'Say “Hey Recall” to save a memory'
        : 'Wake word off');
  }

  Future<void> _finishWakeIntro(bool enable) async {
    setState(() {
      _wakeEnabled = enable;
      _showWakeIntro = false;
    });
    _applyWakeMode();
    await _syncMic();
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

  /// Asks [prefilled] (from voice) or the text field, shows it in chat, and
  /// speaks the answer via Sarvam TTS.
  Future<void> _askQuestion({String? prefilled}) async {
    final q = (prefilled ?? _askController.text).trim();
    if (q.isEmpty || _asking) return;
    FocusScope.of(context).unfocus();
    if (prefilled == null) _askController.clear();

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

    String raw;
    try {
      raw = await _pipe!.ask(q);
    } catch (e) {
      raw = 'Error: $e';
    }
    if (!mounted) return;

    // Separate the answer from its memory citation: speak/show the answer,
    // box the reference below it.
    final isError = raw.startsWith('Error:');
    final (answer, reference) = isError ? (raw, null) : splitAnswer(raw);

    final botMsg = await _chat!.add(ChatMessage(
      timestamp: DateTime.now(),
      fromUser: false,
      text: answer,
      reference: reference,
    ));
    setState(() {
      _messages.add(botMsg);
      _asking = false;
    });
    _scrollToBottom();

    // Voice only the answer (never the citation); best-effort, never blocks.
    if (!isError) {
      unawaited(_pipe!.speak(answer).catchError((_) {}));
    }
  }

  /// Mic button in Ask: tap to start recording, tap again to transcribe
  /// (Sarvam STT) and ask. Releases the wake-word mic while recording.
  Future<void> _toggleVoiceAsk() async {
    if (_voiceAsking) {
      await _voiceSub?.cancel();
      _voiceSub = null;
      setState(() => _voiceAsking = false);
      final samples = Float32List.fromList(_voiceBuf);
      _voiceBuf.clear();
      if (_voiceWasCapturing) await _syncMic(); // restore background listening
      if (samples.isEmpty) return;
      setState(() => _asking = true); // show progress while transcribing
      String q;
      try {
        q = await _pipe!.transcribeQuestion(samples);
      } catch (e) {
        setState(() => _asking = false);
        _toast('Transcription failed');
        return;
      }
      setState(() => _asking = false);
      if (q.isEmpty) {
        _toast('Didn’t catch that');
        return;
      }
      await _askQuestion(prefilled: q);
      return;
    }

    // Start recording — the mic can't be shared, so pause wake-word capture.
    if (!await ensureMicPermission()) {
      _toast('Microphone permission denied');
      return;
    }
    _voiceWasCapturing = _capturing;
    if (_capturing) {
      await _pipe!.stop();
      setState(() => _capturing = false);
    }
    _voiceBuf.clear();
    _voiceSub = micStream().listen((b) => _voiceBuf.addAll(pcm16ToFloat32(b)));
    setState(() => _voiceAsking = true);
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
            hintText: 'ws://10.20.3.196:8000/ws',
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

    // recordSamples needs exclusive mic access — release ours, then restore
    // background listening afterwards.
    final wasCapturing = _capturing;
    if (wasCapturing) {
      await _pipe!.stop();
      setState(() => _capturing = false);
    }
    _toast('Recording 8s — keep talking…');
    final samples = await recordSamples(const Duration(seconds: 8));
    if (samples == null) {
      _toast('Microphone permission denied');
      if (wasCapturing) await _syncMic();
      return;
    }
    await _pipe!.enroll(name, samples);
    if (wasCapturing) await _syncMic();
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
    _wakeSub?.cancel();
    _questionSub?.cancel();
    _voiceSub?.cancel();
    _queryController.dispose();
    _nameController.dispose();
    _askController.dispose();
    _chatScroll.dispose();
    _chat?.dispose();
    _pipe?.dispose();
    super.dispose();
  }

  // ==================== presentation ====================

  /// Connected to the PC hub: actively linked, whether idle or flushing.
  bool get _pcConnected =>
      _syncStatus == 'connected' ||
      _syncStatus == 'synced' ||
      _syncStatus.endsWith('pending');

  void _setTab(int i) {
    setState(() => _tab = i);
    _pipe?.setAskMode(i == 1); // wake-word captures become questions in Ask
  }

  @override
  Widget build(BuildContext context) {
    if (_pipe != null && !_needsSetup && !_showWakeIntro) return _buildReady();

    return Scaffold(
      body: _glowBackground(
        child: Center(
          child: _initError != null
              ? Padding(
                  padding: const EdgeInsets.all(24),
                  child: Text('Failed to start:\n$_initError',
                      textAlign: TextAlign.center),
                )
              : _pipe == null
                  ? _buildLoading()
                  : _needsSetup
                      ? _buildSetup()
                      : _buildWakeIntro(),
        ),
      ),
    );
  }

  /// A subtle top radial glow to give the dark-first screens depth.
  Widget _glowBackground({required Widget child}) {
    final cs = Theme.of(context).colorScheme;
    return DecoratedBox(
      decoration: BoxDecoration(
        gradient: RadialGradient(
          center: const Alignment(0, -0.7),
          radius: 1.2,
          colors: [
            cs.primary.withValues(alpha: 0.12),
            cs.surface,
          ],
          stops: const [0.0, 0.7],
        ),
      ),
      child: child,
    );
  }

  Widget _buildLoading() {
    final cs = Theme.of(context).colorScheme;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        _orbGlyph(96),
        const SizedBox(height: 28),
        Text('Recall',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.w700, letterSpacing: 0.5)),
        const SizedBox(height: 20),
        SizedBox(
          width: 26,
          height: 26,
          child: CircularProgressIndicator(strokeWidth: 2.5, color: cs.primary),
        ),
        const SizedBox(height: 16),
        Text(_status, style: TextStyle(color: cs.onSurfaceVariant)),
      ],
    );
  }

  /// The static, non-animated orb used as a brand glyph on non-ready screens.
  Widget _orbGlyph(double size) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: RadialGradient(
          colors: [Colors.white, cs.primary, AppTheme.accent.withValues(alpha: 0.85)],
          stops: const [0.0, 0.5, 1.0],
        ),
        boxShadow: [
          BoxShadow(color: cs.primary.withValues(alpha: 0.55), blurRadius: 40, spreadRadius: 8),
          BoxShadow(color: AppTheme.accent.withValues(alpha: 0.3), blurRadius: 60, spreadRadius: 2),
        ],
      ),
    );
  }

  /// Main UI once models are loaded and a speaker is enrolled.
  Widget _buildReady() {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        titleSpacing: 16,
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            _orbGlyph(22),
            const SizedBox(width: 10),
            const Text('Recall'),
          ],
        ),
        actions: [
          IconButton(
            tooltip: _wakeEnabled ? '“Hey Recall”: on' : '“Hey Recall”: off',
            isSelected: _wakeEnabled,
            onPressed: _toggleWake,
            icon: const Icon(Icons.hearing_disabled),
            selectedIcon: Icon(Icons.hearing, color: cs.primary),
          ),
          _pcChip(),
          PopupMenuButton<String>(
            tooltip: 'More',
            icon: const Icon(Icons.more_vert),
            onSelected: (v) {
              switch (v) {
                case 'enroll':
                  _enroll();
                case 'download':
                  _downloadModel();
                case 'local':
                  _loadLocalModel();
                case 'clear':
                  _clearChat();
              }
            },
            itemBuilder: (_) => [
              const PopupMenuItem(
                value: 'enroll',
                child: ListTile(
                    leading: Icon(Icons.person_add_alt),
                    title: Text('Enroll speaker'),
                    contentPadding: EdgeInsets.zero),
              ),
              const PopupMenuItem(
                value: 'download',
                child: ListTile(
                    leading: Icon(Icons.cloud_download_outlined),
                    title: Text('Download LLM model'),
                    contentPadding: EdgeInsets.zero),
              ),
              const PopupMenuItem(
                value: 'local',
                child: ListTile(
                    leading: Icon(Icons.folder_open_outlined),
                    title: Text('Load local model'),
                    contentPadding: EdgeInsets.zero),
              ),
              PopupMenuItem(
                value: 'clear',
                enabled: _messages.isNotEmpty,
                child: const ListTile(
                    leading: Icon(Icons.delete_sweep_outlined),
                    title: Text('Clear chat'),
                    contentPadding: EdgeInsets.zero),
              ),
            ],
          ),
          const SizedBox(width: 4),
        ],
      ),
      body: Stack(
        children: [
          IndexedStack(
            index: _tab,
            children: [_buildMemories(), _buildAsk()],
          ),
          Positioned.fill(
            child: ListeningOrb(
              visible: _listening,
              label: _tab == 1 ? 'Ask me…' : 'Listening…',
            ),
          ),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: _setTab,
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.bubble_chart_outlined),
            selectedIcon: Icon(Icons.bubble_chart),
            label: 'Memories',
          ),
          NavigationDestination(
            icon: Icon(Icons.forum_outlined),
            selectedIcon: Icon(Icons.forum),
            label: 'Ask',
          ),
        ],
      ),
      floatingActionButton: _tab == 0 && !_listening && _selected.isEmpty
          ? FloatingActionButton.extended(
              onPressed: _toggleCapture,
              backgroundColor: _manualCapture ? cs.error : cs.primary,
              foregroundColor: _manualCapture ? cs.onError : cs.onPrimary,
              icon: Icon(_manualCapture ? Icons.stop : Icons.mic),
              label: Text(_manualCapture ? 'Stop' : 'Record'),
            )
          : null,
    );
  }

  /// AppBar PC-sync chip: colored dot + short state, tap opens sync settings.
  Widget _pcChip() {
    final cs = Theme.of(context).colorScheme;
    final (color, icon, label) = _pcConnected
        ? (Colors.greenAccent.shade400, Icons.cloud_done, 'PC')
        : _syncStatus == 'connecting'
            ? (Colors.orangeAccent, Icons.cloud_sync, '···')
            : (cs.onSurfaceVariant, Icons.cloud_off, 'off');
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Tooltip(
        message: _syncStatus.isEmpty ? 'PC sync: off' : 'PC: $_syncStatus',
        child: InkWell(
          borderRadius: BorderRadius.circular(30),
          onTap: _syncSettings,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: cs.surfaceContainerHigh,
              borderRadius: BorderRadius.circular(30),
              border: Border.all(color: color.withValues(alpha: 0.5)),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, size: 16, color: color),
                const SizedBox(width: 5),
                Text(label,
                    style: TextStyle(
                        fontSize: 12, fontWeight: FontWeight.w600, color: cs.onSurface)),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ---------------------------------------------------------- Memories tab

  Widget _buildMemories() {
    return Column(
      children: [
        if (_selected.isNotEmpty) _selectionBar() else _statusBar(),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 8),
          child: TextField(
            controller: _queryController,
            textInputAction: TextInputAction.search,
            onSubmitted: (_) => _search(),
            onChanged: (_) => setState(() {}), // toggles the clear button
            decoration: InputDecoration(
              hintText: 'Search memories…',
              prefixIcon: _searching
                  ? const Padding(
                      padding: EdgeInsets.all(12),
                      child: SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2)),
                    )
                  : const Icon(Icons.search),
              suffixIcon: _queryController.text.isEmpty
                  ? null
                  : IconButton(
                      tooltip: 'Clear',
                      icon: const Icon(Icons.close),
                      onPressed: () {
                        _queryController.clear();
                        setState(() {});
                        _refresh();
                      },
                    ),
            ),
          ),
        ),
        Expanded(
          child: _memories.isEmpty
              ? _emptyState(
                  icon: Icons.auto_awesome,
                  title: 'No memories yet',
                  body: _queryController.text.isEmpty
                      ? 'Say “Hey Recall” or tap Record to capture your first memory.'
                      : 'No memories match “${_queryController.text}”.',
                )
              : ListView.separated(
                  padding: const EdgeInsets.fromLTRB(12, 4, 12, 96),
                  itemCount: _memories.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 8),
                  itemBuilder: (_, i) => _memoryCard(_memories[i]),
                ),
        ),
      ],
    );
  }

  Widget _statusBar() {
    final cs = Theme.of(context).colorScheme;
    final recording = _status.toLowerCase().contains('record');
    final listening = _status.toLowerCase().contains('listen');
    final (dot, text) = recording
        ? (cs.error, 'Recording')
        : listening
            ? (Colors.greenAccent.shade400, 'Listening')
            : (cs.onSurfaceVariant, _status);
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 10, 20, 4),
      child: Row(
        children: [
          Container(width: 9, height: 9, decoration: BoxDecoration(color: dot, shape: BoxShape.circle)),
          const SizedBox(width: 8),
          Text(text, style: TextStyle(color: cs.onSurfaceVariant, fontWeight: FontWeight.w500)),
          const Spacer(),
          if (_memories.isNotEmpty)
            Text('${_memories.length} ${_memories.length == 1 ? 'memory' : 'memories'}',
                style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12)),
        ],
      ),
    );
  }

  Widget _selectionBar() {
    final cs = Theme.of(context).colorScheme;
    return Container(
      color: cs.primaryContainer,
      padding: const EdgeInsets.symmetric(horizontal: 4),
      child: Row(
        children: [
          IconButton(
            tooltip: 'Cancel selection',
            icon: const Icon(Icons.close),
            onPressed: () => setState(_selected.clear),
          ),
          Text('${_selected.length} selected',
              style: TextStyle(
                  color: cs.onPrimaryContainer, fontWeight: FontWeight.w600)),
          const Spacer(),
          IconButton(
            tooltip: 'Delete selected',
            icon: const Icon(Icons.delete_outline),
            onPressed: _deleteSelected,
          ),
        ],
      ),
    );
  }

  Widget _memoryCard(Memory m) {
    final cs = Theme.of(context).colorScheme;
    final selected = _selected.contains(m.id);
    final selecting = _selected.isNotEmpty;
    return Card(
      color: selected ? cs.primaryContainer : cs.surfaceContainer,
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: selecting ? () => _toggleSelect(m.id) : null,
        onLongPress: () => _toggleSelect(m.id),
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              selecting
                  ? Padding(
                      padding: const EdgeInsets.only(top: 2, right: 2),
                      child: Icon(
                        selected ? Icons.check_circle : Icons.circle_outlined,
                        color: selected ? cs.primary : cs.onSurfaceVariant,
                      ),
                    )
                  : _avatar(m.speaker),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(m.text,
                        style: const TextStyle(fontSize: 15, height: 1.35)),
                    const SizedBox(height: 6),
                    Row(
                      children: [
                        Icon(Icons.person_outline, size: 13, color: cs.onSurfaceVariant),
                        const SizedBox(width: 4),
                        Text(m.speaker,
                            style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant)),
                        Text('  ·  ${_time(m.timestamp)}',
                            style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant)),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _avatar(String name) {
    final color = _avatarColor(name);
    final initial = name.trim().isEmpty ? '?' : name.trim()[0].toUpperCase();
    return Container(
      width: 38,
      height: 38,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: LinearGradient(
          colors: [color, Color.lerp(color, Colors.black, 0.25)!],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Text(initial,
          style: const TextStyle(
              color: Colors.white, fontWeight: FontWeight.w700, fontSize: 16)),
    );
  }

  // ---------------------------------------------------------- Ask tab

  Widget _buildAsk() {
    final cs = Theme.of(context).colorScheme;
    return Column(
      children: [
        Expanded(
          child: _messages.isEmpty && !_asking
              ? _emptyState(
                  icon: Icons.forum_outlined,
                  title: 'Ask Recall anything',
                  body: 'Ask about your memories by text or voice — I answer '
                      'from what you’ve saved, and read it back aloud.',
                )
              : ListView.builder(
                  controller: _chatScroll,
                  padding: const EdgeInsets.fromLTRB(14, 14, 14, 14),
                  itemCount: _messages.length + (_asking ? 1 : 0),
                  itemBuilder: (context, i) {
                    if (i == _messages.length) return const _TypingBubble();
                    return _buildBubble(_messages[i]);
                  },
                ),
        ),
        SafeArea(
          top: false,
          child: Container(
            decoration: BoxDecoration(
              color: cs.surfaceContainerLow,
              border: Border(top: BorderSide(color: cs.outlineVariant)),
            ),
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.end,
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
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filled(
                  tooltip: 'Send',
                  onPressed: _asking ? null : _askQuestion,
                  icon: const Icon(Icons.arrow_upward),
                ),
                const SizedBox(width: 6),
                IconButton.filled(
                  tooltip: _voiceAsking ? 'Stop & ask' : 'Ask by voice',
                  onPressed: _asking ? null : _toggleVoiceAsk,
                  icon: Icon(_voiceAsking ? Icons.stop : Icons.mic),
                  style: IconButton.styleFrom(
                    backgroundColor: _voiceAsking ? cs.error : cs.secondaryContainer,
                    foregroundColor:
                        _voiceAsking ? cs.onError : cs.onSecondaryContainer,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildBubble(ChatMessage m) {
    final cs = Theme.of(context).colorScheme;
    final maxW = MediaQuery.of(context).size.width * 0.80;
    final ref = m.reference;
    final userRadius = BorderRadius.circular(18).copyWith(bottomRight: const Radius.circular(4));
    final botRadius = BorderRadius.circular(18).copyWith(bottomLeft: const Radius.circular(4));
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Column(
        crossAxisAlignment:
            m.fromUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 11),
            constraints: BoxConstraints(maxWidth: maxW),
            decoration: BoxDecoration(
              gradient: m.fromUser
                  ? LinearGradient(
                      colors: [cs.primary, Color.lerp(cs.primary, AppTheme.accent, 0.45)!],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    )
                  : null,
              color: m.fromUser ? null : cs.surfaceContainerHigh,
              borderRadius: m.fromUser ? userRadius : botRadius,
            ),
            child: SelectableText(
              m.text,
              style: TextStyle(
                color: m.fromUser ? cs.onPrimary : cs.onSurface,
                height: 1.35,
                fontSize: 15,
              ),
            ),
          ),
          if (ref != null && ref.isNotEmpty)
            Container(
              margin: const EdgeInsets.only(top: 6, left: 4),
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
              constraints: BoxConstraints(maxWidth: maxW),
              decoration: BoxDecoration(
                color: cs.surfaceContainerLow,
                border: Border.all(color: cs.outlineVariant),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.bookmark_border, size: 15, color: cs.tertiary),
                  const SizedBox(width: 7),
                  Flexible(
                    child: SelectableText(
                      ref,
                      style: Theme.of(context)
                          .textTheme
                          .bodySmall
                          ?.copyWith(color: cs.onSurfaceVariant),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _emptyState({
    required IconData icon,
    required String title,
    required String body,
  }) {
    final cs = Theme.of(context).colorScheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 84,
              height: 84,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: cs.primaryContainer.withValues(alpha: 0.4),
              ),
              child: Icon(icon, size: 40, color: cs.primary),
            ),
            const SizedBox(height: 20),
            Text(title,
                style: Theme.of(context)
                    .textTheme
                    .titleMedium
                    ?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            Text(body,
                textAlign: TextAlign.center,
                style: TextStyle(color: cs.onSurfaceVariant, height: 1.4)),
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------- onboarding

  Widget _buildSetup() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(28),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Center(child: _orbGlyph(84)),
          const SizedBox(height: 28),
          Text('Set up your voice',
              style: Theme.of(context)
                  .textTheme
                  .headlineSmall
                  ?.copyWith(fontWeight: FontWeight.w700),
              textAlign: TextAlign.center),
          const SizedBox(height: 10),
          Text(
            'Record a few seconds of speech so Recall can recognize you and tag '
            'your memories with your name.',
            textAlign: TextAlign.center,
            style: TextStyle(
                color: Theme.of(context).colorScheme.onSurfaceVariant, height: 1.4),
          ),
          const SizedBox(height: 28),
          TextField(
            controller: _nameController,
            enabled: !_recording,
            textCapitalization: TextCapitalization.words,
            decoration: const InputDecoration(labelText: 'Your name'),
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: _recording ? null : _setupVoice,
            icon: _recording
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2))
                : const Icon(Icons.mic),
            label: Text(_recording ? 'Recording… keep talking' : 'Record my voice (8s)'),
          ),
          const SizedBox(height: 4),
          TextButton(
            onPressed: _recording ? null : () => setState(() => _needsSetup = false),
            child: const Text('Skip for now'),
          ),
        ],
      ),
    );
  }

  /// First-run onboarding for the "Hey Recall" wake word.
  Widget _buildWakeIntro() {
    final cs = Theme.of(context).colorScheme;
    return SingleChildScrollView(
      padding: const EdgeInsets.all(28),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Center(child: _orbGlyph(100)),
          const SizedBox(height: 36),
          Text('Say “Hey Recall”',
              style: Theme.of(context)
                  .textTheme
                  .headlineSmall
                  ?.copyWith(fontWeight: FontWeight.w700),
              textAlign: TextAlign.center),
          const SizedBox(height: 12),
          Text(
            'Recall listens in the background. Whenever you say “Hey Recall”, a '
            'glowing orb rises and Recall saves what you say next to your memories.',
            textAlign: TextAlign.center,
            style: TextStyle(color: cs.onSurfaceVariant, height: 1.45),
          ),
          const SizedBox(height: 32),
          FilledButton.icon(
            onPressed: () => _finishWakeIntro(true),
            icon: const Icon(Icons.hearing),
            label: const Text('Enable “Hey Recall”'),
          ),
          const SizedBox(height: 4),
          TextButton(
            onPressed: () => _finishWakeIntro(false),
            child: const Text('Not now'),
          ),
        ],
      ),
    );
  }

  static Color _avatarColor(String name) {
    const palette = [
      Color(0xFF7C5CFF),
      Color(0xFF22D3EE),
      Color(0xFFFF6B9D),
      Color(0xFF4ADE80),
      Color(0xFFFBBF24),
      Color(0xFFF97316),
    ];
    var h = 7;
    for (final c in name.codeUnits) {
      h = (h * 31 + c) & 0x7fffffff;
    }
    return palette[h % palette.length];
  }

  static String _time(DateTime t) {
    final now = DateTime.now();
    String two(int n) => n.toString().padLeft(2, '0');
    final hm = '${two(t.hour)}:${two(t.minute)}';
    final sameDay = t.year == now.year && t.month == now.month && t.day == now.day;
    if (sameDay) return hm;
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return '${months[t.month - 1]} ${t.day}';
  }
}

/// Animated three-dot "typing" indicator for the assistant's pending answer.
class _TypingBubble extends StatefulWidget {
  const _TypingBubble();

  @override
  State<_TypingBubble> createState() => _TypingBubbleState();
}

class _TypingBubbleState extends State<_TypingBubble>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 1100))
        ..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Align(
      alignment: Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 5),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          color: cs.surfaceContainerHigh,
          borderRadius: BorderRadius.circular(18).copyWith(bottomLeft: const Radius.circular(4)),
        ),
        child: AnimatedBuilder(
          animation: _c,
          builder: (context, _) {
            return Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(3, (i) {
                final phase = (_c.value + i * 0.2) % 1.0;
                final o = 0.3 + 0.7 * (phase < 0.5 ? phase * 2 : (1 - phase) * 2);
                return Padding(
                  padding: EdgeInsets.only(right: i < 2 ? 5 : 0),
                  child: Container(
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: cs.primary.withValues(alpha: o),
                    ),
                  ),
                );
              }),
            );
          },
        ),
      ),
    );
  }
}
