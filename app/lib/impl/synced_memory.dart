import 'dart:async';
import 'dart:convert';

import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../pipeline/vector_store.dart';

/// A [VectorStore] that stores memories locally (delegating to [_local]) and
/// mirrors them to a PC hub over WebSocket.
///
/// Every saved memory is queued in a persistent **outbox**; a memory leaves the
/// outbox only when the PC acknowledges it. So the outbox is exactly "the
/// memories not yet sent" — it survives restarts and is flushed whenever the
/// socket (re)connects. Reads (recent/search) are served locally, so the app
/// works fully offline; the PC just receives a copy.
class SyncedMemory implements VectorStore {
  final VectorStore _local;
  final Database _db;

  final _status = StreamController<String>.broadcast();
  WebSocketChannel? _channel;
  StreamSubscription? _sub;
  Timer? _reconnect;
  String? _serverUrl;
  int _backoffMs = 1000;
  bool _disposed = false;

  SyncedMemory._(this._local, this._db);

  static Future<SyncedMemory> create({required VectorStore local}) async {
    final db = await openDatabase(
      p.join(await getDatabasesPath(), 'sync.db'),
      version: 1,
      onCreate: (db, _) async {
        await db.execute(
          'CREATE TABLE outbox (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)',
        );
        await db.execute(
          'CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)',
        );
      },
    );
    final sm = SyncedMemory._(local, db);
    sm._serverUrl = await sm._loadUrl();
    if (sm._serverUrl != null && sm._serverUrl!.isNotEmpty) sm._connect();
    return sm;
  }

  /// Live sync status for the UI ("connected", "disconnected", "3 pending", …).
  Stream<String> get status => _status.stream;

  String? get serverUrl => _serverUrl;

  /// Sets (and persists) the PC hub URL, e.g. `ws://192.168.1.20:8765`, and
  /// (re)connects. Pass an empty string to disable sync.
  Future<void> setServerUrl(String url) async {
    _serverUrl = url.trim();
    await _db.insert('settings', {'key': 'url', 'value': _serverUrl},
        conflictAlgorithm: ConflictAlgorithm.replace);
    await _closeSocket();
    if (_serverUrl!.isNotEmpty) {
      _backoffMs = 1000;
      _connect();
    } else {
      _emit('disabled');
    }
  }

  @override
  Future<int> add(Memory memory) async {
    final id = await _local.add(memory);
    // Outbox payload: metadata + transcript (the PC re-embeds on its side).
    final payload = jsonEncode({
      'id': id,
      'timestamp': memory.timestamp.toUtc().toIso8601String(),
      'speaker': memory.speaker,
      'text': memory.text,
    });
    await _db.insert('outbox', {'id': id, 'payload': payload},
        conflictAlgorithm: ConflictAlgorithm.replace);
    _flush();
    return id;
  }

  @override
  Future<List<Memory>> recent(int limit) => _local.recent(limit);

  @override
  Future<List<Memory>> search(String query, {int topK = 20}) =>
      _local.search(query, topK: topK);

  // --- sync internals ---

  void _connect() {
    if (_disposed || _serverUrl == null || _serverUrl!.isEmpty) return;
    _emit('connecting');
    try {
      final channel = WebSocketChannel.connect(Uri.parse(_serverUrl!));
      _channel = channel;
      _sub = channel.stream.listen(
        _onMessage,
        onError: (_) => _scheduleReconnect(),
        onDone: _scheduleReconnect,
        cancelOnError: true,
      );
      channel.ready.then((_) {
        _backoffMs = 1000;
        _emit('connected');
        _flush();
      }).catchError((_) {
        _scheduleReconnect();
      });
    } catch (_) {
      _scheduleReconnect();
    }
  }

  /// Sends every outbox entry (the not-yet-sent memories) to the PC.
  Future<void> _flush() async {
    final channel = _channel;
    if (channel == null) return;
    final rows = await _db.query('outbox', orderBy: 'id ASC');
    if (rows.isEmpty) {
      _emit('synced');
      return;
    }
    try {
      for (final row in rows) {
        channel.sink.add(row['payload'] as String);
      }
      _emit('${rows.length} pending');
    } catch (_) {
      _scheduleReconnect();
    }
  }

  void _onMessage(dynamic data) {
    // The PC acknowledges each stored memory: {"type":"ack","id":<id>}.
    try {
      final msg = jsonDecode(data as String);
      if (msg is Map && msg['type'] == 'ack' && msg['id'] is int) {
        _db.delete('outbox', where: 'id = ?', whereArgs: [msg['id']]).then((_) async {
          final pending = Sqflite.firstIntValue(
                  await _db.rawQuery('SELECT COUNT(*) FROM outbox')) ??
              0;
          _emit(pending == 0 ? 'synced' : '$pending pending');
        });
      }
    } catch (_) {
      // ignore malformed messages
    }
  }

  void _scheduleReconnect() {
    _closeSocket();
    if (_disposed || _serverUrl == null || _serverUrl!.isEmpty) return;
    _emit('disconnected');
    _reconnect?.cancel();
    _reconnect = Timer(Duration(milliseconds: _backoffMs), _connect);
    _backoffMs = (_backoffMs * 2).clamp(1000, 30000); // exp backoff, cap 30s
  }

  Future<void> _closeSocket() async {
    _reconnect?.cancel();
    await _sub?.cancel();
    _sub = null;
    await _channel?.sink.close();
    _channel = null;
  }

  Future<String?> _loadUrl() async {
    final rows = await _db.query('settings', where: 'key = ?', whereArgs: ['url']);
    return rows.isEmpty ? null : rows.first['value'] as String?;
  }

  void _emit(String s) {
    if (!_disposed) _status.add(s);
  }

  @override
  Future<void> dispose() async {
    _disposed = true;
    await _closeSocket();
    await _status.close();
    await _local.dispose();
    await _db.close();
  }
}
