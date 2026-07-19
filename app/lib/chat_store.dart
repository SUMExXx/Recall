import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

/// One turn in the Ask chat: either the user's question or Recall's answer.
class ChatMessage {
  final int id;
  final DateTime timestamp;
  final bool fromUser;
  final String text;

  /// Memory citation for an answer (shown in a box under the bubble), or null.
  final String? reference;

  const ChatMessage({
    this.id = 0,
    required this.timestamp,
    required this.fromUser,
    required this.text,
    this.reference,
  });
}

/// Persists the Ask chat history in a local sqflite table so the conversation
/// survives app restarts. Separate DB file from the memory store.
class ChatStore {
  final Database _db;

  ChatStore._(this._db);

  static Future<ChatStore> create() async {
    final db = await openDatabase(
      p.join(await getDatabasesPath(), 'chat.db'),
      version: 2,
      onCreate: (db, _) => db.execute(
        'CREATE TABLE messages ('
        'id INTEGER PRIMARY KEY AUTOINCREMENT, '
        'timestamp INTEGER NOT NULL, '
        'from_user INTEGER NOT NULL, '
        'text TEXT NOT NULL, '
        'reference TEXT)',
      ),
      onUpgrade: (db, from, to) async {
        if (from < 2) {
          await db.execute('ALTER TABLE messages ADD COLUMN reference TEXT');
        }
      },
    );
    return ChatStore._(db);
  }

  /// Appends a message and returns it with its assigned row id.
  Future<ChatMessage> add(ChatMessage m) async {
    final id = await _db.insert('messages', {
      'timestamp': m.timestamp.millisecondsSinceEpoch,
      'from_user': m.fromUser ? 1 : 0,
      'text': m.text,
      'reference': m.reference,
    });
    return ChatMessage(
      id: id,
      timestamp: m.timestamp,
      fromUser: m.fromUser,
      text: m.text,
      reference: m.reference,
    );
  }

  /// Full history, oldest first (chat order).
  Future<List<ChatMessage>> history() async {
    final rows = await _db.query('messages', orderBy: 'id ASC');
    return rows.map(_toMessage).toList();
  }

  /// Removes every message.
  Future<void> clear() => _db.delete('messages');

  static ChatMessage _toMessage(Map<String, Object?> r) => ChatMessage(
        id: r['id'] as int,
        timestamp: DateTime.fromMillisecondsSinceEpoch(r['timestamp'] as int),
        fromUser: (r['from_user'] as int) == 1,
        text: r['text'] as String,
        reference: r['reference'] as String?,
      );

  Future<void> dispose() => _db.close();
}
