import 'dart:typed_data';

/// Float32List <-> BLOB helpers for sqflite storage.

Uint8List floatsToBytes(Float32List f) =>
    f.buffer.asUint8List(f.offsetInBytes, f.lengthInBytes);

/// Copies so the result owns 4-byte-aligned storage regardless of the blob's
/// original byte offset (sqflite may hand back a view into a larger buffer).
Float32List bytesToFloats(Uint8List b) => Uint8List.fromList(b).buffer.asFloat32List();
