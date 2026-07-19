/// Splits an LLM answer into the answer proper and its memory reference.
///
/// Answers come back like "Green. From memory excerpt [b104…]" or
/// "…in the garage. Source: [id] (note: …)". We speak/show only the answer and
/// box the reference separately. Returns `(answer, null)` when there is no
/// recognizable reference.
(String, String?) splitAnswer(String full) {
  final text = full.trim();
  // A reference starts at a citation lead-in ("Source:", "From memory…",
  // "Based on memory…") or a bracketed source id like [b104…] / [id:1].
  final marker = RegExp(
    r'(\bsources?\s*:'
    r'|\bfrom\s+(?:the\s+)?memor(?:y|ies)'
    r'|\bbased on\s+(?:your\s+|the\s+)?memor'
    r'|\[[0-9a-fA-F]{3,}[^\]]*\])',
    caseSensitive: false,
  );
  final m = marker.firstMatch(text);
  if (m == null || m.start == 0) return (text, null);

  final answer =
      text.substring(0, m.start).replaceAll(RegExp(r'[\s—–\-,;]+$'), '').trim();
  final reference = text.substring(m.start).trim();
  if (answer.isEmpty) return (text, null);
  return (answer, reference.isEmpty ? null : reference);
}
