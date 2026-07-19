import 'package:flutter/material.dart';

/// Recall's visual identity: a dark-first "AI assistant" look — electric indigo
/// with a cyan accent, deep near-black surfaces, soft glows, generous rounding.
/// Ships full light + dark schemes so it follows the system setting.
class AppTheme {
  static const Color seed = Color(0xFF7C5CFF); // electric indigo
  static const Color accent = Color(0xFF22D3EE); // cyan glow

  static ThemeData light() => _build(Brightness.light);
  static ThemeData dark() => _build(Brightness.dark);

  static ThemeData _build(Brightness brightness) {
    final isDark = brightness == Brightness.dark;
    var cs = ColorScheme.fromSeed(
      seedColor: seed,
      brightness: brightness,
      tertiary: accent,
    );

    if (isDark) {
      // Deep charcoal surface ladder + a brighter primary so the accent pops
      // against near-black. This is what gives the "AI" glow feel.
      cs = cs.copyWith(
        primary: const Color(0xFF9C8CFF),
        onPrimary: const Color(0xFF1B0F3D),
        primaryContainer: const Color(0xFF2C2358),
        onPrimaryContainer: const Color(0xFFE7DEFF),
        tertiary: accent,
        onTertiary: const Color(0xFF00272E),
        surface: const Color(0xFF0D0D14),
        onSurface: const Color(0xFFEAEAF4),
        onSurfaceVariant: const Color(0xFFA7A7BE),
        surfaceContainerLowest: const Color(0xFF08080D),
        surfaceContainerLow: const Color(0xFF12121C),
        surfaceContainer: const Color(0xFF181824),
        surfaceContainerHigh: const Color(0xFF20202E),
        surfaceContainerHighest: const Color(0xFF29293A),
        outline: const Color(0xFF56566D),
        outlineVariant: const Color(0xFF2A2A3A),
      );
    }

    final radius = BorderRadius.circular(18);

    return ThemeData(
      useMaterial3: true,
      colorScheme: cs,
      scaffoldBackgroundColor: cs.surface,
      splashFactory: InkSparkle.splashFactory,
      appBarTheme: AppBarTheme(
        backgroundColor: cs.surface,
        surfaceTintColor: Colors.transparent,
        scrolledUnderElevation: 0,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: TextStyle(
          color: cs.onSurface,
          fontSize: 22,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.2,
        ),
      ),
      cardTheme: CardThemeData(
        color: cs.surfaceContainer,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      ),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: cs.surfaceContainerLow,
        indicatorColor: cs.primaryContainer,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        height: 68,
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        iconTheme: WidgetStateProperty.resolveWith((states) {
          final selected = states.contains(WidgetState.selected);
          return IconThemeData(
            color: selected ? cs.onPrimaryContainer : cs.onSurfaceVariant,
          );
        }),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: cs.surfaceContainerHigh,
        isDense: true,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        hintStyle: TextStyle(color: cs.onSurfaceVariant),
        border: OutlineInputBorder(borderRadius: radius, borderSide: BorderSide.none),
        enabledBorder:
            OutlineInputBorder(borderRadius: radius, borderSide: BorderSide.none),
        focusedBorder: OutlineInputBorder(
          borderRadius: radius,
          borderSide: BorderSide(color: cs.primary, width: 1.6),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          textStyle: const TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
        ),
      ),
      snackBarTheme: SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
        backgroundColor: cs.surfaceContainerHighest,
        contentTextStyle: TextStyle(color: cs.onSurface),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      ),
      dialogTheme: DialogThemeData(
        backgroundColor: cs.surfaceContainerHigh,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      ),
      dividerTheme: DividerThemeData(color: cs.outlineVariant, thickness: 1, space: 1),
      listTileTheme: ListTileThemeData(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      ),
      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: cs.primary,
        foregroundColor: cs.onPrimary,
      ),
    );
  }
}
