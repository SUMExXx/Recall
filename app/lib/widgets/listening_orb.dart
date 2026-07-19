import 'package:flutter/material.dart';

import '../theme.dart';

/// A glowing ball of light that rises from the bottom of the screen while
/// Recall is actively listening (after the "Hey Recall" wake word) and sinks
/// back down when it stops. Purely visual feedback — ignores pointer events.
class ListeningOrb extends StatefulWidget {
  /// Whether the orb should be visible and pulsing.
  final bool visible;

  /// Caption under the orb.
  final String label;

  const ListeningOrb({super.key, required this.visible, this.label = 'Listening…'});

  @override
  State<ListeningOrb> createState() => _ListeningOrbState();
}

class _ListeningOrbState extends State<ListeningOrb>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1600),
  )..repeat(reverse: true);

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return IgnorePointer(
      child: AnimatedSlide(
        duration: const Duration(milliseconds: 400),
        curve: Curves.easeOutBack,
        offset: widget.visible ? Offset.zero : const Offset(0, 1.8),
        child: AnimatedOpacity(
          duration: const Duration(milliseconds: 250),
          opacity: widget.visible ? 1 : 0,
          child: Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 56),
              child: Semantics(
                liveRegion: true,
                label: widget.visible ? widget.label : null,
                child: AnimatedBuilder(
                  animation: _pulse,
                  builder: (context, _) {
                    final t = _pulse.value; // 0..1, ping-pongs
                    final scale = 0.9 + t * 0.22;
                    return Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        SizedBox(
                          width: 200,
                          height: 200,
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              // Outer halo — soft, wide, breathes with the pulse.
                              Container(
                                width: 200 * scale,
                                height: 200 * scale,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  gradient: RadialGradient(
                                    colors: [
                                      AppTheme.accent.withValues(alpha: 0.18 + t * 0.12),
                                      cs.primary.withValues(alpha: 0.10),
                                      Colors.transparent,
                                    ],
                                    stops: const [0.0, 0.55, 1.0],
                                  ),
                                ),
                              ),
                              // Core orb — white-hot center into indigo/cyan.
                              Container(
                                width: 96 * scale,
                                height: 96 * scale,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  gradient: RadialGradient(
                                    colors: [
                                      Colors.white,
                                      cs.primary,
                                      AppTheme.accent.withValues(alpha: 0.85),
                                    ],
                                    stops: const [0.0, 0.5, 1.0],
                                  ),
                                  boxShadow: [
                                    BoxShadow(
                                      color: cs.primary.withValues(alpha: 0.55),
                                      blurRadius: 34 + t * 30,
                                      spreadRadius: 6 + t * 8,
                                    ),
                                    BoxShadow(
                                      color: AppTheme.accent.withValues(alpha: 0.35),
                                      blurRadius: 50 + t * 30,
                                      spreadRadius: 2,
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
                          decoration: BoxDecoration(
                            color: cs.surfaceContainerHighest.withValues(alpha: 0.9),
                            borderRadius: BorderRadius.circular(30),
                            border: Border.all(color: cs.primary.withValues(alpha: 0.4)),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(Icons.graphic_eq, size: 16, color: cs.primary),
                              const SizedBox(width: 6),
                              Text(
                                widget.label,
                                style: TextStyle(
                                  color: cs.onSurface,
                                  fontWeight: FontWeight.w600,
                                  fontSize: 14,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
