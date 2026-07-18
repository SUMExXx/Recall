import 'package:flutter/material.dart';

/// A glowing ball of light that rises from the bottom of the screen while
/// Recall is actively listening (after the "Hey Recall" wake word) and sinks
/// back down when it stops. Purely visual feedback — ignores pointer events.
class ListeningOrb extends StatefulWidget {
  /// Whether the orb should be visible and pulsing.
  final bool visible;

  const ListeningOrb({super.key, required this.visible});

  @override
  State<ListeningOrb> createState() => _ListeningOrbState();
}

class _ListeningOrbState extends State<ListeningOrb>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulse = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1400),
  )..repeat(reverse: true);

  @override
  void dispose() {
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme.primary;
    return IgnorePointer(
      child: AnimatedSlide(
        duration: const Duration(milliseconds: 350),
        curve: Curves.easeOutBack,
        offset: widget.visible ? Offset.zero : const Offset(0, 1.8),
        child: AnimatedOpacity(
          duration: const Duration(milliseconds: 250),
          opacity: widget.visible ? 1 : 0,
          child: Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 48),
              child: AnimatedBuilder(
                animation: _pulse,
                builder: (context, _) {
                  final t = _pulse.value; // 0..1, ping-pongs
                  final scale = 0.85 + t * 0.28;
                  final glow = 26 + t * 34;
                  return Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 92 * scale,
                        height: 92 * scale,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: RadialGradient(
                            colors: [
                              Colors.white,
                              color,
                              color.withOpacity(0.55),
                            ],
                            stops: const [0.0, 0.55, 1.0],
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: color.withOpacity(0.6),
                              blurRadius: glow,
                              spreadRadius: glow / 3,
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 14),
                      Text(
                        'Listening…',
                        style: TextStyle(
                          color: color,
                          fontWeight: FontWeight.w600,
                          fontSize: 16,
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
    );
  }
}
