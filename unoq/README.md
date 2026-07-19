# Recall on the Arduino UNO Q

This little board is Recall's "ears in the room." It sits there quietly, listens for
you to say **"Hey Arduino!"**, and when it hears it, it starts recording what you
say next and sends it off to be remembered.

Think of it as a tiny, always-listening assistant that only wakes up when you
call it — and shows you what it's doing with a little heart on its LED display.

## What it actually does

1. **Listens quietly** in the background through a microphone.
2. **Wakes up on "Hey Arduino!"** — a little heart animation plays on the LED
   matrix so you know it heard you.
3. **Records for 10 seconds** and streams that audio to the Recall hub (the
   brain running on your PC), which turns it into a memory.
4. **Goes back to sleep** automatically after those 10 seconds — no buttons,
   no timers to set.
5. **Shows you what's going on** the whole time using simple patterns on the
   LED matrix:

   | What you see | What it means |
   |---|---|
   | 🫀 Heart | Idle — just listening for the wake word |
   | ▮ Solid block | Recording your memory right now |
   | ▦ Checkerboard | The hub is searching your memories for an answer |
   | ✓ Checkmark | Found something and recalled it |
   | ✕ Cross | Muted |
   | ⚠ Full blink | Something's wrong (lost connection, etc.) |

## How it connects to the PC

The board itself doesn't do any "thinking" — it just captures your voice and
ships it off. All the real work (turning speech into text, understanding it,
remembering it, answering questions about it) happens on the **PC hub**, which
runs separately on your computer.

The board and the PC talk to each other over Wi-Fi. The board says "someone
just said the wake word, here's what they said next," and the hub replies with
things like "still searching…" or "found it" — which is what makes the LED
show the checkerboard or checkmark. So the two halves — board and PC — are
always in a conversation: the board reports what it hears, the hub reports
what it's doing about it.

## How we actually run it

Getting audio into the board and the app running takes a few steps:

1. **Connect a Microphone based device over Bluetooth.** Instead of a wired
   mic, we pair Bluetooth ear budsnand use it as the audio
   source — its mic picks up the room and streams that audio over to the
   board.

2. With that pipeline running, launch the Recall app
   from App Lab as usual. It reads from that same loopback mic, listens for
   the wake word, and streams to the hub whenever you say "Hey Arduino!"

## A note on the LED logic

The idle/recording state is decided by *this board* — it knows exactly when it
started and stopped listening, so that's always accurate. The searching/
recalled/error states are decided by the *hub*, since only the hub knows when
someone's asking a question or something's gone wrong. Each side sticks to
what it actually knows, so the display never says "recording" when it isn't
and never gets stuck.
