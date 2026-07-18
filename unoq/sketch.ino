// SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
//
// SPDX-License-Identifier: MPL-2.0
//
// Recall — Arduino UNO Q, MCU (real-time) side.
// Implements the "MCU Side" of Architecture doc section 2:
//   * LED state machine (capturing / searching / recalled / muted / error)
//   * MCU watchdog: error blink if the Linux side goes silent
// (Push-button bookmark/forget and the hardware mute switch are intentionally
//  not implemented; the "muted" LED state is still driven by the hub.)
//
// The 12x8 LED matrix is monochrome, so the spec's colours are mapped to
// distinct patterns:  amber(capturing)=solid block, blue(searching)=checker,
// green(recalled)=tick, red(muted)=X, red(error)=full blink.
// The original "Hey Recall" heart flourish (keyword_detected -> wake_up) is
// preserved unchanged.

#include <Arduino.h>
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

#include "heart_frames.h"

Arduino_LED_Matrix matrix;

// ---- LED states (MUST stay in sync with main.py) --------------------------
enum LedState {
  LED_IDLE      = 0,   // heart
  LED_CAPTURING = 1,   // amber -> solid block
  LED_SEARCHING = 2,   // blue  -> checkerboard
  LED_RECALLED  = 3,   // green -> tick
  LED_MUTED     = 4,   // red   -> X
  LED_ERROR     = 5,   // red   -> full blink
};

// ---- State ----------------------------------------------------------------
volatile int  gLedState  = LED_IDLE;
volatile bool gDirty     = true;    // LED needs a re-render
volatile bool gAnimating = false;   // heart flourish is playing; don't repaint

// Watchdog: blink error if the Linux side stops talking to us
const unsigned long BRIDGE_TIMEOUT_MS = 8000;
unsigned long gLastBridgeMs = 0;
bool gEverHeard = false;

// ---------------------------------------------------------------------------
// Frame helpers
// ---------------------------------------------------------------------------
// Pack an 8x12 (row-major) on/off grid into the 3x uint32 the matrix expects
// (pixel 0,0 = MSB of word 0).
void packFrame(const uint8_t px[8][12], uint32_t out[3]) {
  out[0] = out[1] = out[2] = 0;
  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 12; c++) {
      int idx = r * 12 + c;                 // 0..95
      if (px[r][c]) out[idx / 32] |= (uint32_t)1 << (31 - (idx % 32));
    }
  }
}

void renderState(int s) {
  if (s == LED_IDLE) {           // keep the heart as the resting face
    matrix.loadFrame(HeartStatic);
    return;
  }

  uint8_t px[8][12];
  memset(px, 0, sizeof(px));

  switch (s) {
    case LED_CAPTURING:                              // solid block ("recording")
      for (int r = 2; r < 6; r++)
        for (int c = 4; c < 8; c++) px[r][c] = 1;
      break;
    case LED_SEARCHING:                              // checkerboard ("scanning")
      for (int r = 0; r < 8; r++)
        for (int c = 0; c < 12; c++) px[r][c] = (r + c) & 1;
      break;
    case LED_RECALLED: {                             // tick ("recalled")
      const int pts[7][2] = {{5,2},{6,3},{5,4},{4,5},{3,6},{2,7},{1,8}};
      for (int i = 0; i < 7; i++) px[pts[i][0]][pts[i][1]] = 1;
      break;
    }
    case LED_MUTED:                                  // X ("muted / privacy")
    case LED_ERROR: {                                // (error uses full grid below)
      if (s == LED_MUTED) {
        for (int i = 0; i < 8; i++) { px[i][i + 2] = 1; px[i][9 - i] = 1; }
      } else {
        for (int r = 0; r < 8; r++)
          for (int c = 0; c < 12; c++) px[r][c] = 1;
      }
      break;
    }
  }

  uint32_t frame[3];
  packFrame(px, frame);
  matrix.loadFrame(frame);
}

// ---------------------------------------------------------------------------
// RPC callbacks (called by the Linux/Python side over the bridge)
// ---------------------------------------------------------------------------
void wake_up() {                        // preserved: "Hey Recall" heart flourish
  gLastBridgeMs = millis();
  gEverHeard = true;
  gAnimating = true;
  matrix.loadSequence(HeartAnim);
  matrix.playSequence();
  delay(1000);
  matrix.loadFrame(HeartStatic);
  gAnimating = false;
  gDirty = true;                        // let the state machine reassert (e.g. CAPTURING)
}

void set_led_state(int state) {         // hub-driven LED state machine
  gLastBridgeMs = millis();
  gEverHeard = true;
  gLedState = state;
  gDirty = true;
}

void mcu_ping() {                       // heartbeat: Linux side is alive
  gLastBridgeMs = millis();
  gEverHeard = true;
}

// ---------------------------------------------------------------------------
// Setup / loop
// ---------------------------------------------------------------------------
void setup() {
  matrix.begin();
  matrix.clear();
  matrix.loadFrame(HeartStatic);

  Bridge.begin();
  Bridge.provide("keyword_detected", wake_up);          // existing
  Bridge.provide_safe("set_led_state", set_led_state);  // touches the matrix
  Bridge.provide_safe("mcu_ping", mcu_ping);

  gLastBridgeMs = millis();
}

void loop() {
  unsigned long now = millis();

  // ---- MCU watchdog: blink error if the Linux side went silent ----
  if (gEverHeard && (now - gLastBridgeMs > BRIDGE_TIMEOUT_MS)) {
    static unsigned long lastBlink = 0;
    static bool on = false;
    if (now - lastBlink > 400) {
      lastBlink = now;
      on = !on;
      if (on) renderState(LED_ERROR);
      else matrix.clear();
    }
    delay(20);
    return;                                   // hold error until Linux returns
  }

  // ---- Render current LED state on change ----
  if (gDirty && !gAnimating) {
    gDirty = false;
    renderState(gLedState);
  }

  delay(20);
}
