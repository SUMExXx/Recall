// pdf-content.js — injected via chrome.scripting.executeScript({files:[...]})
// into the tab that is showing a PDF.
//
// This only attempts the *fast path*: if the browser's own PDF viewer has
// already rendered a text layer into the page DOM (this happens in
// Firefox's built-in pdf.js viewer, and in some in-page pdf.js-based
// viewers), grab it directly — no network calls, no CDNs, no CSP issues.
//
// If nothing is found here, background.js falls back to downloading the
// PDF bytes itself and parsing them with a bundled copy of pdf.js running
// in the offscreen document. That fallback works for Chrome's built-in
// PDFium viewer too, since that viewer's rendered text isn't reachable
// from a content script at all (it lives in a separate guest view).
//
// The completion value of this script (the resolved value of the async
// IIFE below) becomes `results[0].result` in the caller's
// chrome.scripting.executeScript(...) call.

(async () => {
  function getViewerText() {
    // pdf.js-based viewers (Firefox's built-in viewer, some web-hosted
    // viewers) render each glyph run as a <span> inside a ".textLayer".
    // Chrome's built-in PDFium viewer does NOT do this — its content lives
    // in a separate guest view that content scripts can't reach — so this
    // will correctly come back empty there, and the caller falls back to
    // background-fetched + offscreen-parsed extraction.
    const spans = document.querySelectorAll(".textLayer span, #viewer .page span");
    if (spans.length > 0) {
      const text = Array.from(spans).map(s => s.textContent).join(" ").replace(/\s+/g, " ").trim();
      if (text.length > 20) return text;
    }
    return "";
  }

  return getViewerText() || null;
})();
