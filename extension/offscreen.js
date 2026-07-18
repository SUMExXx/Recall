// offscreen.js — runs in the offscreen document (has full DOM/canvas APIs).
// Runs OCR (Tesseract.js) and PDF parsing (pdf.js) when background asks.
//
// Both libraries are bundled *inside the extension* at lib/tesseract/ and
// lib/pdfjs/, rather than loaded from a CDN. Extension pages enforce a
// default CSP of `script-src 'self' 'wasm-unsafe-eval'`, which silently
// blocks remote <script src> tags and remote Worker() scripts — so the
// previous CDN-based loading here never actually worked; it just hung
// until the caller's timeout fired. Local, same-origin extension
// resources are covered by 'self', so this works reliably instead.

let tesseractWorker = null;
let workerReady = false;

async function getWorker() {
  if (tesseractWorker && workerReady) return tesseractWorker;

  console.log("[Recall][ocr] loading tesseract.min.js…");
  await loadScript(chrome.runtime.getURL("lib/tesseract/tesseract.min.js"));
  console.log("[Recall][ocr] tesseract.min.js loaded, creating worker…");

  tesseractWorker = await Tesseract.createWorker("eng", 1, {
    // Local worker script — avoids the CDN blobURL+importScripts trick,
    // which would otherwise try (and fail, per the CSP note above) to
    // pull worker.min.js from jsdelivr.
    workerPath: chrome.runtime.getURL("lib/tesseract/worker.min.js"),
    workerBlobURL: false,
    // Directory containing the *.wasm.js core bundles (LSTM-only, with and
    // without SIMD — Tesseract.js auto-picks the right one per device).
    corePath: chrome.runtime.getURL("lib/tesseract/core"),
    // langPath used to be left as the Tesseract.js default, which fetches
    // eng.traineddata (~11MB) from cdn.jsdelivr.net at OCR time. On
    // networks that block/throttle jsdelivr that fetch never resolves, so
    // every OCR call just sat there until our 30s timeout fired ("OCR
    // timeout" — this is the bug that was reported). Bundling the
    // trained-data file locally, same as the worker/core files above,
    // removes that network dependency entirely.
    langPath: chrome.runtime.getURL("lib/tesseract"),
    // The core bundle we ship is LSTM-only (tesseract-core-lstm.wasm.js /
    // -simd-lstm.wasm.js — no legacy engine), so request the matching
    // lstm-only trained-data variant. It's also ~4x smaller than the
    // combined legacy+LSTM data the default would otherwise fetch.
    lstmOnly: true,
    logger: (m) => console.log("[Recall][ocr]", m.status, m.progress),
  });
  console.log("[Recall][ocr] worker created and language loaded");
  workerReady = true;
  return tesseractWorker;
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(s);
  });
}

// ======================================================= PDF PARSING ======
// Runs pdf.js against raw PDF bytes handed to us by background.js. We load
// pdf.js from a copy bundled *inside the extension* (lib/pdfjs/) rather
// than a CDN: extension pages enforce a default CSP of
// `script-src 'self' 'wasm-unsafe-eval'`, which blocks remote <script src>
// tags outright, so a CDN load here would silently fail every time.
// Local extension resources count as 'self', so this works reliably.

let pdfjsReady = null;

async function getPdfjs() {
  if (pdfjsReady) return pdfjsReady;
  pdfjsReady = (async () => {
    await loadScript(chrome.runtime.getURL("lib/pdfjs/pdf.min.js"));
    const pdfjsLib = self.pdfjsLib;
    pdfjsLib.GlobalWorkerOptions.workerSrc = chrome.runtime.getURL("lib/pdfjs/pdf.worker.min.js");
    return pdfjsLib;
  })();
  return pdfjsReady;
}

function base64ToUint8Array(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function parsePdfBytes(base64) {
  const pdfjsLib = await getPdfjs();
  const data = base64ToUint8Array(base64);
  const loadingTask = pdfjsLib.getDocument({ data });
  const pdf = await loadingTask.promise;

  const textParts = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    textParts.push(content.items.map(item => item.str).join(" "));
  }
  return textParts.join("\n\n").trim();
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "RUN_OCR") {
    console.log("[Recall][ocr] RUN_OCR received, id=%s", msg.id);
    (async () => {
      try {
        const worker = await getWorker();
        const { data } = await worker.recognize(msg.dataUrl);
        console.log(
          "[Recall] recognizer output (%d chars):",
          (data.text || "").trim().length,
          (data.text || "").trim() || "(empty)"
        );
        // Send result back — background is listening with a specific id
        chrome.runtime.sendMessage({
          type: "OCR_RESULT",
          id: msg.id,
          text: data.text || "",
        });
      } catch (err) {
        console.error("[Recall][ocr] failed:", err);
        chrome.runtime.sendMessage({
          type: "OCR_RESULT",
          id: msg.id,
          error: String(err.message || err),
          text: "",
        });
      }
    })();
    return true;
  }

  if (msg.type === "PARSE_PDF_BYTES") {
    (async () => {
      try {
        const text = await parsePdfBytes(msg.base64);
        chrome.runtime.sendMessage({
          type: "PARSE_PDF_RESULT",
          id: msg.id,
          text,
        });
      } catch (err) {
        chrome.runtime.sendMessage({
          type: "PARSE_PDF_RESULT",
          id: msg.id,
          error: String(err.message || err),
          text: "",
        });
      }
    })();
    return true;
  }
});
