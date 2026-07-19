// background.js — all network calls + chrome API calls that need elevated
// permissions live here. Content scripts message in; we reply.

importScripts("repoGraph.js"); // → global RecallRepoGraph (graph build + layout)

const DEFAULT_HUB_URL = "http://localhost:8000";

// ---------------------------------------------------------------- helpers --
async function getHubUrl() {
  const { hubUrl } = await chrome.storage.sync.get("hubUrl");
  return (hubUrl || DEFAULT_HUB_URL).replace(/\/+$/, "");
}

async function getGithubToken() {
  const { githubToken } = await chrome.storage.sync.get("githubToken");
  return githubToken || "";
}

async function saveMemory(text) {
  const hubUrl = await getHubUrl();
  const res = await fetch(`${hubUrl}/demo/inject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`Hub returned ${res.status}`);
  return res.json();
}

async function saveImageMemory(imageDataUrl, tab) {
  const hubUrl = await getHubUrl();
  const res = await fetch(`${hubUrl}/ingest/image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image: imageDataUrl,
      title: tab?.title ? `Screenshot — ${tab.title}` : "Screenshot",
      source_url: tab?.url || "",
      source_title: tab?.title || "",
    }),
  });
  if (!res.ok) throw new Error(`Hub returned ${res.status}`);
  return res.json();
}

async function testConnection() {
  const hubUrl = await getHubUrl();
  const res = await fetch(`${hubUrl}/state`);
  if (!res.ok) throw new Error(`Hub returned ${res.status}`);
  return res.json();
}

function notify(title, message) {
  chrome.notifications.create({
    type: "basic",
    iconUrl: "icons/icon128.png",
    title,
    message,
  });
}

function withSource(text, tab) {
  const title = tab?.title || "";
  const url = tab?.url || "";
  return `${text.trim()}\n\n— from "${title}" (${url})`;
}

// Chunk a long string into ≤maxLen pieces, splitting on paragraph/sentence
// boundaries so each chunk is a coherent fact.
function chunkText(text, maxLen = 2000) {
  const chunks = [];
  const paragraphs = text.split(/\n{2,}/).map(p => p.trim()).filter(Boolean);
  let current = "";
  for (const para of paragraphs) {
    if ((current + "\n\n" + para).length > maxLen && current) {
      chunks.push(current.trim());
      current = para;
    } else {
      current = current ? current + "\n\n" + para : para;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

// ================================================== GITHUB REPO INGESTION ==
// Fetches a repo's file tree + contents straight from the GitHub REST API
// (this is where network calls belong — see README notes on CORS/CSP),
// then hands the file list to repoGraph.js to build a graph + LLM context
// text, exactly mirroring DiagramStudio's repo_ingestion.py pipeline but
// running client-side with a lightweight regex-based graph instead of a
// full AST parse.

function parseGithubRepoUrl(url) {
  try {
    const u = new URL(url);
    if (!/(^|\.)github\.com$/.test(u.hostname)) return null;
    const parts = u.pathname.split("/").filter(Boolean);
    const RESERVED = new Set([
      "settings", "notifications", "marketplace", "explore", "topics",
      "trending", "collections", "sponsors", "codespaces", "issues",
      "pulls", "dashboard", "new", "organizations", "about", "apps",
    ]);
    if (parts.length < 2 || RESERVED.has(parts[0])) return null;
    const [owner, repo] = parts;
    // If URL is .../tree/<branch>/... capture the branch hint
    let branchHint = null;
    if (parts[2] === "tree" && parts[3]) branchHint = parts[3];
    return { owner, repo: repo.replace(/\.git$/, ""), branchHint };
  } catch {
    return null;
  }
}

async function ghFetch(url, token) {
  const headers = { Accept: "application/vnd.github+json" };
  if (token) headers.Authorization = `token ${token}`;
  const res = await fetch(url, { headers });
  return res;
}

async function getRepoTree(owner, repo, branch, token) {
  const url = `https://api.github.com/repos/${owner}/${repo}/git/trees/${encodeURIComponent(branch)}?recursive=1`;
  const res = await ghFetch(url, token);
  if (!res.ok) throw new Error(`branch "${branch}" (${res.status})`);
  const data = await res.json();
  return (data.tree || []).filter(
    item => item.type === "blob" &&
      RecallRepoGraph.shouldIncludePath(item.path) &&
      (item.size || 0) <= RecallRepoGraph.MAX_FILE_SIZE
  );
}

async function getFileContent(owner, repo, path, branch, token) {
  const url = `https://api.github.com/repos/${owner}/${repo}/contents/${encodeURIComponent(path).replace(/%2F/g, "/")}?ref=${encodeURIComponent(branch)}`;
  const res = await ghFetch(url, token);
  if (!res.ok) return "";
  const data = await res.json();
  if (data.encoding === "base64" && data.content) {
    try {
      const binary = atob(data.content.replace(/\n/g, ""));
      const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
      return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
    } catch {
      return "";
    }
  }
  return data.content || "";
}

async function mapWithConcurrency(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      try {
        results[i] = await fn(items[i]);
      } catch {
        results[i] = null;
      }
    }
  }
  await Promise.all(new Array(Math.min(limit, items.length)).fill(0).map(worker));
  return results;
}

async function ingestGithubRepo(owner, repo, branchHint) {
  const token = await getGithubToken();
  const candidates = [];
  if (branchHint) candidates.push(branchHint);
  for (const b of ["main", "master", "develop"]) if (b !== branchHint) candidates.push(b);

  let branch = null, blobs = null, lastErr = null;
  for (const candidate of candidates) {
    try {
      blobs = await getRepoTree(owner, repo, candidate, token);
      branch = candidate;
      break;
    } catch (err) {
      lastErr = err;
    }
  }
  if (!branch) {
    throw new Error(
      `Could not read ${owner}/${repo} (tried ${candidates.join(", ")}). ` +
      `${lastErr ? lastErr.message : ""} If it's private, add a GitHub token in Settings.`
    );
  }

  const totalInTree = blobs.length;
  const capped = blobs.slice(0, RecallRepoGraph.MAX_GRAPH_FILES);

  const fetched = await mapWithConcurrency(capped, 8, async (blob) => {
    const content = await getFileContent(owner, repo, blob.path, branch, token);
    if (!content) return null;
    return {
      path: blob.path,
      language: RecallRepoGraph.detectLanguage(blob.path),
      size: blob.size || 0,
      content,
    };
  });

  const files = fetched.filter(Boolean);
  const skipped = capped.length - files.length + (totalInTree - blobs.length);

  const graph = RecallRepoGraph.buildGraphFromFiles(files);
  const meta = {
    repo: `${owner}/${repo}`,
    branch,
    fileCount: files.length,
    totalFiles: totalInTree,
    skipped,
  };
  const contextText = RecallRepoGraph.formatContextText(graph.nodes, graph.edges, meta);

  return { graph, contextText, meta };
}

// ---------------------------------------------------------- context menus --
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "recall-save-selection",
      title: 'Save "%s" to Recall',
      contexts: ["selection"],
    });
    chrome.contextMenus.create({
      id: "recall-screenshot",
      title: "Screenshot → Recall memory",
      contexts: ["page"],
    });
    chrome.contextMenus.create({
      id: "recall-save-github-repo",
      title: "Save this GitHub repo to Recall",
      contexts: ["page"],
      documentUrlPatterns: ["https://github.com/*"],
    });
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "recall-save-selection" && info.selectionText) {
    try {
      await saveMemory(withSource(info.selectionText, tab));
      notify("Saved to Recall", info.selectionText.slice(0, 120));
    } catch (err) {
      notify("Recall save failed", String(err.message || err));
    }
  } else if (info.menuItemId === "recall-screenshot") {
    handleScreenshotOCR(tab);
  } else if (info.menuItemId === "recall-save-github-repo") {
    chrome.tabs.sendMessage(tab.id, { type: "OPEN_GITHUB_INGEST" });
  }
});

// ---------------------------------------------------- screenshot + OCR ----
async function handleScreenshotOCR(tab) {
  try {
    // Capture the visible tab as a base64 PNG
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });

    // Tell the content script to show an area-select overlay, then extract text
    chrome.tabs.sendMessage(tab.id, {
      type: "START_SCREENSHOT_SELECT",
      dataUrl,
    });
  } catch (err) {
    notify("Screenshot failed", String(err.message || err));
  }
}

// Make sure content.js is alive in a tab before we try to message it —
// needed for file:// tabs and tabs the content script hasn't loaded into
// yet (e.g. it was opened before the extension was installed/reloaded).
async function ensureContentScriptInTab(tabId) {
  const alreadyThere = await new Promise((resolve) => {
    chrome.tabs.sendMessage(tabId, { type: "PING" }, (res) => {
      resolve(!chrome.runtime.lastError && !!res);
    });
  });
  if (alreadyThere) return true;
  try {
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
    await new Promise((r) => setTimeout(r, 150)); // let it initialise
    return true;
  } catch {
    return false;
  }
}

// The popup-driven screenshot flow. This is fired-and-forgotten by
// popup.js right before it calls window.close() — everything from here
// on (capture, content-script check, opening the selection dialog, and
// any error reporting) happens in the background so the popup closing
// early can never leave the flow half-started.
async function handleScreenshotFlow(tabId) {
  try {
    const tab = await chrome.tabs.get(tabId);
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
    const ready = await ensureContentScriptInTab(tabId);
    if (!ready) {
      notify(
        "Recall",
        "Couldn't start the screenshot tool on this page — reload the tab and try again."
      );
      return;
    }
    chrome.tabs.sendMessage(tabId, { type: "START_SCREENSHOT_SELECT", dataUrl });
  } catch (err) {
    notify("Screenshot failed", String(err.message || err));
  }
}

// Run Tesseract OCR via an offscreen document (needs DOM APIs)
let offscreenCreated = false;
async function ensureOffscreen() {
  if (offscreenCreated) return;
  const existing = await chrome.offscreen.hasDocument?.().catch(() => false);
  if (existing) { offscreenCreated = true; return; }
  console.log("[Recall] creating offscreen document for OCR/PDF…");
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["BLOBS", "WORKERS"],
    justification: "Run Tesseract OCR (via a Worker) on screenshots and pdf.js on PDF bytes",
  });
  offscreenCreated = true;
}

async function ocrImage(croppedDataUrl) {
  console.log("[Recall] running text extraction on selected region…");
  await ensureOffscreen();
  return new Promise((resolve, reject) => {
    const id = Math.random().toString(36).slice(2);
    function listener(msg) {
      if (msg.type === "OCR_RESULT" && msg.id === id) {
        chrome.runtime.onMessage.removeListener(listener);
        if (msg.error) reject(new Error(msg.error));
        else resolve(msg.text);
      }
    }
    chrome.runtime.onMessage.addListener(listener);
    chrome.runtime.sendMessage({ type: "RUN_OCR", id, dataUrl: croppedDataUrl });
    // First-ever OCR call has to compile the wasm core and decode the
    // bundled trained-data file, so give it more headroom than a warm
    // call would need. If this still times out, open the offscreen
    // document's own DevTools console (chrome://extensions → Recall →
    // "Inspect views: offscreen.html") — that's where [Recall][ocr]
    // step-by-step logs land, not the page console.
    setTimeout(() => {
      chrome.runtime.onMessage.removeListener(listener);
      reject(new Error("OCR timeout"));
    }, 25000);
  });
}

// -------------------------------------------------------- PDF extraction --
// Two-stage strategy:
//   1. Fast path (pdf-content.js, run in the page): works only when a
//      pdf.js-style text layer is already in the page DOM (Firefox's
//      built-in viewer, some web-hosted viewers). Instant, no network.
//   2. Fallback (here): fetch the PDF bytes ourselves — the extension's
//      host_permissions let us fetch cross-origin without hitting the page's
//      CORS restrictions — then hand the bytes to the offscreen document,
//      which parses them with a *locally bundled* copy of pdf.js. This is
//      the path that actually handles Chrome's built-in PDF viewer, since
//      its rendered text isn't reachable from a content script at all.

function arrayBufferToBase64(buf) {
  let binary = "";
  const bytes = new Uint8Array(buf);
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

async function fetchPdfBytesAsBase64(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download PDF (${res.status})`);
  const buf = await res.arrayBuffer();
  return arrayBufferToBase64(buf);
}

async function parsePdfViaOffscreen(base64) {
  await ensureOffscreen();
  return new Promise((resolve, reject) => {
    const id = Math.random().toString(36).slice(2);
    function listener(msg) {
      if (msg.type === "PARSE_PDF_RESULT" && msg.id === id) {
        chrome.runtime.onMessage.removeListener(listener);
        if (msg.error) reject(new Error(msg.error));
        else resolve(msg.text);
      }
    }
    chrome.runtime.onMessage.addListener(listener);
    chrome.runtime.sendMessage({ type: "PARSE_PDF_BYTES", id, base64 });
    setTimeout(() => {
      chrome.runtime.onMessage.removeListener(listener);
      reject(new Error("PDF parse timeout"));
    }, 60000);
  });
}

// Try the DOM text-layer fast path first, then fall back to fetch+parse.
async function extractPdfText(tabId, url) {
  // Stage 1: ask the page itself (works for pdf.js-flavored viewers only).
  try {
    const [{ result } = {}] = await chrome.scripting.executeScript({
      target: { tabId },
      files: ["pdf-content.js"],
    });
    if (result && result.trim().length > 20) return result.trim();
  } catch {
    // Injection can fail (e.g. on chrome:// or restricted pages) — fall through.
  }

  // Stage 2: download the bytes ourselves and parse with bundled pdf.js.
  const base64 = await fetchPdfBytesAsBase64(url);
  const text = await parsePdfViaOffscreen(base64);
  if (!text || text.trim().length < 20) {
    throw new Error("Could not extract text from this PDF (it may be a scanned image — try 'Screenshot → Memory' instead).");
  }
  return text.trim();
}

// ------------------------------------------ ingest whole page in chunks --
async function ingestPageText(pageText, tab, onProgress) {
  const chunks = chunkText(pageText);
  let saved = 0;
  for (const chunk of chunks) {
    const text = withSource(chunk, tab);
    await saveMemory(text);
    saved++;
    if (onProgress) onProgress(saved, chunks.length);
  }
  return { saved, total: chunks.length };
}

// -------------------------------------------------------- message router --
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      switch (msg.type) {

        // --- original: save a memory ---
        case "SAVE_MEMORY": {
          const text = msg.withSource
            ? withSource(msg.text, sender.tab || msg.tab)
            : msg.text.trim();
          const result = await saveMemory(text);
          sendResponse({ ok: true, result });
          break;
        }

        // --- new: send the raw cropped screenshot image directly to the
        // hub's /ingest/image endpoint — no local OCR needed ---
        case "SAVE_SCREENSHOT_IMAGE": {
          const tab = sender.tab || msg.tab;
          try {
            const result = await saveImageMemory(msg.imageDataUrl, tab);
            sendResponse({ ok: result.ok, description: result.description, error: result.error });
          } catch (err) {
            sendResponse({ ok: false, error: String(err.message || err) });
          }
          break;
        }


        // dialog all happen here, decoupled from the popup's lifetime ---
        case "START_SCREENSHOT_FLOW": {
          handleScreenshotFlow(msg.tabId); // don't await — popup already closed
          sendResponse({ ok: true });
          break;
        }

        // --- screenshot capture (legacy path, kept for compatibility) ---
        case "TAKE_SCREENSHOT": {
          // Triggered from popup or content script
          const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
          const dataUrl = await chrome.tabs.captureVisibleTab(activeTab.windowId, { format: "png" });
          sendResponse({ ok: true, dataUrl });
          break;
        }

        case "OCR_AND_SAVE": {
          // Legacy: OCR + immediate save (kept for context-menu path)
          const text = await ocrImage(msg.croppedDataUrl);
          if (!text || !text.trim()) {
            sendResponse({ ok: false, error: "No text found in the selected area" });
            break;
          }
          const tab = sender.tab || msg.tab;
          const memory = withSource(text.trim(), tab);
          await saveMemory(memory);
          sendResponse({ ok: true, text: text.trim() });
          break;
        }

        case "OCR_REGION": {
          // Reads text out of a cropped screenshot region and returns it;
          // the caller (content.js) saves it as a memory immediately —
          // there's no separate review step in the UI.
          const text = await ocrImage(msg.croppedDataUrl);
          console.log(
            "[Recall] text extracted from screenshot region (%d chars):",
            text ? text.trim().length : 0,
            text ? text.trim() : "(empty)"
          );
          if (!text || !text.trim()) {
            sendResponse({ ok: false, error: "No text found in the selected area" });
            break;
          }
          sendResponse({ ok: true, text: text.trim() });
          break;
        }

        // --- PDF text extraction: extract ONLY, no save. The caller shows
        // the text in the review popup so the user can see/edit exactly
        // what's about to be sent before it goes anywhere. ---
        case "EXTRACT_PDF": {
          const tab = sender.tab || msg.tab;
          const text = await extractPdfText(tab.id, tab.url);
          sendResponse({ ok: true, text });
          break;
        }

        // --- PDF text save: called once the user has reviewed (and
        // possibly edited) the extracted text in the popup and confirms. ---
        case "SAVE_PDF_TEXT": {
          const tab = sender.tab || msg.tab;
          const result = await ingestPageText(msg.text, tab);
          sendResponse({ ok: true, ...result });
          break;
        }

        // --- GitHub repo ingest: build graph + LLM context text ---
        case "INGEST_GITHUB_REPO": {
          // msg.owner, msg.repo, msg.branchHint
          const { graph, contextText, meta } = await ingestGithubRepo(
            msg.owner, msg.repo, msg.branchHint
          );
          sendResponse({ ok: true, graph, contextText, meta });
          break;
        }

        // --- send the generated repo context text to Recall memory ---
        case "SAVE_GITHUB_CONTEXT": {
          const tab = sender.tab || msg.tab;
          const result = await ingestPageText(msg.contextText, tab);
          sendResponse({ ok: true, ...result });
          break;
        }

        // --- utility ---
        case "TEST_CONNECTION": {
          const result = await testConnection();
          sendResponse({ ok: true, result });
          break;
        }
        case "GET_HUB_URL": {
          sendResponse({ ok: true, hubUrl: await getHubUrl() });
          break;
        }
        default:
          sendResponse({ ok: false, error: `Unknown message type: ${msg.type}` });
      }
    } catch (err) {
      sendResponse({ ok: false, error: String(err.message || err) });
    }
  })();
  return true;
});