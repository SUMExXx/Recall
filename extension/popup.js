const statusDot  = document.getElementById("status-dot");
const ledLabel   = document.getElementById("led-label");
const statusText = document.getElementById("status-text");
const hubUrlEl   = document.getElementById("hub-url");
const manualText = document.getElementById("manual-text");
const saveBtn    = document.getElementById("save-btn");
const saveStatus = document.getElementById("save-status");
const featureStatus = document.getElementById("feature-status");

function send(msg) {
  return new Promise((resolve) => chrome.runtime.sendMessage(msg, resolve));
}

function setLed(state) {
  // states: checking | ok | error
  statusDot.className = "led " + state;
  ledLabel.textContent = state === "ok" ? "live" : state === "error" ? "offline" : "…";
}

function setConnectionStatus(ok, text) {
  setLed(ok ? "ok" : "error");
  statusText.textContent = text;
  statusText.className = "status-bar " + (ok ? "ok" : "err");
}

function setFeatureStatus(msg, type = "") {
  featureStatus.textContent = msg;
  featureStatus.className = "feature-status " + type;
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

// ── connection status ─────────────────────────────────────────────────────────
async function refreshStatus() {
  setLed("checking");
  statusText.textContent = "checking hub…";
  statusText.className = "status-bar";

  const { hubUrl } = await send({ type: "GET_HUB_URL" });
  hubUrlEl.textContent = hubUrl || "";

  const res = await send({ type: "TEST_CONNECTION" });
  if (res?.ok) {
    setConnectionStatus(true, "connected · " + (hubUrl || "localhost:8000"));
  } else {
    setConnectionStatus(false, "hub unreachable — check settings");
  }
}

// ── Screenshot → Memory ───────────────────────────────────────────────────────
document.getElementById("btn-screenshot").addEventListener("click", async () => {
  const tab = await getActiveTab();
  if (!tab?.id) { setFeatureStatus("couldn't find the active tab", "err"); return; }
  chrome.runtime.sendMessage({ type: "START_SCREENSHOT_FLOW", tabId: tab.id });
  window.close();
});

// ── Save PDF ──────────────────────────────────────────────────────────────────
document.getElementById("btn-save-pdf").addEventListener("click", async () => {
  const tab = await getActiveTab();
  const url = tab?.url || "";
  const isPdf = url.toLowerCase().endsWith(".pdf") ||
    url.includes("application/pdf") ||
    tab?.title?.toLowerCase().includes(".pdf");

  if (!isPdf) {
    setFeatureStatus("current tab doesn't look like a PDF", "err");
    return;
  }

  setFeatureStatus("extracting PDF text…");

  const result = await send({
    type: "EXTRACT_PDF",
    tab: { id: tab.id, title: tab.title, url },
  });

  if (!result?.ok) {
    setFeatureStatus("PDF failed: " + (result?.error || "unknown"), "err");
    return;
  }

  chrome.tabs.sendMessage(tab.id, {
    type: "OPEN_PDF_PREVIEW",
    text: result.text,
    tab: { title: tab.title, url },
  }, () => {
    if (chrome.runtime.lastError) {
      setFeatureStatus("extracted, but review popup failed — reload tab", "err");
    } else {
      setFeatureStatus("review the extracted text on the page →", "ok");
      window.close();
    }
  });
});

// ── GitHub repo ───────────────────────────────────────────────────────────────
document.getElementById("btn-save-github").addEventListener("click", async () => {
  const tab = await getActiveTab();
  const url = tab?.url || "";
  let isRepoPage = false;
  try {
    const u = new URL(url);
    const parts = u.pathname.split("/").filter(Boolean);
    isRepoPage = u.hostname === "github.com" && parts.length >= 2;
  } catch { /* not a valid URL */ }

  if (!isRepoPage) {
    setFeatureStatus("open a GitHub repo page first", "err");
    return;
  }

  chrome.tabs.sendMessage(tab.id, { type: "OPEN_GITHUB_INGEST" }, (res) => {
    if (chrome.runtime.lastError) {
      setFeatureStatus("reload the GitHub tab and try again", "err");
    } else {
      window.close();
    }
  });
});

// ── manual text save ──────────────────────────────────────────────────────────
saveBtn.addEventListener("click", async () => {
  const text = manualText.value.trim();
  if (!text) return;
  saveBtn.disabled = true;
  saveBtn.textContent = "saving…";
  saveStatus.textContent = "";
  saveStatus.className = "save-status";

  const res = await send({ type: "SAVE_MEMORY", text, withSource: false });
  if (res?.ok) {
    saveStatus.textContent = "saved to memory ✓";
    saveStatus.className = "save-status ok";
    manualText.value = "";
  } else {
    saveStatus.textContent = "failed — " + (res?.error || "unknown error");
    saveStatus.className = "save-status err";
  }
  saveBtn.disabled = false;
  saveBtn.textContent = "→ save to memory";
});

document.getElementById("open-options").addEventListener("click", (e) => {
  e.preventDefault();
  chrome.runtime.openOptionsPage();
});

refreshStatus();