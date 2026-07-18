const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const hubUrlEl = document.getElementById("hub-url");
const manualText = document.getElementById("manual-text");
const saveBtn = document.getElementById("save-btn");
const saveStatus = document.getElementById("save-status");
const featureStatus = document.getElementById("feature-status");

function send(msg) {
  return new Promise((resolve) => chrome.runtime.sendMessage(msg, resolve));
}

function setFeatureStatus(msg, isError = false) {
  featureStatus.textContent = msg;
  featureStatus.style.color = isError ? "#dc2626" : "#16a34a";
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

// ---- connection status ---------------------------------------------------
async function refreshStatus() {
  const { hubUrl } = await send({ type: "GET_HUB_URL" });
  hubUrlEl.textContent = hubUrl || "";
  const res = await send({ type: "TEST_CONNECTION" });
  if (res?.ok) {
    statusDot.className = "dot dot-ok";
    statusText.textContent = "Connected to the Recall hub.";
  } else {
    statusDot.className = "dot dot-error";
    statusText.textContent = "Can't reach the hub — check Settings.";
  }
}

// ---- Screenshot → Memory -------------------------------------------------
// The popup window is short-lived and Chrome can tear it down mid-flight
// (e.g. once captureVisibleTab shifts focus), which used to leave this
// awaiting a response that never rendered — the popup just sat there.
// Fix: grab the tab id, hand the whole flow to background.js, and close
// immediately. Capture, content-script injection, and opening the
// selection dialog all now happen in the background, decoupled from
// whether this popup document is even still alive.
document.getElementById("btn-screenshot").addEventListener("click", async () => {
  const tab = await getActiveTab();
  if (!tab?.id) { setFeatureStatus("Could not find the active tab.", true); return; }
  chrome.runtime.sendMessage({ type: "START_SCREENSHOT_FLOW", tabId: tab.id });
  window.close();
});

// ---- Save this PDF -------------------------------------------------------
// Extracts the PDF text, then hands it to the content script to show in an
// in-page review popup — nothing is sent to the hub until the user reviews
// (and optionally edits) the text there and clicks Save.
document.getElementById("btn-save-pdf").addEventListener("click", async () => {
  const tab = await getActiveTab();
  const url = tab?.url || "";
  const isPdf = url.toLowerCase().endsWith(".pdf") ||
    url.includes("application/pdf") ||
    tab?.title?.toLowerCase().includes(".pdf");

  if (!isPdf) {
    setFeatureStatus("Current tab doesn't look like a PDF.", true);
    return;
  }

  setFeatureStatus("Extracting PDF text…");

  const result = await send({
    type: "EXTRACT_PDF",
    tab: { id: tab.id, title: tab.title, url },
  });

  if (!result?.ok) {
    setFeatureStatus("PDF failed: " + (result?.error || "unknown"), true);
    return;
  }

  // Open the review popup on the page itself, pre-filled with the
  // extracted text, so the user can see exactly what's about to be sent.
  chrome.tabs.sendMessage(tab.id, {
    type: "OPEN_PDF_PREVIEW",
    text: result.text,
    tab: { title: tab.title, url },
  }, () => {
    if (chrome.runtime.lastError) {
      setFeatureStatus("Extracted, but couldn't open the review popup — reload the tab and try again.", true);
    } else {
      setFeatureStatus("Review the extracted text on the page →");
      window.close(); // close the toolbar popup so the review popup is visible
    }
  });
});

// ---- Manual text save ----------------------------------------------------
saveBtn.addEventListener("click", async () => {
  const text = manualText.value.trim();
  if (!text) return;
  saveBtn.disabled = true;
  saveBtn.textContent = "Saving…";
  saveStatus.textContent = "";
  const res = await send({ type: "SAVE_MEMORY", text, withSource: false });
  if (res?.ok) {
    saveStatus.textContent = "Saved ✓";
    saveStatus.style.color = "#16a34a";
    manualText.value = "";
  } else {
    saveStatus.textContent = "Failed — " + (res?.error || "unknown error");
    saveStatus.style.color = "#dc2626";
  }
  saveBtn.disabled = false;
  saveBtn.textContent = "Save memory";
});

// ---- Save GitHub repo ------------------------------------------------------
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
    setFeatureStatus("Open a GitHub repo page first.", true);
    return;
  }

  chrome.tabs.sendMessage(tab.id, { type: "OPEN_GITHUB_INGEST" }, (res) => {
    if (chrome.runtime.lastError) {
      setFeatureStatus("Reload the GitHub tab and try again.", true);
    } else {
      window.close(); // close popup so the ingest dialog is visible
    }
  });
});

document.getElementById("open-options").addEventListener("click", (e) => {
  e.preventDefault();
  chrome.runtime.openOptionsPage();
});

refreshStatus();
