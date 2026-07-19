// content.js — runs on every page.
// Features:
//   1. Text selection → "Save to Recall" pill → editable save panel (original)
//   2. Screenshot area-select overlay — draw a rect, OCR it, save as memory

(() => {
  const HOST_ID = "recall-save-host";
  if (document.getElementById(HOST_ID)) return;

  const host = document.createElement("div");
  host.id = HOST_ID;
  host.style.all = "initial";
  document.documentElement.appendChild(host);
  const root = host.attachShadow({ mode: "open" });

  // ---------------------------------------------------------------- styles --
  const style = document.createElement("style");
  style.textContent = `
    :host { all: initial; font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }

    /* ---- shared ---- */
    .hidden { display: none !important; }

    /* ---- selection pill ---- */
    .pill {
      position: absolute; display: none; align-items: center; gap: 6px;
      background: #12100c; color: #fff; padding: 6px 12px;
      border-radius: 999px; font: 500 13px/1.2 inherit;
      cursor: pointer; box-shadow: 0 4px 14px rgba(0,0,0,.25);
      z-index: 2147483646; user-select: none; transition: transform .1s;
    }
    .pill:hover { transform: scale(1.05); background: #24243e; }
    .pill svg { width: 14px; height: 14px; flex: none; }

    /* ---- generic panel ---- */
    .panel {
      position: fixed; display: none; flex-direction: column;
      width: 360px; max-width: 95vw;
      background: #1b1812; border-radius: 14px;
      box-shadow: 0 16px 48px rgba(0,0,0,.3);
      color: #ece7db; z-index: 2147483647;
      overflow: hidden; border: 1px solid #e8e8ee;
    }
    .panel-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 10px 14px; background: #12100c; color: #fff;
      font-weight: 600; font-size: 12px; letter-spacing: .03em;
      flex: none;
    }
    .panel-header button {
      background: none; border: none; color: #fff; opacity: .7;
      cursor: pointer; font-size: 18px; line-height: 1; padding: 0;
    }
    .panel-header button:hover { opacity: 1; }
    .panel textarea {
      border: none; resize: vertical; padding: 10px 14px;
      font: inherit; min-height: 80px; max-height: 200px;
      outline: none; color: #ece7db; flex: none;
    }
    .panel-footer {
      display: flex; align-items: center; justify-content: flex-end;
      padding: 8px 14px 10px; border-top: 1px solid #f0f0f0; gap: 8px; flex: none;
    }
    .source-toggle {
      display: flex; align-items: center; gap: 5px;
      font-size: 11px; color: #a39b8a; cursor: pointer; margin-right: auto;
    }
    .btn {
      border: none; border-radius: 8px; padding: 7px 14px;
      font: 600 12px/1 inherit; cursor: pointer;
    }
    .btn-primary { background: #f0a030; color: #12100c; }
    .btn-primary:hover { background: #d48a20; }
    .btn-primary:disabled { background: rgba(240,160,48,.4); cursor: default; }
    .btn-ghost { background: #221e16; color: #ece7db; }
    .btn-ghost:hover { background: #2a2418; }
    .status-msg { font-size: 11px; padding: 4px 14px 6px; color: #57c87b; display: none; flex: none; }
    .status-msg.error { color: #e8837c; }

    /* ---- pdf preview panel ---- */
    .pdf-backdrop {
      position: fixed; inset: 0; z-index: 2147483646;
      background: rgba(17, 17, 27, 0); backdrop-filter: blur(0px);
      display: none; align-items: flex-start; justify-content: center;
      padding: 6vh 20px 20px; box-sizing: border-box; overflow-y: auto;
      transition: background .18s ease, backdrop-filter .18s ease;
    }
    .pdf-backdrop.open {
      background: rgba(17, 17, 27, .45); backdrop-filter: blur(2px);
    }
    #pdf-panel {
      position: relative; display: flex; width: 560px; max-width: 100%;
      max-height: 88vh; margin: 0; top: auto; left: auto; transform: none;
      border-radius: 16px; border: 1px solid #332d21;
      box-shadow: 0 24px 60px rgba(0,0,0,.6), 0 2px 8px rgba(0,0,0,.3);
      opacity: 0; transform: translateY(10px) scale(.98);
      transition: opacity .16s ease, transform .16s ease;
    }
    .pdf-backdrop.open #pdf-panel { opacity: 1; transform: translateY(0) scale(1); }

    #pdf-panel .panel-header {
      background: #1b1812; color: #ece7db; padding: 16px 18px 14px;
      border-bottom: 1px solid #332d21; align-items: flex-start;
    }
    .pdf-header-title { display: flex; gap: 10px; align-items: center; }
    .pdf-header-icon {
      width: 30px; height: 30px; flex: none; border-radius: 8px;
      background: #f0a030; display: flex; align-items: center; justify-content: center;
    }
    .pdf-header-icon svg { width: 16px; height: 16px; color: #fff; }
    .pdf-header-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
    .pdf-header-text .t1 { font: 700 13.5px/1.2 inherit; letter-spacing: 0; color: #ece7db; }
    .pdf-header-text .t2 {
      font: 400 11px/1.2 inherit; color: #a39b8a; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis; max-width: 320px;
    }
    #pdf-panel .panel-header button {
      color: #a39b8a; width: 26px; height: 26px; border-radius: 7px;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px; opacity: 1; transition: background .12s, color .12s;
    }
    #pdf-panel .panel-header button:hover { background: #2a2418; color: #ece7db; }

    .pdf-meta {
      display: flex; align-items: center; justify-content: space-between;
      gap: 10px; padding: 10px 18px; flex: none;
    }
    .pdf-meta-badge {
      display: inline-flex; align-items: center; gap: 6px;
      background: rgba(240,160,48,.1); color: #f0a030; border: 1px solid rgba(240,160,48,.25);
      border-radius: 999px; padding: 4px 10px 4px 8px;
      font: 600 11px/1 inherit;
    }
    .pdf-meta-badge svg { width: 12px; height: 12px; flex: none; }
    .pdf-meta-note { font: 400 11px/1.3 inherit; color: #a39b8a; text-align: right; }

    #pdf-panel .pdf-text-wrap { padding: 0 18px 4px; flex: 1; min-height: 0; display: flex; }
    #pdf-panel textarea {
      min-height: 260px; max-height: 48vh; flex: 1; font-size: 12.5px;
      line-height: 1.6; white-space: pre-wrap; resize: none;
      background: #221e16; border: 1px solid #332d21; border-radius: 10px;
      padding: 12px 13px; color: #ece7db; scrollbar-width: thin;
      scrollbar-color: #d3cff0 transparent;
    }
    #pdf-panel textarea:focus { border-color: #c9c3f2; background: #1b1812; }
    #pdf-panel textarea::-webkit-scrollbar { width: 8px; }
    #pdf-panel textarea::-webkit-scrollbar-thumb { background: #332d21; border-radius: 8px; }
    #pdf-panel textarea::-webkit-scrollbar-thumb:hover { background: #4a4030; }

    #pdf-panel .panel-footer {
      padding: 12px 18px 16px; border-top: 1px solid #332d21; gap: 8px;
    }
    #pdf-panel .btn { padding: 8px 16px; border-radius: 9px; font-size: 12.5px; }
    #pdf-panel .btn-primary {
      display: inline-flex; align-items: center; gap: 6px;
      box-shadow: 0 2px 8px rgba(240,160,48,.35);
    }
    #pdf-panel .btn-primary svg { width: 13px; height: 13px; }
    #pdf-panel .btn-primary:disabled { box-shadow: none; }
    #pdf-panel .status-msg { padding: 0 18px 10px; }

    /* ---- screenshot crop dialog ---- */
    .sc-backdrop {
      position: fixed; inset: 0; z-index: 2147483646;
      background: rgba(10,10,20,.72); backdrop-filter: blur(3px);
      display: none; align-items: center; justify-content: center;
      padding: 12px; box-sizing: border-box;
    }
    .sc-backdrop.open { display: flex; }

    .sc-dialog {
      background: #12100c; border-radius: 16px;
      box-shadow: 0 24px 64px rgba(0,0,0,.6);
      display: flex; flex-direction: column;
      width: min(97vw, 1440px);
      height: 88vh; max-height: 88vh;
      overflow: hidden; border: 1px solid rgba(255,255,255,.08);
    }

    .sc-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,.08);
      flex: none;
    }
    .sc-title { color: #fff; font: 600 13px/1 inherit; display: flex; align-items: center; gap: 8px; }
    .sc-title-badge {
      background: #f0a030; color: #12100c; font: 700 10px/1 inherit;
      padding: 3px 7px; border-radius: 999px; letter-spacing: .04em;
    }
    .sc-hint { color: rgba(255,255,255,.45); font: 400 11px/1 inherit; }
    .sc-close {
      background: rgba(255,255,255,.08); border: none; color: rgba(255,255,255,.6);
      width: 26px; height: 26px; border-radius: 7px; cursor: pointer;
      font-size: 16px; line-height: 1; display: flex; align-items: center; justify-content: center;
      transition: background .12s, color .12s;
    }
    .sc-close:hover { background: rgba(255,255,255,.15); color: #fff; }

    .sc-canvas-wrap {
      position: relative; flex: 1; min-height: 0;
      display: flex; align-items: center; justify-content: center;
      padding: 14px; background: #0d0b08; overflow: hidden;
    }
    .sc-canvas-wrap canvas {
      display: block; max-width: 100%; max-height: 100%;
      user-select: none;
    }

    .sc-selection {
      position: absolute; border: 2px solid #f0a030;
      box-shadow: 0 0 0 9999px rgba(0,0,0,.55);
      pointer-events: auto; display: none; box-sizing: border-box;
      cursor: move;
    }
    .sc-selection.dragging {
      background-image:
        repeating-linear-gradient(90deg, transparent 0 calc(33.333% - 1px), rgba(255,255,255,.28) calc(33.333% - 1px) 33.333%),
        repeating-linear-gradient(0deg, transparent 0 calc(33.333% - 1px), rgba(255,255,255,.28) calc(33.333% - 1px) 33.333%);
    }
    .sc-handle {
      position: absolute; width: 10px; height: 10px;
      background: #1b1812; border: 2px solid #f0a030; border-radius: 2px;
      pointer-events: all; box-sizing: border-box;
    }
    .sc-handle.nw { top:-5px;left:-5px;cursor:nw-resize; }
    .sc-handle.ne { top:-5px;right:-5px;cursor:ne-resize; }
    .sc-handle.sw { bottom:-5px;left:-5px;cursor:sw-resize; }
    .sc-handle.se { bottom:-5px;right:-5px;cursor:se-resize; }
    .sc-handle.n  { top:-5px;left:calc(50% - 5px);cursor:n-resize; }
    .sc-handle.s  { bottom:-5px;left:calc(50% - 5px);cursor:s-resize; }
    .sc-handle.w  { top:calc(50% - 5px);left:-5px;cursor:w-resize; }
    .sc-handle.e  { top:calc(50% - 5px);right:-5px;cursor:e-resize; }
    .sc-sel-label {
      position:absolute;top:4px;left:4px;
      background:rgba(240,160,48,.85);color:#12100c;
      font:600 10px/1 inherit;padding:3px 6px;border-radius:4px;
      pointer-events:none;white-space:nowrap;
    }

    .sc-footer { border-top:1px solid rgba(255,255,255,.08); padding:10px 16px; flex:none; }

    .sc-ocr-area { display:none; flex-direction:column; gap:8px; margin-bottom:10px; }
    .sc-ocr-area.visible { display:flex; }
    .sc-ocr-label { color:rgba(255,255,255,.5);font:600 10px/1 inherit;letter-spacing:.05em; }
    .sc-ocr-text {
      background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);
      border-radius:8px;color:#e8e8f0;font:13px/1.55 inherit;
      padding:10px 12px;resize:vertical;min-height:72px;max-height:180px;
      outline:none;width:100%;box-sizing:border-box;
    }
    .sc-ocr-text:focus { border-color:rgba(240,160,48,.5);background:rgba(240,160,48,.05); }

    .sc-actions { display:flex;align-items:center;gap:8px; }
    .sc-actions-hint { color:rgba(255,255,255,.35);font:400 11px/1 inherit;margin-right:auto; }
    .sc-btn {
      border:none;border-radius:8px;padding:8px 16px;
      font:600 12px/1 inherit;cursor:pointer;
      display:inline-flex;align-items:center;gap:6px;
    }
    .sc-btn:disabled { opacity:.45;cursor:default; }
    .sc-btn-ocr { background:#221e16;color:#a39b8a;border:1px solid #332d21; }
    .sc-btn-ocr:hover:not(:disabled) { background:#2a2418;color:#ece7db; }
    .sc-btn-send { background:#f0a030;color:#12100c;box-shadow:0 2px 8px rgba(240,160,48,.35); }
    .sc-btn-send:hover:not(:disabled) { background:#d48a20; }
    .sc-btn-cancel { background:rgba(255,255,255,.07);color:rgba(255,255,255,.6); }
    .sc-btn-cancel:hover:not(:disabled) { background:rgba(255,255,255,.12);color:#fff; }
    .sc-status { font:400 11px/1 inherit;padding-left:4px; }
    .sc-status.ok { color:#57c87b; }
    .sc-status.err { color:#e8837c; }
  `;
  root.appendChild(style);

  // ============================================================ SAVE PANEL ==
  const pill = document.createElement("div");
  pill.className = "pill";
  pill.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="9"/><path d="M12 8v4l3 2"/>
    </svg>
    <span>Save to Recall</span>`;
  root.appendChild(pill);

  const savePanel = document.createElement("div");
  savePanel.className = "panel";
  savePanel.id = "save-panel";
  savePanel.innerHTML = `
    <div class="panel-header">
      <span>New Recall memory</span>
      <button data-action="close-save">&times;</button>
    </div>
    <textarea data-role="save-text"></textarea>
    <div class="status-msg" data-role="save-status"></div>
    <div class="panel-footer">
      <label class="source-toggle">
        <input type="checkbox" data-role="include-source" checked />
        Include page source
      </label>
      <button class="btn btn-ghost" data-action="close-save">Cancel</button>
      <button class="btn btn-primary" data-action="save-memory">Save memory</button>
    </div>`;
  root.appendChild(savePanel);

  const saveTextarea = savePanel.querySelector('[data-role="save-text"]');
  const includeSourceEl = savePanel.querySelector('[data-role="include-source"]');
  const saveStatusEl = savePanel.querySelector('[data-role="save-status"]');
  const saveBtnEl = savePanel.querySelector('[data-action="save-memory"]');

  // ======================================================== PDF PREVIEW ==
  // Shows exactly what text was extracted from the PDF before anything is
  // sent to the hub. The user can edit it, then confirm — or cancel.
  const pdfBackdrop = document.createElement("div");
  pdfBackdrop.className = "pdf-backdrop";
  root.appendChild(pdfBackdrop);

  const pdfPanel = document.createElement("div");
  pdfPanel.className = "panel";
  pdfPanel.id = "pdf-panel";
  pdfPanel.innerHTML = `
    <div class="panel-header">
      <div class="pdf-header-title">
        <div class="pdf-header-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <path d="M14 2v6h6"/>
          </svg>
        </div>
        <div class="pdf-header-text">
          <span class="t1">Review text extracted from PDF</span>
          <span class="t2" data-role="pdf-source"></span>
        </div>
      </div>
      <button data-action="close-pdf">&times;</button>
    </div>
    <div class="pdf-meta">
      <span class="pdf-meta-badge">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/></svg>
        <span data-role="pdf-count">0 characters</span>
      </span>
      <span class="pdf-meta-note">Edit anything below — this is exactly what gets sent.</span>
    </div>
    <div class="pdf-text-wrap"><textarea data-role="pdf-text" spellcheck="false"></textarea></div>
    <div class="status-msg" data-role="pdf-status"></div>
    <div class="panel-footer">
      <button class="btn btn-ghost" data-action="close-pdf">Cancel</button>
      <button class="btn btn-primary" data-action="save-pdf">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/></svg>
        <span>Send to Recall</span>
      </button>
    </div>`;
  pdfBackdrop.appendChild(pdfPanel);

  const pdfTextarea = pdfPanel.querySelector('[data-role="pdf-text"]');
  const pdfCountEl = pdfPanel.querySelector('[data-role="pdf-count"]');
  const pdfSourceEl = pdfPanel.querySelector('[data-role="pdf-source"]');
  const pdfStatusEl = pdfPanel.querySelector('[data-role="pdf-status"]');
  const pdfSaveBtn = pdfPanel.querySelector('[data-action="save-pdf"]');
  const pdfSaveBtnLabel = pdfSaveBtn.querySelector("span");
  let pdfSourceTab = null;

  function updatePdfMeta() {
    pdfCountEl.textContent = `${pdfTextarea.value.length.toLocaleString()} characters`;
  }

  function openPdfPreview(text, tab) {
    pdfSourceTab = tab || null;
    pdfTextarea.value = text || "";
    pdfSourceEl.textContent = tab?.title || tab?.url || "";
    pdfStatusEl.style.display = "none";
    pdfSaveBtn.disabled = false;
    pdfSaveBtnLabel.textContent = "Send to Recall";
    updatePdfMeta();
    pdfBackdrop.style.display = "flex";
    // force a layout tick so the CSS transition on .open actually runs
    requestAnimationFrame(() => pdfBackdrop.classList.add("open"));
    pdfTextarea.focus();
  }

  function closePdfPreview() {
    pdfBackdrop.classList.remove("open");
    setTimeout(() => { pdfBackdrop.style.display = "none"; }, 160);
  }

  pdfTextarea.addEventListener("input", updatePdfMeta);

  pdfBackdrop.addEventListener("mousedown", (e) => {
    if (e.target === pdfBackdrop) closePdfPreview();
  });

  pdfPanel.addEventListener("click", (e) => {
    const action = e.target.closest("[data-action]")?.getAttribute("data-action");
    if (action === "close-pdf") { closePdfPreview(); return; }
    if (action === "save-pdf") {
      const text = pdfTextarea.value.trim();
      if (!text) return;
      pdfSaveBtn.disabled = true;
      pdfSaveBtnLabel.textContent = "Sending…";
      chrome.runtime.sendMessage({
        type: "SAVE_PDF_TEXT",
        text,
        tab: pdfSourceTab || { title: document.title, url: location.href },
      }, (res) => {
        pdfStatusEl.style.display = "block";
        if (res?.ok) {
          pdfStatusEl.textContent = `Saved ✓ (${res.total} chunk${res.total !== 1 ? "s" : ""} → Recall)`;
          pdfStatusEl.className = "status-msg";
          setTimeout(closePdfPreview, 900);
        } else {
          pdfStatusEl.textContent = "Failed — " + (res?.error || "check hub connection in Settings.");
          pdfStatusEl.className = "status-msg error";
          pdfSaveBtn.disabled = false;
          pdfSaveBtnLabel.textContent = "Try again";
        }
      });
    }
  });

  // ======================================================= SCREENSHOT DIALOG
  const scBackdrop = document.createElement("div");
  scBackdrop.className = "sc-backdrop";
  scBackdrop.innerHTML = `
    <div class="sc-dialog">
      <div class="sc-header">
        <div class="sc-title">
          <span class="sc-title-badge">SCREENSHOT</span>
          <span>Drag to select a region, then save</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px">
          <span class="sc-hint" data-role="sc-hint">Loading…</span>
          <button class="sc-close" data-action="sc-close">&times;</button>
        </div>
      </div>
      <div class="sc-canvas-wrap" data-role="sc-canvas-wrap">
        <canvas data-role="sc-canvas"></canvas>
        <div class="sc-selection" data-role="sc-selection">
          <div class="sc-sel-label" data-role="sc-sel-label"></div>
          <div class="sc-handle nw" data-h="nw"></div>
          <div class="sc-handle ne" data-h="ne"></div>
          <div class="sc-handle sw" data-h="sw"></div>
          <div class="sc-handle se" data-h="se"></div>
          <div class="sc-handle n"  data-h="n"></div>
          <div class="sc-handle s"  data-h="s"></div>
          <div class="sc-handle w"  data-h="w"></div>
          <div class="sc-handle e"  data-h="e"></div>
        </div>
      </div>
      <div class="sc-footer">
        <div class="sc-actions">
          <span class="sc-actions-hint" data-role="sc-actions-hint">Loading screenshot…</span>
          <span class="sc-status" data-role="sc-status"></span>
          <button class="sc-btn sc-btn-cancel" data-action="sc-close">Cancel</button>
          <button class="sc-btn sc-btn-send" data-action="sc-send" disabled>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/></svg>
            Save to Recall
          </button>
        </div>
      </div>
    </div>`;
  root.appendChild(scBackdrop);

  const scCanvas      = scBackdrop.querySelector("[data-role='sc-canvas']");
  const scCanvasWrap  = scBackdrop.querySelector("[data-role='sc-canvas-wrap']");
  const scSelection   = scBackdrop.querySelector("[data-role='sc-selection']");
  const scSelLabel    = scBackdrop.querySelector("[data-role='sc-sel-label']");
  const scHint        = scBackdrop.querySelector("[data-role='sc-hint']");
  const scActionsHint = scBackdrop.querySelector("[data-role='sc-actions-hint']");
  const scStatus      = scBackdrop.querySelector("[data-role='sc-status']");
  const scBtnSend     = scBackdrop.querySelector("[data-action='sc-send']");

  // ================================================================ STATE ==
  let lastSelectionRect = null;
  let hidePillTimeout = null;
  let scDataUrl = null;   // full-page screenshot
  let scImgNatural = { w: 1, h: 1 }; // native image dimensions
  let scCropRect = null; // { x,y,w,h } in CANVAS-pixel space
  let scDragging = false;
  let scDragStart = null;
  let scActiveHandle = null; // which resize handle is being dragged

  // ============================================================= UTILITIES ==
  function positionNear(el, rect) {
    const top = window.scrollY + rect.bottom + 8;
    const left = Math.max(8, Math.min(rect.left, window.innerWidth - 380));
    el.style.top = `${top}px`;
    el.style.left = `${left}px`;
    el.style.position = "absolute";
  }

  // ============================================================ SAVE PANEL ==
  function openSavePanel(text) {
    saveTextarea.value = text;
    saveStatusEl.style.display = "none";
    saveBtnEl.disabled = false;
    saveBtnEl.textContent = "Save memory";
    if (lastSelectionRect) positionNear(savePanel, lastSelectionRect);
    else { savePanel.style.top = "80px"; savePanel.style.left = "20px"; savePanel.style.position = "fixed"; }
    savePanel.style.display = "flex";
    pill.style.display = "none";
    saveTextarea.focus();
  }

  function closeSavePanel() { savePanel.style.display = "none"; }

  savePanel.addEventListener("click", async (e) => {
    const action = e.target.getAttribute("data-action");
    if (action === "close-save") { closeSavePanel(); return; }
    if (action === "save-memory") {
      const text = saveTextarea.value.trim();
      if (!text) return;
      saveBtnEl.disabled = true;
      saveBtnEl.textContent = "Saving…";
      chrome.runtime.sendMessage({
        type: "SAVE_MEMORY", text,
        withSource: includeSourceEl.checked,
        tab: { title: document.title, url: location.href },
      }, (res) => {
        saveStatusEl.style.display = "block";
        if (res?.ok) {
          saveStatusEl.textContent = "Saved ✓";
          saveStatusEl.className = "status-msg";
          setTimeout(closeSavePanel, 700);
        } else {
          saveStatusEl.textContent = "Failed — check hub connection in Settings.";
          saveStatusEl.className = "status-msg error";
          saveBtnEl.disabled = false;
          saveBtnEl.textContent = "Try again";
        }
      });
    }
  });

  // ===================================================== SCREENSHOT DIALOG ==

  // Draw the full screenshot onto the canvas, preserving aspect ratio
  function scDrawImage(onReady) {
    if (!scDataUrl) return;
    const img = new Image();
    img.onload = () => {
      scImgNatural = { w: img.naturalWidth, h: img.naturalHeight };
      const wrap = scCanvasWrap.getBoundingClientRect();
      const maxW = wrap.width  - 24;
      const maxH = wrap.height - 24;
      // Show the screenshot at up to 80% of its real pixel size — large
      // enough to actually read while cropping — but still shrink further
      // if the dialog doesn't have room for that.
      const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 0.8);
      scCanvas.width  = Math.round(img.naturalWidth  * scale);
      scCanvas.height = Math.round(img.naturalHeight * scale);
      const ctx = scCanvas.getContext("2d");
      ctx.drawImage(img, 0, 0, scCanvas.width, scCanvas.height);
      if (onReady) onReady();
    };
    img.src = scDataUrl;
  }

  // Default crop box: the whole image, just like a normal image-editor
  // crop tool — the user narrows it down by dragging the handles/edges.
  function scInitFullSelection() {
    scCropRect = { x: 0, y: 0, w: scCanvas.width, h: scCanvas.height };
    scApplySelection(scCropRect);
    scBtnSend.disabled = false;
    scActionsHint.textContent = "Drag the corners to crop, then save";
    scHint.textContent = "Drag the corners or edges to adjust the crop";
  }

  // Convert canvas-element-relative coords to the position/size of the overlay div
  function scApplySelection(rect) {
    if (!rect) { scSelection.style.display = "none"; return; }
    const cr = scCanvas.getBoundingClientRect();
    const wr = scCanvasWrap.getBoundingClientRect();
    const left   = cr.left - wr.left + rect.x;
    const top    = cr.top  - wr.top  + rect.y;
    scSelection.style.display = "block";
    scSelection.style.left    = left + "px";
    scSelection.style.top     = top  + "px";
    scSelection.style.width   = rect.w + "px";
    scSelection.style.height  = rect.h + "px";
    // label shows native pixel dimensions
    const scale = scImgNatural.w / scCanvas.width;
    const nw = Math.round(rect.w * scale);
    const nh = Math.round(rect.h * scale);
    scSelLabel.textContent = nw + " × " + nh + " px";
  }

  function scResetCrop() {
    scCropRect = null;
    scSelection.style.display = "none";
    scBtnSend.disabled = true;
    scActionsHint.textContent = "Loading screenshot…";
    scStatus.textContent = "";
    scStatus.className = "sc-status";
    scHint.textContent = "Loading…";
  }

  function openScreenshotDialog(dataUrl) {
    scDataUrl = dataUrl;
    scResetCrop();
    scBackdrop.classList.add("open");
    // draw after layout so getBoundingClientRect is valid
    requestAnimationFrame(() => scDrawImage(scInitFullSelection));
  }

  function closeScreenshotDialog() {
    scBackdrop.classList.remove("open");
    scDataUrl = null;
    scCropRect = null;
    scDragging = false;
    scDragStart = null;
    scActiveHandle = null;
  }

  // ---- pointer helpers ----
  function canvasRelative(e) {
    const r = scCanvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

  // ---- resize handle drag ----
  scSelection.querySelectorAll(".sc-handle").forEach(h => {
    h.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      scActiveHandle = h.dataset.h;
      scDragStart = canvasRelative(e);
      scDragging = true;
      scSelection.classList.add("dragging");
      e.stopPropagation();
      e.preventDefault();
    });
  });

  // ---- move selection (drag inside the crop box) ----
  scSelection.addEventListener("mousedown", (e) => {
    if (e.button !== 0 || e.target.classList.contains("sc-handle")) return;
    scActiveHandle = "move";
    scDragStart = canvasRelative(e);
    scDragging = true;
    scSelection.classList.add("dragging");
    e.stopPropagation();
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!scDragging || !scActiveHandle) return;
    const p = canvasRelative(e);
    const cw = scCanvas.width, ch = scCanvas.height;

    if (scActiveHandle === "move") {
      const dx = p.x - scDragStart.x;
      const dy = p.y - scDragStart.y;
      scDragStart = p;
      scCropRect.x = clamp(scCropRect.x + dx, 0, cw - scCropRect.w);
      scCropRect.y = clamp(scCropRect.y + dy, 0, ch - scCropRect.h);
    } else {
      // resize from a corner/edge handle
      const r = Object.assign({}, scCropRect);
      const dx = p.x - scDragStart.x;
      const dy = p.y - scDragStart.y;
      scDragStart = p;
      const h = scActiveHandle;
      if (h.includes("e")) { r.w = clamp(r.w + dx, 10, cw - r.x); }
      if (h.includes("s")) { r.h = clamp(r.h + dy, 10, ch - r.y); }
      if (h.includes("w")) { const nw = clamp(r.w - dx, 10, r.x + r.w); r.x = r.x + r.w - nw; r.w = nw; }
      if (h.includes("n")) { const nh = clamp(r.h - dy, 10, r.y + r.h); r.y = r.y + r.h - nh; r.h = nh; }
      scCropRect = r;
    }
    scApplySelection(scCropRect);
  });

  document.addEventListener("mouseup", (e) => {
    if (!scDragging) return;
    scDragging = false;
    scActiveHandle = null;
    scSelection.classList.remove("dragging");
    scBtnSend.disabled = false;
    scActionsHint.textContent = "Drag the corners to adjust, then save";
    scHint.textContent = "Drag the corners or edges to adjust the crop";
  });

  // ---- Save button ----
  // Single action: read the text out of the selected region and save it to
  // Recall. This happens behind the scenes — there's no separate
  // "extract"/"OCR" step shown in the UI, only a crop and a save.
  scBtnSend.addEventListener("click", async () => {
    if (!scCropRect || !scDataUrl) return;
    scBtnSend.disabled = true;
    scBtnSend.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/></svg> Saving…`;
    scStatus.textContent = "";
    scStatus.className = "sc-status";

    // crop from the NATIVE-resolution image
    const scale = scImgNatural.w / scCanvas.width;
    const nx = Math.round(scCropRect.x * scale);
    const ny = Math.round(scCropRect.y * scale);
    const nw = Math.round(scCropRect.w * scale);
    const nh = Math.round(scCropRect.h * scale);

    const crop = document.createElement("canvas");
    crop.width = nw; crop.height = nh;
    const img = new Image();
    img.onload = () => {
      crop.getContext("2d").drawImage(img, nx, ny, nw, nh, 0, 0, nw, nh);
      const croppedDataUrl = crop.toDataURL("image/png");

      chrome.runtime.sendMessage({
        type: "SAVE_SCREENSHOT_IMAGE",
        imageDataUrl: croppedDataUrl,
        tab: { title: document.title, url: location.href },
      }, (res) => {
        console.log("[Recall] image save result:", res);
        if (res?.ok) {
          scStatus.textContent = "Saved to Recall ✓";
          scStatus.className = "sc-status ok";
          setTimeout(closeScreenshotDialog, 900);
        } else {
          scStatus.textContent = "Save failed — " + (res?.error || "check hub connection");
          scStatus.className = "sc-status err";
          scBtnSend.disabled = false;
          scBtnSend.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/></svg> Save to Recall`;
        }
      });
    };
    img.src = scDataUrl;
  });

  // ---- close / backdrop click ----
  scBackdrop.querySelectorAll("[data-action='sc-close']").forEach(el => {
    el.addEventListener("click", closeScreenshotDialog);
  });
  scBackdrop.addEventListener("mousedown", (e) => {
    if (e.target === scBackdrop) closeScreenshotDialog();
  });

  // redraw canvas if window resizes while dialog is open
  window.addEventListener("resize", () => {
    if (scBackdrop.classList.contains("open") && scDataUrl) {
      const prevRect = scCropRect;
      const prevW = scCanvas.width, prevH = scCanvas.height;
      scDrawImage(() => {
        if (prevRect && prevW && prevH) {
          const sx = scCanvas.width / prevW;
          const sy = scCanvas.height / prevH;
          scCropRect = {
            x: prevRect.x * sx, y: prevRect.y * sy,
            w: prevRect.w * sx, h: prevRect.h * sy,
          };
          scApplySelection(scCropRect);
        }
      });
    }
  });

  // ================================================ SELECTION PILL LOGIC ==
  document.addEventListener("mouseup", (e) => {
    if (host.contains(e.target)) return;
    clearTimeout(hidePillTimeout);
    hidePillTimeout = setTimeout(() => {
      const sel = window.getSelection();
      const text = sel?.toString().trim() || "";
      if (text) {
        const range = sel.getRangeAt(0);
        lastSelectionRect = range.getBoundingClientRect();
        positionNear(pill, lastSelectionRect);
        pill.style.display = "flex";
      } else {
        pill.style.display = "none";
      }
    }, 10);
  });

  document.addEventListener("mousedown", (e) => {
    if (!host.contains(e.target)) {
      pill.style.display = "none";
      closeSavePanel();
    }
  });

  pill.addEventListener("click", () => {
    const text = window.getSelection()?.toString().trim() || "";
    if (text) openSavePanel(text);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeSavePanel();
      closePdfPreview();
      closeScreenshotDialog();
    }
  });

  // ============================================ MESSAGES FROM BACKGROUND ==
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === "PING") { sendResponse({ ok: true }); return; } // used by popup to detect presence
    if (msg.type === "START_SCREENSHOT_SELECT") openScreenshotDialog(msg.dataUrl);
    if (msg.type === "OPEN_PDF_PREVIEW") {
      openPdfPreview(msg.text, msg.tab);
      sendResponse({ ok: true });
    }
  });

})();