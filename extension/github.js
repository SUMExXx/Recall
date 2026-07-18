// github.js — runs on github.com. Adds a "Save repo to Recall" trigger on
// repo pages: ingests the repo (via background.js → GitHub API), builds a
// graph + LLM context text (via repoGraph.js), and shows a dialog with a
// Graph tab and a Context tab (copy + send-to-memory).

(() => {
  const HOST_ID = "recall-github-host";
  if (document.getElementById(HOST_ID)) return;

  const RG = window.RecallRepoGraph;
  if (!RG) return; // repoGraph.js failed to load — bail quietly

  const host = document.createElement("div");
  host.id = HOST_ID;
  host.style.all = "initial";
  document.documentElement.appendChild(host);
  const root = host.attachShadow({ mode: "open" });

  // ---------------------------------------------------------------- styles --
  const style = document.createElement("style");
  style.textContent = `
    :host { all: initial; font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .hidden { display: none !important; }

    /* ---- floating trigger button ---- */
    .trigger {
      position: fixed; bottom: 20px; right: 20px; z-index: 2147483646;
      display: flex; align-items: center; gap: 7px;
      background: #1a1a2e; color: #fff; padding: 10px 16px;
      border-radius: 999px; font: 600 13px/1.2 inherit; cursor: pointer;
      box-shadow: 0 6px 20px rgba(0,0,0,.3); user-select: none;
      transition: transform .12s, background .12s; border: none;
    }
    .trigger:hover { transform: scale(1.04); background: #24243e; }
    .trigger svg { width: 15px; height: 15px; flex: none; }
    .trigger.busy { cursor: default; opacity: .85; }
    .trigger .spinner {
      width: 13px; height: 13px; border: 2px solid rgba(255,255,255,.35);
      border-top-color: #fff; border-radius: 50%; animation: rg-spin .7s linear infinite;
    }
    @keyframes rg-spin { to { transform: rotate(360deg); } }

    .toast {
      position: fixed; bottom: 72px; right: 20px; z-index: 2147483646;
      background: #111; color: #fff; padding: 8px 14px; border-radius: 8px;
      font-size: 12px; max-width: 320px; box-shadow: 0 6px 20px rgba(0,0,0,.3);
    }
    .toast.error { background: #dc2626; }

    /* ---- dialog ---- */
    .overlay {
      position: fixed; inset: 0; z-index: 2147483647;
      background: rgba(15,15,25,.55); display: none;
      align-items: center; justify-content: center;
    }
    .dialog {
      width: min(920px, 94vw); height: min(640px, 88vh);
      background: #fff; border-radius: 14px; overflow: hidden;
      display: flex; flex-direction: column;
      box-shadow: 0 24px 64px rgba(0,0,0,.35); color: #1a1a2e;
    }
    .dialog-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 12px 16px; background: #1a1a2e; color: #fff; flex: none;
    }
    .dialog-header .title { font-weight: 700; font-size: 13px; }
    .dialog-header .sub { font-weight: 400; opacity: .7; font-size: 11.5px; margin-left: 8px; }
    .dialog-header button {
      background: none; border: none; color: #fff; opacity: .75;
      cursor: pointer; font-size: 20px; line-height: 1; padding: 0 2px;
    }
    .dialog-header button:hover { opacity: 1; }

    .tabs {
      display: flex; border-bottom: 1px solid #e5e7eb; background: #fafafa; flex: none;
    }
    .tab-btn {
      padding: 10px 18px; font: 600 12.5px inherit; background: none; border: none;
      cursor: pointer; color: #6b7280; border-bottom: 2px solid transparent;
    }
    .tab-btn.active { color: #111; border-bottom-color: #111; }

    .tab-body { flex: 1; overflow: hidden; display: flex; }
    .tab-panel { flex: 1; display: none; overflow: hidden; flex-direction: column; }
    .tab-panel.active { display: flex; }

    /* graph tab */
    .graph-toolbar {
      display: flex; align-items: center; gap: 10px; padding: 6px 12px;
      background: rgba(248,250,252,.9); border-bottom: 1px solid #e5e7eb; flex: none;
      font-size: 11.5px; color: #374151; font-family: monospace;
    }
    .graph-toolbar .legend { display: flex; gap: 8px; flex-wrap: wrap; margin-left: auto; }
    .legend-item { display: flex; align-items: center; gap: 4px; font-family: inherit; font-size: 10.5px; color: #4b5563; }
    .legend-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
    .graph-canvas-wrap { flex: 1; position: relative; overflow: hidden; background: #f8fafc; cursor: grab; }
    .graph-canvas-wrap.dragging { cursor: grabbing; }
    .tooltip-panel {
      position: absolute; bottom: 10px; left: 10px; max-width: 320px;
      background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
      padding: 10px 12px; box-shadow: 0 8px 24px rgba(0,0,0,.12); display: none;
    }
    .tooltip-panel .name { font-weight: 700; font-size: 13px; margin-bottom: 2px; }
    .tooltip-panel .meta { font-size: 11px; color: #6b7280; }

    /* context tab */
    .ctx-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 14px; border-bottom: 1px solid #e5e7eb; background: #fafafa; flex: none;
    }
    .ctx-header span { font-size: 11px; color: #6b7280; font-family: monospace; }
    .btn {
      border: none; border-radius: 8px; padding: 7px 14px;
      font: 600 12px/1 inherit; cursor: pointer;
    }
    .btn-primary { background: #4f46e5; color: #fff; }
    .btn-primary:hover { background: #4338ca; }
    .btn-primary:disabled { background: #b7b3f5; cursor: default; }
    .btn-ghost { background: #f2f2f5; color: #444; }
    .btn-ghost:hover { background: #e6e6ea; }
    .ctx-body { flex: 1; overflow: auto; padding: 14px; }
    .ctx-body pre {
      margin: 0; font-size: 11.5px; font-family: monospace;
      color: #1f2937; line-height: 1.7; white-space: pre-wrap; word-break: break-word;
    }
    .ctx-footer {
      display: flex; align-items: center; gap: 10px; padding: 10px 14px;
      border-top: 1px solid #e5e7eb; flex: none; background: #fafafa;
    }
    .ctx-status { font-size: 11.5px; color: #16a34a; }
    .ctx-status.error { color: #dc2626; }
  `;
  root.appendChild(style);

  // ============================================================= TRIGGER ==
  const trigger = document.createElement("button");
  trigger.className = "trigger";
  trigger.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.7c-2.78.6-3.37-1.34-3.37-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.9 1.52 2.34 1.08 2.91.83.09-.65.35-1.08.63-1.33-2.22-.25-4.56-1.11-4.56-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.65 0 0 .84-.27 2.75 1.02a9.6 9.6 0 0 1 5 0c1.91-1.3 2.75-1.02 2.75-1.02.55 1.38.2 2.4.1 2.65.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.86v2.76c0 .27.18.58.69.48A10 10 0 0 0 12 2Z"/>
    </svg>
    <span data-role="trigger-label">Save repo to Recall</span>`;
  root.appendChild(trigger);

  const toast = document.createElement("div");
  toast.className = "toast hidden";
  root.appendChild(toast);

  function showToast(msg, isError) {
    toast.textContent = msg;
    toast.className = `toast${isError ? " error" : ""}`;
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => toast.classList.add("hidden"), 3200);
  }

  function setBusy(busy, label) {
    trigger.classList.toggle("busy", busy);
    trigger.disabled = busy;
    const labelEl = trigger.querySelector('[data-role="trigger-label"]');
    labelEl.textContent = label || "Save repo to Recall";
    const svgEl = trigger.querySelector("svg");
    if (busy) {
      svgEl.outerHTML = '<div class="spinner"></div>';
    } else if (!trigger.querySelector("svg")) {
      trigger.querySelector(".spinner")?.replaceWith(svgFromMarkup());
    }
  }
  function svgFromMarkup() {
    const tmp = document.createElement("div");
    tmp.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.7c-2.78.6-3.37-1.34-3.37-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.9 1.52 2.34 1.08 2.91.83.09-.65.35-1.08.63-1.33-2.22-.25-4.56-1.11-4.56-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.65 0 0 .84-.27 2.75 1.02a9.6 9.6 0 0 1 5 0c1.91-1.3 2.75-1.02 2.75-1.02.55 1.38.2 2.4.1 2.65.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.86v2.76c0 .27.18.58.69.48A10 10 0 0 0 12 2Z"/></svg>`;
    return tmp.firstChild;
  }

  // ============================================================= DIALOG ==
  const overlay = document.createElement("div");
  overlay.className = "overlay";
  overlay.innerHTML = `
    <div class="dialog" role="dialog" aria-label="Repo → Recall memory">
      <div class="dialog-header">
        <div><span class="title" data-role="dlg-title">Repo graph</span><span class="sub" data-role="dlg-sub"></span></div>
        <button data-action="close-dialog" aria-label="Close">&times;</button>
      </div>
      <div class="tabs" role="tablist">
        <button class="tab-btn active" data-tab="graph" role="tab">Graph</button>
        <button class="tab-btn" data-tab="context" role="tab">Context text</button>
      </div>
      <div class="tab-body">
        <div class="tab-panel active" data-panel="graph">
          <div class="graph-toolbar">
            <span data-role="graph-counts">0 nodes • 0 edges</span>
            <div class="legend" data-role="legend"></div>
          </div>
          <div class="graph-canvas-wrap" data-role="canvas-wrap" tabindex="0">
            <svg data-role="svg" width="100%" height="100%"></svg>
            <div class="tooltip-panel" data-role="tooltip">
              <div class="name" data-role="tooltip-name"></div>
              <div class="meta" data-role="tooltip-meta"></div>
            </div>
          </div>
        </div>
        <div class="tab-panel" data-panel="context">
          <div class="ctx-header">
            <span data-role="ctx-count">0 characters</span>
            <button class="btn btn-ghost" data-action="copy-context">Copy</button>
          </div>
          <div class="ctx-body"><pre data-role="ctx-text"></pre></div>
          <div class="ctx-footer">
            <button class="btn btn-primary" data-action="send-memory">Send to memory</button>
            <span class="ctx-status" data-role="ctx-status"></span>
          </div>
        </div>
      </div>
    </div>`;
  root.appendChild(overlay);

  const dlgTitle = overlay.querySelector('[data-role="dlg-title"]');
  const dlgSub = overlay.querySelector('[data-role="dlg-sub"]');
  const svgEl = overlay.querySelector('[data-role="svg"]');
  const canvasWrap = overlay.querySelector('[data-role="canvas-wrap"]');
  const graphCounts = overlay.querySelector('[data-role="graph-counts"]');
  const legendEl = overlay.querySelector('[data-role="legend"]');
  const tooltipEl = overlay.querySelector('[data-role="tooltip"]');
  const tooltipName = overlay.querySelector('[data-role="tooltip-name"]');
  const tooltipMeta = overlay.querySelector('[data-role="tooltip-meta"]');
  const ctxCount = overlay.querySelector('[data-role="ctx-count"]');
  const ctxText = overlay.querySelector('[data-role="ctx-text"]');
  const ctxStatus = overlay.querySelector('[data-role="ctx-status"]');

  let currentContextText = "";
  let currentGraph = { nodes: [], edges: [] };
  let currentTab = null;

  function closeDialog() { overlay.style.display = "none"; }

  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeDialog();
    const action = e.target.getAttribute("data-action");
    if (action === "close-dialog") closeDialog();
    if (action === "copy-context") {
      navigator.clipboard.writeText(currentContextText || "");
      ctxStatus.textContent = "Copied ✓";
      ctxStatus.className = "ctx-status";
      setTimeout(() => (ctxStatus.textContent = ""), 1600);
    }
    if (action === "send-memory") sendContextToMemory();
  });

  overlay.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      overlay.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      overlay.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      overlay.querySelector(`[data-panel="${btn.dataset.tab}"]`).classList.add("active");
    });
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && overlay.style.display !== "none") closeDialog();
  });

  async function sendContextToMemory() {
    const btn = overlay.querySelector('[data-action="send-memory"]');
    btn.disabled = true;
    btn.textContent = "Sending…";
    ctxStatus.textContent = "";
    chrome.runtime.sendMessage({
      type: "SAVE_GITHUB_CONTEXT",
      contextText: currentContextText,
      tab: { title: document.title, url: location.href },
    }, (res) => {
      btn.disabled = false;
      btn.textContent = "Send to memory";
      if (res?.ok) {
        ctxStatus.textContent = `Saved to Recall memory (${res.total} chunk${res.total !== 1 ? "s" : ""}) ✓`;
        ctxStatus.className = "ctx-status";
      } else {
        ctxStatus.textContent = "Failed — " + (res?.error || "check hub connection in Settings.");
        ctxStatus.className = "ctx-status error";
      }
    });
  }

  // --------------------------------------------------------- graph render --
  function renderLegend(nodes) {
    const cats = [...new Set(nodes.map(n => n.category))];
    legendEl.innerHTML = cats.map(c => `
      <span class="legend-item">
        <span class="legend-dot" style="background:${RG.CATEGORY_COLORS[c] || "#9ca3af"}"></span>${c}
      </span>`).join("");
  }

  function renderGraph(graph) {
    const { nodes, edges } = graph;
    graphCounts.textContent = `${nodes.length} nodes • ${edges.length} edges`;
    renderLegend(nodes);

    const rect = canvasWrap.getBoundingClientRect();
    const W = rect.width || 760, H = rect.height || 460;
    const positions = RG.computeLayout(nodes, edges, 42, W, H);
    const degreeMap = RG.computeDegreeMap(nodes, edges);
    const maxDeg = Math.max(...(degreeMap.size ? [...degreeMap.values()] : [1]), 1);
    const radiusOf = (id) => 6 + 12 * Math.sqrt((degreeMap.get(id) || 0) / maxDeg);

    let pan = { x: 0, y: 0 }, zoom = 1;
    let selectedId = null;

    function draw() {
      const nodeById = new Map(nodes.map(n => [n.id, n]));
      const edgeSvg = edges.map(e => {
        const sp = positions[e.source], tp = positions[e.target];
        if (!sp || !tp) return "";
        let stroke = "#d1d5db", w = 1, opacity = 0.55;
        if (selectedId) {
          if (e.source === selectedId) { stroke = "#f97316"; w = 2; opacity = 1; }
          else if (e.target === selectedId) { stroke = "#3b82f6"; w = 2; opacity = 1; }
          else opacity = 0.12;
        }
        return `<line x1="${sp.x}" y1="${sp.y}" x2="${tp.x}" y2="${tp.y}" stroke="${stroke}" stroke-width="${w}" opacity="${opacity}" />`;
      }).join("");

      const neighborSet = selectedId ? new Set(
        edges.filter(e => e.source === selectedId || e.target === selectedId)
          .flatMap(e => [e.source, e.target])
      ) : null;

      const nodeSvg = nodes.map(n => {
        const p = positions[n.id];
        if (!p) return "";
        const color = RG.CATEGORY_COLORS[n.category] || "#9ca3af";
        const r = radiusOf(n.id);
        const isSel = selectedId === n.id;
        const opacity = selectedId ? (isSel || neighborSet.has(n.id) ? 1 : 0.15) : 1;
        const label = RG.trunc(n.name, 16);
        return `
          <g data-node-id="${encodeURIComponent(n.id)}" style="cursor:pointer" opacity="${opacity}">
            <circle cx="${p.x}" cy="${p.y}" r="${r + 4}" fill="transparent" />
            <circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${color}" fill-opacity="0.2"
              stroke="${isSel ? "#111" : color}" stroke-width="${isSel ? 2.5 : 1.4}" />
            <text x="${p.x}" y="${p.y + r + 11}" text-anchor="middle" font-size="9.5" fill="#374151"
              style="pointer-events:none;font-family:system-ui;user-select:none">${label}</text>
          </g>`;
      }).join("");

      svgEl.innerHTML = `<g transform="translate(${pan.x},${pan.y}) scale(${zoom})">${edgeSvg}${nodeSvg}</g>`;

      svgEl.querySelectorAll("[data-node-id]").forEach(g => {
        g.addEventListener("click", (e) => {
          e.stopPropagation();
          const id = decodeURIComponent(g.getAttribute("data-node-id"));
          selectedId = selectedId === id ? null : id;
          if (selectedId) {
            const n = nodeById.get(selectedId);
            tooltipName.textContent = n.name;
            tooltipMeta.textContent = `${n.category} • ${n.file || ""}`;
            tooltipEl.style.display = "block";
          } else {
            tooltipEl.style.display = "none";
          }
          draw();
        });
      });
    }

    draw();

    // pan + zoom
    let dragging = false, dragStart = null;
    canvasWrap.onmousedown = (e) => {
      dragging = true;
      canvasWrap.classList.add("dragging");
      dragStart = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    };
    canvasWrap.onmousemove = (e) => {
      if (!dragging) return;
      pan = { x: e.clientX - dragStart.x, y: e.clientY - dragStart.y };
      draw();
    };
    canvasWrap.onmouseup = canvasWrap.onmouseleave = () => {
      dragging = false;
      canvasWrap.classList.remove("dragging");
    };
    canvasWrap.onwheel = (e) => {
      e.preventDefault();
      zoom = Math.max(0.3, Math.min(3, zoom * (e.deltaY < 0 ? 1.12 : 0.89)));
      draw();
    };
    canvasWrap.onclick = () => {
      if (dragging) return;
      selectedId = null;
      tooltipEl.style.display = "none";
      draw();
    };
  }

  // --------------------------------------------------------------- ingest --
  function currentRepoInfo() {
    const parts = location.pathname.split("/").filter(Boolean);
    const RESERVED = new Set([
      "settings", "notifications", "marketplace", "explore", "topics",
      "trending", "collections", "sponsors", "codespaces", "issues",
      "pulls", "dashboard", "new", "organizations", "about", "apps",
    ]);
    if (location.hostname !== "github.com" || parts.length < 2 || RESERVED.has(parts[0])) return null;
    const [owner, repo] = parts;
    let branchHint = null;
    if (parts[2] === "tree" && parts[3]) branchHint = parts[3];
    return { owner, repo: repo.replace(/\.git$/, ""), branchHint };
  }

  async function startIngest() {
    const info = currentRepoInfo();
    if (!info) { showToast("This doesn't look like a GitHub repo page.", true); return; }

    setBusy(true, "Ingesting repo…");
    chrome.runtime.sendMessage({
      type: "INGEST_GITHUB_REPO",
      owner: info.owner, repo: info.repo, branchHint: info.branchHint,
    }, (res) => {
      setBusy(false);
      if (!res?.ok) {
        showToast("Ingest failed: " + (res?.error || "unknown error"), true);
        return;
      }
      currentGraph = res.graph;
      currentContextText = res.contextText;

      dlgTitle.textContent = res.meta.repo;
      dlgSub.textContent = ` ${res.meta.branch} • ${res.meta.fileCount} files analyzed`;
      ctxText.textContent = currentContextText;
      ctxCount.textContent = `${currentContextText.length.toLocaleString()} characters`;
      ctxStatus.textContent = "";

      overlay.style.display = "flex";
      // graph tab is active by default; render after layout so canvas has real size
      requestAnimationFrame(() => renderGraph(currentGraph));
    });
  }

  trigger.addEventListener("click", startIngest);

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "OPEN_GITHUB_INGEST") startIngest();
  });

  // Only show the floating trigger on actual repo pages. GitHub is a
  // pjax/turbo SPA, so re-check on navigation instead of relying on reload.
  function syncTriggerVisibility() {
    trigger.classList.toggle("hidden", !currentRepoInfo());
  }
  syncTriggerVisibility();
  let lastHref = location.href;
  setInterval(() => {
    if (location.href !== lastHref) {
      lastHref = location.href;
      syncTriggerVisibility();
    }
  }, 1200);
})();
