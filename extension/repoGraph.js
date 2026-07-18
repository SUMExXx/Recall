// repoGraph.js — shared GitHub-repo → graph + LLM-context builder.
// Loaded in TWO different contexts, so no import/export — attach to global:
//   • background.js  (MV3 service worker, classic script) via importScripts()
//   • github.js       (content script)                    via <script> tag
//
// Output format mirrors DiagramStudio's RepoGraphDialog: each graph node is
// serialized as a "Module { id } / File: ... / name (category) — lines A-B"
// block, blocks separated by a line containing just "\", relationships as
// "Relationship: src -> tgt (relationship)" lines. That keeps the generated
// context text human/LLM-readable *and* round-trippable if it's ever pasted
// back into DiagramStudio.

(function (global) {

  // ------------------------------------------------------------------ consts
  const CATEGORY_COLORS = {
    module:           "#7c3aed",
    class:            "#0ea5e9",
    function:         "#f97316",
    method:           "#f59e0b",
    directory:        "#6b7280",
    external_symbol:  "#a3a3a3",
    next_page_module: "#8b5cf6",
    component:        "#10b981",
    interface:        "#06b6d4",
    type_alias:       "#ec4899",
    variable:         "#84cc16",
    enum:             "#fb923c",
    unknown:          "#9ca3af",
  };

  const IGNORE_DIRS = new Set([
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", "coverage", ".mypy_cache", "vendor",
  ]);
  const IGNORE_EXTENSIONS = new Set([
    "png", "jpg", "jpeg", "gif", "svg", "ico", "webp",
    "mp4", "mp3", "pdf", "zip", "tar", "gz",
    "exe", "dll", "so", "bin", "wasm", "lock", "woff", "woff2", "ttf",
  ]);
  const CODE_EXTENSIONS = new Set([
    "py", "js", "mjs", "cjs", "jsx", "ts", "tsx", "java", "go",
    "rs", "cpp", "c", "h", "cs", "rb", "php",
    "json", "yaml", "yml", "toml", "md", "txt",
    "html", "css", "scss",
  ]);

  const MAX_FILE_SIZE = 500_000;     // bytes, skip anything bigger
  const MAX_GRAPH_FILES = 200;       // cap files considered for the graph
  const MAX_NODES = 140;
  const MAX_EDGES = 400;
  const MAX_SYMBOLS_PER_FILE = 6;

  // ------------------------------------------------------------------ utils
  function trunc(s, n) {
    return s && s.length > n ? s.slice(0, n - 1) + "…" : (s || "");
  }

  function extOf(path) {
    const base = path.split("/").pop() || "";
    const i = base.lastIndexOf(".");
    return i > 0 ? base.slice(i + 1).toLowerCase() : "";
  }

  function shouldIncludePath(path) {
    const parts = path.split("/");
    if (parts.some(p => IGNORE_DIRS.has(p))) return false;
    const ext = extOf(path);
    if (IGNORE_EXTENSIONS.has(ext)) return false;
    if (CODE_EXTENSIONS.size && !CODE_EXTENSIONS.has(ext)) return false;
    return true;
  }

  const LANG_MAP = {
    py: "python", js: "javascript", mjs: "javascript", cjs: "javascript",
    jsx: "javascript", ts: "typescript", tsx: "typescript", java: "java",
    go: "go", rs: "rust", cpp: "cpp", c: "c", cs: "csharp", rb: "ruby",
    php: "php", json: "json", yaml: "yaml", yml: "yaml", toml: "toml",
    md: "markdown", html: "html", css: "css", scss: "scss",
  };

  function detectLanguage(path) {
    return LANG_MAP[extOf(path)] || "text";
  }

  // File-level category: what kind of "module" this file is.
  function fileCategory(path, language) {
    const base = (path.split("/").pop() || "");
    const ext = extOf(path);
    if (ext === "jsx" || ext === "tsx") {
      // Capitalized filename or lives in a components/ dir → component
      const name = base.replace(/\.(jsx|tsx)$/, "");
      if (/^[A-Z]/.test(name) || /\/components?\//i.test(path)) return "component";
      return "module";
    }
    if (["json", "yaml", "yml", "toml", "md", "txt", "css", "scss", "html"].includes(ext)) {
      return "unknown";
    }
    if (/(^|\/)pages\//i.test(path) && (ext === "js" || ext === "ts")) return "next_page_module";
    return "module";
  }

  // ------------------------------------------------------ symbol extraction
  // Best-effort, regex-based (no AST). Good enough to give the graph some
  // structure beyond pure file-to-file edges.
  function extractSymbols(path, content, language) {
    if (!content) return [];
    const lines = content.split("\n");
    const found = [];
    const seen = new Set();

    const push = (name, category, lineIdx) => {
      if (!name || seen.has(name) || found.length >= MAX_SYMBOLS_PER_FILE) return;
      seen.add(name);
      found.push({ name, category, line: lineIdx + 1 });
    };

    const isJsxLike = language === "javascript" || language === "typescript";
    const isPy = language === "python";

    lines.forEach((line, idx) => {
      if (found.length >= MAX_SYMBOLS_PER_FILE) return;

      if (isJsxLike) {
        let m;
        if ((m = line.match(/^\s*export\s+default\s+function\s+([A-Za-z_$][\w$]*)/))) {
          push(m[1], /^[A-Z]/.test(m[1]) ? "component" : "function", idx);
        } else if ((m = line.match(/^\s*export\s+(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/))) {
          push(m[1], /^[A-Z]/.test(m[1]) ? "component" : "function", idx);
        } else if ((m = line.match(/^\s*function\s+([A-Za-z_$][\w$]*)/))) {
          push(m[1], /^[A-Z]/.test(m[1]) ? "component" : "function", idx);
        } else if ((m = line.match(/^\s*export\s+(?:default\s+)?class\s+([A-Za-z_$][\w$]*)/))) {
          push(m[1], "class", idx);
        } else if ((m = line.match(/^\s*class\s+([A-Za-z_$][\w$]*)/))) {
          push(m[1], "class", idx);
        } else if ((m = line.match(/^\s*export\s+(?:const|let|var)\s+([A-Z][\w$]*)\s*=\s*(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>/))) {
          push(m[1], "component", idx); // PascalCase arrow → likely a component
        } else if ((m = line.match(/^\s*export\s+(?:const|let|var)\s+([a-z_$][\w$]*)\s*=\s*(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>/))) {
          push(m[1], "function", idx);
        } else if ((m = line.match(/^\s*export\s+interface\s+([A-Za-z_$][\w$]*)/))) {
          push(m[1], "interface", idx);
        } else if ((m = line.match(/^\s*export\s+type\s+([A-Za-z_$][\w$]*)/))) {
          push(m[1], "type_alias", idx);
        } else if ((m = line.match(/^\s*export\s+enum\s+([A-Za-z_$][\w$]*)/))) {
          push(m[1], "enum", idx);
        }
      } else if (isPy) {
        let m;
        if ((m = line.match(/^\s*class\s+([A-Za-z_]\w*)/))) {
          push(m[1], "class", idx);
        } else if ((m = line.match(/^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)/))) {
          const isMethod = /^\s{2,}/.test(line);
          push(m[1], isMethod ? "method" : "function", idx);
        }
      }
    });

    return found;
  }

  // ------------------------------------------------------------- imports
  function extractImportTargets(path, content, language) {
    if (!content) return [];
    const targets = [];
    if (language === "javascript" || language === "typescript") {
      const re = /(?:import\s+(?:[\s\S]*?)\s+from\s+|import\s+|require\()\s*["'`]([^"'`]+)["'`]/g;
      let m;
      while ((m = re.exec(content)) !== null) targets.push(m[1]);
    } else if (language === "python") {
      const re1 = /^\s*from\s+([.\w]+)\s+import/gm;
      const re2 = /^\s*import\s+([.\w]+)/gm;
      let m;
      while ((m = re1.exec(content)) !== null) targets.push(m[1]);
      while ((m = re2.exec(content)) !== null) targets.push(m[1]);
    }
    return targets;
  }

  function resolvePathTarget(fromPath, target, pathSet) {
    if (!target) return null;
    const dir = fromPath.includes("/") ? fromPath.slice(0, fromPath.lastIndexOf("/")) : "";

    if (target.startsWith(".")) {
      // relative JS/TS import
      const rawParts = (dir ? dir + "/" + target : target).split("/");
      const stack = [];
      for (const part of rawParts) {
        if (part === "" || part === ".") continue;
        if (part === "..") stack.pop();
        else stack.push(part);
      }
      const base = stack.join("/");
      const candidates = [
        base, `${base}.js`, `${base}.jsx`, `${base}.ts`, `${base}.tsx`,
        `${base}/index.js`, `${base}/index.jsx`, `${base}/index.ts`, `${base}/index.tsx`,
      ];
      return candidates.find(c => pathSet.has(c)) || null;
    }

    // python dotted module path, relative "." prefix already handled loosely
    if (target.includes(".") && !target.includes("/")) {
      const asPath = target.replace(/\./g, "/") + ".py";
      if (pathSet.has(asPath)) return asPath;
      const initPath = target.replace(/\./g, "/") + "/__init__.py";
      if (pathSet.has(initPath)) return initPath;
    }
    return null; // external package — not graphed
  }

  // ------------------------------------------------------------ graph build
  function buildGraphFromFiles(files) {
    const usable = files.filter(f => f.content).slice(0, MAX_GRAPH_FILES);
    const pathSet = new Set(usable.map(f => f.path));

    const nodes = [];
    const edges = [];
    const edgeSeen = new Set();

    const addEdge = (source, target, relationship) => {
      if (!source || !target || source === target) return;
      const key = `${source}|${target}|${relationship}`;
      if (edgeSeen.has(key)) return;
      edgeSeen.add(key);
      edges.push({ source, target, relationship });
    };

    for (const f of usable) {
      const language = f.language || detectLanguage(f.path);
      const category = fileCategory(f.path, language);
      const totalLines = f.content.split("\n").length;

      nodes.push({
        id: f.path, name: f.path.split("/").pop(), category,
        file: f.path, line: 1, lineEnd: totalLines,
      });

      // symbols within the file
      const symbols = extractSymbols(f.path, f.content, language);
      for (const sym of symbols) {
        const symId = `${f.path}::${sym.name}`;
        nodes.push({
          id: symId, name: sym.name, category: sym.category,
          file: f.path, line: sym.line, lineEnd: sym.line,
        });
        addEdge(f.path, symId, "defines");
      }

      // import edges (file → file)
      const targets = extractImportTargets(f.path, f.content, language);
      for (const t of targets) {
        const resolved = resolvePathTarget(f.path, t, pathSet);
        if (resolved) addEdge(f.path, resolved, "imports");
      }
    }

    return {
      nodes: nodes.slice(0, MAX_NODES),
      edges: edges.slice(0, MAX_EDGES),
    };
  }

  // --------------------------------------------------------- context format
  function formatContextText(nodes, edges, meta) {
    const header = [
      `# GitHub Repository Context`,
      `Repo: ${meta.repo}`,
      `Branch: ${meta.branch}`,
      `Files analyzed: ${meta.fileCount} of ${meta.totalFiles} (skipped ${meta.skipped})`,
      `Nodes: ${nodes.length}  Edges: ${edges.length}`,
      ``,
    ].join("\n");

    const nodeIds = new Set(nodes.map(n => n.id));
    const blocks = nodes.map(n => {
      const lineStr = n.line === n.lineEnd ? `lines ${n.line}` : `lines ${n.line}-${n.lineEnd}`;
      return [
        `Module {${n.id}}`,
        `File: ${n.file || "None"}`,
        `${n.name} (${n.category}) — ${lineStr}`,
      ].join("\n");
    });

    const relLines = edges
      .filter(e => nodeIds.has(e.source) && nodeIds.has(e.target))
      .map(e => `Relationship: ${e.source} -> ${e.target} (${e.relationship})`);

    return (
      header +
      blocks.join("\n\\\n") +
      (relLines.length ? `\n\n${relLines.join("\n")}\n` : "\n")
    );
  }

  // -------------------------------------------------------------- analysis
  function computeDegreeMap(nodes, edges) {
    const deg = new Map(nodes.map(n => [n.id, 0]));
    edges.forEach(e => {
      if (deg.has(e.source)) deg.set(e.source, deg.get(e.source) + 1);
      if (deg.has(e.target)) deg.set(e.target, deg.get(e.target) + 1);
    });
    return deg;
  }

  // Force-directed layout with a seeded PRNG (ported from DiagramStudio's
  // layout.js so the extension's graph tab feels the same).
  function computeLayout(nodes, edges, seed, W, H) {
    W = W || 760; H = H || 480;
    if (!nodes.length) return {};

    let s = seed || 42;
    const rng = () => { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (s >>> 0) / 0xffffffff; };

    const pos = {};
    nodes.forEach((n, i) => {
      const angle = (i / nodes.length) * Math.PI * 2;
      const r = 100 + rng() * 80;
      pos[n.id] = { x: W / 2 + Math.cos(angle) * r, y: H / 2 + Math.sin(angle) * r };
    });

    const nodeIds = new Set(nodes.map(n => n.id));
    const iters = nodes.length < 20 ? 200 : nodes.length < 50 ? 150 : 80;

    for (let iter = 0; iter < iters; iter++) {
      const forces = {};
      nodes.forEach(n => { forces[n.id] = { x: 0, y: 0 }; });

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const dx = pos[b.id].x - pos[a.id].x, dy = pos[b.id].y - pos[a.id].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const f = Math.min(8000 / (dist * dist), 25);
          const fx = (dx / dist) * f, fy = (dy / dist) * f;
          forces[a.id].x -= fx; forces[a.id].y -= fy;
          forces[b.id].x += fx; forces[b.id].y += fy;
        }
      }

      edges.forEach(e => {
        if (!nodeIds.has(e.source) || !nodeIds.has(e.target)) return;
        const dx = pos[e.target].x - pos[e.source].x, dy = pos[e.target].y - pos[e.source].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = (dist - 50) * 0.04;
        const fx = (dx / dist) * f, fy = (dy / dist) * f;
        forces[e.source].x += fx; forces[e.source].y += fy;
        forces[e.target].x -= fx; forces[e.target].y -= fy;
      });

      nodes.forEach(n => {
        forces[n.id].x += (W / 2 - pos[n.id].x) * 0.008;
        forces[n.id].y += (H / 2 - pos[n.id].y) * 0.008;
      });

      const cooling = 1 - iter / iters;
      nodes.forEach(n => {
        pos[n.id].x += forces[n.id].x * cooling;
        pos[n.id].y += forces[n.id].y * cooling;
      });
    }
    return pos;
  }

  // --------------------------------------------------------------- exports
  global.RecallRepoGraph = {
    CATEGORY_COLORS,
    IGNORE_DIRS, IGNORE_EXTENSIONS, CODE_EXTENSIONS,
    MAX_FILE_SIZE, MAX_GRAPH_FILES,
    shouldIncludePath, detectLanguage, fileCategory,
    buildGraphFromFiles, formatContextText,
    computeDegreeMap, computeLayout, trunc,
  };

})(typeof window !== "undefined" ? window : self);
