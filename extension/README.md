# Recall — Browser Extension

Select text on any webpage → edit it if you want → save it as a Recall
memory. This is a **third capture path** alongside the phone mic and the
`/demo/inject` backdoor: same hub, same pipeline (embed → store → amber LED
→ dashboard), just triggered from a browser selection instead of speech.

## How it works

- **content.js** watches for a text selection on the page and shows a small
  "Save to Recall" pill next to it. Clicking it opens an editable panel
  prefilled with the *exact* selected text — you can save it as-is or edit
  it first.
- Right-click a selection also works, via the "Save to Recall" context menu
  item (saves the exact selection immediately, no edit step).
- **background.js** is the only piece that talks to the network — it POSTs
  `{"text": "..."}` to `POST {hubUrl}/demo/inject` on your PC hub. That
  endpoint runs the exact same path as a live spoken chunk, so it triggers
  the amber "encoding" LED and shows up on the dashboard immediately.
- If "Include page as source" is checked, the saved text gets the page
  title + URL appended, so it's traceable later:
  `"...selected text..." — from "Page Title" (https://...)`.
- The toolbar **popup** shows hub connection status and has a manual
  paste-to-save box for anything not on a webpage.
- **Options page** sets the hub's address — default `http://localhost:8000`,
  but change it to the PC's LAN IP (e.g. `http://192.168.1.42:8000`) for the
  demo hotspot setup, matching how the phone app is configured.

## Install (unpacked, for the hackathon)

1. Start the hub: `cd server && .venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000`
2. Chrome/Edge → `chrome://extensions` → enable **Developer mode** →
   **Load unpacked** → select this `extension/` folder.
3. Click the Recall icon → **Settings** → set the hub URL → **Save & test
   connection**. You should see "Connected to the hub ✓".
4. Go to any webpage, select some text — the "Save to Recall" pill appears.
   Click it, edit if you want, hit **Save memory**.
5. Check `http://<hub>:8000` dashboard — the new memory (and its amber LED
   pulse) should appear within a couple seconds.

## GitHub repo → graph + memory

On any `github.com/owner/repo` page, a **"Save repo to Recall"** button
appears bottom-right (also reachable via the popup's "Save GitHub repo"
button, or right-click → "Save this GitHub repo to Recall"). Clicking it:

1. **background.js** pulls the repo's file tree + contents straight from
   the GitHub REST API (`api.github.com`), trying `<current tree branch>` →
   `main` → `master` → `develop`. Binary/huge/vendored files are filtered
   out (`node_modules`, `.git`, lockfiles, images, etc.).
2. **repoGraph.js** turns those files into a graph: one node per file
   (categorized as `module`/`component`/`next_page_module`/… like
   DiagramStudio), plus lightweight regex-extracted symbol nodes
   (`class`/`function`/`method`/`interface`/`type_alias`/`enum`) linked to
   their file via `defines` edges, and `imports` edges resolved between
   files. This is a regex heuristic, not a full AST parse — good enough for
   structure, not a drop-in replacement for gitvizz.
3. The same module formats that graph into an LLM-ready **context text**
   (`Module {id} / File: … / name (category) — lines …`, `Relationship: src
   -> tgt (relationship)`), matching DiagramStudio's `parseContext.js`
   grammar so it's portable if you ever want to inspect it there too.
4. A dialog opens with two tabs:
   - **Graph** — force-directed node-link view (drag to pan, scroll to
     zoom, click a node to highlight its neighbors + see its file/category).
   - **Context text** — the raw text, with a **Copy** button and a
     **Send to memory** button that POSTs it (chunked, same path as every
     other capture) to `{hubUrl}/demo/inject`.

Private repos or heavier rate limits need a personal access token — add
one in **Settings → GitHub token (optional)**.

## Screenshot → OCR → memory

Same underlying bug as the PDF path, just in a different spot: `offscreen.js`
used to load Tesseract.js from a CDN via `<script src="https://cdnjs...">`.
Extension pages block that by default CSP (`script-src 'self'`), so the
`<script>` tag silently failed to load and every OCR attempt just sat there
until the 30s timeout in `background.js` fired and reported "OCR failed."

Fixed the same way as the PDF library: `tesseract.min.js`, `worker.min.js`,
and the LSTM-only wasm cores (with and without SIMD) are bundled locally at
`lib/tesseract/`, and `createWorker()` is pointed at those local paths
(`workerPath`, `workerBlobURL: false`, `corePath`) instead of the CDN
defaults. The one thing still fetched remotely is the English trained-data
file (`langPath` left at its default, pulled from jsdelivr on first use) —
that's a plain data download from inside the worker, not a script/worker
load, so it isn't affected by the CSP restriction and keeping it remote
avoids bundling multiple megabytes of trained-data into the extension.

## PDF → memory

Open a PDF, click the popup's **"Save this PDF"** button:

1. **background.js** tries the fast path first: it injects `pdf-content.js`
   into the tab, which checks whether the page already has a rendered
   `.textLayer` (true for pdf.js-flavored viewers, e.g. Firefox's built-in
   one). If found, that text is used immediately.
2. Otherwise — which is the normal case in Chrome, since its built-in
   PDFium viewer renders the PDF in a separate guest view a content script
   can't see into — background.js downloads the PDF bytes itself (its
   `host_permissions` let it fetch cross-origin without hitting page CORS)
   and hands them to the **offscreen document**, which parses them with a
   copy of **pdf.js bundled locally** at `lib/pdfjs/`. It's bundled rather
   than loaded from a CDN because extension pages enforce a default CSP of
   `script-src 'self'`, which silently blocks remote `<script src>` tags —
   loading it from inside the extension package sidesteps that entirely.
3. The extracted text is **not** sent anywhere yet. It's handed back to
   `content.js`, which opens an in-page **review popup** showing exactly
   that text (character count included) so you can read/edit it before
   anything is sent. Only clicking **"Send to Recall"** there actually
   chunks it and POSTs it to `/demo/inject`, same as any other capture path.
4. If extraction genuinely fails (e.g. a scanned/image-only PDF with no
   text layer at all), the popup reports that and suggests Screenshot OCR
   instead.

## Notes for the team

- No backend changes were needed — this uses the existing `/demo/inject`
  endpoint, so it's safe to merge without touching `server/`.
- If you'd rather store the source URL as real metadata instead of folded
  into the text (e.g. for a clickable "source" link on the dashboard), that
  needs a small `memory_store.py` change to accept a `source` field
  alongside `text`/`session` — happy to pair on that if the dashboard wants
  it, but the current approach needs zero API changes.
- Network calls only happen from `background.js` (not the content script),
  since MV3 content scripts are still subject to page CORS/CSP while the
  background service worker isn't, given the `host_permissions` grant.
