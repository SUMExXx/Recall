# Recall — Browser Extension

This is the part of Recall that lives in your browser. It lets you grab
anything you're looking at — a highlighted sentence, a whole PDF, a GitHub
repo, even a screenshot — and send it straight into your Recall memory, the
same memory your voice and your phone feed into.

Nothing you save here goes to the cloud. It's sent to your own Recall hub
(the program running on your PC), and that's the only place it goes.

## What it can do

**Save any highlighted text.** Select a sentence or paragraph on any page and
a small "Save to Recall" button pops up next to it. Click it, edit the text
if you want to tidy it up, and save. You can also right-click a selection and
choose "Save to Recall" to save it instantly, no editing step.

If you tick "Include page as source" in the save panel, Recall remembers
which page the text came from, so you can trace it back later.

**Save a whole GitHub repo.** Visit any repo page on github.com and a "Save
repo to Recall" button appears. Recall reads through the repo's files, builds
a map of how they connect (which files define what, which files import from
which), and shows it to you as an interactive diagram — drag it, zoom it,
click a file to see its neighbors. You can also copy or send the whole thing
into memory as plain text, so you can later ask Recall questions like "how
does this repo handle authentication?"

Private repos need a personal GitHub token, which you can add in Settings.

**Save a PDF.** Open a PDF and click "Save this PDF" in the extension popup.
Recall pulls out the text, shows it to you in a review window so you can
check it looks right (and edit it if needed), and only sends it once you
click "Send to Recall." If the PDF is a scanned image with no real text in
it, Recall will tell you and suggest using screenshot capture instead.

**Save a screenshot.** Recall can read text out of an image using OCR (optical
character recognition) — handy for scanned documents, photos of whiteboards,
or anything else that isn't selectable text.

## Setting it up

1. Make sure the Recall hub is running on your PC (see the PC README for
   that part).
2. In Chrome or Edge, go to `chrome://extensions`, turn on **Developer mode**
   in the top right, then click **Load unpacked** and select this
   `extension` folder.
3. Click the Recall icon in your toolbar, open **Settings**, and enter your
   hub's address. If the hub is running on the same computer, the default
   `http://localhost:8000` is correct. If it's running on a different
   computer on the same network (like at a demo booth), use that computer's
   network address instead, e.g. `http://192.168.1.42:8000`.
4. Click **Save & test connection**. You should see "Connected to the hub."

Once that's done, you're set — highlight some text on any page and try it out.

## Checking it worked

Open your hub's dashboard in a browser (the same address you set above). New
memories show up there within a couple of seconds, along with a little amber
light that pulses while Recall is processing what you just sent.

## A note on privacy

The extension only ever talks to your own Recall hub — never to any outside
server. The one exception is GitHub itself, which the extension talks to
directly in order to read public repo files (this is the same as visiting
the repo in your browser). Everything you capture stays on your device
unless you've explicitly turned on cloud features in your Recall settings.
