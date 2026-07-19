const hubInput   = document.getElementById("hub-url");
const tokenInput = document.getElementById("github-token");
const resultEl   = document.getElementById("result");
const saveBtn    = document.getElementById("save");

async function load() {
  const { hubUrl, githubToken } = await chrome.storage.sync.get(["hubUrl", "githubToken"]);
  hubInput.value   = hubUrl   || "http://localhost:8000";
  tokenInput.value = githubToken || "";
}

saveBtn.addEventListener("click", async () => {
  let url = hubInput.value.trim().replace(/\/+$/, "");
  if (!/^https?:\/\//.test(url)) url = `http://${url}`;

  await chrome.storage.sync.set({
    hubUrl: url,
    githubToken: tokenInput.value.trim(),
  });

  resultEl.textContent = "testing connection…";
  resultEl.className = "testing";
  saveBtn.disabled = true;

  try {
    const res = await fetch(`${url}/state`);
    if (!res.ok) throw new Error(`hub returned ${res.status}`);
    resultEl.textContent = "saved · connected to hub ✓";
    resultEl.className = "ok";
  } catch (err) {
    resultEl.textContent = "saved, but can't reach hub: " + err.message;
    resultEl.className = "err";
  } finally {
    saveBtn.disabled = false;
  }
});

load();