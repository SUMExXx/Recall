const input = document.getElementById("hub-url");
const githubTokenInput = document.getElementById("github-token");
const resultEl = document.getElementById("result");

async function load() {
  const { hubUrl, githubToken } = await chrome.storage.sync.get(["hubUrl", "githubToken"]);
  input.value = hubUrl || "http://localhost:8000";
  githubTokenInput.value = githubToken || "";
}

document.getElementById("save").addEventListener("click", async () => {
  let url = input.value.trim().replace(/\/+$/, "");
  if (!/^https?:\/\//.test(url)) url = `http://${url}`;
  await chrome.storage.sync.set({
    hubUrl: url,
    githubToken: githubTokenInput.value.trim(),
  });

  resultEl.textContent = "Testing connection…";
  resultEl.className = "";
  try {
    const res = await fetch(`${url}/state`);
    if (!res.ok) throw new Error(`Hub returned ${res.status}`);
    resultEl.textContent = `Saved. Connected to the hub ✓`;
    resultEl.className = "ok";
  } catch (err) {
    resultEl.textContent = `Saved, but couldn't reach the hub: ${err.message}`;
    resultEl.className = "err";
  }
});

load();
