/* Voice enrollment — static recorder. Uploads to /api/voices/upload, then
   POSTs /api/voices/process to segment + register for training approval. */
(function () {
  "use strict";

  let scripts = null;
  let lang = "en";
  let session = "";
  let takes = []; // {blob, url, done}

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function token() { return localStorage.getItem("userToken"); }

  async function main() {
    try {
      const r = await fetch("scripts.json", { cache: "no-store" });
      scripts = await r.json();
    } catch (e) {
      document.getElementById("gateMsg").textContent =
        "Could not load scripts.json (" + e.message + "). Serve over HTTP.";
      return;
    }
    if (!token()) {
      document.getElementById("gateMsg").innerHTML =
        'Login required: <a href="/ui/auth/auth.html">sign in</a> first, then come back.';
      return;
    }
    document.getElementById("gateSection").classList.add("hidden");
    document.getElementById("setupSection").classList.remove("hidden");
    document.getElementById("startBtn").addEventListener("click", startSession);
    document.getElementById("langSelect").addEventListener("change", (e) => { lang = e.target.value; });
  }

  function startSession() {
    session = document.getElementById("sessionInput").value.trim().replace(/[^\w\-]+/g, "-") || "take1";
    takes = scripts[lang].map(() => ({ blob: null, url: null, rec: null, done: false }));
    renderList();
    document.getElementById("recordSection").classList.remove("hidden");
    updateProg();
  }

  function updateProg() {
    const n = takes.filter((t) => t.done).length;
    document.getElementById("progPill").textContent = n + " / " + takes.length;
    document.getElementById("submitBtn").disabled = n !== takes.length;
  }

  function renderList() {
    const host = document.getElementById("sentList");
    host.innerHTML = "";
    scripts[lang].forEach((text, i) => {
      const card = el("div", "lab-card");
      card.appendChild(el("h4", null, (i + 1) + ". " + text));
      const row = el("p", "lab-meta");
      const recBtn = el("button", "btn-small", "● Record");
      const playBtn = el("button", "btn-small", "▶ Play");
      playBtn.disabled = true;
      const st = el("span", "pill idle", "empty");
      recBtn.addEventListener("click", () => toggleRec(i, recBtn, playBtn, st));
      playBtn.addEventListener("click", () => { new Audio(takes[i].url).play(); });
      row.appendChild(recBtn); row.appendChild(document.createTextNode(" "));
      row.appendChild(playBtn); row.appendChild(document.createTextNode(" "));
      row.appendChild(st);
      card.appendChild(row);
      host.appendChild(card);
    });
    document.getElementById("submitBtn").addEventListener("click", submitAll, { once: true });
  }

  async function toggleRec(i, recBtn, playBtn, st) {
    const t = takes[i];
    if (t.rec) {
      t.rec.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "" });
      const chunks = [];
      rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      rec.onstop = () => {
        stream.getTracks().forEach((tr) => tr.stop());
        if (t.url) URL.revokeObjectURL(t.url);
        t.blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
        t.url = URL.createObjectURL(t.blob);
        t.done = true; t.rec = null;
        recBtn.textContent = "● Re-record";
        playBtn.disabled = false;
        st.textContent = "recorded ✓"; st.className = "pill pass";
        updateProg();
      };
      t.rec = rec;
      rec.start();
      recBtn.textContent = "■ Stop";
      st.textContent = "recording…"; st.className = "pill idle";
    } catch (e) {
      st.textContent = "mic blocked: " + e.message;
      st.className = "pill fail";
    }
  }

  async function submitAll() {
    const msg = document.getElementById("submitMsg");
    msg.textContent = "Uploading 0/" + takes.length + "…";
    const headers = { Authorization: "Bearer " + token() };
    for (let i = 0; i < takes.length; i++) {
      const form = new FormData();
      form.append("file", takes[i].blob, session + "_" + String(i + 1).padStart(2, "0") + ".webm");
      form.append("voice_name", session + "_" + String(i + 1).padStart(2, "0"));
      form.append("speaker_lang", lang);
      const r = await fetch("/api/voices/upload", { method: "POST", headers, body: form });
      if (!r.ok) {
        msg.textContent = "Upload failed at sentence " + (i + 1) + ": HTTP " + r.status +
          " — fix and resubmit (recorded takes are kept).";
        document.getElementById("submitBtn").addEventListener("click", submitAll, { once: true });
        return;
      }
      msg.textContent = "Uploading " + (i + 1) + "/" + takes.length + "…";
    }
    msg.textContent = "Processing…";
    const pr = await fetch("/api/voices/process", {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, headers),
      body: JSON.stringify({
        session, speaker_lang: lang,
        items: scripts[lang].map((text, i) => ({ idx: i + 1, text })),
      }),
    });
    const data = await pr.json().catch(() => ({}));
    const pre = document.getElementById("donePre");
    if (pr.ok) {
      pre.textContent = "Clips: " + data.clips + " (" + data.seconds + "s) → " + data.corpus_dir +
        "\nStatus: READY FOR TRAINING APPROVAL (owner starts the run) — nothing trains by itself.";
    } else {
      pre.textContent = "Processing failed: " + (data.detail || ("HTTP " + pr.status));
    }
    document.getElementById("doneSection").classList.remove("hidden");
    msg.textContent = "Done.";
  }

  document.addEventListener("DOMContentLoaded", main);
})();
