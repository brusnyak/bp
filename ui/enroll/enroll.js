/* Voice enrollment — static recorder. Uploads to /api/voices/upload, then
   POSTs /api/voices/process to segment + register for training approval. */
(function () {
  "use strict";

  let scripts = null;
  let lang = "en";
  let session = "";
  let uploading = false;
  let recordingLang = null;
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
        'Sign in required: <a href="/ui/auth/auth.html?next=/ui/enroll/enroll.html">sign in and return to voice preparation</a>.';
      return;
    }
    document.getElementById("gateSection").classList.add("hidden");
    document.getElementById("setupSection").classList.remove("hidden");
    document.getElementById("startBtn").addEventListener("click", startSession);
    document.getElementById("langSelect").addEventListener("change", (e) => { lang = e.target.value; });
  }

  function startSession() {
    if (uploading) return;
    if (takes.some(t => t.done) && !window.confirm('Start a new set? Unsaved recordings in this tab will be discarded.')) return;
    for (const take of takes) { if (take.rec) take.rec.stop(); if (take.url) URL.revokeObjectURL(take.url); }
    session = document.getElementById("sessionInput").value.trim().replace(/[^\w\-]+/g, "-") || "take1";
    recordingLang = lang;
    takes = scripts[recordingLang].map(() => ({ blob: null, url: null, rec: null, done: false }));
    renderList();
    document.getElementById("recordSection").classList.remove("hidden");
    updateProg();
  }

  function updateProg() {
    const n = takes.filter((t) => t.done).length;
    document.getElementById("progPill").textContent = n + " / " + takes.length;
    document.getElementById("submitBtn").disabled = uploading || n !== takes.length;
  }

  function renderList() {
    const host = document.getElementById("sentList");
    host.innerHTML = "";
    scripts[recordingLang].forEach((text, i) => {
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
    document.getElementById("submitBtn").onclick = submitAll;
  }

  async function toggleRec(i, recBtn, playBtn, st) {
    if (uploading) return;
    const t = takes[i];
    if (t.rec) {
      t.rec.stop();
      return;
    }
    let activeStream = null;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      activeStream = stream;
      const rec = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "" });
      const chunks = [];
      rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      rec.onstop = () => {
        stream.getTracks().forEach((tr) => tr.stop());
        if (t.url) URL.revokeObjectURL(t.url);
        t.blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
        t.url = URL.createObjectURL(t.blob);
        t.done = true; t.rec = null; t.uploaded = false;
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
      if (activeStream) activeStream.getTracks().forEach(track => track.stop());
      st.textContent = "mic blocked: " + e.message;
      st.className = "pill fail";
    }
  }

  async function submitAll() {
    if (uploading || takes.some(t => !t.done)) return;
    uploading = true; updateProg();
    document.getElementById('langSelect').disabled = true;
    document.getElementById('startBtn').disabled = true;
    const msg = document.getElementById("submitMsg");
    const pre = document.getElementById("donePre");
    const headers = { Authorization: "Bearer " + token() };
    try {
      for (let i = 0; i < takes.length; i++) {
        if (takes[i].uploaded) continue;
        msg.textContent = "Uploading sentence " + (i + 1) + "/" + takes.length + "…";
        const form = new FormData();
        form.append("file", takes[i].blob, session + "_" + String(i + 1).padStart(2, "0") + ".webm");
        form.append("voice_name", session + "_" + String(i + 1).padStart(2, "0"));
        form.append("speaker_lang", recordingLang);
        const response = await fetch("/api/voices/upload", { method: "POST", headers, body: form });
        if (!response.ok) throw new Error("Sentence " + (i + 1) + ": HTTP " + response.status);
        takes[i].uploaded = true;
      }
      msg.textContent = "Processing uploaded recordings…";
      const response = await fetch("/api/voices/process", {
        method: "POST", headers: Object.assign({ "Content-Type": "application/json" }, headers),
        body: JSON.stringify({ session, speaker_lang: recordingLang, items: scripts[recordingLang].map((text, i) => ({ idx: i + 1, text })) })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || "Processing HTTP " + response.status);
      pre.textContent = "Clips: " + data.clips + " (" + data.seconds + "s). Recorded and processed; not a trained or qualified clone. Training requires approval, followed by listening checks on target hardware.";
      document.getElementById("doneSection").classList.remove("hidden");
      msg.textContent = "Processing complete. Voice training and qualification remain.";
    } catch (error) {
      msg.textContent = "Not complete: " + error.message + ". Recordings remain in this tab. Retry resumes acknowledged uploads; an upload whose response was lost may need checking on the server.";
    } finally { uploading = false; updateProg(); document.getElementById('langSelect').disabled = false; document.getElementById('startBtn').disabled = false; }
  }
  window.addEventListener('beforeunload', event => {
    if (uploading || takes.some(t => t.done)) { event.preventDefault(); event.returnValue = ''; }
  });

  document.addEventListener("DOMContentLoaded", main);
})();
