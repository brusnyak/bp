/* Voice Lab — static page. Reads library.json only. No backend, no auth. */
(function () {
  "use strict";

  const THRESHOLD = 0.84; // resemblyzer demo reference point (see scripts/voice_similarity_qc.py)

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function audioEl(src) {
    const a = document.createElement("audio");
    a.controls = true;
    a.preload = "metadata";
    a.src = encodeURI(src); // paths contain spaces (e.g. "2_my voice 1.wav"); blob: URLs survive encodeURI
    return a;
  }

  function pill(text, kind) {
    return el("span", "pill " + kind, text);
  }

  function getRatings() {
    try { return JSON.parse(localStorage.getItem("hlas-ratings") || "{}"); }
    catch (e) { return {}; }
  }
  function saveRating(name, patch) {
    const all = getRatings();
    all[name] = Object.assign(all[name] || {}, patch);
    localStorage.setItem("hlas-ratings", JSON.stringify(all));
  }

  function ratingRow(item) {
    const wrap = el("div", "lab-rate");
    const saved = getRatings()[item.name] || {};
    wrap.appendChild(document.createTextNode("sounds like me: "));
    const slider = document.createElement("input");
    slider.type = "range"; slider.min = "0"; slider.max = "100";
    slider.value = saved.sim !== undefined ? saved.sim : "50";
    slider.setAttribute("aria-label", "Similarity to my voice for " + item.name);
    const val = el("strong", null, String(slider.value));
    slider.addEventListener("input", () => {
      val.textContent = slider.value;
      saveRating(item.name, { sim: Number(slider.value) });
    });
    wrap.appendChild(slider); wrap.appendChild(document.createTextNode(" "));
    wrap.appendChild(val);
    [["steadiness", "tremor"], ["hiss", "hiss"], ["muffled", "phone-fog"]].forEach(([key, label]) => {
      wrap.appendChild(document.createTextNode(" " + label + " "));
      const sel = document.createElement("select");
      sel.setAttribute("aria-label", label + " for " + item.name);
      ["?", "1-none", "2-slight", "3-clear", "4-strong"].forEach((o, i) => {
        const opt = document.createElement("option");
        opt.value = String(i); opt.textContent = o;
        sel.appendChild(opt);
      });
      sel.value = String(saved[key] !== undefined ? saved[key] : 0);
      sel.addEventListener("change", () => saveRating(item.name, { [key]: Number(sel.value) }));
      wrap.appendChild(sel);
    });
    return wrap;
  }

  function metaLine(parts) {
    return el("p", "lab-meta", parts.filter(Boolean).join(" · "));
  }

  function itemCard(item, sectionId) {
    const card = el("div", "lab-card");
    const top = el("div", "lab-card-top");
    top.appendChild(el("h4", null, item.name));
    if (sectionId === "qc") {
      if (item.similarity === undefined || item.similarity === null) {
        top.appendChild(pill("not scored", "idle"));
      } else {
        top.appendChild(pill(
          item.similarity >= THRESHOLD ? "✓ pass" : "✗ below",
          item.similarity >= THRESHOLD ? "pass" : "fail"));
      }
    } else if (sectionId === "voices") {
      top.appendChild(pill(
        item.in_registry ? "in registry" : "not registered",
        item.in_registry ? "pass" : "fail"));
    }
    card.appendChild(top);
    if (sectionId === "voices") {
      card.appendChild(metaLine([
        "language: " + (item.language || "?"),
        "file: " + item.file,
      ]));
      if (item.transcript) card.appendChild(metaLine(["transcript: " + item.transcript]));
    } else if (sectionId === "qc") {
      card.appendChild(metaLine([
        "engine: " + (item.engine || "?"),
        "lang: " + (item.language || "?"),
        item.sample_rate ? item.sample_rate + " Hz" : null,
        item.synthesis_latency_s !== undefined && item.synthesis_latency_s !== null
          ? "synth: " + item.synthesis_latency_s + "s" : null,
        item.similarity !== undefined && item.similarity !== null
          ? "similarity: " + item.similarity.toFixed(4) + " (ref " + THRESHOLD + ")" : null,
      ]));
    } else if (sectionId === "test") {
      card.appendChild(metaLine(["file: " + item.file]));
      if (item.transcript) card.appendChild(metaLine(["ground truth: " + item.transcript]));
    } else if (sectionId === "sk_direction") {
      card.appendChild(metaLine([
        "audio: " + (item.audio_s ? item.audio_s + "s" : "?"),
        "SK ref chars: " + (item.ref_chars || "?"),
        "EN ref chars: " + (item.en_ref_chars || "?"),
        "file: " + item.file,
      ]));
      if (item.rungs && Object.keys(item.rungs).length) {
        const tbl = el("table", "matrix-table");
        const thead = el("thead");
        const trh = el("tr");
        ["Rung", "WER", "CER", "MT chrF", "STT Latency", "RTF", "MT Latency"].forEach(h => {
          trh.appendChild(el("th", null, h));
        });
        thead.appendChild(trh);
        tbl.appendChild(thead);
        const tbody = el("tbody");
        for (const [rungName, r] of Object.entries(item.rungs)) {
          const tr = el("tr");
          tr.appendChild(el("td", null, rungName));
          const werTd = el("td", null, (r.wer * 100).toFixed(1) + "%");
          if (r.wer < 0.3) werTd.className = "pass";
          else if (r.wer > 0.5) werTd.className = "fail";
          tr.appendChild(werTd);
          tr.appendChild(el("td", null, (r.cer * 100).toFixed(1) + "%"));
          const chrfTd = el("td", null, String(r.chrf));
          if (r.chrf >= 60) chrfTd.className = "pass";
          tr.appendChild(chrfTd);
          tr.appendChild(el("td", null, r.stt_s + "s"));
          tr.appendChild(el("td", null, String(r.rtf)));
          tr.appendChild(el("td", null, r.mt_s + "s"));
          tbody.appendChild(tr);
        }
        tbl.appendChild(tbody);
        card.appendChild(tbl);
      }
    }
    if (item.meta && Object.keys(item.meta).length) {
      card.appendChild(metaLine(
        Object.entries(item.meta).map(([k, v]) => k + ": " + v)));
    }
    card.appendChild(audioEl(item.file));
    if (sectionId === "qc") card.appendChild(ratingRow(item));
    return card;
  }

  function renderStats(library) {
    const strip = document.getElementById("statStrip");
    const counts = {};
    library.sections.forEach((s) => { counts[s.id] = s.items.length; });
    const scored = (library.sections.find((s) => s.id === "qc") || { items: [] }).items
      .filter((i) => i.similarity !== undefined && i.similarity !== null).length;
    [
      ["Voices", counts.voices || 0],
      ["QC candidates", counts.qc || 0],
      ["SK matrix", counts.sk_direction || 0],
      ["Scored", scored],
      ["Test clips", counts.test || 0],
    ].forEach(([label, n]) => {
      const chip = el("span", "lab-stat", label);
      chip.prepend(el("strong", null, String(n)));
      strip.appendChild(chip);
    });
  }

  function renderLibrary(library) {
    renderStats(library);
    const host = document.getElementById("librarySections");
    library.sections.forEach((section) => {
      const s = document.createElement("section");
      s.appendChild(el("h2", null, section.title + " (" + section.items.length + ")"));
      if (!section.items.length) {
        s.appendChild(el("p", "lab-hint", "Empty."));
      }
      section.items.forEach((item) => s.appendChild(itemCard(item, section.id)));
      host.appendChild(s);
    });
  }

  // Backend-aware upload: if the FastAPI backend answers, staged files can be
  // REALLY uploaded (POST /api/voices/upload) instead of just previewed.
  // Needs a login token (localStorage.userToken from /ui/auth/auth.html);
  // without one the server 401s and we say so plainly.
  let backendUp = false;
  async function probeBackend() {
    try {
      const ctl = new AbortController();
      const t = setTimeout(() => ctl.abort(), 3000);
      const r = await fetch("/api/auth/config", { signal: ctl.signal });
      clearTimeout(t);
      backendUp = r.ok;
    } catch (e) {
      backendUp = false;
    }
    const hint = document.getElementById("backendHint");
    if (hint) {
      hint.textContent = backendUp
        ? "Backend reachable: staged files can be uploaded for real (needs login)."
        : "Backend not reachable: staging only (make run to enable real upload).";
    }
  }

  async function uploadStaged(file, lang, msgEl) {
    const token = localStorage.getItem("userToken");
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("voice_name", file.name.replace(/\.[^.]+$/, ""));
    form.append("speaker_lang", lang);
    try {
      const r = await fetch("/api/voices/upload", {
        method: "POST",
        headers: token ? { Authorization: "Bearer " + token } : {},
        body: form,
      });
      const data = await r.json().catch(() => ({}));
      if (r.ok) {
        msgEl.textContent = "Uploaded ✓ — refresh the manifest: python3 scripts/update_voice_lab_library.py";
        msgEl.className = "lab-meta pass";
      } else if (r.status === 401 || r.status === 403) {
        msgEl.textContent = "Login required: sign in at /ui/auth/auth.html first, then upload again.";
        msgEl.className = "lab-meta fail";
      } else {
        msgEl.textContent = "Upload failed: " + (data.detail || ("HTTP " + r.status));
        msgEl.className = "lab-meta fail";
      }
    } catch (e) {
      msgEl.textContent = "Upload failed: backend unreachable (" + e.message + ")";
      msgEl.className = "lab-meta fail";
    }
  }

  let stagedURLs = [];
  function stageFiles(files) {
    stagedURLs.forEach((u) => URL.revokeObjectURL(u)); // don't leak blobs across re-stages
    stagedURLs = [];
    const list = document.getElementById("uploadList");
    list.innerHTML = "";
    Array.from(files).forEach((f) => {
      const url = URL.createObjectURL(f);
      stagedURLs.push(url);
      const card = el("div", "lab-card");
      const top = el("div", "lab-card-top");
      top.appendChild(el("h4", null, f.name));
      top.appendChild(pill("staged, not saved", "idle"));
      card.appendChild(top);
      card.appendChild(metaLine([
        (f.size / 1024).toFixed(0) + " KB",
        f.type || "unknown type",
      ]));
      card.appendChild(metaLine([
        "next: save to speaker_voices/ → prepare_voice_corpus.py → update_voice_lab_library.py",
      ]));
      card.appendChild(audioEl(url));
      if (backendUp) {
        const row = el("p", "lab-meta");
        const sel = document.createElement("select");
        sel.setAttribute("aria-label", "Speaker language");
        ["en", "sk", "cs", "de"].forEach((l) => {
          const o = document.createElement("option");
          o.value = l;
          o.textContent = l;
          sel.appendChild(o);
        });
        const btn = el("button", "btn-small", "Upload for real");
        const msg = el("p", "lab-meta", "");
        btn.addEventListener("click", () => {
          msg.textContent = "Uploading…";
          uploadStaged(f, sel.value, msg);
        });
        row.appendChild(sel);
        row.appendChild(document.createTextNode(" "));
        row.appendChild(btn);
        card.appendChild(row);
        card.appendChild(msg);
      }
      list.appendChild(card);
    });
  }

  function setupUpload() {
    const zone = document.getElementById("dropzone");
    const input = document.getElementById("uploadInput");
    zone.addEventListener("click", () => input.click());
    input.addEventListener("change", () => { if (input.files.length) stageFiles(input.files); });
    ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    }));
    ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
    }));
    zone.addEventListener("drop", (e) => {
      if (e.dataTransfer && e.dataTransfer.files.length) stageFiles(e.dataTransfer.files);
    });
  }

  // Plan panel: fetched EAGERLY on page load (not on click) so a stuck "loading…"
  // is impossible to confuse with an unfired request — check the server log.
  async function loadPlan() {
    const pre = document.getElementById("planPre");
    try {
      const r = await fetch("../../PLAN.md", { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      pre.textContent = await r.text();
    } catch (e) {
      pre.textContent = "Could not load PLAN.md (" + e.message + "). " +
        "Serve the repo over HTTP (make lab) or read PLAN.md at the repo root.";
    }
  }

  function setupPlanToggle() {
    const btn = document.getElementById("planToggle");
    const panel = document.getElementById("planPanel");
    btn.addEventListener("click", () => {
      const hidden = panel.classList.toggle("hidden");
      btn.textContent = hidden ? "show" : "hide";
    });
  }

  async function main() {
    setupUpload();
    setupPlanToggle();
    loadPlan();
    probeBackend();
    const dl = el("button", "btn-small", "Download my ratings (JSON)");
    dl.addEventListener("click", () => {
      const blob = new Blob([localStorage.getItem("hlas-ratings") || "{}"],
        { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "voice-ratings.json";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    });
    document.getElementById("statStrip").appendChild(dl);
    try {
      const r = await fetch("library.json", { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      renderLibrary(await r.json());
    } catch (e) {
      const host = document.getElementById("librarySections");
      host.appendChild(el("p", "lab-error",
        "Could not load library.json (" + e.message + "). " +
        "If you opened lab.html via file:// and the sections below are empty, serve the repo instead: " +
        "python3 -m http.server 8080  →  http://localhost:8080/ui/voice-lab/lab.html"));
    }
  }

  document.addEventListener("DOMContentLoaded", main);
})();
