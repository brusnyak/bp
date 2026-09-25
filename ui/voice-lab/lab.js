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
    }
    card.appendChild(audioEl(item.file));
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

  async function main() {
    setupUpload();
    try {
      const r = await fetch("library.json");
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
