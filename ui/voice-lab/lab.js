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

  function metaLine(parts) {
    return el("p", "lab-meta", parts.filter(Boolean).join(" · "));
  }

  function itemCard(item, sectionId) {
    const card = el("div", "lab-card");
    card.appendChild(el("h4", null, item.name));
    if (sectionId === "voices") {
      card.appendChild(metaLine([
        "language: " + (item.language || "?"),
        "file: " + item.file,
        item.in_registry ? "in registry ✓" : "NOT in registry ✗",
      ]));
      if (item.transcript) card.appendChild(metaLine(["transcript: " + item.transcript]));
    } else if (sectionId === "qc") {
      card.appendChild(metaLine([
        "engine: " + (item.engine || "?"),
        "lang: " + (item.language || "?"),
        item.sample_rate ? item.sample_rate + " Hz" : null,
        item.synthesis_latency_s !== undefined && item.synthesis_latency_s !== null
          ? "synth: " + item.synthesis_latency_s + "s" : null,
      ]));
      if (item.similarity === undefined || item.similarity === null) {
        card.appendChild(metaLine(["similarity: — (run --score-only in the qc venv)"]));
      } else {
        const p = el("p", "lab-meta",
          "similarity: " + item.similarity.toFixed(4) +
          " (ref " + THRESHOLD + ") " + (item.similarity >= THRESHOLD ? "✓ pass" : "✗ below"));
        p.classList.add(item.similarity >= THRESHOLD ? "pass" : "fail");
        card.appendChild(p);
      }
    } else if (sectionId === "test") {
      card.appendChild(metaLine(["file: " + item.file]));
      if (item.transcript) card.appendChild(metaLine(["ground truth: " + item.transcript]));
    }
    card.appendChild(audioEl(item.file));
    return card;
  }

  function renderLibrary(library) {
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

  function setupUpload() {
    const input = document.getElementById("uploadInput");
    const list = document.getElementById("uploadList");
    input.addEventListener("change", () => {
      list.innerHTML = "";
      Array.from(input.files).forEach((f) => {
        const card = el("div", "lab-card");
        card.appendChild(el("h4", null, f.name));
        card.appendChild(metaLine([
          (f.size / 1024).toFixed(0) + " KB",
          f.type || "unknown type",
          "staged, not saved →",
        ]));
        card.appendChild(metaLine([
          "next: save to speaker_voices/ → prepare_voice_corpus.py → update_voice_lab_library.py",
        ]));
        card.appendChild(audioEl(URL.createObjectURL(f)));
        list.appendChild(card);
      });
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
