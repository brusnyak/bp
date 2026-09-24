/* Voice Lab — read-only eval page. No auth, no mutations. Local use only. */
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
    a.src = src;
    return a;
  }

  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(url + " → HTTP " + r.status);
    return r.json();
  }

  function voiceAudioURL(v) {
    // metadata `path` looks like "speaker_voices/2_my voice 1.wav"
    const p = v.path || ("speaker_voices/" + (v.filename || ""));
    return "/" + encodeURI(p);
  }

  function renderStatus(status) {
    const engBody = document.querySelector("#enginesTable tbody");
    Object.entries(status.engines || {}).forEach(([key, f]) => {
      const tr = document.createElement("tr");
      [key, f.cloning ? "yes" : "no", f.streaming ? "yes" : "no",
       f.requires_speaker_wav ? "yes" : "no"].forEach((t) => tr.appendChild(el("td", null, t)));
      engBody.appendChild(tr);
    });

    const beBody = document.querySelector("#backendsTable tbody");
    Object.entries(status.hardware_backends || {}).forEach(([stage, be]) => {
      const tr = document.createElement("tr");
      tr.appendChild(el("td", null, stage));
      tr.appendChild(el("td", null, String(be)));
      beBody.appendChild(tr);
    });

    const pm = document.getElementById("piperModels");
    (status.piper_models || []).forEach((m) => pm.appendChild(el("li", null, m)));
    if (!(status.piper_models || []).length) pm.appendChild(el("li", null, "(none found)"));
  }

  function renderVoices(voices) {
    const list = document.getElementById("voicesList");
    if (!voices.length) {
      list.appendChild(el("p", "lab-hint", "No voices returned by /api/voices."));
      return;
    }
    voices.forEach((v) => {
      const card = el("div", "lab-card");
      card.appendChild(el("h4", null, v.name || v.id));
      card.appendChild(el("p", "lab-meta",
        "language: " + (v.language || "?") + " · file: " + (v.filename || v.path || "?")));
      if (v.transcribed_text) card.appendChild(el("p", "lab-meta", "transcript: " + v.transcribed_text));
      card.appendChild(audioEl(voiceAudioURL(v)));
      list.appendChild(card);
    });
  }

  function renderQC(status) {
    const list = document.getElementById("qcList");
    const cands = status.qc_candidates || [];
    if (!cands.length) {
      list.appendChild(el("p", "lab-hint",
        "No candidates.json — run: venv/bin/python scripts/voice_similarity_qc.py --synthesize-only"));
      return;
    }
    const scores = {};
    (status.qc_scores || []).forEach((s) => { scores[s.label] = s.cosine_similarity; });
    if (!status.qc_scored) {
      const note = el("p", "lab-hint",
        "Not scored yet — run --score-only in the qc venv, then reload.");
      list.appendChild(note);
    }
    cands.forEach((c) => {
      const card = el("div", "lab-card");
      card.appendChild(el("h4", null, c.label));
      card.appendChild(el("p", "lab-meta",
        "engine: " + c.engine + " · lang: " + c.language + " · " + c.sample_rate + " Hz" +
        " · synth: " + c.synthesis_latency_s + "s"));
      const sim = scores[c.label];
      if (sim === undefined) {
        card.appendChild(el("p", "lab-meta", "similarity: — (not scored)"));
      } else {
        const p = el("p", "lab-meta", "similarity: " + sim.toFixed(4) +
          " (ref " + THRESHOLD + ") " + (sim >= THRESHOLD ? "✓ pass" : "✗ below"));
        p.classList.add(sim >= THRESHOLD ? "pass" : "fail");
        card.appendChild(p);
      }
      card.appendChild(audioEl("/voice_qc/" + encodeURIComponent(c.label) + ".wav"));
      list.appendChild(card);
    });
  }

  async function main() {
    try {
      const status = await getJSON("/api/voice-lab/status");
      renderStatus(status);
      renderQC(status);
    } catch (e) {
      const err = document.getElementById("statusError");
      err.textContent = "Status failed: " + e.message + " (is the backend running?)";
      err.classList.remove("hidden");
    }
    try {
      renderVoices(await getJSON("/api/voices"));
    } catch (e) {
      document.getElementById("voicesList")
        .appendChild(el("p", "lab-error", "Voices failed: " + e.message));
    }
  }

  document.addEventListener("DOMContentLoaded", main);
})();
