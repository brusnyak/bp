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
    } else if (sectionId === "conversation") {
      const meta = item.meta || {};
      const direction = item.name.startsWith("en_sk") ? "English → Slovak" : "Slovak → English";
      top.querySelector("h4").textContent = direction + " · " + item.name.split("_").pop();
      if (meta.src) {
        const source = el("p", "turn-copy");
        source.appendChild(el("strong", null, "Heard: "));
        source.appendChild(document.createTextNode(meta.src));
        card.appendChild(source);
      }
      if (meta.tgt) {
        const target = el("p", "turn-copy");
        target.appendChild(el("strong", null, "Translated: "));
        target.appendChild(document.createTextNode(meta.tgt));
        card.appendChild(target);
      }
      card.appendChild(metaLine([
        meta.stt_s !== undefined ? "STT " + meta.stt_s + "s" : null,
        meta.mt_sentence_s !== undefined ? "MT " + meta.mt_sentence_s + "s" : null,
        meta.tts_s !== undefined ? "TTS " + meta.tts_s + "s" : null,
        meta.audio_s !== undefined ? "audio " + meta.audio_s + "s" : null,
        meta.note || null,
      ]));
    }
    if (sectionId !== "conversation" && item.meta && Object.keys(item.meta).length) {
      card.appendChild(metaLine(
        Object.entries(item.meta).map(([k, v]) => k + ": " + v)));
    }
    if (sectionId === "sk_direction") {
      // Long source recordings stay one click away — the matrix table is the story.
      const det = document.createElement("details");
      det.className = "lab-more";
      det.appendChild(el("summary", null, "Play source recording (" + item.name + ")"));
      det.appendChild(audioEl(item.file));
      card.appendChild(det);
    } else if (!(sectionId === "stt_input" && !item.file)) {
      card.appendChild(audioEl(item.file));
    }
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

  function evidenceSection(title, lead, id) {
    const section = document.createElement("section");
    section.className = "evidence-section";
    section.id = id;
    section.appendChild(el("h2", null, title));
    if (lead) section.appendChild(el("p", "evidence-lead", lead));
    return section;
  }

  function cardGrid(items, sectionId, className) {
    const grid = el("div", className || "comparison-grid");
    items.forEach((item) => grid.appendChild(itemCard(item, sectionId)));
    return grid;
  }

  function renderQc(items, host) {
    const section = evidenceSection(
      "Voice comparison",
      "Same Slovak sentence, same fast Piper runtime. The comparison is about voice character and clarity; the timings below are synthesis time, not total translation latency.",
      "voice-comparison"
    );
    const generic = items.find((item) => item.name === "sandbox_sk_SK-lili-medium_sk");
    const personal = items.find((item) => item.name === "sandbox_sk_SK-personal-male-medium_sk");
    if (generic && personal) {
      section.appendChild(cardGrid([generic, personal], "qc"));
    } else {
      section.appendChild(el("p", "lab-hint", "No matched generic/personal Slovak pair is available in this manifest yet."));
    }
    const rest = items.filter((item) => item !== generic && item !== personal);
    if (rest.length) {
      const details = document.createElement("details");
      details.className = "lab-more";
      details.appendChild(el("summary", null, "Other QC candidates (" + rest.length + ")"));
      const list = el("div", "lab-card-list");
      rest.forEach((item) => list.appendChild(itemCard(item, "qc")));
      details.appendChild(list);
      section.appendChild(details);
    }
    host.appendChild(section);
  }

  function renderConversation(items, host) {
    const section = evidenceSection(
      "Conference turn playback",
      "Saved output from real pipeline components. Use this when the room is not suitable for an extended microphone conversation.",
      "conference-playback"
    );
    section.appendChild(cardGrid(items, "conversation", "conversation-grid"));
    host.appendChild(section);
  }

  function renderMatrix(items, host) {
    const section = evidenceSection(
      "Slovak recognition and translation evidence",
      "This matrix makes the trade-off explicit: generic Whisper is faster, while the Slovak-tuned small model preserves substantially more of the spoken content.",
      "recognition-evidence"
    );
    items.forEach((item) => section.appendChild(itemCard(item, "sk_direction")));
    host.appendChild(section);
  }

  function renderCharts(items, host) {
    if (!items.length) return;
    const section = evidenceSection(
      "Measured charts",
      "Rendered from the same JSON the tables below read — no hand-typed numbers. Rebuild with: python3 scripts/build_demo_charts.py",
      "measured-charts"
    );
    const grid = el("div", "comparison-grid");
    items.forEach((item) => {
      const card = el("div", "lab-card");
      card.appendChild(el("h4", null, item.name));
      const img = document.createElement("img");
      img.src = encodeURI(item.file);
      img.alt = item.caption || item.name;
      img.loading = "lazy";
      img.className = "lab-chart";
      card.appendChild(img);
      if (item.caption) card.appendChild(metaLine([item.caption]));
      grid.appendChild(card);
    });
    section.appendChild(grid);
    host.appendChild(section);
  }

  function renderSttInput(items, host) {
    if (!items.length) return;
    const section = evidenceSection(
      "STT input control: clean synth vs microphone",
      "Same Slovak sentences synthesized by Piper (known-perfect text), then recognized. " +
      "The personal fine-tuned voice is harder to recognize than the generic one — " +
      "voice quality and SK→EN accuracy are linked, not separate issues.",
      "stt-input-control"
    );
    const matrix = items.find((i) => i.name === "_matrix");
    if (matrix && matrix.matrix) {
      const card = el("div", "lab-card");
      card.appendChild(el("h4", null, "Clean-synth WER by input voice and STT rung"));
      const tbl = el("table", "matrix-table");
      const thead = el("thead");
      const trh = el("tr");
      ["STT input", "Rung", "WER", "CER", "STT latency", "Audio"].forEach((h) => {
        trh.appendChild(el("th", null, h));
      });
      thead.appendChild(trh);
      tbl.appendChild(thead);
      const tbody = el("tbody");
      Object.entries(matrix.matrix).forEach(([key, r]) => {
        const [voice, rung] = key.split("/");
        const tr = el("tr");
        tr.appendChild(el("td", null, voice));
        tr.appendChild(el("td", null, rung));
        const werTd = el("td", null, (r.wer * 100).toFixed(1) + "%");
        if (r.wer < 0.3) werTd.className = "pass";
        else if (r.wer > 0.5) werTd.className = "fail";
        tr.appendChild(werTd);
        tr.appendChild(el("td", null, (r.cer * 100).toFixed(1) + "%"));
        tr.appendChild(el("td", null, r.stt_s + "s"));
        tr.appendChild(el("td", null, r.audio_s + "s"));
        tbody.appendChild(tr);
      });
      tbl.appendChild(tbody);
      card.appendChild(tbl);
      if (matrix.ref_chars) card.appendChild(metaLine(["reference text: " + matrix.ref_chars + " chars, 200-char proofread SK passage"]));
      section.appendChild(card);
    }
    const clips = items.filter((i) => i.name !== "_matrix" && i.file);
    if (clips.length) {
      section.appendChild(el("p", "lab-hint", "Listen: the exact audio the recognizer was scored on."));
      section.appendChild(cardGrid(clips.map((c) => ({ name: c.name, file: c.file })), "test", "lab-card-list"));
    }
    host.appendChild(section);
  }

  function renderMeeting(items, host) {
    if (!items.length) return;
    const section = evidenceSection(
      "Simulated bilingual meeting",
      "Scripted turns read by the personal voices, pipelined sentence-by-sentence while the speaker " +
      "continues — chunk playbacks overlap later speech, like live simultaneous interpretation. " +
      "Dark blue is speech, the dashed outline is live STT, the dot marks first audio, green is " +
      "translated playback on the other speaker's lane. One player below holds the whole mix.",
      "simulated-meeting"
    );
    const gantt = el("div", "lab-card");
    gantt.appendChild(el("h4", null, "Meeting timeline"));
    const img = document.createElement("img");
    img.src = "charts/gantt_meeting.png";
    img.alt = "Meeting Gantt chart: two speaker lanes over time with pipeline stages";
    img.loading = "lazy";
    img.className = "lab-chart";
    gantt.appendChild(img);
    section.appendChild(gantt);
    const mix = items.find((i) => i.name === "_mix");
    if (mix) {
      const card = el("div", "lab-card");
      card.appendChild(el("h4", null, "Whole conversation (" + (mix.meta.total_s || "?") + "s, one player)"));
      card.appendChild(audioEl(mix.file));
      const chapters = mix.meta.chapters || [];
      if (chapters.length) {
        const tbl = el("table", "matrix-table");
        const thead = el("thead");
        const trh = el("tr");
        ["Turn", "Speaker", "Direction", "Speech at", "Playback at"].forEach((h) => {
          trh.appendChild(el("th", null, h));
        });
        thead.appendChild(trh);
        tbl.appendChild(thead);
        const tbody = el("tbody");
        chapters.forEach((c) => {
          const tr = el("tr");
          [c.turn, c.speaker, c.direction, c.speech_at + "s", c.playback_at + "s"].forEach((v) => {
            tr.appendChild(el("td", null, String(v)));
          });
          tbody.appendChild(tr);
        });
        tbl.appendChild(tbody);
        card.appendChild(tbl);
      }
      if (mix.meta.timing_model) {
        card.appendChild(metaLine(["timing model: " + mix.meta.timing_model + " (first word ~0.1s, first audio ~0.2s after speech end)"]));
      }
      section.appendChild(card);
    }
    items.filter((i) => i.name !== "_mix").forEach((item) => {
      const card = el("div", "lab-card");
      const meta = item.meta || {};
      const top = el("div", "lab-card-top");
      top.appendChild(el("h4", null, "Turn " + item.name + " · Speaker " + (meta.speaker || "?")));
      top.appendChild(pill(meta.direction || "", "idle"));
      card.appendChild(top);
      [["Script", meta.script], ["Heard", meta.heard], ["Translated", meta.translation]].forEach(([k, v]) => {
        if (v) {
          const p = el("p", "turn-copy");
          p.appendChild(el("strong", null, k + ": "));
          p.appendChild(document.createTextNode(v));
          card.appendChild(p);
        }
      });
      card.appendChild(metaLine([
        meta.starts_at !== undefined ? "starts at " + meta.starts_at + "s" : null,
        meta.speech_s !== undefined ? "speech " + meta.speech_s + "s" : null,
        meta.stt_s !== undefined ? "STT " + meta.stt_s + "s" : null,
        meta.mt_s !== undefined ? "MT " + meta.mt_s + "s" : null,
        meta.tts_s !== undefined ? "TTS " + meta.tts_s + "s" : null,
        meta.playback_s !== undefined ? "playback " + meta.playback_s + "s" : null,
        meta.first_word_s != null ? "first word " + meta.first_word_s + "s" : null,
        meta.first_audio_s != null ? "first audio " + meta.first_audio_s + "s" : null,
        meta.streaming ? "✓ streams (playback starts before speech ends)" : null,
        meta.live ? "live /ws measurement" : null,
      ]));
      if (item.file) card.appendChild(audioEl(item.file));
      section.appendChild(card);
    });
    host.appendChild(section);
  }

  function renderAudioList(items, host, title, lead, id) {    if (!items.length) return;
    const section = evidenceSection(title, lead, id);
    if (id === "model-spikes") {
      const card = el("div", "lab-card");
      card.appendChild(el("h4", null, "Engine comparison"));
      const tbl = el("table", "matrix-table");
      const thead = el("thead");
      const trh = el("tr");
      ["Engine × clip", "RTF", "chrF / WER", "Audio"].forEach((h) => {
        trh.appendChild(el("th", null, h));
      });
      thead.appendChild(trh);
      tbl.appendChild(thead);
      const tbody = el("tbody");
      items.forEach((item) => {
        const meta = item.meta || {};
        const tr = el("tr");
        tr.appendChild(el("td", null, item.name));
        const rtfTd = el("td", null, meta.rtf !== undefined && meta.rtf !== null ? String(meta.rtf) : "—");
        if (meta.rtf !== undefined && meta.rtf !== null && meta.rtf < 1) rtfTd.className = "pass";
        tr.appendChild(rtfTd);
        const q = meta.mt_chrf !== undefined && meta.mt_chrf !== null ? "chrF " + meta.mt_chrf
          : (meta.qc_wer !== undefined && meta.qc_wer !== null ? "WER " + (meta.qc_wer * 100).toFixed(1) + "%" : "—");
        tr.appendChild(el("td", null, q));
        tr.appendChild(el("td", null, meta.audio_s !== undefined && meta.audio_s !== null ? meta.audio_s + "s" : "—"));
        tbody.appendChild(tr);
      });
      tbl.appendChild(tbody);
      card.appendChild(tbl);
      section.appendChild(card);
    }
    items.forEach((item) => {
      const card = el("div", "lab-card");
      card.appendChild(el("h4", null, item.name));
      const meta = item.meta || {};
      const lines = Object.entries(meta).filter(([, v]) => v !== undefined && v !== null && v !== "");
      if (lines.length) {
        card.appendChild(metaLine(lines.map(([k, v]) => k + ": " + v)));
      }
      card.appendChild(audioEl(item.file));
      section.appendChild(card);
    });
    host.appendChild(section);
  }

  function renderLibrary(library) {
    renderStats(library);
    const voices = library.sections.find((section) => section.id === "voices") || { items: [] };
    const voicesPanel = document.getElementById("voicesPanel");
    document.getElementById("voicesCount").textContent = "(" + voices.items.length + ")";
    if (voices.items.length) {
      const featured = voices.items.filter((i) => i.featured);
      const rest = voices.items.filter((i) => !i.featured);
      voicesPanel.appendChild(cardGrid(featured.length ? featured : voices.items, "voices", "voice-grid"));
      if (featured.length && rest.length) {
        const det = document.createElement("details");
        det.className = "lab-more";
        det.appendChild(el("summary", null, "Other recordings (" + rest.length + ")"));
        const list = el("div", "lab-card-list");
        rest.forEach((item) => list.appendChild(itemCard(item, "voices")));
        det.appendChild(list);
        voicesPanel.appendChild(det);
      }
    } else voicesPanel.appendChild(el("p", "lab-hint", "No reference recordings in this manifest."));

    const host = document.getElementById("labEvidence");
    const qc = library.sections.find((section) => section.id === "qc");
    const conversation = library.sections.find((section) => section.id === "conversation");
    const matrix = library.sections.find((section) => section.id === "sk_direction");
    if (qc) renderQc(qc.items, host);
    if (library.sections.find((section) => section.id === "charts"))
      renderCharts(library.sections.find((section) => section.id === "charts").items, host);
    if (conversation) renderConversation(conversation.items, host);
    if (matrix) renderMatrix(matrix.items, host);
    const sttInput = library.sections.find((section) => section.id === "stt_input");
    if (sttInput) renderSttInput(sttInput.items, host);
    const zeroshot = library.sections.find((section) => section.id === "zeroshot");
    if (zeroshot) renderAudioList(zeroshot.items, host,
      "Zero-shot voice clones",
      "OmniVoice clones from the new v2b recordings (isolated eval, not the live pipeline). Judge blind against the Piper pair above.",
      "zero-shot-clones");
    const demoAudio = library.sections.find((section) => section.id === "demo_audio");
    if (demoAudio) renderAudioList(demoAudio.items, host,
      "Demo turns (offline fallback)",
      "Full EN↔SK pipeline receipts rendered offline — playable even if the live server dies on stage.",
      "demo-turns");
    const meeting = library.sections.find((section) => section.id === "meeting");
    if (meeting) renderMeeting(meeting.items, host);
    const corpus = library.sections.find((section) => section.id === "corpus");
    if (corpus) renderAudioList(corpus.items, host,
      "Piper training corpus",
      "Bulk OmniVoice clones of the v2b voice (~6 min, mean QC WER 0.05) — the material a fresh Piper voice will train on. Not live voices, corpus only.",
      "training-corpus");
    const spikes = library.sections.find((section) => section.id === "spikes");
    if (spikes) renderAudioList(spikes.items, host,
      "New-model spikes",
      "Isolated-eval results on fixed clips: RTF, chrF/WER, hypothesis text, output audio. Kill reasons recorded in PLAN.md.",
      "model-spikes");
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
    setupDisclosure("planToggle", "planPanel", "Show plan", "Hide plan");
    setupDisclosure("voicesToggle", "voicesPanel", "Show recordings", "Hide recordings");
    setupDisclosure("uploadToggle", "uploadPanel", "Stage audio", "Hide staging");
  }

  function setupDisclosure(buttonId, panelId, showLabel, hideLabel) {
    const btn = document.getElementById(buttonId);
    const panel = document.getElementById(panelId);
    btn.addEventListener("click", () => {
      const hidden = panel.classList.toggle("hidden");
      btn.textContent = hidden ? showLabel : hideLabel;
      btn.setAttribute("aria-expanded", String(!hidden));
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
      const host = document.getElementById("labEvidence");
      host.appendChild(el("p", "lab-error",
        "Could not load library.json (" + e.message + "). " +
        "If you opened lab.html via file:// and the sections below are empty, serve the repo instead: " +
        "python3 -m http.server 8080  →  http://localhost:8080/ui/voice-lab/lab.html"));
    }
  }

  document.addEventListener("DOMContentLoaded", main);
})();
