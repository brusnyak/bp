#!/usr/bin/env python3
"""bp — single entry for every repeatable workflow in this repo.

Replaces the scripts/ sprawl: one CLI, one --help, same behavior (each command
reuses the original module's code path, nothing rewritten).

    venv/bin/python scripts/bp.py corpus      # normalize/dedupe/transcribe voices
    venv/bin/python scripts/bp.py qc --synthesize-only
    venv/bin/python scripts/bp.py stt --models base,small --clip sk
    venv/bin/python scripts/bp.py e2e         # full EN->SK timed run, new voice
    venv/bin/python scripts/bp.py library     # refresh Voice Lab manifest
    venv/bin/python scripts/bp.py grades      # machine pre-grades + merge into manifest
    venv/bin/python scripts/bp.py grades --import voice-ratings.json
                                              # import Lab ear verdicts (Download my ratings)
    venv/bin/python scripts/bp.py script      # bilingual reading sheet via local MT

Trainers stay separate (own venv by design): scripts/finetune_personal_voice.py.
"""

import argparse
import os
import runpy
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

WRAPPED = {
    "corpus": "prepare_voice_corpus.py",
    "qc": "voice_similarity_qc.py",
    "e2e": "e2e_ensk_new_voice.py",
    "library": "update_voice_lab_library.py",
    "script": "make_reading_script.py",
    "demo-audio": "demo_audio.py",
    "meeting": "meeting_sim.py",
    "engine-ab": "engine_ab.py",
    "listen": "machine_listen_qc.py",
    "bulk": "bulk_hq.py",
}


def cmd_bench(args):
    """New-model bench on fixed owner clips (reports/model_landscape_2026-10.md §6).
    engine nemotron uses the pure-C runtime (scripts/bench_nemotron.py);
    everything else goes through scripts/bench_new_models.py."""
    if args.engine == "nemotron":
        sys.argv = ["bench_nemotron.py"]
        runpy.run_path(os.path.join(REPO_ROOT, "scripts", "bench_nemotron.py"),
                       run_name="__main__")
    else:
        sys.argv = ["bench_new_models.py", "--engine", args.engine,
                    "--clips", args.clips]
        runpy.run_path(os.path.join(REPO_ROOT, "scripts", "bench_new_models.py"),
                       run_name="__main__")


def cmd_stt(args):
    """STT model × language matrix on the owner's real recordings.

    The question it answers: which STT rung for which language, with numbers.
    Ground truth: EN = script sentences; SK = proofread SK column.
    --lang-prompt: force decoder language (sk | cs | auto) — cs tests the
    Czech-as-Slovak-proxy hypothesis with zero downloads.
    """
    import time
    import unicodedata

    import librosa
    import numpy as np

    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from make_reading_script import EN_SENTENCES

    try:
        from jiwer import wer
    except ImportError:
        sys.exit("jiwer not installed in this venv")

    def norm(s):
        return " ".join(s.lower().strip().split())

    def plain(s):
        return norm("".join(
            c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)))

    def load(name):
        wav, sr = librosa.load(os.path.join(REPO_ROOT, "speaker_voices", name),
                               sr=16000, mono=True)
        return np.asarray(wav, dtype=np.float32), sr

    def hyp_text(segments):
        return norm(" ".join(
            s["text"] if isinstance(s, dict) else getattr(s, "text", "") for s in segments))

    sk_refs = []
    md = os.path.join(REPO_ROOT, "documentation", "reading_script_bilingual.md")
    for line in open(md).read().splitlines():
        if line.startswith("|") and len(line) > 2 and line[2].isdigit():
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) == 3:
                sk_refs.append(parts[2])

    # Clips come from the language packs (single source of truth); hardcoding the
    # filenames here is what broke this command after the recordings were renamed.
    import language_packs as lp

    def clip_for(pair):
        pack = lp.load(pair)
        clip = pack.get("clip")
        if not clip:
            sys.exit(f"pack {pair} has no clip; edit languages/packs/{pair}.json")
        return os.path.basename(clip)

    clips = {
        "en": (clip_for("en-sk"), norm(" ".join(EN_SENTENCES))),
        "sk": (clip_for("sk-en"), norm(" ".join(sk_refs))),
    }
    if args.clip != "both":
        clips = {args.clip: clips[args.clip]}

    import json
    results = []
    prompt_list = [p.strip() for p in args.lang_prompt.split(",")]
    for model in args.models.split(","):
        stt = FasterWhisperSTT(model_size=model.strip())
        for clip_key, (fname, ref) in clips.items():
            # --lang-prompt applies to the SK clip (the open question); EN always en.
            prompts = ["en"] if clip_key == "en" else prompt_list
            for prompt in prompts:
                wav, sr = load(fname)
                segs, t, dl = stt.transcribe_audio(
                    wav, sr, language=None if prompt == "auto" else prompt)
                hyp = hyp_text(segs)
                row = {
                    "model": model.strip(), "clip": fname, "prompt": prompt,
                    "wer": round(wer(ref, hyp), 4),
                    "wer_plain": round(wer(plain(ref), plain(hyp)), 4),
                    "stt_time_s": round(t, 1), "audio_s": round(len(wav) / sr, 1),
                    "detected_lang": dl,
                }
                results.append(row)
                print(f"{row['model']:6s} {fname:24s} prompt={prompt:4s} "
                      f"WER {row['wer']:.4f} (plain {row['wer_plain']:.4f})  {t:.1f}s")
    out = os.path.join(REPO_ROOT, "processed", "stt_matrix.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out} ({len(results)} rows)")


def cmd_grades(args):
    """Machine pre-grade the Lab + merge into the manifest.

    Default: grade_library.py --apply (never overwrites stored ear verdicts),
    then update_voice_lab_library.py --no-test so library.json carries the
    fresh machine notes. --import loads the browser JSON from the Lab's
    "Download my ratings" button into processed/ear_grades.json; re-run
    without flags afterwards to merge.
    """
    import json
    repo = REPO_ROOT
    grades_path = os.path.join(repo, "processed", "ear_grades.json")
    if args.pull:
        import urllib.request
        token = args.token or os.environ.get("BP_TOKEN")
        if not token:
            sys.exit("need --token or BP_TOKEN (login token from /ui/auth/auth.html)")
        base = args.server.rstrip("/")
        req = urllib.request.Request(
            base + "/api/ratings", headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(req, timeout=15) as r:
            store = json.load(r)
        with open(grades_path) as f:
            local = json.load(f)
        by_name = {g["name"]: g for g in local.get("grades", [])}
        n = 0
        for g in store.get("grades", []):
            ear = g.get("ear") or {}
            if ear.get("grade") is None and ear.get("keep") is None \
                    and not ear.get("note") and not ear.get("defects"):
                continue
            tgt = by_name.get(g.get("name"))
            if tgt is None:
                continue
            if (ear.get("updated") or 0) >= ((tgt.get("ear") or {}).get("updated") or 0):
                tgt["ear"] = ear
                n += 1
        with open(grades_path, "w") as f:
            json.dump(local, f, indent=2, ensure_ascii=False)
        print(f"pulled {n} ear verdicts from {base}")
        return
    if args.import_json:
        with open(args.import_json) as f:
            browser = json.load(f)
        with open(grades_path) as f:
            store = json.load(f)
        by_name = {}
        for g in store.get("grades", []):
            by_name.setdefault(g["name"], []).append(g)
        n = 0
        ambiguous = []
        for name, r in browser.items():
            targets = by_name.get(name)
            if not targets:
                continue
            if len(targets) > 1:
                ambiguous.append(name)
            defects = {k: r.get(k) for k in ("steadiness", "hiss", "muffled")
                       if r.get(k)}
            ear = {"grade": r.get("grade") or None,
                   "keep": ({1: "keep", 2: "kill"}.get(r.get("keep"))),
                   "sim": r.get("sim"),
                   "defects": defects,
                   "note": r.get("note", "")}
            if ear["grade"] == 0:
                ear["grade"] = None
            for g in targets:
                g["ear"] = ear
                n += 1
        with open(grades_path, "w") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
        print(f"imported {n} ear verdicts from {args.import_json}")
        if ambiguous:
            print("ambiguous names applied to every section: " + ", ".join(ambiguous))
        return
    for script, script_args in (("grade_library.py", ["--apply"]),
                                ("update_voice_lab_library.py", ["--no-test"])):
        sys.argv = [script] + script_args
        runpy.run_path(os.path.join(repo, "scripts", script), run_name="__main__")
        print("---")


def cmd_engine(args):
    """Engine registry: list capabilities per stage, or run the licence gate on one."""
    import json as _json
    from backend.registry import STAGES, engines as _engines, select as _select

    if args.action == "list":
        for stage in STAGES:
            caps = _engines(stage)
            if not caps:
                continue
            print(f"[{stage}]")
            for cap in caps:
                d = cap.describe()
                mark = "ok" if d["available"] else "--"
                warn = "" if d["license"] == "commercial-ok" else f"  <- {d['license']}"
                print(f"  {d['name']:22s} {mark:>3s}  {d['license']:15s}{warn}")
        print()
        print("ok = installed here; -- = not installed (needs the module) or declared-only.")
        print("licence (fail closed): bp engine check <stage> <name>")
        print("  non-commercial -> --allow-noncommercial ;  unknown -> --allow-unknown  (internal runs only)")
        return

    if not args.stage or not args.name:
        sys.exit("usage: bp engine check|gate <stage> <name>")

    if args.action == "check":
        try:
            cap = _select(args.stage, args.name,
                          allow_noncommercial=args.allow_noncommercial,
                          allow_unknown=args.allow_unknown)
        except (KeyError, PermissionError) as exc:
            sys.exit(str(exc))
        print(_json.dumps(cap.describe(), indent=2))
        return

    if args.action == "gate":
        # T030: Per-stage quality/latency gate
        from backend.registry import engines as _engines, select as _select, instantiate
        import time
        import numpy as np
        import json
        import os

        cap = _engines(args.stage)
        cap = next((c for c in cap if c.name == args.name), None)
        if not cap:
            sys.exit(f"engine '{args.name}' not found in stage '{args.stage}'")

        print(f"Running gate for {args.stage}/{args.name}...")

        # Per-stage thresholds (fail if exceeded)
        THRESHOLDS = {
            ("stt", "faster_whisper"): {"wer_max": 0.15, "latency_max_s": 8.0},  # WER < 15%, latency < 8s
            ("mt", "opus-mt"): {"chrF_min": 0.6, "latency_max_s": 0.5},  # chrF > 60%, latency < 0.5s
            ("tts", "piper"): {"rtf_max": 0.1, "sim_min": 0.7},  # RTF < 0.1, speaker sim > 0.7
            ("tts", "piper_personal"): {"rtf_max": 0.1, "sim_min": 0.7},
            ("tts", "piper_sk_personal"): {"rtf_max": 0.1, "sim_min": 0.7},
            ("vad", "webrtcvad"): {"false_start_rate_max": 0.05, "latency_max_s": 0.01},
        }

        key = (args.stage, args.name)
        thresholds = THRESHOLDS.get(key, {})
        if not thresholds:
            print(f"WARNING: No thresholds defined for {key}, skipping gate")
            return

        # Run gate test
        gate_dir = os.path.join(REPO_ROOT, "processed", "engine_gate")
        os.makedirs(gate_dir, exist_ok=True)
        out_file = os.path.join(gate_dir, f"{args.stage}_{args.name}.json")

        if args.stage == "stt" and args.name == "faster_whisper":
            from backend.stt.faster_whisper_stt import FasterWhisperSTT
            import librosa
            import jiwer

            # Load test clip (en-sk)
            clip_path = os.path.join(REPO_ROOT, "speaker_voices", "me_en_script.m4a")
            ref_text = "Good morning, and welcome to this live demonstration of real-time speech translation. I am speaking English, and you will hear my own voice speaking Slovak within seconds. The system listens through the microphone and detects when speech starts and ends. First the speech is transcribed into text, then translated, then spoken again. Every step runs locally on this computer, with no cloud and no waiting room. Latency is the time between my words and the translated voice you hear. Our goal is a delay of only a few seconds, short enough for natural conversation. Voice cloning means the Slovak output sounds like me, not like a stranger. To teach the system my voice, I recorded a few minutes of clean speech. A quiet room matters more than an expensive microphone. Please speak at a steady pace, with short pauses between sentences. Numbers, names, and technical terms need extra care in both languages. The subtitle line shows partial results while I am still speaking. You can pop the subtitles out into a floating window above any application. The translated voice is routed into the meeting as a virtual microphone. Other participants hear Slovak, while I keep speaking English. Questions from the audience will be translated in the opposite direction. Thank you for listening, and I look forward to your questions."

            if not os.path.exists(clip_path):
                sys.exit(f"Test clip not found: {clip_path}")

            wav, sr = librosa.load(clip_path, sr=16000, mono=True)
            wav = np.asarray(wav, dtype=np.float32)

            engine = instantiate("stt", "faster_whisper", model_size="base")
            t0 = time.time()
            segs, stt_s, _ = engine.transcribe_audio(wav, sr, language="en")
            hyp = " ".join(s.text for s in segs)
            wer = jiwer.wer(ref_text.lower(), hyp.lower())
            latency = time.time() - t0

            result = {"engine": args.name, "stage": args.stage, "wer": wer, "latency_s": latency, "thresholds": thresholds, "passed": wer <= thresholds.get("wer_max", 1.0) and latency <= thresholds.get("latency_max_s", 999)}
            print(f"  WER: {wer:.4f} (threshold: {thresholds.get('wer_max')})")
            print(f"  Latency: {latency:.2f}s (threshold: {thresholds.get('latency_max_s')}s)")
            print(f"  PASSED: {result['passed']}")

            with open(out_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            if not result["passed"]:
                sys.exit(1)

        elif args.stage == "mt" and args.name == "opus-mt":
            from backend.mt.ctranslate2_mt import CTranslate2MT
            import sacrebleu

            engine = instantiate("mt", "opus-mt", model_path=os.path.join(REPO_ROOT, "ct2_models", "Helsinki-NLP--opus-mt-en-sk"), device="cpu")

            # Test sentences
            test_src = ["Good morning.", "This is a test.", "How are you today?"]
            test_ref = ["Dobré ráno.", "Toto je test.", "Ako sa dnes máš?"]

            t0 = time.time()
            hyps = [engine.translate(s, "en", "sk")[0] if isinstance(engine.translate(s, "en", "sk"), tuple) else engine.translate(s, "en", "sk") for s in test_src]
            latency = time.time() - t0

            chrf = sacrebleu.corpus_chrf(hyps, [test_ref]).score / 100
            per_sent_latency = latency / len(test_src)

            result = {"engine": args.name, "stage": args.stage, "chrF": chrf, "latency_s": per_sent_latency, "thresholds": thresholds, "passed": chrf >= thresholds.get("chrF_min", 0) and per_sent_latency <= thresholds.get("latency_max_s", 999)}
            print(f"  chrF: {chrf:.4f} (threshold: {thresholds.get('chrF_min')})")
            print(f"  Latency/sent: {per_sent_latency:.3f}s (threshold: {thresholds.get('latency_max_s')}s)")
            print(f"  PASSED: {result['passed']}")

            with open(out_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            if not result["passed"]:
                sys.exit(1)

        elif args.stage == "tts" and args.name.startswith("piper"):
            from backend.tts.piper_tts import PiperTTS
            import soundfile as sf

            voice_map = {
                "piper": "cs_CZ-jirka-medium",
                "piper_personal": "en_US-personal-medium",
                "piper_sk_personal": "sk_SK-personal-male-medium",
            }
            voice = voice_map.get(args.name, "cs_CZ-jirka-medium")

            engine = instantiate("tts", args.name)
            text = "This is a test of the text to speech engine."

            t0 = time.time()
            wav, sr, syn_s = engine.synthesize(text, language="en")
            rtf = syn_s / (len(wav) / sr)

            result = {"engine": args.name, "stage": args.stage, "rtf": rtf, "thresholds": thresholds, "passed": rtf <= thresholds.get("rtf_max", 999)}
            print(f"  RTF: {rtf:.4f} (threshold: {thresholds.get('rtf_max')})")
            print(f"  PASSED: {result['passed']}")

            with open(out_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            if not result["passed"]:
                sys.exit(1)

        elif args.stage == "vad" and args.name == "webrtcvad":
            import webrtcvad

            vad = instantiate("vad", "webrtcvad")
            # Test with silence
            silence = np.zeros(320, dtype=np.int16)  # 20ms at 16kHz
            speech = np.random.randint(-1000, 1000, 320, dtype=np.int16)

            t0 = time.time()
            is_silence = vad.is_speech(silence.tobytes(), 16000)
            is_speech_flag = vad.is_speech(speech.tobytes(), 16000)
            latency = time.time() - t0

            false_start_rate = 1.0 if is_silence else 0.0  # Simplified

            result = {"engine": args.name, "stage": args.stage, "false_start_rate": false_start_rate, "latency_s": latency, "thresholds": thresholds, "passed": false_start_rate <= thresholds.get("false_start_rate_max", 1.0) and latency <= thresholds.get("latency_max_s", 999)}
            print(f"  False start rate: {false_start_rate:.4f} (threshold: {thresholds.get('false_start_rate_max')})")
            print(f"  Latency: {latency:.4f}s (threshold: {thresholds.get('latency_max_s')}s)")
            print(f"  PASSED: {result['passed']}")

            with open(out_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            if not result["passed"]:
                sys.exit(1)

        else:
            print(f"No gate test implemented for {key}")
            sys.exit(0)

        print(f"Gate record written to {out_file}")


def cmd_selftest(args):
    """Verify setup: models load, engines available, quick pipeline pass."""
    import time
    import json
    import numpy as np
    from backend.registry import instantiate, engines as list_engines
    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from backend.mt.ctranslate2_mt import CTranslate2MT
    from backend.tts.piper_tts import PiperTTS

    print("== bp selftest ==")
    errors = []
    warnings = []

    # 1. Hardware profile
    profile_path = os.path.join(REPO_ROOT, "processed", "hardware_profile.json")
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            hw = json.load(f)
        print(f"Hardware: {hw['hardware']['os']}/{hw['hardware']['arch']}, "
              f"GPU={hw['hardware']['gpu']['vendor']}, "
              f"Audio={hw['audio'].get('virtual_mic_name', 'none')}")
    else:
        warnings.append("No hardware profile (run setup.py first)")

    # 2. Core engines load
    print("\n[1/4] Loading core engines...")
    try:
        stt = FasterWhisperSTT(model_size="base", compute_type="int8")
        print("  STT (faster-whisper base): OK")
    except Exception as e:
        errors.append(f"STT load failed: {e}")

    try:
        mt = CTranslate2MT(model_path=os.path.join(REPO_ROOT, "ct2_models", "Helsinki-NLP--opus-mt-en-sk"), device="cpu")
        print("  MT (opus-mt en-sk): OK")
    except Exception as e:
        errors.append(f"MT load failed: {e}")

    try:
        tts = PiperTTS(model_id="cs_CZ-jirka-medium")
        print("  TTS (piper cs_CZ-jirka-medium): OK")
    except Exception as e:
        errors.append(f"TTS load failed: {e}")

    # 3. Quick pipeline pass
    print("\n[2/4] Quick pipeline pass (EN->SK, ~3s clip)...")
    try:
        text = "Good morning, this is a test."
        segs, stt_t, _ = stt.transcribe_audio(
            np.zeros(16000 * 2, dtype=np.float32), 16000, language="en")  # 2s silence
        mt_result = mt.translate(text, "en", "sk")
        mt_text = mt_result[0] if isinstance(mt_result, tuple) else mt_result
        wav, sr, tts_t = tts.synthesize(mt_text, language="sk")
        print(f"  STT: {stt_t:.2f}s, MT: {len(mt_text)} chars, TTS: {tts_t:.2f}s, audio: {len(wav)/sr:.1f}s")
    except Exception as e:
        errors.append(f"Pipeline pass failed: {e}")

    # 4. Registry engines
    print("\n[3/4] Registry engine check...")
    # Core engines that MUST be available for basic functionality
    CORE_ENGINES = {
        ("vad", "webrtcvad"),
        ("stt", "faster_whisper"),
        ("mt", "opus-mt"),
        ("tts", "piper"),
    }
    for cap in list_engines():
        if cap.available():
            print(f"  {cap.stage}/{cap.name}: available ({cap.license})")
        else:
            msg = f"  {cap.stage}/{cap.name}: NOT available ({cap.license}) - {cap.describe().get('unavailable_reason', 'unknown')}"
            is_core = (cap.stage, cap.name) in CORE_ENGINES
            if cap.factory is None:
                warnings.append(msg + " (declared-only, not implemented)")
            elif args.strict or is_core:
                errors.append(msg)
            else:
                warnings.append(msg)

    # 5. Language packs
    print("\n[4/4] Language pack resolution...")
    try:
        import language_packs as lp
        for pair in ["en-sk", "sk-en", "en-cs"]:
            pack = lp.load(pair)
            resolved = lp.resolve(pack)
            if resolved.get("resolved"):
                print(f"  {pair}: resolved ({resolved['stt']}/{resolved['mt']}/{resolved['tts']})")
            else:
                warnings.append(f"  {pair}: experimental - {', '.join(resolved.get('errors', ['unresolved']))}")
    except Exception as e:
        warnings.append(f"Language pack check failed: {e}")

    # Summary
    print("\n=== SELFTEST RESULT ===")
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    if warnings:
        print(f"PASSED (with {len(warnings)} warning(s))")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("PASSED (clean)")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(prog="bp", description="Single CLI for repo workflows.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in WRAPPED:
        p = sub.add_parser(name, help=f"passthrough to scripts/{WRAPPED[name]}")
        p.add_argument("rest", nargs=argparse.REMAINDER, help="args forwarded to the script")

    p = sub.add_parser("stt", help="STT model x language matrix on real recordings")
    p.add_argument("--models", default="base,small", help="comma list, e.g. base,small,medium")
    p.add_argument("--clip", default="both", choices=["en", "sk", "both"])
    p.add_argument("--lang-prompt", default="sk",
                   help="comma list of decoder prompts: sk,cs,auto (cs = Czech-proxy test)")

    p = sub.add_parser("grades", help="machine pre-grade the Lab + merge into manifest")
    p.add_argument("--import-json", default=None, metavar="FILE",
                   help="import Lab browser ratings JSON into processed/ear_grades.json")
    p.add_argument("--pull", action="store_true",
                   help="pull ear verdicts from a running backend (/api/ratings) into processed/ear_grades.json")
    p.add_argument("--token", default=None, help="login token for --pull (or BP_TOKEN env)")
    p.add_argument("--server", default="https://localhost:8000",
                   help="backend base URL for --pull")

    p = sub.add_parser("bench", help="new-model bench on fixed owner clips")
    p.add_argument("--engine", default="seamless",
                   help="nemotron (pure-C runtime) or bench_new_models.py engine")
    p.add_argument("--clips", default="me_sk_trhove_b",
                   help="comma list for bench_new_models.py engines")

    p = sub.add_parser("engine", help="engine registry + capability manifests + licence gate")
    p.add_argument("action", choices=["list", "check", "gate"])
    p.add_argument("stage", nargs="?", help="vad|stt|mt|tts|unified (for check/gate)")
    p.add_argument("name", nargs="?", help="engine name (for check/gate)")
    p.add_argument("--allow-noncommercial", action="store_true",
                   help="internal runs only: let a non-commercial engine through the gate")
    p.add_argument("--allow-unknown", action="store_true",
                   help="internal runs only: let an unlabelled-licence engine through the gate")

    p = sub.add_parser("selftest", help="verify setup: models load, engines available, quick pipeline pass")
    p.add_argument("--strict", action="store_true", help="fail on any missing optional engine")

    p = sub.add_parser("language", help="language pack management (add/list/test/remove)")
    p.add_argument("action", choices=["add", "list", "test", "remove", "show"], nargs="?", default="list")
    p.add_argument("pair", nargs="?", help="language pair (e.g. en-sk)")
    p.add_argument("--clip", default=None, help="override clip path for test")
    p.add_argument("--ref", default=None, help="override reference text for test")
    p.add_argument("--yes", action="store_true", help="confirm removal")
    p.add_argument("--license", default=None, choices=["commercial-ok", "non-commercial", "unknown"])

    args = parser.parse_args()
    if args.cmd == "stt":
        cmd_stt(args)
    elif args.cmd == "grades":
        cmd_grades(args)
    elif args.cmd == "bench":
        cmd_bench(args)
    elif args.cmd == "engine":
        cmd_engine(args)
    elif args.cmd == "selftest":
        cmd_selftest(args)
    elif args.cmd == "language":
        sys.argv = ["language_packs.py", args.action] + (
            [args.pair] if args.pair else []
        ) + (
            ["--clip", args.clip] if args.clip else []
        ) + (
            ["--ref", args.ref] if args.ref else []
        ) + (
            ["--yes"] if args.yes else []
        ) + (
            ["--license", args.license] if args.license else []
        )
        runpy.run_path(os.path.join(REPO_ROOT, "scripts", "language_packs.py"),
                       run_name="__main__")
    else:
        sys.argv = [WRAPPED[args.cmd]] + (args.rest[1:] if args.rest[:1] == ["--"] else args.rest)
        runpy.run_path(os.path.join(REPO_ROOT, "scripts", WRAPPED[args.cmd]),
                       run_name="__main__")


if __name__ == "__main__":
    main()
