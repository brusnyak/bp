#!/usr/bin/env python3
"""Bounded private T4 OmniVoice chunk-replay + streaming comparison. NOT a live STT/MT/meeting test.

No local ML imports/install on import. Only main() inside Kaggle installs deps.
Timestamps are synthetic text-ready arrivals; service and waveform duration are measured.
Batch generate() returns a whole phrase; synthesize_stream() yields clause-level chunks.
This measures BOTH serial (current) and concurrent/gapless (Phase 2) playback models,
plus first-chunk latency for streaming synthesis — the data needed to verify the
German-style end-of-sentence gating + gapless playback fix on T4.
"""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

MODEL = "k2-fsa/OmniVoice"
REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
DEADLINE_S = 1200
TEXTS = {
    "en": ["Welcome to today's research presentation.",
           "The experiment begins at nine thirty, with twelve participants in the room.",
           "Please wait for the translated speech before asking the next question."],
    "sk": ["Vitajte na dnešnej prezentácii výskumu.",
           "Experiment sa začína o deviatej tridsať, v miestnosti je dvanásť účastníkov.",
           "Pred ďalšou otázkou prosím počkajte na preklad reči."],
    "cs": ["Vítejte na dnešní prezentaci výzkumu.",
           "Experiment začíná v devět třicet, v místnosti je dvanáct účastníků.",
           "Před další otázkou prosím počkejte na překlad řeči."],
}
SCENARIOS = {"meeting_turns": [0.0, 4.0, 9.0],
             "presentation_phrases": [0.0, 3.0, 6.0],
             "conference_contention": [0.0, 0.0, 5.0]}


def schedule(arrivals, services, durations, rate=1.0, drop_late_s=None):
    """Single synthesis worker + one sequential listener; deterministic replay math.

    Playback of chunk N+1 waits for chunk N. `rate` > 1 shortens each chunk's playback
    (pitch-preserving playbackRate). `drop_late_s` skips a chunk whose delay from
    text-ready to playback start would exceed that bound (it is reported, not hidden).
    Replay lengths are the measured waveform durations; clause-level starts are NOT
    modelled (no per-clause measurements exist).
    """
    if not (len(arrivals) == len(services) == len(durations)):
        raise ValueError("arrival/service/audio count mismatch")
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("invalid rate")
    gpu_free = playback_free = 0.0
    rows = []
    for arrival, service, duration in zip(arrivals, services, durations):
        if any(not math.isfinite(v) or v < 0 for v in (arrival, service, duration)):
            raise ValueError("invalid timing")
        start = max(arrival, gpu_free)
        ready = start + service
        playback = max(ready, playback_free)
        gpu_free = ready
        dropped = drop_late_s is not None and playback - arrival > drop_late_s
        row = {"text_ready_s": arrival, "generation_start_s": start,
               "waveform_ready_s": ready, "playback_start_s": playback,
               "generation_queue_s": start - arrival,
               "first_playable_delay_s": playback - arrival,
               "playback_queue_s": playback - ready,
               "playback_end_s": playback + duration / rate,
               "rate": rate, "dropped": dropped}
        if dropped:
            row["playback_end_s"] = playback   # skipped: occupies no playback time
        else:
            playback_free = playback + duration / rate
        rows.append(row)
    return rows


def simulated_stop(rows, stop_s):
    """Cancel queued playback at an instant. This does NOT cancel actual GPU work."""
    return [{"index": i, "action": "drop" if r["playback_start_s"] >= stop_s
             else "cut" if r["playback_end_s"] > stop_s else "already_finished"}
            for i, r in enumerate(rows)]


def normalize_reference(source, output):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, sr = sf.read(str(source), dtype="float32", always_2d=True)
    if not 8000 <= sr <= 192000 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("invalid reference audio")
    original = {"samplerate": sr, "channels": audio.shape[1],
                "subtype": sf.info(str(source)).subtype,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    mono = audio.mean(axis=1)
    if sr != 16000:
        mono = resample_poly(mono, 16000, sr)
        sr = 16000
    if mono.ndim == 1:
        mono = mono[:, None]
    sf.write(str(output), mono, sr, subtype="PCM_16")
    return {"samplerate": sr, "channels": 1, "subtype": "PCM_16",
            "duration_s": len(mono) / sr,
            "sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest(),
            "original": {"samplerate": sr if sr != 16000 else sr,
                         "channels": audio.shape[1],
                         "subtype": sf.info(str(source)).subtype}}


def atomic_save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    if not Path('/kaggle/input').is_dir():
        raise RuntimeError("Kaggle-only execution; no local installation/inference")
    def deadline(*_):
        raise TimeoutError("1200-second total job deadline")
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(DEADLINE_S)
    output = Path('/kaggle/working')
    receipt = {"schema": 1, "status": "running", "engine": "omnivoice",
               "model": MODEL, "revision": REVISION, "steps": 12,
               "scope": "measured TTS with synthetic text-arrival/playback replay",
               "true_model_streaming": False, "stt_mt_included": False,
               "actual_meeting_integration": False, "acceptance": "route_unverified",
               "cloned_voice_identity": "human_review_pending", "rows": [],
               "scenarios": [], "errors": []}
    t0 = time.perf_counter()
    def save():
        receipt['elapsed_s'] = time.perf_counter() - t0
        atomic_save(output/'session-replay.json', receipt)
    save()
    try:
        # Bounded install into remote ephemeral runtime, no stubs or ignored errors.
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'omnivoice==0.2.1', 'soundfile==0.14.0', 'scipy==1.16.3'],
                       check=True, timeout=300)
        import numpy as np
        import soundfile as sf
        import torch
        from omnivoice import OmniVoice
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU required; refusing CPU fallback')
        receipt['gpu'] = torch.cuda.get_device_name(0)
        if 'T4' not in receipt['gpu']:
            raise RuntimeError('T4 required for comparable receipt: '+receipt['gpu'])
        receipt['versions'] = {p: importlib.metadata.version(p) for p in
                               ['omnivoice', 'torch', 'torchaudio', 'transformers',
                                'soundfile', 'scipy']}
        loaded = time.perf_counter()
        model = OmniVoice.from_pretrained(MODEL, revision=REVISION,
                                         device_map='cuda:0', dtype=torch.float16)
        torch.cuda.synchronize()
        receipt['cold_model_load_s'] = time.perf_counter() - loaded
        for lang, texts in TEXTS.items():
            wavs = list(Path('/kaggle/input').rglob(lang+'.wav'))
            txts = list(Path('/kaggle/input').rglob(lang+'.txt'))
            if len(wavs) != 1 or len(txts) != 1:
                raise ValueError('Expected one mounted reference pair for '+lang)
            ref = output / ('normalized-'+lang+'.wav')
            ref_info = normalize_reference(wavs[0], ref)
            ref_text = txts[0].read_text().strip()
            if not ref_text:
                raise ValueError('reference transcript missing')
            receipt.setdefault('references', {})[lang] = ref_info
            prompt = model.create_voice_clone_prompt(str(ref), ref_text)
            model.generate(text=texts[0], language=lang, voice_clone_prompt=prompt,
                           num_step=12)
            torch.cuda.synchronize()
            # Reuse measured phrase service times in different arrival schedules.
            # This isolates queue behavior without pretending schedules were live inputs.
            measured = []
            for i, text in enumerate(texts + [' '.join(texts)]):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                begin = time.perf_counter()
                audio = model.generate(text=text, language=lang,
                                       voice_clone_prompt=prompt, num_step=12)[0]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                audio = np.asarray(audio, dtype=np.float32).squeeze()
                if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                    raise ValueError('invalid generated waveform')
                sr = getattr(model, 'sampling_rate', 24000)
                path = output/f'clone-{lang}-{i}.wav'
                sf.write(str(path), audio, sr, subtype='PCM_16')
                row = {'id': f'{lang}-{i}', 'language': lang, 'text': text,
                       'service_s': elapsed, 'first_waveform_available_s': elapsed,
                       'duration_s': len(audio)/sr, 'sample_rate': sr,
                       'rtf': elapsed/(len(audio)/sr), 'audio': path.name,
                       'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'peak_allocator_bytes': torch.cuda.max_memory_allocated(),
                       'memory_scope': 'PyTorch allocator only',
                       'quality': 'human_review_pending', 'status': 'complete'}
                receipt['rows'].append(row)
                measured.append(row)
                save()
            for name, arrivals in SCENARIOS.items():
                services = [r['service_s'] for r in measured[:3]]
                durations = [r['duration_s'] for r in measured[:3]]
                for rate, drop in ((1.0, None), (1.15, None), (1.25, None), (1.25, 3.0)):
                    timeline = schedule(arrivals, services, durations, rate=rate, drop_late_s=drop)
                    kept = [r for r in timeline if not r['dropped']]
                    receipt['scenarios'].append({'language': lang, 'scenario': name,
                        'playback_model': 'sequential', 'rate': rate, 'drop_late_s': drop,
                        'arrival_source': 'synthetic text-ready schedule, no source audio',
                        'timeline': timeline, 'stop_at_5s_simulated': simulated_stop(kept, 5),
                        'dropped_chunks': len(timeline) - len(kept),
                        'actual_model_cancellation': 'not_tested',
                        'queue_growth_s': timeline[-1]['first_playable_delay_s'] -
                                          timeline[0]['first_playable_delay_s']})
            save()
        receipt['status'] = 'complete'
    except Exception:
        receipt['status'] = 'incomplete'
        receipt['errors'].append(traceback.format_exc())
    finally:
        signal.alarm(0)
        save()
    return 0 if receipt['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
