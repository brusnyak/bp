"""Private bounded Canary ASR/AST smoke; existing dataset only, no new audio uploads.
ASR WER uses source-language reference length. AST is unscored without a
reviewed target-language reference. A short dataset smoke never promotes a route.
Model API: https://huggingface.co/nvidia/canary-1b-v2
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def wer(reference, hypothesis):
    a, b = normalize(reference).split(), normalize(hypothesis).split()
    if not a:
        raise ValueError("Empty reference cannot be scored")
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(prev[j] + 1, row[-1] + 1, prev[j - 1] + (x != y)))
        prev = row
    return prev[-1] / len(a)


def smoke_complete(result):
    expected = {('sk', 'sk'), ('sk', 'en'), ('cs', 'cs'), ('cs', 'en'), ('en', 'en')}
    rows = result.get('rows', [])
    seen = {(r.get('source_lang'), r.get('target_lang')) for r in rows}
    return not result.get('errors') and len(rows) == len(expected) and seen == expected and all(
        r.get('status') == 'complete' for r in rows
    )


def main():
    start = time.time()
    out = Path('/kaggle/working/canary-validation.json')
    result = {'schema': 1, 'model': 'nvidia/canary-1b-v2', 'rows': [],
              'scope': 'short existing dataset smoke; not a real-mic release gate',
              'vram_limit_gib': 8, 'errors': []}
    def save():
        result['elapsed_s'] = round(time.time() - start, 2)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    save()
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                        'git+https://github.com/huggingface/transformers', 'accelerate',
                        'soundfile', 'librosa'], check=True, timeout=480)
        import torch
        import soundfile as sf
        import librosa
        from huggingface_hub import model_info
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; CPU fallback refused')
        result['gpu'] = torch.cuda.get_device_name(0)
        torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats()
        result['versions'] = {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'accelerate']}
        revision = model_info(result['model']).sha
        result['revision'] = revision
        save()
        processor = AutoProcessor.from_pretrained(result['model'], revision=revision)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(result['model'], revision=revision,
                                                       torch_dtype=torch.float16).to('cuda').eval()
        for lang in ['sk', 'cs', 'en']:
            files = sorted(Path('/kaggle/input').rglob(lang + '.wav'))
            if not files:
                result['rows'].append({'lang': lang, 'status': 'missing_clip'}); save(); continue
            clip = files[0]
            reference = clip.with_suffix('.txt')
            if not reference.exists():
                result['rows'].append({'lang': lang, 'status': 'missing_reference'}); save(); continue
            ref = reference.read_text().strip()
            audio, _ = librosa.load(clip, sr=16000, mono=True)
            identity = {'clip': str(clip), 'sha256': hashlib.sha256(clip.read_bytes()).hexdigest(),
                        'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
                        'duration_s': round(len(audio)/16000, 3), 'source_lang': lang}
            for target in ([lang, 'en'] if lang != 'en' else ['en']):
                row = dict(identity, target_lang=target, task='asr' if lang == target else 'ast')
                try:
                    inputs = processor.apply_transcription_request(audio=audio, source_language=lang,
                                                                   target_language=target).to(model.device)
                    # Match floating input tensors to model dtype while preserving token ids.
                    inputs = {k: v.to(dtype=model.dtype) if torch.is_tensor(v) and v.is_floating_point() else v
                              for k, v in inputs.items()}
                    torch.cuda.synchronize(); t = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=256)
                    torch.cuda.synchronize()
                    row['latency_s'] = round(time.perf_counter() - t, 4)
                    decoded = processor.batch_decode(generated, skip_special_tokens=True)
                    row['output'] = decoded[0]
                    row['rtf'] = round(row['latency_s'] / identity['duration_s'], 4)
                    if lang == target:
                        row.update(reference=ref, wer=round(wer(ref, row['output']), 4),
                                   score_status='short_clip_only_reference_completeness_unreviewed')
                    else:
                        row['score_status'] = 'unscored_no_reviewed_English_reference'
                    row['status'] = 'complete'
                except Exception:
                    row.update(status='error', error=traceback.format_exc())
                result['rows'].append(row)
                result['peak_allocated_gib'] = round(torch.cuda.max_memory_allocated()/1024**3, 3)
                result['peak_reserved_gib'] = round(torch.cuda.max_memory_reserved()/1024**3, 3)
                save()
    except Exception:
        result['errors'].append(traceback.format_exc())
    result['acceptance'] = 'smoke_complete_route_unverified' if smoke_complete(result) else 'incomplete'
    save()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not smoke_complete(result):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
