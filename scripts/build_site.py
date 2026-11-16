#!/usr/bin/env python3
"""Build the standalone product page from its template, CSS and saved receipts.

Run from any directory: python3 scripts/build_site.py. No inference or network.
"""
import json
import math
import re
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'documentation/site'
EV = ROOT / 'documentation/evidence/2026-11-15'


def receipt(name):
    return json.loads((EV / name).read_text(encoding='utf-8'))


def score(name):
    text = (EV / name).read_text(encoding='utf-8')
    return [float(re.search(pattern, text)[1]) for pattern in (
        r'median lag <= [\d.]+ s: ([\d.-]+)',
        r'p90 lag <= [\d.]+ s: ([\d.-]+)',
        r'lag growth <= [\d.]+ s/min: ([\d.-]+)',
        r'recognition error <= .*?: ([\d.]+)',
    )]


def span(values, digits=2, scale=1):
    return f'{min(values)*scale:.{digits}f}–{max(values)*scale:.{digits}f}'


def main():
    current, previous = score('D_wait05.score.txt'), score('A_baseline.score.txt')
    xtts = receipt('kaggle_xtts_t4_results.json')['tts']
    streaming = [case['median'] for lang in xtts.values()
                 for key, case in lang.get('cases', {}).items() if key.endswith('_streaming')]
    xvoice = receipt('kaggle_xvoice_t4_results.json')['tts']
    stage1 = [case for case in xvoice.values() if case['model'] == 'stage1']
    short16 = [c for k, c in xvoice.items() if 'stage1_nfe16' in k and k.endswith('_short')]
    short32 = [c for k, c in xvoice.items() if 'stage1_nfe32' in k and k.endswith('_short')]
    higgs = [case['median'] for lang in receipt('kaggle_higgs_t4_fp16_results.json')['tts'].values()
             for case in lang['cases'].values()]
    values = {
        'css': (ROOT / 'ui' / 'tokens.css').read_text(encoding='utf-8') + '\n' + (SITE / 'site.css').read_text(encoding='utf-8'),
        'lag': f'{current[0]:.2f}', 'p90': f'{current[1]:.2f}',
        'drift': f'{current[2]:.2f}', 'wer': f'{current[3]:.1f}',
        'previous_lag': f'{previous[0]:.2f}', 'previous_p90': f'{previous[1]:.2f}',
        'xtts_first': f"{min(c['first_audio_latency_s'] for c in streaming):.2f}–{math.ceil(max(c['first_audio_latency_s'] for c in streaming)*100)/100:.2f}",
        'xtts_rtf': span([c['rtf'] for c in streaming]),
        'xvoice16': span([c['wall_s'] for c in short16], 1),
        'xvoice32': span([c['wall_s'] for c in short32], 1),
        'xvoice_rtf': span([c['rtf'] for c in stage1 if c['nfe_step'] == 16]),
        'xvoice_wer': span([r['asr']['wer'] for c in stage1 for r in c['repeats']], 0, 100),
        'higgs_rtf': span([c['rtf'] for c in higgs], 1),
    }
    page = Template((SITE / 'index.template.html').read_text(encoding='utf-8')).substitute(values)
    (SITE / 'index.html').write_text(page, encoding='utf-8')
    print(f'wrote documentation/site/index.html ({len(page.encode())} bytes)')


if __name__ == '__main__':
    main()
