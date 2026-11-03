"""Re-score historical JSON without rerunning inference or printing transcripts."""
import argparse
import hashlib
import json
from pathlib import Path
from validate import normalize, wer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('result', type=Path)
    args = ap.parse_args()
    raw = args.result.read_bytes()
    data = json.loads(raw)
    output = {'source_sha256': hashlib.sha256(raw).hexdigest(),
              'scope': 'rescored saved strings; inference not rerun; reference completeness unreviewed',
              'rows': []}
    for name, row in data['floor'].items():
        output['rows'].append({'name': name, 'reported_wer': row.get('wer'),
                               'reference_words': len(normalize(row['ref_text']).split()),
                               'hypothesis_words': len(normalize(row['asr_text']).split()),
                               'corrected_wer': round(wer(row['ref_text'], row['asr_text']), 4)})
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
