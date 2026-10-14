"""Read-only remote scoring check; never launches or submits a notebook."""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def classify(row):
    if row.get('errorDescription'):
        return 'FAILED', None
    if row.get('status') != 'COMPLETE':
        return 'AWAITING_SCORING', None
    try:
        score = float(row.get('publicScore'))
    except (TypeError, ValueError):
        return 'AWAITING_SCORE', None
    if not math.isfinite(score) or not 0 <= score <= 1:
        return 'INVALID_SCORE', None
    return 'SCORED', score


def main():
    sys.path.insert(0, str(HERE.parent / 'convnext1005'))
    from run_scale_probe import prepare_dns
    from kaggle.api.kaggle_api_extended import KaggleApi
    receipt = json.loads((HERE / 'submission.json').read_text())
    prepare_dns()
    api = KaggleApi()
    api.authenticate()
    rows = [s.to_dict() for s in api.competition_submissions('rsna-knee-abnormality-detection')]
    matches = [r for r in rows if str(r.get('ref')) == str(receipt['response']['ref'])]
    assert len(matches) == 1, 'Submission not found uniquely; preserve prior scoring record'
    row = matches[0]
    assert receipt['kernel'] + '?' in row.get('url', '')
    assert 'scriptVersionId=357879252' in row['url'], 'Unexpected scored version'
    state, score = classify(row)
    record = dict(submission_ref=row['ref'], kernel=receipt['kernel'], version=receipt['version'],
                  scriptVersionId=357879252, checked_at=datetime.now(timezone.utc).isoformat(),
                  state=state, api_status=row.get('status'), publicScore=score,
                  errorDescription=row.get('errorDescription'), raw=row,
                  protected_fallback=57012367, fallback_score=0.950,
                  delta_vs_fallback=None if score is None else round(score - 0.950, 6),
                  final_selection_changed=False)
    (HERE / 'scoring.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
