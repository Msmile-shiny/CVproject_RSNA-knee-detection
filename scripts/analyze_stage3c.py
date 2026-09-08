"""Recompute the matched-study 2x2 experiment; no training or label edits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    frames = {}
    rows = {}
    for name in ('v5s1', 'v5s3a', 'v5s3b', 'v5s3c'):
        path = ROOT / 'results' / name
        df = pd.read_csv(path / 'gold_validation_predictions_s42.csv').set_index('StudyInstanceUID').sort_index()
        assert df.index.is_unique and len(df) == 58
        frames[name] = df
        targets = [c[5:] for c in df if c.startswith('true_')]
        assert len(targets) == 12
        rows[name] = {t: roc_auc_score(df['true_' + t], df['prob_' + t]) for t in targets}
        rows[name]['macro'] = float(np.mean(list(rows[name].values())))
        if name != 'v5s1':
            pd.testing.assert_frame_equal(df[['true_' + t for t in targets]], frames['v5s1'][['true_' + t for t in targets]])
    deltas = {t: rows['v5s3c'][t] - rows['v5s1'][t] for t in rows['v5s1']}
    # Paired study bootstrap: descriptive uncertainty, NOT independent validation.
    rng = np.random.default_rng(42)
    a, b = frames['v5s1'], frames['v5s3c']
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(a), len(a))
        y = a[['true_' + t for t in targets]].values[idx]
        if any(len(np.unique(y[:, j])) < 2 for j in range(12)):
            continue
        pa = a[['prob_' + t for t in targets]].values[idx]
        pb = b[['prob_' + t for t in targets]].values[idx]
        boot.append(roc_auc_score(y, pb) - roc_auc_score(y, pa))
    result = {'auc': rows, 'delta_3c_minus_v5s1': deltas,
              'paired_bootstrap_95pct': np.quantile(boot, [.025, .975]).tolist(),
              'bootstrap_valid_replicates': len(boot),
              'public_user_reported': {'v5s3a': .818, 'v5s3c': .807},
              'caveat': '58 gold studies are a reused development set.'}
    out = ROOT / 'results/v5s3c/comparison.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
