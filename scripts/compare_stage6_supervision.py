"""Compare Stage 6B and 6C on identical cases and both fixed label targets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))


def frame(path):
    result = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    assert result.index.is_unique, path
    assert np.isfinite(result.to_numpy(float)).all(), path
    return result


def compare(base, candidate, metadata, old_labels, new_labels):
    a, b = [json.loads((directory / 'run_receipt.json').read_text()) for directory in (base, candidate)]
    for receipt in (a, b):
        assert receipt['status'] == 'PILOT_COMPLETE' and receipt['completed_epochs'] == 24
        assert receipt['gold_independent'] is False
    for key in ('weights_sha256', 'group_unit', 'train_metadata_sha256', 'series_metadata_sha256',
                'train_count', 'val_count', 'gold_count'):
        assert a[key] == b[key], key
    for key in ('size', 'windows', 'batch_size', 'fold', 'seed', 'pooling', 'backbone_lr', 'head_lr'):
        assert a['config'][key] == b['config'][key], key
    assert a['config']['pooling'] == 'mean'
    for receipt, path in ((a, old_labels), (b, new_labels)):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['labels_sha256'], path
    raw = metadata.read_bytes()
    assert a['train_metadata_sha256'] in (hashlib.sha256(raw).hexdigest(),
                                           hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest())
    gold_a, gold_b = [frame(d / 'gold_development_predictions.csv') for d in (base, candidate)]
    val_a, val_b = [frame(d / 'pseudo_validation_predictions.csv') for d in (base, candidate)]
    assert len(gold_a) == len(gold_b) == 58 and len(val_a) == len(val_b) == 852
    assert set(gold_a.index) == set(gold_b.index) and set(val_a.index) == set(val_b.index)
    assert list(gold_a.columns) == list(gold_b.columns) == list(val_a.columns) == list(val_b.columns)
    gold_b, val_b = gold_b.loc[gold_a.index], val_b.loc[val_a.index]
    cols = list(gold_a.columns)
    truth = pd.read_csv(metadata, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[gold_a.index, cols]
    assert np.isin(truth.to_numpy(float), [0, 1]).all()
    classes = []
    for col in cols:
        y = truth[col].to_numpy(float)
        old, new = auc(y, gold_a[col].to_numpy(float)), auc(y, gold_b[col].to_numpy(float))
        assert abs(old - a['gold_auc_by_class'][col]) < 1e-8
        assert abs(new - b['gold_auc_by_class'][col]) < 1e-8
        classes.append({'target': col, 'positives': int(y.sum()), 'baseline_auc': old,
                        'candidate_auc': new, 'delta': new - old,
                        'rank_correlation': float(gold_a[col].rank().corr(gold_b[col].rank()))})
    label_a, label_b = [frame(path) for path in (old_labels, new_labels)]
    assert set(val_a.index) <= set(label_a.index) and set(val_a.index) <= set(label_b.index)
    common_mse = {}
    for name, labels in (('old', label_a), ('candidate', label_b)):
        target = labels.loc[val_a.index, ['prob_' + col for col in cols]].to_numpy(float)
        assert np.isfinite(target).all()
        common_mse[name] = {
            'baseline': float(np.mean((val_a.to_numpy(float) - target) ** 2)),
            'candidate': float(np.mean((val_b.to_numpy(float) - target) ** 2)),
        }
    assert abs(common_mse['old']['baseline'] - a['pseudo_validation_mse']) < 1e-6
    assert abs(common_mse['candidate']['candidate'] - b['pseudo_validation_mse']) < 1e-6
    y, pa, pb = truth.to_numpy(float), gold_a.to_numpy(float), gold_b.to_numpy(float)
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(3000):
        idx = rng.integers(len(y), size=len(y))
        pair = [(auc(y[idx, j], pa[idx, j]), auc(y[idx, j], pb[idx, j])) for j in range(len(cols))]
        if all(x is not None and z is not None for x, z in pair):
            deltas.append(float(np.mean([z - x for x, z in pair])))
    return {'gold_cases': len(gold_a), 'validation_cases': len(val_a), 'gold_independent': False,
            'baseline_macro_auc': float(np.mean([r['baseline_auc'] for r in classes])),
            'candidate_macro_auc': float(np.mean([r['candidate_auc'] for r in classes])),
            'descriptive_paired_bootstrap_delta_ci95': np.quantile(deltas, [.025, .975]).tolist(),
            'bootstrap_valid': len(deltas), 'validation_mse_on_same_target': common_mse,
            'classes': classes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'candidate', 'metadata', 'old_labels', 'new_labels'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.metadata, args.old_labels, args.new_labels)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'classes'}, indent=2))
