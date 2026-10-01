"""Paired equal-epoch Stage 6 input ablation; Gold is development only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def auc(y, p):
    positive, negative = p[y == 1], p[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    return float(((positive[:, None] > negative).sum() + .5 * (positive[:, None] == negative).sum()) /
                 (len(positive) * len(negative)))


def read(path):
    table = pd.read_csv(path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    assert table.index.is_unique and np.isfinite(table.to_numpy(float)).all(), path
    return table


def compare(reference, candidate, metadata, dimension, epochs=12):
    a, b = [json.loads((folder / 'run_receipt.json').read_text()) for folder in (reference, candidate)]
    for receipt in (a, b):
        assert receipt['status'] == 'PILOT_COMPLETE' and receipt['completed_epochs'] == epochs
        assert receipt['config']['epochs'] == epochs
        assert receipt['device'] == 'cuda' and receipt['first_backbone_gradient_norm'] > 0
        assert receipt['gold_independent'] is False
    for key in ('labels_sha256', 'weights_sha256', 'group_unit', 'train_metadata_sha256',
                'series_metadata_sha256', 'train_count', 'val_count', 'gold_count'):
        assert a[key] == b[key], key
    assert a['config']['implementation_sha256'] == b['config']['implementation_sha256']
    fixed = ('batch_size', 'fold', 'seed', 'pooling', 'backbone_lr', 'head_lr', 'smoke', 'epochs')
    for key in fixed:
        assert a['config'][key] == b['config'][key], key
    assert dimension in ('windows', 'size')
    assert all(a['config'][key] == b['config'][key] for key in ('windows', 'size') if key != dimension)
    assert a['config'][dimension] != b['config'][dimension]
    raw = metadata.read_bytes()
    assert a['train_metadata_sha256'] in (hashlib.sha256(raw).hexdigest(),
                                           hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest())
    gold_a, gold_b = [read(folder / 'gold_development_predictions.csv') for folder in (reference, candidate)]
    val_a, val_b = [read(folder / 'pseudo_validation_predictions.csv') for folder in (reference, candidate)]
    assert len(gold_a) == len(gold_b) == 58 and len(val_a) == len(val_b) == 852
    assert set(gold_a.index) == set(gold_b.index) and set(val_a.index) == set(val_b.index)
    assert list(gold_a.columns) == list(gold_b.columns) == list(val_a.columns) == list(val_b.columns)
    gold_b, val_b = gold_b.loc[gold_a.index], val_b.loc[val_a.index]
    truth = pd.read_csv(metadata, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[gold_a.index, gold_a.columns]
    assert np.isin(truth.to_numpy(float), [0, 1]).all()
    rows = []
    for col in gold_a.columns:
        y = truth[col].to_numpy(float)
        old, new = auc(y, gold_a[col].to_numpy(float)), auc(y, gold_b[col].to_numpy(float))
        assert abs(old - a['gold_auc_by_class'][col]) < 1e-8
        assert abs(new - b['gold_auc_by_class'][col]) < 1e-8
        rows.append({'target': col, 'positives': int(y.sum()), 'reference_auc': old,
                     'candidate_auc': new, 'delta': new - old,
                     'rank_correlation': float(gold_a[col].rank().corr(gold_b[col].rank()))})
    histories = [json.loads((folder / 'history.json').read_text()) for folder in (reference, candidate)]
    assert len(histories[0]) == len(histories[1]) == epochs
    y, pa, pb = truth.to_numpy(float), gold_a.to_numpy(float), gold_b.to_numpy(float)
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(3000):
        idx = rng.integers(len(y), size=len(y))
        paired = [(auc(y[idx, j], pa[idx, j]), auc(y[idx, j], pb[idx, j])) for j in range(y.shape[1])]
        if all(x is not None and z is not None for x, z in paired):
            deltas.append(float(np.mean([z - x for x, z in paired])))
    return {'epochs': epochs, 'changed_dimension': dimension, 'reference_value': a['config'][dimension],
            'candidate_value': b['config'][dimension], 'gold_cases': 58, 'validation_cases': 852,
            'gold_independent': False,
            'reference_macro_auc': float(np.mean([row['reference_auc'] for row in rows])),
            'candidate_macro_auc': float(np.mean([row['candidate_auc'] for row in rows])),
            'descriptive_paired_bootstrap_delta_ci95': np.quantile(deltas, [.025, .975]).tolist(),
            'validation_mse': {'reference': a['pseudo_validation_mse'], 'candidate': b['pseudo_validation_mse']},
            'final_training_loss': {'reference': histories[0][-1]['loss'], 'candidate': histories[1][-1]['loss']},
            'classes': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('--dimension', choices=('windows', 'size'), required=True)
    parser.add_argument('--epochs', type=int, choices=(12, 24), default=12)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.metadata, args.dimension, args.epochs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'classes'}, indent=2))
