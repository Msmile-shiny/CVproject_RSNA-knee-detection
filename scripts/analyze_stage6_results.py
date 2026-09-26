"""Read-only prediction audit; JSON output, no checkpoint or threshold selection."""
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
    return float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))


def analyze(old, new, truth_path):
    receipts = [json.loads((d/'run_receipt.json').read_text()) for d in (old, new)]
    assert [r['completed_epochs'] for r in receipts] == [12, 24]
    for key in ('labels_sha256', 'weights_sha256', 'group_unit', 'train_metadata_sha256', 'series_metadata_sha256'):
        assert receipts[0][key] == receipts[1][key], key
    raw=truth_path.read_bytes()
    # Git's Windows checkout may change only LF to CRLF. No other normalization.
    assert receipts[0]['train_metadata_sha256'] in [hashlib.sha256(raw).hexdigest(),hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()]
    frames = [pd.read_csv(d/'gold_development_predictions.csv', dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID') for d in (old, new)]
    a, b = frames
    assert a.index.is_unique and b.index.is_unique and set(a.index) == set(b.index)
    assert list(a.columns) == list(b.columns)
    b = b.loc[a.index]
    truth = pd.read_csv(truth_path, dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[a.index, a.columns]
    y = truth.to_numpy(float)
    assert np.isin(y, [0,1]).all()
    for f in (a,b):
        assert np.isfinite(f.to_numpy()).all() and ((f>=0)&(f<=1)).all().all()
    rows=[]
    for col in a.columns:
        x,z=auc(truth[col].to_numpy(),a[col].to_numpy()),auc(truth[col].to_numpy(),b[col].to_numpy())
        assert abs(x-receipts[0]['gold_auc_by_class'][col])<1e-8
        assert abs(z-receipts[1]['gold_auc_by_class'][col])<1e-8
        rows.append(dict(target=col, positives=int(truth[col].sum()),auc12=x,auc24=z,delta=z-x))
    # Paired case bootstrap is descriptive only: Gold is reused and non-independent.
    rng=np.random.default_rng(42); differences=[]
    pa,pb=a.to_numpy(),b.to_numpy()
    for _ in range(2000):
        idx=rng.integers(len(a),size=len(a))
        pairs=[(auc(y[idx,j],pa[idx,j]),auc(y[idx,j],pb[idx,j])) for j in range(len(a.columns))]
        if all(x is not None and z is not None for x,z in pairs):
            differences.append(float(np.mean([z-x for x,z in pairs])))
    histories=[json.loads((d/'history.json').read_text()) for d in (old,new)]
    assert histories[1][:12] == histories[0], 'Continuation history mismatch'
    return dict(gold_cases=len(a),gold_independent=False,classes=rows,
                macro_auc12=float(np.mean([r['auc12'] for r in rows])),
                macro_auc24=float(np.mean([r['auc24'] for r in rows])),
                descriptive_paired_bootstrap_delta_ci95=np.quantile(differences,[.025,.975]).tolist(),
                bootstrap_valid=len(differences),bootstrap_seed=42,
                loss12=histories[0][-1]['loss'],loss24=histories[1][-1]['loss'],
                mse12=receipts[0]['pseudo_validation_mse'],mse24=receipts[1]['pseudo_validation_mse'],
                seconds_continuation=receipts[1]['seconds'],
                decision='No epoch-48 run or leaderboard blend; audit inputs and supervision next')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('old',type=Path);p.add_argument('new',type=Path);p.add_argument('truth',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args(); result=analyze(args.old,args.new,args.truth)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
