"""UID-aligned development rank-blend diagnostics; never writes a submission."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def compare(parent_path, member_path, truth_path, repeats=1000):
    frames=[pd.read_csv(p,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID') for p in (parent_path,member_path,truth_path)]
    parent,member,truth=frames
    for f in frames:
        if not f.index.is_unique: raise ValueError('Duplicate UID')
        if set(f.index)!=set(truth.index): raise ValueError('UID sets differ; refusing inner-join deletion')
        if set(f.columns)!=set(truth.columns): raise ValueError('Target columns differ')
    parent=parent.loc[truth.index,truth.columns]; member=member.loc[truth.index,truth.columns]
    for f in (parent,member,truth):
        v=f.to_numpy(float)
        if not np.isfinite(v).all() or (v<0).any() or (v>1).any(): raise ValueError('Invalid numeric values')
    y=truth.to_numpy(float)
    if not np.isin(y,[0,1]).all(): raise ValueError('Gold truth must be binary')
    p=parent.rank(pct=True).to_numpy(); m=member.rank(pct=True).to_numpy()
    def auc(a, yy=y):
        return np.mean([roc_auc_score(yy[:,j],a[:,j]) for j in range(a.shape[1])])
    base=auc(p); rng=np.random.default_rng(42); draws=[]
    for _ in range(repeats):
        idx=rng.integers(0,len(y),len(y))
        if all(len(np.unique(y[idx,j]))==2 for j in range(y.shape[1])): draws.append(idx)
    rows=[]
    for alpha in (0.,.02,.05):
        blend=(1-alpha)*p+alpha*m
        ds=[auc(blend[idx],y[idx])-auc(p[idx],y[idx]) for idx in draws]
        rows.append(dict(alpha=alpha,auc=auc(blend),delta=auc(blend)-base,
                         paired_bootstrap_ci95=np.quantile(ds,[.025,.975]).tolist() if ds else None))
    return dict(parent_auc=base,member_auc=auc(m),rows=rows,bootstrap_valid=len(draws),
                per_class_rank_correlation={c:float(np.corrcoef(p[:,j],m[:,j])[0,1]) for j,c in enumerate(truth.columns)},
                limitation='Development only: Gold influenced historical labels and model selection; parent Gold exclusion not verified. Bootstrap cannot remove selection bias.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--parent',required=True); p.add_argument('--member',required=True); p.add_argument('--truth',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); report=compare(a.parent,a.member,a.truth)
    Path(a.out).write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
