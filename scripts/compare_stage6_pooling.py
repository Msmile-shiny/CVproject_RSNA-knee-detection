"""Paired 24-epoch mean/attention analysis on the fixed Gold development set."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def auc(y,p):
    positive=p[y==1];negative=p[y==0]
    if not len(positive) or not len(negative):return None
    return float(((positive[:,None]>negative).sum()+.5*(positive[:,None]==negative).sum())/(len(positive)*len(negative)))


def compare(mean_dir,attention_dir,truth_path):
    dirs=[Path(mean_dir),Path(attention_dir)]
    receipts=[json.loads((d/'run_receipt.json').read_text()) for d in dirs]
    for r in receipts:
        assert r['status']=='PILOT_COMPLETE' and r['completed_epochs']==24
        assert r['gold_independent'] is False
    for key in ('labels_sha256','weights_sha256','group_unit','train_metadata_sha256','series_metadata_sha256','train_count','val_count','gold_count'):
        assert receipts[0][key]==receipts[1][key],key
    for key in ('size','windows','batch_size','fold','seed','backbone_lr','head_lr'):
        assert receipts[0]['config'][key]==receipts[1]['config'][key],key
    assert receipts[0]['config']['pooling']=='mean' and receipts[1]['config']['pooling']=='attention'
    raw=Path(truth_path).read_bytes()
    assert receipts[0]['train_metadata_sha256'] in (hashlib.sha256(raw).hexdigest(),hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest())
    frames=[pd.read_csv(d/'gold_development_predictions.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID') for d in dirs]
    mean,attention=frames
    assert mean.index.is_unique and attention.index.is_unique and set(mean.index)==set(attention.index)
    assert list(mean.columns)==list(attention.columns)
    attention=attention.loc[mean.index]
    truth=pd.read_csv(truth_path,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[mean.index,mean.columns]
    assert len(mean)==58 and np.isin(truth.to_numpy(),[0,1]).all()
    for frame in (mean,attention):
        values=frame.to_numpy()
        assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
    classes=[]
    for col in mean:
        y=truth[col].to_numpy(float);a=mean[col].to_numpy(float);b=attention[col].to_numpy(float)
        old,new=auc(y,a),auc(y,b)
        assert abs(old-receipts[0]['gold_auc_by_class'][col])<1e-8
        assert abs(new-receipts[1]['gold_auc_by_class'][col])<1e-8
        pos=y==1;neg=~pos
        pa=a[pos,None]-a[None,neg]
        pb=b[pos,None]-b[None,neg]
        classes.append(dict(target=col,positives=int(pos.sum()),negatives=int(neg.sum()),
                            mean_auc=old,attention_auc=new,delta=new-old,
                            corrected_pairs=int(((pa<=0)&(pb>0)).sum()),
                            spoiled_pairs=int(((pa>0)&(pb<=0)).sum()),
                            rank_correlation=float(mean[col].rank().corr(attention[col].rank()))))
    y=truth.to_numpy(float);a=mean.to_numpy(float);b=attention.to_numpy(float)
    rng=np.random.default_rng(42);deltas=[]
    for _ in range(3000):
        idx=rng.integers(len(y),size=len(y));pairs=[(auc(y[idx,j],a[idx,j]),auc(y[idx,j],b[idx,j])) for j in range(y.shape[1])]
        if all(old is not None and new is not None for old,new in pairs):
            deltas.append(float(np.mean([new-old for old,new in pairs])))
    hist=[json.loads((d/'history.json').read_text()) for d in dirs]
    assert len(hist[0])==len(hist[1])==24
    return dict(cases=len(mean),class_count=len(classes),gold_independent=False,
                mean_macro_auc=float(np.mean([c['mean_auc'] for c in classes])),
                attention_macro_auc=float(np.mean([c['attention_auc'] for c in classes])),
                descriptive_paired_bootstrap_delta_ci95=np.quantile(deltas,[.025,.975]).tolist(),
                bootstrap_valid=len(deltas),bootstrap_seed=42,
                mean_loss24=hist[0][-1]['loss'],attention_loss24=hist[1][-1]['loss'],
                mean_pseudo_mse=receipts[0]['pseudo_validation_mse'],
                attention_pseudo_mse=receipts[1]['pseudo_validation_mse'],
                classes=classes)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mean',type=Path);p.add_argument('attention',type=Path);p.add_argument('truth',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args();result=compare(args.mean,args.attention,args.truth)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('mean_macro_auc','attention_macro_auc','descriptive_paired_bootstrap_delta_ci95','bootstrap_valid')},indent=2))
