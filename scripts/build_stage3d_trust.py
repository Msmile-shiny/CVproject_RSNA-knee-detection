"""Build auxiliary ranking eligibility, keeping historical v5 labels untouched.

Rule-selected candidates, not verified clinical truth. Entire reports containing
known uncertainty markers are conservatively excluded from ranking only.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from report_extractor_v2 import TARGETS, UNCERTAIN, normalize

ROOT = Path(__file__).resolve().parents[1]
LABEL_SHA = 'c13adffaabf4f8e518abb038282bb1aa09baac7652a9165e030710c457d0be6a'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {'labels': ROOT / 'data/processed/v5_labels.csv',
             'text': ROOT / 'data/processed/report_labels_v2.csv',
             'teacher': ROOT / 'kaggle_dataset/archive/oof.npz',
             'metadata': ROOT / 'data/metadata/train.csv',
             'extractor': ROOT / 'scripts/report_extractor_v2.py'}
    assert digest(paths['labels']) == LABEL_SHA, 'Historical labels changed'
    meta = pd.read_csv(paths['metadata']).set_index('StudyInstanceUID')
    txt = pd.read_csv(paths['text']).set_index('StudyInstanceUID')
    z = np.load(paths['teacher'], allow_pickle=True)
    assert list(z['targets'].astype(str)) == TARGETS
    pred = pd.DataFrame(z['pred'], index=z['ids'].astype(str), columns=TARGETS)
    for frame in (meta, txt, pred):
        assert frame.index.is_unique and set(frame.index) == set(meta.index)
    ids = meta.index[~meta[TARGETS].notna().all(axis=1)].sort_values()
    txt, pred = txt.loc[ids], pred.loc[ids]
    assert np.isfinite(pred.values).all() and ((pred.values >= 0) & (pred.values <= 1)).all()
    uncertain = meta.loc[ids, 'Report'].fillna('').map(lambda s: bool(UNCERTAIN.search(normalize(s)))).values
    state = np.zeros((len(ids), len(TARGETS)), dtype=np.int8)
    stats = {}
    for j, t in enumerate(TARGETS):
        p, n = txt[t + '__npos'].values, txt[t + '__nneg'].values
        score, conf, teacher = txt[t].values, txt[t + '__conf'].values, pred[t].values
        pos = (p > 0) & (n == 0) & (score >= .8) & (conf >= .7) & (teacher >= .8) & ~uncertain
        neg = (n > 0) & (p == 0) & (score <= .2) & (conf >= .55) & (teacher <= .2) & ~uncertain
        state[pos, j], state[neg, j] = 1, -1
        pp, pn = pos.mean(), neg.mean()
        stats[t] = {'positive': int(pos.sum()), 'negative': int(neg.sum()),
                    'eligible_pairs': int(pos.sum() * neg.sum()),
                    'approx_batch6_pair_probability': float(1 - (1-pp)**6 - (1-pn)**6 + (1-pp-pn)**6)}
    out = ROOT / 'data/processed/stage3d_trust'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'stage3d_trust.npz', ids=ids.values.astype(str), targets=np.array(TARGETS), state=state)
    manifest = {'sources_sha256': {k: digest(p) for k, p in paths.items()},
                'asset_sha256': digest(out / 'stage3d_trust.npz'), 'n_train': len(ids),
                'uncertain_reports_excluded': int(uncertain.sum()), 'stats': stats,
                'teacher_crossfit_provenance': 'unverified: source lacks per-study fold training manifest',
                'rules': 'positive: npos>0,nneg=0,text>=.8,conf>=.7,teacher>=.8; negative: nneg>0,npos=0,text<=.2,conf>=.55,teacher<=.2; exclude known uncertain reports and all gold'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
