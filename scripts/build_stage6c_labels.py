"""Stage 6C: fixed 30% report-evidence probability control.

Only probabilities on singly mentioned, non-Gold findings change. Training
weights, masks and all Gold rows are byte-value equivalent to the v5 source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
         'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
UID='StudyInstanceUID'
V5_SHA='c13adffaabf4f8e518abb038282bb1aa09baac7652a9165e030710c457d0be6a'
REPORT_SHA='5ecb0bf45498da7f0cf0f9d0ff5f1bc4ec3e62613444892a05256cb9ae64b385'
ALPHA=.30


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(v5_path, report_path, gold_path, output):
    assert sha(v5_path)==V5_SHA and sha(report_path)==REPORT_SHA
    old=pd.read_csv(v5_path,dtype={UID:str})
    report=pd.read_csv(report_path,dtype={UID:str}).set_index(UID)
    gold=pd.read_csv(gold_path,dtype={UID:str}).set_index(UID)
    assert old[UID].is_unique and report.index.is_unique and gold.index.is_unique
    assert set(old[UID])==set(report.index)==set(gold.index)
    gold_ids=set(gold.index[gold[TARGETS].notna().all(axis=1)])
    assert len(gold_ids)==58
    aligned=report.loc[old[UID]].reset_index()
    new=old.copy()
    audit=[]
    for target in TARGETS:
        plus=pd.to_numeric(aligned[target+'__npos'],errors='raise').to_numpy()
        minus=pd.to_numeric(aligned[target+'__nneg'],errors='raise').to_numpy()
        score=pd.to_numeric(aligned[target],errors='raise').to_numpy(float)
        old_prob=old['prob_'+target].to_numpy(float)
        eligible=(plus>0)^(minus>0)
        eligible &= ~old[UID].isin(gold_ids).to_numpy()
        assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
        changed=old_prob.copy()
        changed[eligible]=(1-ALPHA)*old_prob[eligible]+ALPHA*score[eligible]
        new['prob_'+target]=changed
        audit.append(dict(target=target,positive_only=int(((plus>0)&(minus==0)&eligible).sum()),
                          negative_only=int(((minus>0)&(plus==0)&eligible).sum()),
                          no_mention=int(((plus==0)&(minus==0)&~old[UID].isin(gold_ids).to_numpy()).sum()),
                          conflicting=int(((plus>0)&(minus>0)&~old[UID].isin(gold_ids).to_numpy()).sum()),
                          changed=int(eligible.sum()),mean_absolute_change=float(np.abs(changed-old_prob).mean())))
    nonprob=[c for c in old if not c.startswith('prob_')]
    assert new[nonprob].equals(old[nonprob])
    assert new.loc[old[UID].isin(gold_ids)].equals(old.loc[old[UID].isin(gold_ids)])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    new.to_csv(output,index=False,float_format='%.10g')
    reread=pd.read_csv(output,dtype={UID:str})
    assert reread.shape==old.shape and list(reread.columns)==list(old.columns)
    for target in TARGETS:
        for prefix in ('weight_','mask_'):
            np.testing.assert_allclose(reread[prefix+target],old[prefix+target],atol=1e-9,rtol=0)
        assert np.isfinite(reread['prob_'+target]).all() and reread['prob_'+target].between(0,1).all()
    receipt=dict(source_v5_sha256=V5_SHA,report_sha256=REPORT_SHA,
                 output_sha256=sha(output),alpha=ALPHA,gold_rows_preserved=58,
                 unchanged_columns=nonprob,eligibility='non-Gold and exactly one of npos/nneg > 0',
                 targets=audit,training_effect='probabilities only; original weight/mask retained')
    (output.parent/'stage6c_label_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--v5',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--gold',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=build(a.v5,a.report,a.gold,a.output)
    print(json.dumps({k:r[k] for k in ('output_sha256','alpha','gold_rows_preserved')},indent=2))
