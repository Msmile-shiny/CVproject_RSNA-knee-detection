"""Validate downloaded visible outputs before selecting a completed version to score."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_outer70 import ROOT, PARENT_SHA, CHANGED

LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
SPECIAL = {'ACL':.75,'Medial Meniscus':.8,'Lateral Meniscus':1.,'Lateral OA':.75,'Fracture':.75}
FILES = ['submission.csv','baseline60_replay.csv','transformer_branch_rank.csv','coat_raptor_branch.csv']


def validate_frames(frames):
    ids = None
    for name, frame in frames.items():
        if list(frame.columns) != ['StudyInstanceUID']+LABELS:
            raise ValueError(name+': column order mismatch')
        uid = frame.StudyInstanceUID
        if frame.empty or uid.isna().any() or uid.duplicated().any() or (uid.astype(str).str.len()==0).any():
            raise ValueError(name+': invalid study identifiers')
        if ids is None:
            ids = uid.tolist()
        elif uid.tolist() != ids:
            raise ValueError(name+': study order mismatch')
        a = frame[LABELS].to_numpy(dtype=float)
        if not np.isfinite(a).all() or (a<0).any() or (a>1).any():
            raise ValueError(name+': invalid probabilities/ranks')
    tr, cr = frames[FILES[2]], frames[FILES[3]]
    for filename, default in [(FILES[0],.7),(FILES[1],.6)]:
        expected = tr[LABELS].copy()
        for label in LABELS:
            w = SPECIAL.get(label,default)
            expected[label] = (1-w)*tr[label]+w*cr[label]
        expected = expected.rank(method='average',pct=True)
        if not np.allclose(expected,frames[filename][LABELS],rtol=0,atol=1e-12):
            raise ValueError(filename+': fusion formula mismatch')


def check(output):
    receipt = json.loads((output/'outer70_receipt.json').read_text(encoding='utf-8'))
    build = json.loads((ROOT/'build_receipt.json').read_text(encoding='utf-8'))
    if receipt.get('status')!='COMPLETE' or receipt.get('ready_for_scoring') is not True:
        raise ValueError('Incomplete candidate receipt')
    if receipt.get('parent_notebook_sha256')!=PARENT_SHA or receipt.get('candidate_core_sha256')!=build['candidate_core_sha256']:
        raise ValueError('Candidate source identity mismatch')
    if receipt.get('changed_labels')!=CHANGED or receipt.get('default_weight_after')!=.7 or receipt.get('default_weight_before')!=.6:
        raise ValueError('Candidate recipe mismatch')
    elapsed = float(receipt.get('elapsed_seconds', float('nan')))
    if not np.isfinite(elapsed) or not 0<elapsed<9*3600:
        raise ValueError('Missing/invalid/over-budget runtime')
    parent = json.loads((output/'sprint_fourway_receipt.json').read_text(encoding='utf-8'))
    if parent.get('ready_for_scoring') is not True or set(parent.get('family_members',[]))!={'resgated_top3','global96_top3','d4_swa3','repairv1_top3'}:
        raise ValueError('Incomplete parent family')
    audit = json.loads((output/'btkd_v559_complete.json').read_text(encoding='utf-8'))
    flags = ('fail','fallback','unavailable','reject','incomplete','partial','mismatch','dropped','neutral')
    issues = [e for e in audit.get('events',[]) if any(f in e.get('kind','').lower() for f in flags)]
    if issues:
        raise ValueError('Runtime events require review: '+str(issues)[:500])
    frames = {}
    for name in FILES:
        p = output/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=receipt['hashes'].get(name):
            raise ValueError('Output hash mismatch: '+name)
        frames[name] = pd.read_csv(p,dtype={'StudyInstanceUID':str},float_precision='round_trip')
    validate_frames(frames)
    if len(frames[FILES[0]]) != receipt.get('study_count'):
        raise ValueError('Study count mismatch')
    return dict(status='VISIBLE_OUTPUT_CHECK_PASS',studies=receipt['study_count'],elapsed_seconds=elapsed,
                hidden_runtime_verified=False, score_verified=False,
                note='Use the matching completed Notebook version; do not upload visible CSV as the competition submission.')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(json.dumps(check(p.parse_args().output),indent=2))
