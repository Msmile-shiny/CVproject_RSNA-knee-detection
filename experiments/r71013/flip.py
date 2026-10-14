"""One bounded R7 mirror inference experiment; no ratio search."""
import argparse, ast, hashlib, json
from pathlib import Path
import run
BASE=Path(__file__).resolve().parent
REF='easoncyy/rsna-r7-training-mirror'

def build():
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    out=BASE/'flip-run'
    assert not (out/'launch_attempt.json').exists()
    built=json.loads((BASE/'build.json').read_text())
    raw=(BASE/'candidate/r7.ipynb').read_bytes()
    assert run.sha(raw)==built['source_sha256']
    nb=json.loads(raw)
    nb['cells'].insert(6,nbformat.v4.new_code_cell("assert '--flip-tta' not in WORKER_ARGS\nWORKER_ARGS += ['--flip-tta']\nprint('R7_MIRROR_ENABLED: training transform, reindexed classes, MCL plain only')\n"))
    # Keep evidence from both workers; refuse silently missing mirror execution.
    nb['cells'][-1]['source']=''.join(nb['cells'][-1]['source'])+"\n_flip_logs=sorted(WORK.glob('worker*_log.txt'))\nassert len(_flip_logs)==2\nassert all('FLIP_TTA_READY' in p.read_text() for p in _flip_logs), 'Mirror execution unverified'\n"
    for i,c in enumerate(nb['cells']):
        c['id']=f'r7-mirror-{i}'
        if c['cell_type']=='code':ast.parse(TransformerManager().transform_cell(''.join(c['source'])))
    nbformat.validate(nbformat.from_dict(nb))
    candidate=out/'candidate';candidate.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(nb,ensure_ascii=False,indent=1).encode()
    (candidate/'r7.ipynb').write_bytes(payload)
    meta=json.loads((BASE/'candidate/kernel-metadata.json').read_text())
    meta.update(id=REF,title='RSNA R7 Training Mirror')
    (candidate/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    built.update(source_sha256=run.sha(payload),parent_sha256=run.sha(raw),inference_unchanged=False,protected_submission=57144024,protected_score=.956,hypothesis='Training-consistent mirror, class permutation, MCL plain; no weight or calibration changes')
    (out/'build.json').write_text(json.dumps(built,indent=2));print(built)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','launch','collect']);p.add_argument('--submit',action='store_true');a=p.parse_args()
    if a.action=='build':build()
    else:
        run.HERE=BASE/'flip-run';run.REF=REF
        if a.submit:
            import numpy as np
            import pandas as pd
            assert a.action=='collect'
            out=run.ROOT/'results/rsna-r7-training-mirror_v1'
            parent=run.ROOT/'results/r7_public_v1'
            for folder,receipt_path in [(out,BASE/'flip-run/verified.json'),(parent,BASE/'verified.json')]:
                receipt=json.loads(receipt_path.read_text())['receipt']
                for name in ['ours_sub.csv','raptor_sub.csv']:
                    assert run.sha((folder/name).read_bytes())==receipt['hashes'][name]
            x=pd.read_csv(parent/'ours_sub.csv').set_index('StudyInstanceUID')
            y=pd.read_csv(out/'ours_sub.csv').set_index('StudyInstanceUID').loc[x.index]
            np.testing.assert_allclose(x.MCL,y.MCL,rtol=0,atol=1e-7)
            assert (parent/'raptor_sub.csv').read_bytes()==(out/'raptor_sub.csv').read_bytes()
            logs=list(out.glob('worker*_log.txt'))
            assert len(logs)==2 and all('FLIP_TTA_READY' in p.read_text() and 'WORKER_DONE' in p.read_text() for p in logs)
            (BASE/'flip-run/comparison.json').write_text(json.dumps(dict(status='PASS',mcl_max_abs=float(abs(x.MCL-y.MCL).max()),raptor_visible_ranks_equal=True,hidden_budget_proven=False),indent=2))
        run.online(a.action,a.submit)
