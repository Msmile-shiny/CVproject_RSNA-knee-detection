"""Isolated execution wrapper for the single-change control."""
import argparse
import json
import run

def compare(base):
    import numpy as np
    import pandas as pd
    parent=json.loads((base/'candidate/r7.ipynb').read_bytes())
    child=json.loads((base/'no-fluid-run/candidate/r7.ipynb').read_bytes())
    assert len(child['cells'])==len(parent['cells'])+1
    assert 'del WORKER_ARGS[_fluid_i:_fluid_i+2]' in ''.join(child['cells'][6]['source'])
    del child['cells'][6]
    codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    assert codes(parent)==codes(child), 'Unexpected inference changes'
    old=run.ROOT/'results/r7_public_v1'
    new=run.ROOT/'results/rsna-r7-no-fluid-control_v1'
    report={}
    for name in ['ours_sub.csv','submission.csv']:
        for folder,receipt_path in [(old,base/'verified.json'),(new,base/'no-fluid-run/verified.json')]:
            receipt=json.loads(receipt_path.read_text())['receipt']
            assert run.sha((folder/name).read_bytes())==receipt['hashes'][name]
        a=pd.read_csv(old/name,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
        b=pd.read_csv(new/name,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[a.index]
        assert list(a.columns)==list(b.columns)
        cols=[c for c in a.columns if c!='Effusion']
        np.testing.assert_allclose(a[cols],b[cols],atol=1e-7,rtol=0)
        report[name]=(a-b).abs().max().to_dict()
    record=dict(status='PASS',visible_cases=len(a),max_abs_differences=report,hidden_equivalence_proven=False)
    (base/'no-fluid-run/comparison.json').write_text(json.dumps(record,indent=2))
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['launch','collect'])
    p.add_argument('--submit',action='store_true')
    args=p.parse_args()
    base=run.HERE
    verified=json.loads((base/'verified.json').read_text())
    assert verified['status']=='PASS' and verified['version']==1
    run.HERE=base/'no-fluid-run'
    run.REF='easoncyy/rsna-r7-no-fluid-control'
    if args.submit:
        assert args.action=='collect'
        print(compare(base))
    run.online(args.action,args.submit)
