"""Audit existing user run, then submit exact version at most once."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtri,expit
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'convnext1005'))
from run_solo import LABELS
from run_scale_probe import prepare_dns
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

if __name__=='__main__':
    raw=next((ROOT/'results/research1009/current95').glob('*.ipynb')).read_bytes()
    nb=json.loads(raw);source=''.join(nb['cells'][9]['source']);tree=ast.parse(source)
    def literal(name):
        values=[ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        assert len(values)==1,name
        return values[0]
    weights=literal('OPTIMAL_WEIGHTS');peers=literal('PEERS')
    out=ROOT/'results/research1009/current95-output'
    log=(out/'rsna-knee-lb-0-95.log').read_text(encoding='utf8')
    assert '[Apex Tier 3 SUCCESS] Written SuperStack' in log
    assert 'FALLBACK]' not in log and 'CRITICAL EXCEPTION]' not in log
    assert 'Successfully adopted mathematically optimal weights' not in log, 'Must audit actual adopted CV weights'
    frames=[pd.read_csv(out/n,dtype={'StudyInstanceUID':str},float_precision='round_trip') for n in ['_sota_0949.csv','_own.csv','submission.csv']]
    a,r,f=frames
    for frame in frames:
        assert frame.columns.tolist()==LABELS and frame.StudyInstanceUID.is_unique
        assert len(frame)>0 and set(frame.StudyInstanceUID)==set(a.StudyInstanceUID)
        v=frame[LABELS[1:]].to_numpy();assert np.isfinite(v).all() and (v>=0).all() and (v<=1).all()
    r=r.set_index('StudyInstanceUID').loc[a.StudyInstanceUID].reset_index()
    f=f.set_index('StudyInstanceUID').loc[a.StudyInstanceUID].reset_index()
    ar=a[LABELS[1:]].rank(pct=True);rr=r[LABELS[1:]].rank(pct=True);expected=ar.copy()
    for label in LABELS[1:]:
        w=weights.get(label,.15)
        expected[label]=.5*((1-w)*ar[label]+w*rr[label])+.5*expit((1-w)*ndtri(np.clip(ar[label],1e-5,1-1e-5))+w*ndtri(np.clip(rr[label],1e-5,1-1e-5)))+(rr[label]-.5)*1e-4
    expected=expected.rank(pct=True)
    for name,connections in peers.items():
        total=sum(connections.values())
        expected[name]=.98*expected[name]+.02*sum(w*expected[label] for label,w in connections.items())/total
    expected=expected.rank(pct=True)
    np.testing.assert_allclose(f[LABELS[1:]],expected,rtol=1e-12,atol=1e-15)
    prepare_dns();api=KaggleApi();api.authenticate();kernel='easoncyy/rsna-knee-lb-0-95'
    req=ApiGetKernelRequest();req.user_name,req.kernel_slug=kernel.split('/')
    with api.build_kaggle_client() as client:remote=client.kernels.kernels_api_client.get_kernel(req)
    version=remote.metadata.to_dict()['currentVersionNumber'];assert version==1
    codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    assert codes(nb)==codes(json.loads(remote.blob.source))
    assert api.kernels_status(kernel).to_dict()['status']=='COMPLETE'
    recent=api.competition_submissions('rsna-knee-abnormality-detection')
    existing=[r.to_dict() for r in recent if '/'+kernel.split('/')[1]+'?' in r.to_dict().get('url','')]
    state=HERE/'current95-run';state.mkdir(exist_ok=True)
    verified=dict(status='PASS',kernel=kernel,version=version,source_sha256=hashlib.sha256(raw).hexdigest(),
                  hashes={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['submission.csv','_sota_0949.csv','_own.csv']},
                  note='Original user recipe, including order-dependent peer graph; not adopted into protected baseline',decode_coverage_verified=False)
    (state/'verified.json').write_text(json.dumps(verified,indent=2));print(verified,flush=True)
    if existing:
        assert len(existing)==1 and 'scriptVersionId=356785385' in existing[0]['url'], 'Review ambiguous existing submissions'
        (state/'submission.json').write_text(json.dumps(dict(kernel=kernel,version=version,response=existing[0],reconciled_existing=True),indent=2))
        print('Existing user submission reconciled; no new submission');raise SystemExit(0)
    if '--submit' in sys.argv:
        assert not (state/'submission_attempt.json').exists() and not (state/'submission.json').exists()
        (state/'submission_attempt.json').write_text(json.dumps(verified,indent=2))
        response=api.competition_submit_code(file_name='submission.csv',message='Existing user 0.95+ v1: source, completed branch and exact fusion output audited; score not assumed',competition='rsna-knee-abnormality-detection',kernel=kernel,kernel_version=version).to_dict()
        (state/'submission.json').write_text(json.dumps(dict(kernel=kernel,version=version,response=response),indent=2));print(response)
