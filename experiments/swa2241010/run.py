import argparse,ast,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtri,expit
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'convnext1005'))
from run_scale_probe import prepare_dns
from run_solo import LABELS
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

def verify_frames(out,receipt):
    frames={n:pd.read_csv(out/n,dtype={'StudyInstanceUID':str},float_precision='round_trip') for n in receipt['hashes']}
    parent=frames['parent950.csv'];ids=parent.StudyInstanceUID
    for name,f in frames.items():
        assert f.columns.tolist()==LABELS and f.StudyInstanceUID.is_unique
        assert len(f)==receipt['study_count']>0 and set(f.StudyInstanceUID)==set(ids)
        v=f[LABELS[1:]].to_numpy();assert np.isfinite(v).all() and (v>=0).all() and (v<=1).all()
        frames[name]=f.set_index('StudyInstanceUID').loc[ids].reset_index()
    # Recompute parent950's original nonlinear + sequential peer recipe to reject a fallback parent.
    nb=json.loads((HERE/'upstream/parent950.ipynb').read_bytes());tree=ast.parse(''.join(nb['cells'][9]['source']))
    def literal(name):
        vals=[ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        assert len(vals)==1;return vals[0]
    ar=frames['_sota_0949.csv'][LABELS[1:]].rank(pct=True);rr=frames['_own.csv'][LABELS[1:]].rank(pct=True);e=ar.copy()
    for label in LABELS[1:]:
        w=literal('OPTIMAL_WEIGHTS')[label]
        e[label]=.5*((1-w)*ar[label]+w*rr[label])+.5*expit((1-w)*ndtri(np.clip(ar[label],1e-5,1-1e-5))+w*ndtri(np.clip(rr[label],1e-5,1-1e-5)))+(rr[label]-.5)*1e-4
    e=e.rank(pct=True)
    for name,links in literal('PEERS').items():e[name]=.98*e[name]+.02*sum(w*e[k] for k,w in links.items())/sum(links.values())
    np.testing.assert_allclose(parent[LABELS[1:]],e.rank(pct=True),rtol=1e-12,atol=1e-15)
    expected=(.85*parent[LABELS[1:]].rank(pct=True)+.15*frames['independent224.csv'][LABELS[1:]].rank(pct=True)).rank(pct=True)
    np.testing.assert_allclose(frames['submission.csv'][LABELS[1:]],expected,rtol=1e-12,atol=1e-15)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','collect']);p.add_argument('--submit',action='store_true');args=p.parse_args()
    built=json.loads((HERE/'build.json').read_text());raw=(HERE/'candidate/fusion.ipynb').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==built['source_sha256']
    prepare_dns();a=KaggleApi();a.authenticate();kernel='easoncyy/rsna-swa224-fixed15'
    if args.action=='launch':
        assert not (HERE/'launch_attempt.json').exists() and not (HERE/'launch.json').exists()
        assert not any(k.ref==kernel for k in a.kernels_list(mine=True,search='rsna-swa224-fixed15',page_size=100))
        (HERE/'launch_attempt.json').write_text(json.dumps(built,indent=2))
        response=a.kernels_push(str(HERE/'candidate'),acc='NvidiaTeslaT4').to_dict()
        (HERE/'launch.json').write_text(json.dumps(dict(build=built,response=response),indent=2));print(response)
    else:
        launch=json.loads((HERE/'launch.json').read_text());kernel=launch['response']['ref'].removeprefix('/code/');version=launch['response']['versionNumber']
        assert launch['build']['source_sha256']==built['source_sha256']
        status=a.kernels_status(kernel).to_dict();print(status,flush=True)
        if status['status'] not in ['COMPLETE','ERROR']:raise SystemExit(0)
        out=HERE.parents[1]/'results'/f'swa224_fixed15_v{version}';out.mkdir(parents=True,exist_ok=True)
        a.kernels_output(kernel,path=str(out),file_pattern=r'^(swa224_receipt\.json|submission\.csv|parent950\.csv|independent224\.csv|_sota_0949\.csv|_own\.csv)$',page_size=100)
        assert status['status']=='COMPLETE',f'Read error log {out}; no automatic retry'
        req=ApiGetKernelRequest();req.user_name,req.kernel_slug=kernel.split('/')
        with a.build_kaggle_client() as c:remote=c.kernels.kernels_api_client.get_kernel(req)
        assert remote.metadata.to_dict()['currentVersionNumber']==version
        codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
        assert codes(json.loads(raw))==codes(json.loads(remote.blob.source))
        receipt=json.loads((out/'swa224_receipt.json').read_text())
        assert receipt['status']=='COMPLETE' and receipt['weight224']==.15 and receipt['parent_submission']==57012367
        assert receipt['source224_sha256']==built['source224_sha256'] and 0<receipt['elapsed_seconds']<8.5*3600
        for name,h in receipt['hashes'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
        verify_frames(out,receipt)
        verified=dict(status='PASS',kernel=kernel,version=version,receipt=receipt)
        (HERE/'verified.json').write_text(json.dumps(verified,indent=2));print(verified)
        if args.submit:
            if (HERE/'submission.json').exists():print('Already submitted');raise SystemExit(0)
            assert not (HERE/'submission_attempt.json').exists(),'Reconcile prior attempt'
            existing=[s.to_dict() for s in a.competition_submissions('rsna-knee-abnormality-detection') if kernel+'?' in s.to_dict().get('url','')]
            assert not existing,'Existing remote submission: reconcile'
            (HERE/'submission_attempt.json').write_text(json.dumps(verified,indent=2))
            response=a.competition_submit_code(file_name='submission.csv',message='0.950 parent plus independent 224 checkpoint fixed15%; exact source and two-stage formula verified',competition='rsna-knee-abnormality-detection',kernel=kernel,kernel_version=version).to_dict()
            (HERE/'submission.json').write_text(json.dumps(dict(kernel=kernel,version=version,response=response),indent=2));print(response)
