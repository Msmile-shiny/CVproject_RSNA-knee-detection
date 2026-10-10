"""Launch once; collect and validate before any separately authorized submission."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'convnext1005'))
from run_scale_probe import prepare_dns
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','collect']);p.add_argument('--submit',action='store_true');p.add_argument('--variant',choices=['paired','probit'],default='paired');args=p.parse_args()
    state=HERE if args.variant=='paired' else HERE/'probit-run'
    state.mkdir(exist_ok=True)
    folder=HERE/('candidate' if args.variant=='paired' else 'probit-candidate')
    built=json.loads((HERE/'build.json').read_text())
    if args.variant=='probit':
        built.update(json.loads((HERE/'probit_build.json').read_text()))
    raw=(folder/'apex.ipynb').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==built['source_sha256']
    prepare_dns();api=KaggleApi();api.authenticate()
    if args.action=='launch':
        assert not (state/'launch_attempt.json').exists() and not (state/'launch.json').exists()
        kernel=json.loads((folder/'kernel-metadata.json').read_text())['id']
        assert not any(k.ref==kernel for k in api.kernels_list(mine=True,search=kernel.split('/')[1],page_size=100))
        (state/'launch_attempt.json').write_text(json.dumps(built,indent=2))
        response=api.kernels_push(str(folder),acc='NvidiaTeslaT4').to_dict()
        (state/'launch.json').write_text(json.dumps(dict(build=built,response=response),indent=2));print(response)
    else:
        launch=json.loads((state/'launch.json').read_text());kernel=launch['response']['ref'].removeprefix('/code/');version=launch['response']['versionNumber']
        assert built['source_sha256']==launch['build']['source_sha256']
        status=api.kernels_status(kernel).to_dict();print(status,flush=True)
        if status['status'] not in ['COMPLETE','ERROR']:raise SystemExit(0)
        out=HERE.parents[1]/'results'/f'apex_{args.variant}_v{version}';out.mkdir(parents=True,exist_ok=True)
        api.kernels_output(kernel,path=str(out),file_pattern=r'^(apex_receipt\.json|submission\.csv|_sota_0949\.csv|_own\.csv)$',page_size=100)
        assert status['status']=='COMPLETE',f'Inspect {out}; do not automatically retry'
        req=ApiGetKernelRequest();req.user_name,req.kernel_slug=kernel.split('/')
        with api.build_kaggle_client() as c:remote=c.kernels.kernels_api_client.get_kernel(req)
        assert remote.metadata.to_dict()['currentVersionNumber']==version
        codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
        assert codes(json.loads(raw))==codes(json.loads(remote.blob.source))
        receipt=json.loads((out/'apex_receipt.json').read_text())
        assert receipt['status']=='COMPLETE' and receipt['full_fusion'] and receipt['reader_models']==3
        assert 0<receipt['elapsed_seconds']<8.5*3600
        if args.variant=='probit':
            baseline=json.loads((HERE/'verified.json').read_text())['receipt']
            assert receipt['reader_weights']==baseline['reader_weights']
            assert receipt['routing']==baseline['routing']
        assert receipt['reader_source_sha256']==built['reader_sha256']
        assert receipt['swa_sha256']=='7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
        for name,digest in receipt['hashes'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
        import pandas as pd
        import numpy as np
        frames=[pd.read_csv(out/n,dtype={'StudyInstanceUID':str},float_precision='round_trip') for n in ['_sota_0949.csv','_own.csv','submission.csv']]
        a,r,f=frames
        from run_solo import LABELS
        for frame in frames:
            assert frame.columns.tolist()==LABELS and frame.StudyInstanceUID.is_unique
            assert len(frame)==receipt['study_count']>0 and set(frame.StudyInstanceUID)==set(a.StudyInstanceUID)
            v=frame[LABELS[1:]].to_numpy();assert np.isfinite(v).all() and (v>=0).all() and (v<=1).all()
        r=r.set_index('StudyInstanceUID').loc[a.StudyInstanceUID].reset_index()
        f=f.set_index('StudyInstanceUID').loc[a.StudyInstanceUID].reset_index()
        ar=a[LABELS[1:]].rank(pct=True);rr=r[LABELS[1:]].rank(pct=True)
        for label in LABELS[1:]:
            w=receipt['routing'][label]
            expect=((1-w)*ar[label]+w*rr[label]+(rr[label]-.5)*1e-4).rank(pct=True)
            if args.variant=='probit':
                from probit_candidate import blend
                expect=pd.Series(blend(ar[label].to_numpy(),rr[label].to_numpy(),w)).rank(pct=True)
            np.testing.assert_allclose(f[label],expect,rtol=1e-12,atol=1e-15)
        verified=dict(status='PASS',kernel=kernel,version=version,receipt=receipt,submitted=False)
        (state/'verified.json').write_text(json.dumps(verified,indent=2));print(verified)
        if args.submit:
            record=state/'submission.json';attempt=state/'submission_attempt.json'
            if record.exists():print('Already submitted',record.read_text());raise SystemExit(0)
            assert not attempt.exists(), 'Reconcile prior attempt before any retry'
            attempt.write_text(json.dumps(verified,indent=2))
            message='Apex '+args.variant+': scored image path, fixed routing; full fusion and variant formula verified'
            response=api.competition_submit_code(file_name='submission.csv',message=message,competition='rsna-knee-abnormality-detection',kernel=kernel,kernel_version=version).to_dict()
            record.write_text(json.dumps(dict(kernel=kernel,version=version,response=response),indent=2));print(response)
