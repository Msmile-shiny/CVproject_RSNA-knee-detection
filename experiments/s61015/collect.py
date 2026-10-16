"""Verify the exact S6 candidate and submit once, without changing final choices."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT/'experiments/convnext1005'))
from run_scale_probe import prepare_dns
from run_solo import LABELS
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main(submit=False):
    launch=json.loads((HERE/'launch.json').read_text())
    ref=launch['response']['ref'].removeprefix('/code/')
    version=launch['response']['versionNumber']
    prepare_dns();api=KaggleApi();api.authenticate()
    status=api.kernels_status(ref).to_dict();print(status,flush=True)
    (HERE/'status.json').write_text(json.dumps(status,indent=2))
    if status['status'] not in ('COMPLETE','ERROR'):return
    out=ROOT/'results'/f's6_all_data_v{version}'
    out.mkdir(parents=True,exist_ok=True)
    api.kernels_output(ref,path=str(out),file_pattern=r'.*(log|receipt|csv|error).*',page_size=100)
    assert status['status']=='COMPLETE', 'Inspect logs before retry'
    source=HERE/'candidate/s6.ipynb'
    assert sha(source)==launch['build']['source_sha256']
    request=ApiGetKernelRequest();request.user_name,request.kernel_slug=ref.split('/')
    with api.build_kaggle_client() as client: remote=client.kernels.kernels_api_client.get_kernel(request)
    assert remote.metadata.to_dict()['currentVersionNumber']==version
    codes=lambda nb:[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert codes(json.loads(source.read_bytes()))==codes(json.loads(remote.blob.source))
    receipt=json.loads((out/'s6_fusion_receipt.json').read_text())
    assert receipt['status']=='PASS' and receipt['weights']==[.5,.4,.1]
    assert 0<receipt['elapsed_seconds']<8.5*3600
    for name,digest in receipt['hashes'].items():assert sha(out/name)==digest,name
    s6=json.loads((out/'s6_member/s6_receipt.json').read_text())
    assert s6['status']=='PASS' and not s6['B']['failed_after_retry']
    assert s6['model']['sha256']=='0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'
    blend=json.loads((out/'blend_receipt.json').read_text())
    assert blend['raptor_failed']==blend['ours_failed']==0
    frames={};ids=None
    for name in ['submission.csv','parent956.csv','raptor_raw.csv','ours_sub.csv','s6_member/s6_raw.csv']:
        f=pd.read_csv(out/name,dtype={'StudyInstanceUID':str},float_precision='round_trip')
        assert list(f.columns)==LABELS and f.StudyInstanceUID.is_unique and len(f)==receipt['rows']>0
        if ids is None:ids=f.StudyInstanceUID.tolist()
        assert set(ids)==set(f.StudyInstanceUID)
        f=f.set_index('StudyInstanceUID').loc[ids]
        assert np.isfinite(f.values).all() and ((f.values>=0)&(f.values<=1)).all()
        frames[name]=f
    a,b,c=[frames[n].rank(pct=True) for n in ['raptor_raw.csv','ours_sub.csv','s6_member/s6_raw.csv']]
    np.testing.assert_allclose(frames['parent956.csv'],.5*a+.5*b,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(frames['submission.csv'],.5*a+.4*b+.1*c,rtol=1e-7,atol=1e-8)
    verified=dict(status='PASS',kernel=ref,version=version,receipt=receipt,visible_only=True)
    (HERE/'verified.json').write_text(json.dumps(verified,indent=2));print(verified,flush=True)
    if not submit or (HERE/'submission.json').exists():return
    submissions=[s.to_dict() for s in api.competition_submissions('rsna-knee-abnormality-detection')]
    assert not any(ref+'?' in s.get('url','') for s in submissions),'Existing submission: reconcile first'
    assert not (HERE/'submission_attempt.json').exists(),'Prior attempt: reconcile first'
    (HERE/'submission_attempt.json').write_text(json.dumps(verified,indent=2))
    result=api.competition_submit_code(file_name='submission.csv',message='Verified R7 + new S6 all-data: rank weights .50 Raptor .40 R7 .10 S6; complete members',competition='rsna-knee-abnormality-detection',kernel=ref,kernel_version=version).to_dict()
    (HERE/'submission.json').write_text(json.dumps(dict(kernel=ref,version=version,response=result),indent=2));print(result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true')
    main(parser.parse_args().submit)
