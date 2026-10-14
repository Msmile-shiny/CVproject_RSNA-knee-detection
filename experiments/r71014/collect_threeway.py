"""Collect exact threeway version, independently verify formula, submit once."""
import argparse, hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main(submit):
    import numpy as np
    import pandas as pd
    sys.path.insert(0,str(HERE.parent/'convnext1005'))
    from run_scale_probe import prepare_dns
    from run_solo import LABELS
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    launch=json.loads((HERE/'launch.json').read_text())
    ref=launch['response']['ref'].removeprefix('/code/');version=launch['response']['versionNumber']
    source=HERE/'candidate/threeway.ipynb'
    assert sha(source)==launch['build']['source_sha256']
    prepare_dns();a=KaggleApi();a.authenticate()
    status=a.kernels_status(ref).to_dict();print(status,flush=True)
    if status.get('status') not in ('COMPLETE','ERROR'):return
    out=ROOT/'results'/f'r7_threeway_v{version}';out.mkdir(parents=True,exist_ok=True)
    a.kernels_output(ref,path=str(out),file_pattern=r'^(.*receipt\.json|.*\.csv|worker.*\.txt)$',page_size=100)
    assert status['status']=='COMPLETE',f'Inspect {out}; do not restart automatically'
    q=ApiGetKernelRequest();q.user_name,q.kernel_slug=ref.split('/')
    with a.build_kaggle_client() as c:r=c.kernels.kernels_api_client.get_kernel(q)
    assert r.metadata.to_dict()['currentVersionNumber']==version
    codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    assert codes(json.loads(source.read_bytes()))==codes(json.loads(r.blob.source))
    receipt=json.loads((out/'threeway_receipt.json').read_text())
    assert receipt['status']=='PASS' and receipt['complete_three_members']
    assert receipt['weights']==[.325,.175,.5] and 0<receipt['elapsed_seconds']<8.8*3600
    for name,h in receipt['hashes'].items():assert sha(out/name)==h,name
    frames={}
    ids=None
    for name in ['submission.csv','raptor_raw.csv','raptor_sub.csv','raptor224_sub.csv','raptor224_raw.csv','ours_sub.csv']:
        f=pd.read_csv(out/name,dtype={'StudyInstanceUID':str},float_precision='round_trip')
        assert f.columns.tolist()==LABELS and f.StudyInstanceUID.is_unique and len(f)==receipt['rows']>0
        if ids is None:ids=f.StudyInstanceUID.tolist()
        assert set(f.StudyInstanceUID)==set(ids)
        f=f.set_index('StudyInstanceUID').loc[ids]
        assert np.isfinite(f.values).all() and ((f.values>=0)&(f.values<=1)).all()
        frames[name]=f
    for name in ['raptor_raw.csv','raptor224_raw.csv']:assert not (frames[name].values==.5).all(axis=1).any()
    # The archived 224 source uses ordinal argsort ranks in [0,1], not pandas pct ranks.
    original224=pd.read_csv(out/'raptor224_raw.csv',dtype={'StudyInstanceUID':str},float_precision='round_trip').set_index('StudyInstanceUID')
    rank224=original224.to_numpy().argsort(0).argsort(0).astype(np.float64)/max(1,len(original224)-1)
    rank224=pd.DataFrame(rank224,index=original224.index,columns=original224.columns).loc[ids]
    np.testing.assert_allclose(rank224,frames['raptor224_sub.csv'],atol=1e-7,rtol=0)
    expected=.325*frames['raptor_raw.csv'].rank(pct=True)+.175*frames['raptor224_sub.csv'].rank(pct=True)+.5*frames['ours_sub.csv'].rank(pct=True)
    np.testing.assert_allclose(frames['submission.csv'],expected.astype(np.float32),atol=1e-8,rtol=1e-7)
    verified=dict(status='PASS',kernel=ref,version=version,receipt=receipt)
    (HERE/'verified.json').write_text(json.dumps(verified,indent=2));print(verified,flush=True)
    if not submit:return
    if (HERE/'submission.json').exists():print('Already submitted');return
    assert not (HERE/'submission_attempt.json').exists(),'Reconcile prior attempt first'
    assert not any(ref+'?' in s.to_dict().get('url','') for s in a.competition_submissions('rsna-knee-abnormality-detection'))
    (HERE/'submission_attempt.json').write_text(json.dumps(verified,indent=2))
    try:
        response=a.competition_submit_code(file_name='submission.csv',message='Complete R7 three-model reproduction: 384 .325, 224 .175, R7 .5; no missing members; independently verified',competition='rsna-knee-abnormality-detection',kernel=ref,kernel_version=version).to_dict()
    except Exception as exc:
        resp=getattr(exc,'response',None)
        (HERE/'submission_error.json').write_text(json.dumps(dict(http_status=getattr(resp,'status_code',None),body=getattr(resp,'text',str(exc))),indent=2))
        raise
    (HERE/'submission.json').write_text(json.dumps(dict(kernel=ref,version=version,response=response),indent=2));print(response)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');main(p.parse_args().submit)
