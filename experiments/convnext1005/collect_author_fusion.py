"""Verify actual launched version and fixed fusion; submit at most once."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from run_scale_probe import prepare_dns
from run_solo import LABELS

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    launch=json.loads((HERE/'author_fusion_launch.json').read_text())
    kernel=launch['response']['ref'].removeprefix('/code/');version=launch['response']['versionNumber']
    raw=(HERE/'author-fusion30/fusion.ipynb').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==launch['build']['source_sha256']
    prepare_dns();api=KaggleApi();api.authenticate()
    status=api.kernels_status(kernel).to_dict();print(status,flush=True)
    if status['status'] not in ['COMPLETE','ERROR']:raise SystemExit(0)
    out=HERE.parents[1]/'results'/f'cnx_author_fusion_v{version}';out.mkdir(parents=True,exist_ok=True)
    api.kernels_output(kernel,path=str(out),file_pattern=r'^(author_fusion_receipt\.json|author_control_receipt\.json|submission\.csv|parent943\.csv|author929\.csv)$',page_size=100)
    assert status['status']=='COMPLETE',f'Inspect error log in {out}'
    request=ApiGetKernelRequest();request.user_name,request.kernel_slug=kernel.split('/')
    with api.build_kaggle_client() as client:remote=client.kernels.kernels_api_client.get_kernel(request)
    assert remote.metadata.to_dict()['currentVersionNumber']==version
    codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    assert codes(json.loads(raw))==codes(json.loads(remote.blob.source))
    receipt=json.loads((out/'author_fusion_receipt.json').read_text())
    assert receipt['status']=='COMPLETE' and receipt['reader_weight']==.3
    assert receipt['parent_submission']==56700487 and receipt['reader_submission']==56910183
    assert 0<receipt['elapsed_seconds']<8.5*3600
    for name,sha in receipt['hashes'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
    reader_receipt=json.loads((out/'author_control_receipt.json').read_text())
    assert reader_receipt['status']=='COMPLETE' and reader_receipt['model_count']==3 and not reader_receipt['parent_used']
    assert reader_receipt['source_hashes']==json.loads((HERE/'author_control_build.json').read_text())['expected_sources']
    assert reader_receipt['checkpoint_hashes']==json.loads((HERE/'scale_result.json').read_text())['scale_receipt']['checkpoint_hashes']
    assert hashlib.sha256((out/'author929.csv').read_bytes()).hexdigest()==reader_receipt['submission_sha256']
    frames=[pd.read_csv(out/n,dtype={'StudyInstanceUID':str},float_precision='round_trip') for n in ['parent943.csv','author929.csv','submission.csv']]
    p,r,f=frames
    for frame in frames:
        assert frame.columns.tolist()==LABELS and frame.StudyInstanceUID.is_unique
        assert len(frame)==receipt['study_count']>0 and set(frame.StudyInstanceUID)==set(p.StudyInstanceUID)
        values=frame[LABELS[1:]].to_numpy();assert np.isfinite(values).all() and (values>=0).all() and (values<=1).all()
    r=r.set_index('StudyInstanceUID').loc[p.StudyInstanceUID].reset_index()
    assert f.StudyInstanceUID.tolist()==p.StudyInstanceUID.tolist()
    expected=(.7*p[LABELS[1:]].rank(pct=True)+.3*r[LABELS[1:]].rank(pct=True)).rank(pct=True)
    np.testing.assert_allclose(f[LABELS[1:]],expected,rtol=1e-12,atol=1e-15)
    verified=dict(status='PASS',kernel=kernel,version=version,study_count=len(f),elapsed_seconds=receipt['elapsed_seconds'],score_verified=False,decode_coverage_verified=False)
    (HERE/'author_fusion_verified.json').write_text(json.dumps(verified,indent=2));print(verified)
    if args.submit:
        record=HERE/'author_fusion_submission.json';attempt=HERE/'author_fusion_submission_attempt.json'
        if record.exists():print('Already submitted',record.read_text());raise SystemExit(0)
        assert not attempt.exists(),'Reconcile previous submit attempt before retry'
        attempt.write_text(json.dumps(verified,indent=2))
        response=api.competition_submit_code(file_name='submission.csv',message='Fixed 30% scored author reader + unchanged 0.943 parent; exact sources and rank blend verified',competition='rsna-knee-abnormality-detection',kernel=kernel,kernel_version=version).to_dict()
        record.write_text(json.dumps(dict(verification=verified,response=response),indent=2));print(response)
