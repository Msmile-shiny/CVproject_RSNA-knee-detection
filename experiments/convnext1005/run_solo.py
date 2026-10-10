"""Version-pinned launch, verification and one submission of the standalone reader."""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from run_scale_probe import prepare_dns

HERE=Path(__file__).resolve().parent
KERNEL='easoncyy/rsna-cnx-threefold-solo'
LABELS=['StudyInstanceUID','ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

def verify(out,built):
    receipt=json.loads((out/'solo_receipt.json').read_text())
    assert receipt['status']=='COMPLETE' and receipt['model_count']==3 and receipt['folds']==[0,1,2]
    assert receipt['parent_used'] is False and receipt['stress_pass']
    assert receipt['fallback']==0 and receipt['partial_predictions']==0
    assert receipt['precision']=='fp16' and receipt['batch_studies']==4 and receipt['windows_per_slot']==12
    assert receipt['source_hashes']==built['sources']
    expected={
        'cnxt_v0_fold0.pt':'0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80',
        'cnxt_v0_fold1.pt':'61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb',
        'cnxt_v0_fold2.pt':'f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6'}
    assert receipt['checkpoint_hashes']==expected
    assert (out/'submission.csv').read_bytes()==(out/'cnx3_raw.csv').read_bytes()
    assert hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()==receipt['submission_sha256']==receipt['raw_sha256']
    with (out/'submission.csv').open(newline='') as handle:
        reader=csv.DictReader(handle);rows=list(reader);assert reader.fieldnames==LABELS
    assert len(rows)==receipt['study_count']>0
    assert len(set(r['StudyInstanceUID'] for r in rows))==len(rows)
    assert all(math.isfinite(float(r[k])) and 0<=float(r[k])<=1 for r in rows for k in LABELS[1:])
    return dict(status='PASS',studies=len(rows),elapsed_seconds=receipt['elapsed_seconds'],score_verified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','collect']);p.add_argument('--submit',action='store_true');args=p.parse_args()
    prepare_dns();api=KaggleApi();api.authenticate()
    built=json.loads((HERE/'solo_build.json').read_text())
    raw=(HERE/'solo/solo.ipynb').read_bytes();assert hashlib.sha256(raw).hexdigest()==built['source_sha256']
    if args.action=='launch':
        assert not (HERE/'solo_attempt.json').exists() and not (HERE/'solo_launch.json').exists()
        assert not any(k.ref==KERNEL for k in api.kernels_list(mine=True,search='rsna-cnx-threefold-solo',page_size=100))
        payload=dict(kernel=KERNEL,source_sha256=built['source_sha256'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (HERE/'solo_attempt.json').write_text(json.dumps(payload,indent=2))
        response=api.kernels_push(str(HERE/'solo'),acc='NvidiaTeslaT4').to_dict()
        (HERE/'solo_launch.json').write_text(json.dumps(dict(**payload,response=response),indent=2));print(response)
    else:
        launch=json.loads((HERE/'solo_launch.json').read_text());kernel=launch['response']['ref'].removeprefix('/code/');version=launch['response']['versionNumber']
        state=api.kernels_status(kernel).to_dict();print(state,flush=True)
        if state['status'] not in ['COMPLETE','ERROR']:raise SystemExit(0)
        out=HERE.parents[1]/'results'/f'cnx_solo_v{version}';out.mkdir(parents=True,exist_ok=True)
        api.kernels_output(kernel,path=str(out),file_pattern=r'^(solo_receipt\.json|solo_worker\.log|cnx3_raw\.csv|submission\.csv)$',page_size=100)
        assert state['status']=='COMPLETE',f'Kernel failed; inspect {out}'
        req=ApiGetKernelRequest();req.user_name,req.kernel_slug=kernel.split('/')
        with api.build_kaggle_client() as client: remote=client.kernels.kernels_api_client.get_kernel(req)
        assert remote.metadata.to_dict()['currentVersionNumber']==version
        codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
        assert codes(json.loads(raw))==codes(json.loads(remote.blob.source))
        result=verify(out,built);(HERE/'solo_verified.json').write_text(json.dumps(result,indent=2));print(result)
        if args.submit:
            record=HERE/'solo_submission.json';attempt=HERE/'solo_submission_attempt.json'
            if record.exists():print('Already submitted',record.read_text());raise SystemExit(0)
            assert not attempt.exists(),'Reconcile prior attempt before retry'
            payload=dict(kernel=kernel,version=version,verified=result)
            attempt.write_text(json.dumps(payload,indent=2))
            response=api.competition_submit_code(file_name='submission.csv',message='Standalone public ConvNeXt threefold reader; no parent blend; strict complete inference',competition='rsna-knee-abnormality-detection',kernel=kernel,kernel_version=version).to_dict()
            record.write_text(json.dumps(dict(**payload,response=response),indent=2));print(response)
