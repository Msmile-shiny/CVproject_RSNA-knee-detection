"""Launch/verify/submit one version of the upstream execution control."""
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
from run_solo import LABELS

HERE=Path(__file__).resolve().parent
KERNEL='easoncyy/rsna-cnx-author-control'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','collect']);p.add_argument('--submit',action='store_true');args=p.parse_args()
    built=json.loads((HERE/'author_control_build.json').read_text())
    raw=(HERE/'author-control/control.ipynb').read_bytes();assert hashlib.sha256(raw).hexdigest()==built['source_sha256']
    prepare_dns();api=KaggleApi();api.authenticate()
    if args.action=='launch':
        assert not (HERE/'author_control_attempt.json').exists() and not (HERE/'author_control_launch.json').exists()
        assert not any(k.ref==KERNEL for k in api.kernels_list(mine=True,search='rsna-cnx-author-control',page_size=100))
        payload=dict(kernel=KERNEL,source_sha256=built['source_sha256'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (HERE/'author_control_attempt.json').write_text(json.dumps(payload,indent=2))
        response=api.kernels_push(str(HERE/'author-control'),acc='NvidiaTeslaT4').to_dict()
        (HERE/'author_control_launch.json').write_text(json.dumps(dict(**payload,response=response),indent=2));print(response)
    else:
        launch=json.loads((HERE/'author_control_launch.json').read_text());kernel=launch['response']['ref'].removeprefix('/code/');version=launch['response']['versionNumber']
        status=api.kernels_status(kernel).to_dict();print(status,flush=True)
        if status['status'] not in ['COMPLETE','ERROR']:raise SystemExit(0)
        out=HERE.parents[1]/'results'/f'cnx_author_control_v{version}';out.mkdir(parents=True,exist_ok=True)
        api.kernels_output(kernel,path=str(out),file_pattern=r'^(author_control_receipt\.json|submission\.csv|_own\.csv)$',page_size=100)
        assert status['status']=='COMPLETE',f'Inspect error log in {out}'
        request=ApiGetKernelRequest();request.user_name,request.kernel_slug=kernel.split('/')
        with api.build_kaggle_client() as client:remote=client.kernels.kernels_api_client.get_kernel(request)
        assert remote.metadata.to_dict()['currentVersionNumber']==version
        codes=lambda nb:[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        assert codes(json.loads(raw))==codes(json.loads(remote.blob.source))
        receipt=json.loads((out/'author_control_receipt.json').read_text())
        assert receipt['status']=='COMPLETE' and receipt['model_count']==3 and receipt['parent_used'] is False
        assert receipt['source_hashes']==built['expected_sources']
        assert receipt['checkpoint_hashes']==json.loads((HERE/'scale_result.json').read_text())['scale_receipt']['checkpoint_hashes']
        assert hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()==receipt['submission_sha256']
        assert hashlib.sha256((out/'_own.csv').read_bytes()).hexdigest()==receipt['raw_sha256']
        with (out/'submission.csv').open(newline='') as handle:
            reader=csv.DictReader(handle);rows=list(reader);assert reader.fieldnames==LABELS
        with (out/'_own.csv').open(newline='') as handle:own={r['StudyInstanceUID']:r for r in csv.DictReader(handle)}
        assert len(rows)==len(own)==receipt['study_count']>0
        assert len({r['StudyInstanceUID'] for r in rows})==len(rows)
        assert {r['StudyInstanceUID'] for r in rows}==set(own)
        for row in rows:
            for label in LABELS[1:]:
                value=float(row[label]);assert math.isfinite(value) and 0<=value<=1
                assert math.isclose(value,float(own[row['StudyInstanceUID']][label]),rel_tol=1e-12,abs_tol=1e-15)
        verified=dict(status='PASS',study_count=len(rows),model_count=3,author_sources_unchanged=True,
                      decode_coverage_verified=False,missing_series_policy='unaltered upstream',score_verified=False)
        (HERE/'author_control_verified.json').write_text(json.dumps(verified,indent=2));print(verified)
        if args.submit:
            record=HERE/'author_control_submission.json';attempt=HERE/'author_control_submission_attempt.json'
            if record.exists():print('Already submitted',record.read_text());raise SystemExit(0)
            assert not attempt.exists(),'Reconcile existing submission attempt'
            payload=dict(kernel=kernel,version=version,verification=verified)
            attempt.write_text(json.dumps(payload,indent=2))
            response=api.competition_submit_code(file_name='submission.csv',message='Author v5 execution control, RUN_STACK=False; unchanged author inference; three hashes verified',competition='rsna-knee-abnormality-detection',kernel=kernel,kernel_version=version).to_dict()
            record.write_text(json.dumps(dict(**payload,response=response),indent=2));print(response)
