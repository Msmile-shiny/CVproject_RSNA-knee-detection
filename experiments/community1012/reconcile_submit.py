"""Reconcile the rejected v1 attempt before a single diagnostic retry."""
import json,sys
from pathlib import Path
import requests
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'convnext1005'))
from run_scale_probe import prepare_dns
from kaggle.api.kaggle_api_extended import KaggleApi
prepare_dns();a=KaggleApi();a.authenticate()
verified=json.loads((HERE/'verified.json').read_text())
kernel=verified['kernel'];version=verified['version']
rows=[s.to_dict() for s in a.competition_submissions('rsna-knee-abnormality-detection')]
existing=[s for s in rows if kernel+'?' in s.get('url','')]
if existing:
    assert len(existing)==1
    result=dict(kernel=kernel,version=version,response={'ref':existing[0]['ref']},reconciled=existing[0])
    (HERE/'submission.json').write_text(json.dumps(result,indent=2));print(result)
else:
    retry=HERE/f'diagnostic_retry_v{version}.json'
    assert not retry.exists(),'Already retried; inspect recorded error'
    if version==1:assert not (HERE/'diagnostic_retry.json').exists()
    retry.write_text(json.dumps(dict(kernel=kernel,version=version,existing_submissions=existing),indent=2))
    try:
        response=a.competition_submit_code(file_name='submission.csv',message='Public Compact ensemble: verified source, coverage and blend',competition='rsna-knee-abnormality-detection',kernel=kernel,kernel_version=version).to_dict()
    except requests.HTTPError as exc:
        result=dict(status='REJECTED',http_status=exc.response.status_code,body=exc.response.text)
        (HERE/f'submission_error_v{version}.json').write_text(json.dumps(result,indent=2));print(result)
        raise SystemExit(1)
    else:
        result=dict(kernel=kernel,version=version,response=response)
        (HERE/'submission.json').write_text(json.dumps(result,indent=2));print(result)
