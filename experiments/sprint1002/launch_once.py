"""Launch the prepared ablation once, recording the exact Kaggle version."""
import datetime
import json
from kaggle.api.kaggle_api_extended import KaggleApi
from build_outer70 import ROOT
from preflight import check

if __name__ == '__main__':
    check()
    attempt=ROOT/'launch_attempt.json'
    if attempt.exists():
        raise SystemExit('Launch already attempted; reconcile launch_receipt.json and Kaggle before retrying')
    api=KaggleApi();api.authenticate()
    matches=api.kernels_list(mine=True,search='outer70',page_size=100)
    if matches:
        raise SystemExit('Existing Outer70 kernels found; inspect before starting another version')
    attempt.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'kernel':'easoncyy/rsna-sprint-outer70','accelerator':'NvidiaTeslaT4'},indent=2))
    result=api.kernels_push(str(ROOT/'outer70'),acc='NvidiaTeslaT4')
    payload=result.to_dict()
    (ROOT/'launch_receipt.json').write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8')
    print(json.dumps(payload,indent=2,default=str))
