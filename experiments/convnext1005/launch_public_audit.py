"""One CPU public-data diagnostic, with durable duplicate guard."""
import datetime
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    attempt=HERE/'public_audit_attempt.json'
    assert not attempt.exists(), 'Already attempted; reconcile first'
    folder=HERE/'public-audit'
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    assert meta['enable_gpu'] is False and meta['enable_internet'] is False
    api=KaggleApi();api.authenticate()
    assert not any(k.ref==meta['id'] for k in api.kernels_list(mine=True,search='rsna-cnx-public-data-audit',page_size=100))
    payload=dict(kernel=meta['id'],gpu=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 source_sha256=hashlib.sha256((folder/meta['code_file']).read_bytes()).hexdigest())
    attempt.write_text(json.dumps(payload,indent=2))
    response=api.kernels_push(str(folder)).to_dict()
    (HERE/'public_audit_launch.json').write_text(json.dumps(dict(**payload,response=response),indent=2))
    print(json.dumps(response))
