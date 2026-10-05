"""Launch a built candidate once; exact source hash and duplicate guards."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args()
    folder=args.folder.resolve()
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    record=folder.parent/'launch_receipt.json'
    attempt=folder.parent/'launch_attempt.json'
    if attempt.exists() or record.exists(): raise SystemExit('Already attempted; reconcile before retry')
    assert meta['enable_gpu'] and not meta['enable_internet'] and meta['is_private']
    import ast
    raw=(folder/meta['code_file']).read_bytes()
    nb=json.loads(raw)
    for c in nb['cells']:
        if c['cell_type']=='code': ast.parse(''.join(c['source']))
    api=KaggleApi();api.authenticate()
    if any(k.ref==meta['id'] for k in api.kernels_list(mine=True,search=meta['id'].split('/')[-1],page_size=100)):
        raise SystemExit('Kernel exists: do not duplicate')
    payload=dict(kernel=meta['id'],notebook_sha256=hashlib.sha256(raw).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accelerator='NvidiaTeslaT4')
    attempt.write_text(json.dumps(payload,indent=2))
    result=api.kernels_push(str(folder),acc='NvidiaTeslaT4').to_dict()
    record.write_text(json.dumps(dict(**payload,response=result),indent=2))
    print(json.dumps(result))
