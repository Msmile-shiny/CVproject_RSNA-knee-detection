"""One-shot CPU probe launch; never submit to competition."""
import datetime
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

root = Path(__file__).resolve().parent
attempt = root / 'cpu_launch_attempt.json'
if attempt.exists():
    raise SystemExit('Already attempted: reconcile status, do not duplicate')
meta = json.loads((root / 'cpu-probe/kernel-metadata.json').read_text())
assert meta['enable_gpu'] is False and meta['enable_internet'] is False
api = KaggleApi()
api.authenticate()
if any(k.ref == meta['id'] for k in api.kernels_list(mine=True, search='meniscus-cpu-gate', page_size=100)):
    raise SystemExit('Kernel exists; reconcile before launch')
attempt.write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'kernel': meta['id'], 'gpu': False}, indent=2))
result = api.kernels_push(str(root / 'cpu-probe')).to_dict()
(root / 'cpu_launch_receipt.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
