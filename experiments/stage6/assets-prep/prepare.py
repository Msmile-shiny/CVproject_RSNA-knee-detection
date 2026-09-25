"""CPU-only asset preparation; no competition data. Training remains offline."""
import hashlib
import json
import urllib.request
from pathlib import Path

url='https://download.pytorch.org/models/resnet34-b627a593.pth'
path=Path('/kaggle/working/resnet34-b627a593.pth')
urllib.request.urlretrieve(url,path)
digest=hashlib.sha256(path.read_bytes()).hexdigest()
assert digest=='b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f'
Path('/kaggle/working/asset_receipt.json').write_text(json.dumps(dict(source=url,sha256=digest,bytes=path.stat().st_size),indent=2))
print('Official weight verified:',digest)
