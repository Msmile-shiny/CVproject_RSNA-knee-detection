"""Archive official Kaggle metadata and source for a bounded public shortlist."""
import json,sys,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'convnext1005'))
from run_scale_probe import prepare_dns
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
refs=['heliosli/blend-mine-in-private-lb-luck-to-all-upvoters','goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-954','bliverrigpdoge/velci-kcd25-20']
prepare_dns();a=KaggleApi();a.authenticate()
for ref in refs:
    folder=HERE/'upstream'/ref.split('/')[0];folder.mkdir(parents=True,exist_ok=True)
    a.kernels_pull(ref,path=str(folder),metadata=True)
    req=ApiGetKernelRequest();req.user_name,req.kernel_slug=ref.split('/')
    with a.build_kaggle_client() as c:r=c.kernels.kernels_api_client.get_kernel(req)
    data=r.metadata.to_dict();(folder/'api-metadata.json').write_text(json.dumps(data,indent=2),encoding='utf8')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.ipynb')}
    (folder/'provenance.json').write_text(json.dumps(dict(ref=ref,source_hashes=files),indent=2))
    print(json.dumps(dict(ref=ref,metadata=data,files=files)),flush=True)
