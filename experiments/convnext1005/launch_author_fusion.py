"""Launch one fixed fusion version, never retry an uncertain push."""
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from run_scale_probe import prepare_dns

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    build=json.loads((HERE/'author_fusion_build.json').read_text())
    assert hashlib.sha256((HERE/'author-fusion30/fusion.ipynb').read_bytes()).hexdigest()==build['source_sha256']
    attempt=HERE/'author_fusion_attempt.json'
    record=HERE/'author_fusion_launch.json'
    assert not attempt.exists() and not record.exists(), 'Reconcile previous launch before retry'
    prepare_dns();a=KaggleApi();a.authenticate()
    kernel='easoncyy/rsna-cnx-author-fusion30'
    assert not any(k.ref==kernel for k in a.kernels_list(mine=True,search='rsna-cnx-author-fusion30',page_size=100))
    attempt.write_text(json.dumps(build,indent=2))
    response=a.kernels_push(str(HERE/'author-fusion30'),acc='NvidiaTeslaT4').to_dict()
    record.write_text(json.dumps(dict(build=build,response=response),indent=2))
    print(response)
