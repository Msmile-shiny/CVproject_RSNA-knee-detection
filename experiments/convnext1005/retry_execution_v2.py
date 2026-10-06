"""One authorized execution-only repair, never retry a pending/successful submission."""
import datetime
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    marker=HERE/'repair_v2_attempt.json'
    assert not marker.exists(), 'Repair already attempted; reconcile rather than repush'
    old=json.loads((HERE/'launch_receipt.json').read_text())
    assert old['response']['versionNumber']==1
    api=KaggleApi();api.authenticate()
    submissions=api.competition_submissions('rsna-knee-abnormality-detection')
    failed=next(s for s in submissions if s.ref==56856316)
    assert failed.error_description and not failed.public_score, 'Only repair the failed v1'
    assert api.kernels_status('easoncyy/rsna-sprint-three-fold-convnext-30').to_dict()['status']=='COMPLETE'
    raw=(HERE/'cnx3fold30/cnx3fold30.ipynb').read_bytes()
    payload=dict(kernel='easoncyy/rsna-sprint-three-fold-convnext-30',
                 notebook_sha256=hashlib.sha256(raw).hexdigest(),
                 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 accelerator='NvidiaTeslaT4',repair='Bounded single-process loader and explicit CUDA allocator initialization; root cause unconfirmed')
    marker.write_text(json.dumps(payload,indent=2))
    result=api.kernels_push(str(HERE/'cnx3fold30'),acc='NvidiaTeslaT4').to_dict()
    (HERE/'launch_receipt.json').write_text(json.dumps(dict(**payload,response=result),indent=2))
    assert result['versionNumber']==2, 'Unexpected version: reconcile before submission'
    for name in ['submission_attempt.json','submission_receipt.json','visible_check_receipt.json']:
        original=HERE/name
        assert original.read_bytes()==(HERE/'failed-v1'/name).read_bytes()
        original.replace(HERE/'failed-v1'/name)
    print(json.dumps(result))
