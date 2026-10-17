import json,sys
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'convnext1005'))
from run_scale_probe import prepare_dns
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
prepare_dns();a=KaggleApi();a.authenticate()
rows=[];token='';seen=set()
with a.build_kaggle_client() as client:
    while True:
        q=ApiListSubmissionsRequest();q.competition_name='rsna-knee-abnormality-detection';q.page_size=100;q.page_token=token
        r=client.competitions.competition_api_client.list_submissions(q).to_dict()
        rows.extend(r.get('submissions',[]));token=r.get('nextPageToken','')
        if not token:break
        assert token not in seen;seen.add(token)
    q=ApiGetKernelRequest();q.user_name='heliosli';q.kernel_slug='blend-mine-in-private-lb-luck-to-all-upvoters'
    r=client.kernels.kernels_api_client.get_kernel(q)
    (HERE/'upstream_metadata.json').write_text(json.dumps(r.metadata.to_dict(),indent=2))
    (HERE/'upstream.ipynb').write_text(r.blob.source,encoding='utf8')
competition=a.competitions_list(search='rsna-knee-abnormality-detection').to_dict()
latest=[k.to_dict() for k in a.kernels_list(competition='rsna-knee-abnormality-detection',sort_by='dateRun',page_size=30)]
record=dict(checked_at=datetime.now(timezone.utc).isoformat(),submissions=rows,competition=competition,latest=latest)
(HERE/'audit.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print('SUBMISSIONS',len(rows))
for s in rows:print(s.get('ref'),s.get('status'),s.get('publicScore'),s.get('errorDescription'),s.get('url'))
print('COMPETITION',json.dumps(competition));print('UPSTREAM',json.dumps(r.metadata.to_dict()));print('LATEST',json.dumps(latest))
