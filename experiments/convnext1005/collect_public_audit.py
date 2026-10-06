"""Collect only the recorded CPU diagnostic version; never submit a prediction."""
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    record=json.loads((HERE/'public_audit_launch.json').read_text())
    kernel=record['response']['ref'].removeprefix('/code/')
    version=record['response']['versionNumber']
    api=KaggleApi();api.authenticate()
    status=api.kernels_status(kernel).to_dict(); print(status,flush=True)
    if status['status'] not in ('COMPLETE','ERROR'): raise SystemExit(0)
    out=HERE.parents[1]/'results'/f'cnx_public_audit_v{version}'
    out.mkdir(parents=True,exist_ok=True)
    api.kernels_output(kernel,path=str(out),file_pattern=r'^(public_data_audit\.json|public_data_series\.jsonl)$',page_size=100)
    if status['status']=='ERROR': raise RuntimeError(f'CPU diagnostic failed: inspect log at {out}')
    request=ApiGetKernelRequest();request.user_name,request.kernel_slug=kernel.split('/')
    with api.build_kaggle_client() as client: remote=client.kernels.kernels_api_client.get_kernel(request)
    assert remote.metadata.to_dict()['currentVersionNumber']==version
    raw=(HERE/'public-audit/audit.ipynb').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==record['source_sha256']
    codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    assert codes(json.loads(raw))==codes(json.loads(remote.blob.source))
    summary=json.loads((out/'public_data_audit.json').read_text())
    assert summary['complete'] and summary['processed']==summary['total_selected']
    assert summary['preprocessing_sha256']=='4491963e4d2156625c9cf79051f4a25c32d2d761f1f07df3117a439ed8029ae2'
    summary.update(kernel=kernel,version=version,remote_source_verified=True)
    (HERE/'public_audit_result.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,ensure_ascii=True))
