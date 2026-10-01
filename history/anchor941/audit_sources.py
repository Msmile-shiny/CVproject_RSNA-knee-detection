"""Fetch exact source versions and metadata without downloading model weights."""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

ROOT = Path(__file__).resolve().parent
api = KaggleApi()
api.authenticate()

def pull(ref, version, folder):
    owner, slug = ref.split('/')
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = owner, slug
    if version:
        request.version_label = str(version)
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(request)
    out = ROOT / folder
    out.mkdir(exist_ok=True)
    source = response.blob.source
    (out / 'source.ipynb').write_text(source, encoding='utf-8')
    record = {'requested_ref': ref, 'requested_version': version,
              'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'metadata': response.metadata.to_dict()}
    (out / 'source_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(ref, version, 'OK', response.metadata.current_version_number,
          record['source_sha256'][:16], flush=True)
    return response

def audit_dependency(item):
    kind, ref = item
    local = KaggleApi()
    local.authenticate()
    try:
        version = None
        if kind == 'dataset':
            req = ApiGetDatasetRequest()
            req.owner_slug, req.dataset_slug = ref.split('/')
            with local.build_kaggle_client() as client:
                response = client.datasets.dataset_api_client.get_dataset(req)
            version = response.current_version_number
        token, files = None, []
        while True:
            if kind == 'dataset':
                response = local.dataset_list_files(f'{ref}/{version}', page_token=token, page_size=100)
            else:
                response = local.kernels_list_files(ref, page_token=token, page_size=100)
            data = response.to_dict()
            files.extend(data.get('files', data.get('datasetFiles', [])))
            token = data.get('nextPageToken')
            if not token:
                break
        result = dict(kind=kind, ref=ref, version=version, files=files, accessible=True)
        print(kind, ref, 'version', version, 'files', len(files), flush=True)
        return result
    except Exception as exc:
        print(ref, type(exc).__name__, str(exc)[:150], flush=True)
        return dict(kind=kind, ref=ref, accessible=False, error=str(exc))

if __name__ == '__main__' and '--dependencies' in sys.argv:
    meta = json.loads((ROOT / 'fast-locked/source_record.json').read_text())['metadata']
    jobs = [('dataset', ref) for ref in meta['datasetDataSources']]
    jobs += [('kernel', ref) for ref in meta['kernelDataSources']]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(audit_dependency, jobs))
    (ROOT / 'dependency_audit.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    assert all(r['accessible'] and r['files'] for r in records), 'An asset is unavailable or empty'
elif __name__ == '__main__':
    for ref, version, folder in [
        ('evgendvorkin/rsna-baseline', 14, 'source-v14'),
        ('mattiaangeli/bend-the-knee-to-the-dinosaurs', 32, 'source-v32'),
        ('jiweiliu/rsna-knee-fast-2xt4-inference', None, 'fast-locked')]:
        try:
            pull(ref, version, folder)
        except Exception as exc:
            print(ref, version, type(exc).__name__, str(exc)[:240], flush=True)
