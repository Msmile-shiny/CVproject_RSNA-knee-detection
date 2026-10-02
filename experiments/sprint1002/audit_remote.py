"""Read-only Kaggle asset and submission inventory. Never starts compute."""
import concurrent.futures
import datetime
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent


def inventory(kind, ref):
    api = KaggleApi()
    api.authenticate()
    files, token, pages = [], None, 0
    try:
        while True:
            method = {'dataset':api.dataset_list_files, 'kernel':api.kernels_list_files,
                      'model':api.model_instance_version_files}[kind]
            response = method(ref, page_token=token, page_size=500)
            d = response.to_dict()
            batch = d.get('datasetFiles', d.get('files', []))
            files.extend(batch)
            token = d.get('nextPageToken')
            pages += 1
            if not token:
                break
            if pages >= 100:
                raise RuntimeError('Pagination exceeded 100 pages')
        return dict(kind=kind, ref=ref, status='LISTED', files=files, pages=pages)
    except Exception as e:
        return dict(kind=kind, ref=ref, status='ERROR', error=type(e).__name__+': '+str(e)[:250])


def main():
    meta = json.loads((ROOT/'outer70/kernel-metadata.json').read_text())
    jobs = [(kind, ref) for kind,key in [('dataset','dataset_sources'),('kernel','kernel_sources'),('model','model_sources')] for ref in meta[key]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assets = list(pool.map(lambda x:inventory(*x), jobs))
    api = KaggleApi(); api.authenticate()
    submissions = api.competition_submissions('rsna-knee-abnormality-detection', page_size=100)
    records = [s.to_dict() for s in submissions]
    keep = [s for s in records if str(s.get('ref')) in {'56700487','56706632','56454576'}]
    report = dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  gpu_launched=False, assets=assets, protected_submissions=keep,
                  note='Listings verify access and names, not checkpoint byte hashes.')
    (ROOT/'remote_audit.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    for a in assets:
        print(a['kind'],a['ref'],a['status'],len(a.get('files',[])),a.get('error',''))
    print('protected_submissions',json.dumps(keep,default=str))
    if any(a['status'] != 'LISTED' or not a['files'] for a in assets):
        raise SystemExit('One or more listings need inspection')


if __name__ == '__main__':
    main()
