"""Collect and validate a completed Outer70 run, optionally submit it once."""
import argparse
import json
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from build_outer70 import ROOT, code_cells
from check_result import check

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');args=p.parse_args()
    api=KaggleApi();api.authenticate()
    kernel='easoncyy/rsna-sprint-outer70'
    version=json.loads((ROOT/'launch_receipt.json').read_text())['versionNumber']
    status=api.kernels_status(kernel).to_dict()
    print(status,flush=True)
    if status.get('status')!='COMPLETE':
        raise SystemExit('Run not complete; do not submit')
    out=ROOT.parents[1]/'results/outer70_20261004'
    out.mkdir(parents=True,exist_ok=True)
    source=out/'verified-source.ipynb'
    request=ApiGetKernelRequest();request.user_name='easoncyy';request.kernel_slug='rsna-sprint-outer70'
    with api.build_kaggle_client() as client:
        remote=client.kernels.kernels_api_client.get_kernel(request)
    metadata=remote.metadata.to_dict()
    if metadata.get('currentVersionNumber')!=version:
        raise RuntimeError('Latest version differs from recorded launch; do not use its outputs')
    source.write_text(remote.blob.source,encoding='utf-8')
    (out/'verified_metadata.json').write_text(json.dumps(metadata,indent=2))
    local=json.loads((ROOT/'outer70/outer70.ipynb').read_text(encoding='utf-8'))
    pulled=json.loads(source.read_text(encoding='utf-8'))
    if code_cells(local)!=code_cells(pulled):
        raise RuntimeError('Submitted version code differs from prepared notebook')
    pattern=r'^(outer70_receipt\.json|sprint_fourway_receipt\.json|btkd_v559_complete\.json|submission\.csv|baseline60_replay\.csv|transformer_branch_rank\.csv|coat_raptor_branch\.csv)$'
    api.kernels_output(kernel,path=str(out),file_pattern=pattern,page_size=500)
    checked=check(out)
    (ROOT/'visible_check_receipt.json').write_text(json.dumps(checked,indent=2))
    print(checked,flush=True)
    if args.submit:
        record=ROOT/'submission_receipt.json';attempt=ROOT/'submission_attempt.json'
        if record.exists():
            print(record.read_text());raise SystemExit(0)
        if attempt.exists():
            raise SystemExit('Reconcile previous submission attempt before retry')
        attempt.write_text(json.dumps(dict(kernel=kernel,version=version),indent=2))
        response=api.competition_submit_code(file_name='submission.csv',
            message='Outer70: change seven default outer weights 0.60 to 0.70; five overrides unchanged; exact 0.943 fourway parent',
            competition='rsna-knee-abnormality-detection',kernel=kernel,kernel_version=version)
        receipt=dict(kernel=kernel,version=version,response=response.to_dict(),visible_check=checked)
        record.write_text(json.dumps(receipt,indent=2,default=str))
        print(json.dumps(receipt,indent=2,default=str))
