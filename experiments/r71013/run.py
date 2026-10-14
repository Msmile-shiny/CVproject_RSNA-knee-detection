"""Isolated R7 reproduction. Preserve the scored R6 baseline."""
import argparse, ast, copy, hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REF='easoncyy/rsna-r7-public-reproduction'
UPSTREAM='jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble'
sha=lambda b:hashlib.sha256(b).hexdigest()

def build():
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    assert not (HERE/'launch_attempt.json').exists(), 'Do not mutate launched candidate'
    raw=next((HERE/'upstream').glob('*.ipynb')).read_bytes()
    nb=json.loads(raw)
    manifest_raw=(HERE/'assets/manifest.json').read_bytes()
    manifest=json.loads(manifest_raw)
    assert manifest['bundle_id']=='8c7b666e27dc0e8f'
    # Reuse already tested output guards without changing upstream inference.
    previous=json.loads((HERE.parent/'community1012/candidate/compact.ipynb').read_bytes())
    guard=''.join(previous['cells'][0]['source']).replace('rsna-knee-compact-v1','rsna-knee-compact-r7cnxfsss')
    old_manifest=sha((HERE.parent/'community1012/assets/manifest.json').read_bytes())
    guard=guard.replace(old_manifest,sha(manifest_raw))
    audit=''.join(previous['cells'][-1]['source']).replace('heliosli/blend-mine-in-private-lb-luck-to-all-upvoters',UPSTREAM)
    original=[ ''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    embedded=ast.parse(original[2])
    sources=next(ast.literal_eval(n.value) for n in embedded.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets))
    for name,source in sources.items():
        ast.parse(source,filename=name)
        assert sha(source.encode())==manifest['src_sha256'][name], name
    nb['cells']=[nbformat.v4.new_code_cell(guard)]+nb['cells']+[nbformat.v4.new_code_cell(audit)]
    for i,c in enumerate(nb['cells']):
        c['id']=f'r7-public-{i}'
        if c['cell_type']=='code':
            c.update(outputs=[],execution_count=None)
            ast.parse(TransformerManager().transform_cell(''.join(c['source'])))
    assert [''.join(c['source']) for c in nb['cells'][1:-1] if c['cell_type']=='code']==original
    nb['nbformat_minor']=5
    nbformat.validate(nbformat.from_dict(nb))
    out=HERE/'candidate';out.mkdir(exist_ok=True)
    payload=json.dumps(nb,ensure_ascii=False,indent=1).encode()
    (out/'r7.ipynb').write_bytes(payload)
    meta=json.loads((HERE/'upstream/kernel-metadata.json').read_text())
    meta.pop('id_no',None)
    meta.update(id=REF,title='RSNA R7 Public Reproduction',code_file='r7.ipynb',is_private=True,enable_internet=False)
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    record=dict(upstream=UPSTREAM,upstream_version=1,upstream_sha256=sha(raw),source_sha256=sha(payload),manifest_sha256=sha(manifest_raw),bundle_id=manifest['bundle_id'],inference_unchanged=True,embedded_files=len(sources),protected_submission=57131878,protected_score=.954)
    (HERE/'build.json').write_text(json.dumps(record,indent=2));print(record)

def online(action,submit=False):
    sys.path.insert(0,str(ROOT/'experiments/convnext1005'))
    from run_scale_probe import prepare_dns
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    prepare_dns();a=KaggleApi();a.authenticate()
    built=json.loads((HERE/'build.json').read_text())
    raw=(HERE/'candidate/r7.ipynb').read_bytes();assert sha(raw)==built['source_sha256']
    if action=='launch':
        assert not (HERE/'launch_attempt.json').exists()
        assert not any(k.ref==REF for k in a.kernels_list(mine=True,search='rsna-r7-public-reproduction',page_size=100))
        (HERE/'launch_attempt.json').write_text(json.dumps(built,indent=2))
        r=a.kernels_push(str(HERE/'candidate'),acc='NvidiaTeslaT4').to_dict()
        (HERE/'launch.json').write_text(json.dumps(dict(build=built,response=r),indent=2));print(r)
        assert not r.get('invalidDatasetSources'),r
        return
    launch=json.loads((HERE/'launch.json').read_text());version=launch['response']['versionNumber']
    status=a.kernels_status(REF).to_dict();print(status,flush=True)
    if status.get('status') not in ('COMPLETE','ERROR'):return
    out=ROOT/'results'/f'{REF.split("/")[1]}_v{version}';out.mkdir(parents=True,exist_ok=True)
    a.kernels_output(REF,path=str(out),file_pattern=r'^(.*receipt\.json|coverage\.json|submission\.csv|raptor_raw\.csv|raptor_sub\.csv|ours_sub\.csv|worker.*\.txt)$',page_size=100)
    assert status['status']=='COMPLETE', 'Inspect logs; no automatic retries'
    q=ApiGetKernelRequest();q.user_name,q.kernel_slug=REF.split('/')
    with a.build_kaggle_client() as c:r=c.kernels.kernels_api_client.get_kernel(q)
    assert r.metadata.to_dict()['currentVersionNumber']==version
    codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    assert codes(json.loads(raw))==codes(json.loads(r.blob.source))
    receipt=json.loads((out/'public_receipt.json').read_text())
    assert receipt['status']=='PASS' and receipt['manifest_sha256']==built['manifest_sha256'] and receipt['bundle_id']==built['bundle_id']
    assert 0<receipt['elapsed_seconds']<8.8*3600
    for name,h in receipt['hashes'].items():assert sha((out/name).read_bytes())==h
    import importlib.util
    spec=importlib.util.spec_from_file_location('compact_gates',ROOT/'experiments/community1012/run.py')
    gates=importlib.util.module_from_spec(spec);spec.loader.exec_module(gates);gates.verify(out,receipt)
    verified=dict(version=version,kernel=REF,status='PASS',receipt=receipt)
    (HERE/'verified.json').write_text(json.dumps(verified,indent=2));print(verified)
    if not submit:return
    if (HERE/'submission.json').exists():print('Already submitted');return
    assert not (HERE/'submission_attempt.json').exists(),'Reconcile previous attempt; never resend blindly'
    assert not any(REF+'?' in s.to_dict().get('url','') for s in a.competition_submissions('rsna-knee-abnormality-detection'))
    (HERE/'submission_attempt.json').write_text(json.dumps(verified,indent=2))
    try:
        message=('R7 no-fluid control: only Effusion volume correction disabled; non-Effusion visible outputs checked; fixed 50/50 blend' if REF.endswith('no-fluid-control') else 'R7 public reproduction: unchanged inference, pinned manifest, four folds and stackers; fixed 50/50 rank blend')
        if REF.endswith('training-mirror'):message='R7 training mirror: original four folds and stackers; MCL plain, reindexed mirror logits; fixed 50/50 blend'
        response=a.competition_submit_code(file_name='submission.csv',message=message,competition='rsna-knee-abnormality-detection',kernel=REF,kernel_version=version).to_dict()
    except Exception as exc:
        r=getattr(exc,'response',None)
        (HERE/'submission_error.json').write_text(json.dumps(dict(http_status=getattr(r,'status_code',None),body=getattr(r,'text',str(exc))),indent=2))
        raise
    (HERE/'submission.json').write_text(json.dumps(dict(kernel=REF,version=version,response=response),indent=2));print(response)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','launch','collect']);p.add_argument('--submit',action='store_true');args=p.parse_args()
    if args.action=='build':build()
    else:online(args.action,args.submit)
