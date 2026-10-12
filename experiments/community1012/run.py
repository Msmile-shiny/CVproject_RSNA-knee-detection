"""Run the archived public compact ensemble, with explicit provenance and coverage."""
import argparse,ast,copy,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sha=lambda b:hashlib.sha256(b).hexdigest()

def build():
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    folder=HERE/'upstream/heliosli';path=next(folder.glob('*.ipynb'));raw=path.read_bytes()
    provenance=json.loads((folder/'provenance.json').read_text());assert sha(raw)==provenance['source_hashes'][path.name]
    parent=json.loads(raw);nb=copy.deepcopy(parent)
    manifest=(HERE/'assets/manifest.json').read_bytes()
    guard='''import hashlib as _pub_hash, time as _pub_time
from pathlib import Path as _PubPath
_pub_started=_pub_time.monotonic()
_pub_manifests=list(_PubPath('/kaggle/input').glob('**/rsna-knee-compact-r6cnxfsss/manifest.json'))
assert len(_pub_manifests)==1, _pub_manifests
assert _pub_hash.sha256(_pub_manifests[0].read_bytes()).hexdigest()==__SHA__, 'Compact manifest version drift'
'''.replace('__SHA__',repr(sha(manifest)))
    audit='''# Verify complete ensemble output, not only that a CSV exists.
assert PROBE_CLASS is None
assert blend['both_failed']==0
assert blend['ours_failed']<=0.03*blend['rows'], 'Compact coverage below 97%; refusing degraded ensemble'
assert blend['raptor_failed']<=0.03*blend['rows']
assert W_RAPTOR==0.5 and W_OURS==0.5 and all(v==0.5 for v in W_RAPTOR_C.values())
_pub_elapsed=_pub_time.monotonic()-_pub_started
assert 0<_pub_elapsed<8.8*3600
_pub_coverage=dict(ids=test_ids,raptor_failed=rap_fail.tolist(),compact_failed=our_fail.tolist())
(WORK/'coverage.json').write_text(json.dumps(_pub_coverage))
_pub_files=['submission.csv','raptor_raw.csv','raptor_sub.csv','ours_sub.csv','coverage.json','blend_receipt.json','submission_receipt.json']
_pub_receipt=dict(status='PASS',source_ref='heliosli/blend-mine-in-private-lb-luck-to-all-upvoters',
    manifest_sha256=_pub_hash.sha256(_pub_manifests[0].read_bytes()).hexdigest(),
    bundle_id=BUNDLE_ID,elapsed_seconds=_pub_elapsed,study_count=len(test_ids),blend=blend,
    decode_coverage_verified=False,hashes={name:_pub_hash.sha256((WORK/name).read_bytes()).hexdigest() for name in _pub_files})
(WORK/'public_receipt.json').write_text(json.dumps(_pub_receipt,indent=2))
print('PUBLIC_COMPACT_VERIFIED',_pub_receipt)
'''
    nb['cells']=[nbformat.v4.new_code_cell(guard)]+nb['cells']+[nbformat.v4.new_code_cell(audit)]
    for i,c in enumerate(nb['cells']):
        c['id']=f'compact-public-{i}'
        if c['cell_type']=='code':
            c.update(outputs=[],execution_count=None)
            ast.parse(TransformerManager().transform_cell(''.join(c['source'])))
    nb['nbformat_minor']=5;nbformat.validate(nbformat.from_dict(nb))
    codes=lambda n:[''.join(c['source']) for c in n if c['cell_type']=='code']
    assert codes(nb['cells'][1:-1])==codes(parent['cells'])
    # Validate embedded Python without executing downloaded source locally.
    tree=ast.parse(''.join(parent['cells'][5]['source']))
    sources=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets))
    for name,source in sources.items():ast.parse(source,filename=name)
    out=HERE/'candidate';out.mkdir(exist_ok=True)
    payload=json.dumps(nb,ensure_ascii=False,indent=1).encode();(out/'compact.ipynb').write_bytes(payload)
    meta=json.loads((folder/'kernel-metadata.json').read_text());meta.pop('id_no',None)
    meta.update(id='easoncyy/rsna-compact-public-ensemble',title='RSNA Compact Public Ensemble',code_file='compact.ipynb',is_private=True)
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    result=dict(source_sha256=sha(payload),upstream_sha256=sha(raw),upstream_version=2,manifest_sha256=sha(manifest),image_and_blend_code_unchanged=True,max_failed_fraction=.03,embedded_files=len(sources))
    (HERE/'build.json').write_text(json.dumps(result,indent=2));print(result)

def verify(out,receipt):
    import numpy as np
    import pandas as pd
    coverage=json.loads((out/'coverage.json').read_text());ids=coverage['ids'];n=len(ids)
    assert n==receipt['study_count']>0 and len(set(ids))==n
    rf=np.array(coverage['raptor_failed'],dtype=bool);cf=np.array(coverage['compact_failed'],dtype=bool)
    assert len(rf)==n and len(cf)==n and not (rf&cf).any() and max(rf.sum(),cf.sum())<=.03*n
    frames={name:pd.read_csv(out/name,dtype={'StudyInstanceUID':str},float_precision='round_trip') for name in ['submission.csv','raptor_raw.csv','raptor_sub.csv','ours_sub.csv']}
    sys.path.insert(0,str(HERE.parent/'convnext1005'));from run_solo import LABELS
    for name,f in frames.items():
        assert f.columns.tolist()==LABELS and f.StudyInstanceUID.is_unique and set(f.StudyInstanceUID)==set(ids)
        f=f.set_index('StudyInstanceUID').loc[ids];v=f.to_numpy()
        assert np.isfinite(v).all() and (v>=0).all() and (v<=1).all();frames[name]=f
    assert np.array_equal(rf,(frames['raptor_raw.csv'].to_numpy()==.5).all(axis=1))
    def ranks(f,fail):
        v=np.full(f.shape,np.nan);v[~fail]=f.loc[~fail].rank(pct=True).to_numpy();return v
    rr=ranks(frames['raptor_raw.csv'],rf);cc=ranks(frames['ours_sub.csv'],cf)
    expected=.5*rr+.5*cc;expected[rf]=cc[rf];expected[cf]=rr[cf]
    np.testing.assert_allclose(frames['submission.csv'],expected.astype(np.float32),rtol=1e-7,atol=1e-8)
    assert receipt['blend']['ours_failed']==int(cf.sum()) and receipt['blend']['raptor_failed']==int(rf.sum())

def online(action,submit):
    sys.path.insert(0,str(HERE.parent/'convnext1005'))
    from run_scale_probe import prepare_dns
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    built=json.loads((HERE/'build.json').read_text());raw=(HERE/'candidate/compact.ipynb').read_bytes();assert sha(raw)==built['source_sha256']
    prepare_dns();a=KaggleApi();a.authenticate();kernel='easoncyy/rsna-compact-public-ensemble'
    if action=='launch':
        assert not (HERE/'launch_attempt.json').exists()
        assert not any(k.ref==kernel for k in a.kernels_list(mine=True,search='rsna-compact-public-ensemble',page_size=100))
        (HERE/'launch_attempt.json').write_text(json.dumps(built,indent=2))
        response=a.kernels_push(str(HERE/'candidate'),acc='NvidiaTeslaT4').to_dict()
        (HERE/'launch.json').write_text(json.dumps(dict(build=built,response=response),indent=2));print(response);return
    launch=json.loads((HERE/'launch.json').read_text());kernel=launch['response']['ref'].removeprefix('/code/');version=launch['response']['versionNumber']
    status=a.kernels_status(kernel).to_dict();print(status,flush=True)
    if status['status'] not in ['COMPLETE','ERROR']:return
    out=ROOT/'results'/f'compact_public_v{version}';out.mkdir(parents=True,exist_ok=True)
    a.kernels_output(kernel,path=str(out),file_pattern=r'^(.*receipt\.json|coverage\.json|submission\.csv|raptor_raw\.csv|raptor_sub\.csv|ours_sub\.csv|worker.*\.txt)$',page_size=100)
    assert status['status']=='COMPLETE',f'Inspect logs {out}; no automatic restart'
    req=ApiGetKernelRequest();req.user_name,req.kernel_slug=kernel.split('/')
    with a.build_kaggle_client() as c:remote=c.kernels.kernels_api_client.get_kernel(req)
    assert remote.metadata.to_dict()['currentVersionNumber']==version
    codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    assert codes(json.loads(raw))==codes(json.loads(remote.blob.source))
    receipt=json.loads((out/'public_receipt.json').read_text())
    assert receipt['status']=='PASS' and receipt['manifest_sha256']==built['manifest_sha256']
    assert receipt['bundle_id']=='209a32c8b50a6b1c' and 0<receipt['elapsed_seconds']<8.8*3600
    for name,h in receipt['hashes'].items():assert sha((out/name).read_bytes())==h
    verify(out,receipt)
    verified=dict(status='PASS',kernel=kernel,version=version,receipt=receipt)
    (HERE/'verified.json').write_text(json.dumps(verified,indent=2));print(verified)
    if submit:
        if (HERE/'submission.json').exists():print('Already submitted');return
        assert not (HERE/'submission_attempt.json').exists(),'Reconcile previous attempt'
        assert not any(kernel+'?' in s.to_dict().get('url','') for s in a.competition_submissions('rsna-knee-abnormality-detection'))
        (HERE/'submission_attempt.json').write_text(json.dumps(verified,indent=2))
        response=a.competition_submit_code(file_name='submission.csv',message='Public Helios compact four-fold ensemble plus Raptor; archived source, pinned manifest, coverage and rank formula verified',competition='rsna-knee-abnormality-detection',kernel=kernel,kernel_version=version).to_dict()
        (HERE/'submission.json').write_text(json.dumps(dict(kernel=kernel,version=version,response=response),indent=2));print(response)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','launch','collect']);p.add_argument('--submit',action='store_true');args=p.parse_args()
    if args.action=='build':build()
    else:online(args.action,args.submit)
