"""Controlled reader-view upgrade of the scored 0.950 notebook."""
import argparse, ast, copy, hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
digest=lambda b:hashlib.sha256(b).hexdigest()

def build():
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    raw=(ROOT/'experiments/swa2241010/upstream/parent950.ipynb').read_bytes()
    assert digest(raw)=='16ce7c78f860b6d07331c8d39c6a65414cde200462a64c984b04bd8efc167289'
    parent=json.loads(raw);nb=copy.deepcopy(parent)
    apexraw=(ROOT/'experiments/apex1009/candidate/apex.ipynb').read_bytes()
    assert digest(apexraw)==json.loads((ROOT/'experiments/apex1009/build.json').read_text())['source_sha256']
    apex=json.loads(apexraw)
    reader=''.join(apex['cells'][9]['source'])
    assert reader.startswith('%%writefile /kaggle/working/own_src/infer.py')
    nb['cells'][8]['source']=reader
    assert all(nb['cells'][i]==parent['cells'][i] for i in range(len(parent['cells'])) if i!=8)
    guard=copy.deepcopy(apex['cells'][0])
    audit='''import hashlib as _r8_h, json as _r8_j
from pathlib import Path as _r8_P
assert len(_o_ck)==3
_r8_root=_r8_P('/kaggle/working')
_r8_names=['submission.csv','_sota_0949.csv','_own.csv']
_r8_receipt=dict(status='COMPLETE',study_count=len(sub),parent_submission=57012367,
    elapsed_seconds=_audit_time.monotonic()-_audit_started,
    reader_source_sha256=_r8_h.sha256((_r8_root/'own_src/infer.py').read_bytes()).hexdigest(),
    reader_weights=_audit_weights,swa_sha256=digest,decode_coverage_verified=False,
    hashes={n:_r8_h.sha256((_r8_root/n).read_bytes()).hexdigest() for n in _r8_names})
assert 0<_r8_receipt['elapsed_seconds']<8.5*3600
(_r8_root/'reader8_receipt.json').write_text(_r8_j.dumps(_r8_receipt,indent=2))
print('READER8 COMPLETE',_r8_receipt)
'''
    nb['cells']=[guard]+nb['cells']+[nbformat.v4.new_code_cell(audit)]
    for i,c in enumerate(nb['cells']):
        c['id']=f'reader8-{i}'
        if c['cell_type']=='code':
            c.update(outputs=[],execution_count=None)
            ast.parse(TransformerManager().transform_cell(''.join(c['source'])))
    nb['nbformat_minor']=5;nbformat.validate(nbformat.from_dict(nb))
    folder=HERE/'candidate';folder.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(nb,ensure_ascii=False,indent=1).encode()
    (folder/'reader8.ipynb').write_bytes(payload)
    meta=json.loads((ROOT/'experiments/swa2241010/upstream/parent-metadata.json').read_text())
    meta.pop('id_no',None)
    meta.update(id='easoncyy/rsna-parent950-reader8',title='RSNA Parent950 Reader8',code_file='reader8.ipynb',is_private=True)
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    result=dict(source_sha256=digest(payload),reader_sha256=digest((reader.split('\n',1)[1].rstrip('\n')+'\n').encode()),parent_sha256=digest(raw),changed_parent_cells=[8],fusion_unchanged=True)
    (HERE/'build.json').write_text(json.dumps(result,indent=2));print(result)

def verify(out,receipt,use_peers=True):
    import numpy as np
    import pandas as pd
    from scipy.special import ndtri,expit
    sys.path.insert(0,str(ROOT/'experiments/convnext1005'))
    from run_solo import LABELS
    frames={n:pd.read_csv(out/n,dtype={'StudyInstanceUID':str},float_precision='round_trip') for n in receipt['hashes']}
    ids=frames['submission.csv'].StudyInstanceUID
    for n,f in frames.items():
        assert f.columns.tolist()==LABELS and f.StudyInstanceUID.is_unique
        assert len(f)==receipt['study_count']>0 and set(f.StudyInstanceUID)==set(ids)
        v=f[LABELS[1:]].to_numpy();assert np.isfinite(v).all() and (v>=0).all() and (v<=1).all()
        frames[n]=f.set_index('StudyInstanceUID').loc[ids].reset_index()
    parent=json.loads((ROOT/'experiments/swa2241010/upstream/parent950.ipynb').read_bytes())
    tree=ast.parse(''.join(parent['cells'][9]['source']))
    def literal(name):
        values=[ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        assert len(values)==1;return values[0]
    a=frames['_sota_0949.csv'][LABELS[1:]].rank(pct=True);r=frames['_own.csv'][LABELS[1:]].rank(pct=True);e=a.copy()
    for label in LABELS[1:]:
        w=literal('OPTIMAL_WEIGHTS')[label]
        e[label]=.5*((1-w)*a[label]+w*r[label])+.5*expit((1-w)*ndtri(np.clip(a[label],1e-5,1-1e-5))+w*ndtri(np.clip(r[label],1e-5,1-1e-5)))+(r[label]-.5)*1e-4
    e=e.rank(pct=True)
    if use_peers:
        for name,links in literal('PEERS').items():e[name]=.98*e[name]+.02*sum(w*e[k] for k,w in links.items())/sum(links.values())
    np.testing.assert_allclose(frames['submission.csv'][LABELS[1:]],e.rank(pct=True),rtol=1e-12,atol=1e-15)

def online(action,submit):
    sys.path.insert(0,str(ROOT/'experiments/convnext1005'))
    from run_scale_probe import prepare_dns
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    built=json.loads((HERE/'build.json').read_text());raw=(HERE/'candidate/reader8.ipynb').read_bytes()
    assert digest(raw)==built['source_sha256']
    prepare_dns();a=KaggleApi();a.authenticate();kernel=built.get('kernel','easoncyy/rsna-parent950-reader8')
    if action=='launch':
        assert not (HERE/'launch_attempt.json').exists()
        assert not any(k.ref==kernel for k in a.kernels_list(mine=True,search='rsna-parent950-reader8',page_size=100))
        (HERE/'launch_attempt.json').write_text(json.dumps(built,indent=2))
        response=a.kernels_push(str(HERE/'candidate'),acc='NvidiaTeslaT4').to_dict()
        (HERE/'launch.json').write_text(json.dumps(dict(build=built,response=response),indent=2));print(response);return
    launch=json.loads((HERE/'launch.json').read_text());version=launch['response']['versionNumber']
    kernel=launch['response']['ref'].removeprefix('/code/')
    status=a.kernels_status(kernel).to_dict();print(status,flush=True)
    if status['status'] not in ['COMPLETE','ERROR']:return
    prefix=built.get('output_prefix','reader8')
    out=ROOT/'results'/f'{prefix}_v{version}';out.mkdir(parents=True,exist_ok=True)
    a.kernels_output(kernel,path=str(out),file_pattern=r'^(reader8_receipt\.json|submission\.csv|_sota_0949\.csv|_own\.csv)$',page_size=100)
    assert status['status']=='COMPLETE',f'Inspect logs in {out}'
    req=ApiGetKernelRequest();req.user_name,req.kernel_slug=kernel.split('/')
    with a.build_kaggle_client() as c:remote=c.kernels.kernels_api_client.get_kernel(req)
    assert remote.metadata.to_dict()['currentVersionNumber']==version
    codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    assert codes(json.loads(raw))==codes(json.loads(remote.blob.source))
    receipt=json.loads((out/'reader8_receipt.json').read_text())
    assert receipt['reader_source_sha256']==built['reader_sha256'] and receipt['status']=='COMPLETE'
    assert 0<receipt['elapsed_seconds']<8.5*3600
    for n,h in receipt['hashes'].items():assert digest((out/n).read_bytes())==h
    verify(out,receipt,use_peers=built.get('use_peers',True))
    verified=dict(status='PASS',kernel=kernel,version=version,receipt=receipt)
    (HERE/'verified.json').write_text(json.dumps(verified,indent=2));print(verified)
    if submit:
        if (HERE/'submission.json').exists():print('Already submitted');return
        assert not (HERE/'submission_attempt.json').exists(),'Reconcile prior attempt'
        assert not any(kernel+'?' in s.to_dict().get('url','') for s in a.competition_submissions('rsna-knee-abnormality-detection'))
        (HERE/'submission_attempt.json').write_text(json.dumps(verified,indent=2))
        response=a.competition_submit_code(file_name='submission.csv',message=built.get('submission_message','0.950 parent with scored Apex 8-view reader; fusion and slot routing unchanged'),competition='rsna-knee-abnormality-detection',kernel=kernel,kernel_version=version).to_dict()
        (HERE/'submission.json').write_text(json.dumps(dict(kernel=kernel,version=version,response=response),indent=2));print(response)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','launch','collect']);p.add_argument('--submit',action='store_true');args=p.parse_args()
    if args.action=='build':build()
    else:online(args.action,args.submit)
