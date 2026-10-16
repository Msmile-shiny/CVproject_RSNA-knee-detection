"""Build and launch one fixed R7 + new all-data S6 candidate; never auto-submit."""
import argparse
import ast
import base64
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REF = 'easoncyy/rsna-r7-s6-all-data'
sha = lambda b: hashlib.sha256(b).hexdigest()


def build():
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    assert not (HERE / 'launch_attempt.json').exists()
    parent = (ROOT / 'experiments/r71013/candidate/r7.ipynb').read_bytes()
    assert sha(parent) == '3aa159ab50bc7f8828a291fc297e3af2ab68985d331f052779ba1539179c7c3f'
    source = (ROOT / 'experiments/assets1015/nartaa-notebook/rsna-knee-s6fall-solo.py').read_bytes()
    adapter = (HERE / 'adapter.py').read_bytes()
    ast.parse(source); ast.parse(adapter)
    nb = nbformat.reads(parent.decode(), as_version=4)
    code = '''import base64, subprocess, sys, gc, hashlib
assert not rap_fail.any() and not our_fail.any(), 'Parent member incomplete'
# Remove the parent-only final filename before attempting the new member.
(WORK/'submission.csv').replace(WORK/'parent956.csv')
_source=base64.b64decode(SOURCE_B64)
_adapter=base64.b64decode(ADAPTER_B64)
assert hashlib.sha256(_source).hexdigest()==SOURCE_SHA
(WORK/'s6_author.py').write_bytes(_source)
(WORK/'s6_adapter.py').write_bytes(_adapter)
gc.collect()
import torch
torch.cuda.empty_cache()
_remaining=int(8.5*3600-(_pub_time.monotonic()-_pub_started))
assert _remaining>3000, 'Insufficient total runtime budget'
subprocess.run([sys.executable,'-u',str(WORK/'s6_adapter.py'),str(WORK/'s6_author.py'),str(_remaining-300)],check=True,timeout=_remaining)
_s6receipt=json.loads((WORK/'s6_member/s6_receipt.json').read_text())
assert _s6receipt['status']=='PASS' and not _s6receipt['B']['failed_after_retry']
assert _s6receipt['model']['sha256']=='0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'
_s6path=WORK/'s6_member/s6_raw.csv'
assert hashlib.sha256(_s6path.read_bytes()).hexdigest()==_s6receipt['raw_sha256']
_s6=pd.read_csv(_s6path,dtype={'StudyInstanceUID':str})
assert list(_s6.columns)==['StudyInstanceUID']+list(labels)
assert _s6.StudyInstanceUID.is_unique and set(_s6.StudyInstanceUID)==set(test_ids)
_s6=_s6.set_index('StudyInstanceUID').loc[test_ids,labels]
assert np.isfinite(_s6.values).all() and ((_s6.values>=0)&(_s6.values<=1)).all()
_parent=pd.read_csv(WORK/'parent956.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[test_ids,labels]
np.testing.assert_allclose(_parent.values,(.5*raw.rank(pct=True)+.5*ours.rank(pct=True)).values,rtol=1e-6,atol=1e-7)
_final=.5*raw.rank(pct=True)+.4*ours.rank(pct=True)+.1*_s6.rank(pct=True)
_final.index.name='StudyInstanceUID'
assert np.isfinite(_final.values).all()
_elapsed=_pub_time.monotonic()-_pub_started
assert _elapsed<8.5*3600
_final.to_csv(WORK/'submission.csv')
_files=['submission.csv','parent956.csv','raptor_raw.csv','ours_sub.csv','s6_member/s6_raw.csv','s6_member/s6_receipt.json']
_receipt=dict(status='PASS',weights=[.5,.4,.1],rows=len(test_ids),elapsed_seconds=_elapsed,decode_coverage_verified=False,source_sha256=SOURCE_SHA,hashes={n:hashlib.sha256((WORK/n).read_bytes()).hexdigest() for n in _files})
(WORK/'s6_fusion_receipt.json').write_text(json.dumps(_receipt,indent=2))
print('S6_FUSION_VERIFIED',_receipt)
'''
    code = code.replace('SOURCE_B64', repr(base64.b64encode(source).decode())).replace('ADAPTER_B64', repr(base64.b64encode(adapter).decode())).replace('SOURCE_SHA', repr(sha(source)))
    nb.cells.append(nbformat.v4.new_code_cell(code))
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs=[]; cell.execution_count=None
            ast.parse(TransformerManager().transform_cell(cell.source))
    nbformat.validate(nb)
    out=HERE/'candidate'; out.mkdir(exist_ok=True)
    nbformat.write(nb,out/'s6.ipynb')
    meta=json.loads((ROOT/'experiments/r71013/candidate/kernel-metadata.json').read_text())
    meta.update(id=REF,title='RSNA R7 S6 All Data',code_file='s6.ipynb')
    meta['dataset_sources'].append('nartaa/rsna-knee-s6fall-weights')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    record=dict(parent_submission=57144024,parent_sha256=sha(parent),source_sha256=sha((out/'s6.ipynb').read_bytes()),author_sha256=sha(source),adapter_sha256=sha(adapter),weights=[.5,.4,.1],status='BUILT_NOT_RUN')
    (HERE/'build.json').write_text(json.dumps(record,indent=2)); print(record)


def launch():
    sys.path.insert(0,str(ROOT/'experiments/convnext1005'))
    from run_scale_probe import prepare_dns
    from kaggle.api.kaggle_api_extended import KaggleApi
    assert not (HERE/'launch_attempt.json').exists(), 'Reconcile existing attempt first'
    record=json.loads((HERE/'build.json').read_text())
    assert sha((HERE/'candidate/s6.ipynb').read_bytes())==record['source_sha256']
    prepare_dns(); api=KaggleApi(); api.authenticate()
    assert not any(k.ref==REF for k in api.kernels_list(mine=True,search='rsna-r7-s6-all-data',page_size=100))
    (HERE/'launch_attempt.json').write_text(json.dumps(record,indent=2))
    result=api.kernels_push(str(HERE/'candidate'),acc='NvidiaTeslaT4').to_dict()
    (HERE/'launch.json').write_text(json.dumps(dict(build=record,response=result),indent=2)); print(result)
    assert not result.get('invalidDatasetSources'), result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','launch'])
    build() if parser.parse_args().action=='build' else launch()
