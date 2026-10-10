"""Fixed 15% independent 224-crop checkpoint addition to scored 0.950 parent."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import nbformat
from IPython.core.inputtransformer2 import TransformerManager
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()

def build():
    parentraw=(HERE/'upstream/parent950.ipynb').read_bytes()
    assert sha(parentraw)=='16ce7c78f860b6d07331c8d39c6a65414cde200462a64c984b04bd8efc167289'
    parent=json.loads(parentraw)
    upstream=json.loads((HERE/'upstream/rsna-knee-0945-efficient-224crop.ipynb').read_bytes())
    guard=''.join(upstream['cells'][3]['source']);source=''.join(upstream['cells'][4]['source'])
    before=sha(source.encode())
    manifest=json.loads((HERE/'upstream/checkpoint-manifest.json').read_text())
    assert before==manifest['efficiency']['source_sha256'], 'Published source digest mismatch'
    # Reject study-level neutral fallbacks; preserve successful image computations.
    old='vol = np.zeros((MAXS, IMG, IMG), np.uint8); mask = np.zeros(MAXS, np.uint8)'
    assert source.count(old)==1;source=source.replace(old,'raise')
    old='np.full((N, len(LAB)), 0.5, np.float32)'
    assert source.count(old)==1;source=source.replace(old,'np.full((N, len(LAB)), np.nan, np.float32)')
    marker='    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe).'
    assert source.count(marker)==1
    source=source.replace(marker,"    assert done_n[0] == N and all(np.isfinite(p).all() for p in arm_probs), 'Incomplete 224 inference; no neutral fallback allowed'\n"+marker)
    script=guard+'\n'+source
    ast.parse(script)
    start='''import time as _s224_time, subprocess as _s224_sp, sys as _s224_sys
from pathlib import Path as _S224Path
_s224_started=_s224_time.monotonic()
_s224_script=__SCRIPT__
_S224Path('/kaggle/working/independent224.py').write_text(_s224_script,encoding='utf8')
_s224_sp.run([_s224_sys.executable,'/kaggle/working/independent224.py'],check=True,timeout=2*3600)
_S224Path('/kaggle/working/independent224.csv').write_bytes(_S224Path('/kaggle/working/submission.csv').read_bytes())
'''.replace('__SCRIPT__',repr(script))
    end='''import pandas as _s224_pd, numpy as _s224_np, json as _s224_json, hashlib as _s224_hash
_s224_root=_S224Path('/kaggle/working')
(_s224_root/'parent950.csv').write_bytes((_s224_root/'submission.csv').read_bytes())
_s224_a=_s224_pd.read_csv(_s224_root/'parent950.csv',dtype={'StudyInstanceUID':str},float_precision='round_trip')
_s224_b=_s224_pd.read_csv(_s224_root/'independent224.csv',dtype={'StudyInstanceUID':str},float_precision='round_trip')
assert _s224_a.columns.tolist()==_s224_b.columns.tolist()
assert _s224_a.StudyInstanceUID.is_unique and _s224_b.StudyInstanceUID.is_unique
assert len(_s224_a)>0 and set(_s224_a.StudyInstanceUID)==set(_s224_b.StudyInstanceUID)
_s224_b=_s224_b.set_index('StudyInstanceUID').loc[_s224_a.StudyInstanceUID].reset_index()
_s224_labels=_s224_a.columns.tolist()[1:]
assert len(_s224_labels)==12
for _s224_frame in [_s224_a,_s224_b]:
    _s224_v=_s224_frame[_s224_labels].to_numpy()
    assert _s224_np.isfinite(_s224_v).all() and (_s224_v>=0).all() and (_s224_v<=1).all()
_s224_final=_s224_a.copy()
_s224_final[_s224_labels]=(.85*_s224_a[_s224_labels].rank(pct=True)+.15*_s224_b[_s224_labels].rank(pct=True)).rank(pct=True)
_s224_final.to_csv(_s224_root/'submission.csv',index=False)
_s224_receipt=dict(status='COMPLETE',parent_submission=57012367,weight224=.15,study_count=len(_s224_a),
    elapsed_seconds=_s224_time.monotonic()-_s224_started,decode_coverage_verified=False,
    source224_sha256=_s224_hash.sha256((_s224_root/'independent224.py').read_bytes()).hexdigest(),
    hashes={n:_s224_hash.sha256((_s224_root/n).read_bytes()).hexdigest() for n in ['submission.csv','parent950.csv','independent224.csv','_sota_0949.csv','_own.csv']})
assert _s224_receipt['elapsed_seconds']<8.5*3600
(_s224_root/'swa224_receipt.json').write_text(_s224_json.dumps(_s224_receipt,indent=2))
print('SWA224 FIXED FUSION',_s224_receipt)
'''
    nb=copy.deepcopy(parent);nb['cells']=[nbformat.v4.new_code_cell(start)]+nb['cells']+[nbformat.v4.new_code_cell(end)]
    for i,c in enumerate(nb['cells']):
        c['id']=f'swa224-{i}'
        if c['cell_type']=='code':
            c.update(outputs=[],execution_count=None)
            ast.parse(TransformerManager().transform_cell(''.join(c['source'])))
    nb['nbformat_minor']=5;nbformat.validate(nbformat.from_dict(nb))
    assert [''.join(c['source']) for c in nb['cells'][1:-1]]==[''.join(c['source']) for c in parent['cells']]
    out=HERE/'candidate';out.mkdir(exist_ok=True)
    raw=json.dumps(nb,ensure_ascii=False,indent=1).encode();(out/'fusion.ipynb').write_bytes(raw)
    meta=json.loads((HERE/'upstream/parent-metadata.json').read_text());meta.pop('id_no',None)
    meta.update(id='easoncyy/rsna-swa224-fixed15',title='RSNA SWA224 Fixed15',code_file='fusion.ipynb',is_private=True)
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    result=dict(status='BUILT_NOT_RUN',source_sha256=sha(raw),parent_sha256=sha(parentraw),source224_sha256=sha(script.encode()),original224_sha256=before,weight224=.15,parent_code_unchanged=True)
    (HERE/'build.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':build()
