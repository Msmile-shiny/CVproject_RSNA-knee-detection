"""Preserve both successful notebook code sequences; fixed 30% rank blend."""
import copy
import hashlib
import json
from pathlib import Path
import nbformat
from IPython.core.inputtransformer2 import TransformerManager
import ast

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'experiments/sprint0930/fourway-public'

def build():
    sha = lambda b: hashlib.sha256(b).hexdigest()
    author_raw = (HERE/'author-control/control.ipynb').read_bytes()
    parent_raw = (PARENT/'sprint-fourway.ipynb').read_bytes()
    assert sha(author_raw) == json.loads((HERE/'author_control_build.json').read_text())['source_sha256']
    assert sha(parent_raw) == 'd0165cf3d144b6be4e05ece77f80b233c27393cdbae67cb06c0d03b96f178505'
    scored = json.loads((HERE/'author_control_scoring.json').read_text())
    assert scored['outcome'] == 'SCORED' and scored['submission'] == 56910183
    author, parent = json.loads(author_raw), json.loads(parent_raw)
    begin = nbformat.v4.new_code_cell('import time as _af_time\n_af_started = _af_time.monotonic()')
    save = nbformat.v4.new_code_cell("""from pathlib import Path as _AFPath
assert _audit_receipt['status']=='COMPLETE' and not _audit_receipt['parent_used']
_AFPath('/kaggle/working/author929.csv').write_bytes(_AFPath('/kaggle/working/submission.csv').read_bytes())
""")
    finish = nbformat.v4.new_code_cell("""import pandas as _af_pd, numpy as _af_np, hashlib as _af_hash, json as _af_json
assert _sprint_receipt['ready_for_scoring'], 'Parent did not pass its own gates'
_af_root = _AFPath('/kaggle/working')
(_af_root/'parent943.csv').write_bytes((_af_root/'submission.csv').read_bytes())
_af_parent = _af_pd.read_csv(_af_root/'parent943.csv',dtype={'StudyInstanceUID':str},float_precision='round_trip')
_af_reader = _af_pd.read_csv(_af_root/'author929.csv',dtype={'StudyInstanceUID':str},float_precision='round_trip')
assert _af_parent.columns.tolist()==_af_reader.columns.tolist()
assert _af_parent.StudyInstanceUID.is_unique and _af_reader.StudyInstanceUID.is_unique
assert len(_af_parent)>0 and set(_af_parent.StudyInstanceUID)==set(_af_reader.StudyInstanceUID)
_af_reader = _af_reader.set_index('StudyInstanceUID').loc[_af_parent.StudyInstanceUID].reset_index()
_af_labels = _af_parent.columns.tolist()[1:]
assert len(_af_labels)==12
for _af_frame in [_af_parent,_af_reader]:
    _af_v=_af_frame[_af_labels].to_numpy()
    assert _af_np.isfinite(_af_v).all() and (_af_v>=0).all() and (_af_v<=1).all()
    if len(_af_frame)>10: assert (_af_frame[_af_labels].nunique()>1).all()
_af_final=_af_parent.copy()
_af_final[_af_labels]=(0.7*_af_parent[_af_labels].rank(method='average',pct=True)+0.3*_af_reader[_af_labels].rank(method='average',pct=True)).rank(method='average',pct=True)
_af_final.to_csv(_af_root/'submission.csv',index=False)
_af_elapsed=_af_time.monotonic()-_af_started
assert _af_elapsed < 8.5*3600, 'Cumulative runtime budget exceeded'
_af_receipt=dict(status='COMPLETE',reader_weight=0.3,parent_submission=56700487,reader_submission=56910183,
    study_count=len(_af_final),elapsed_seconds=_af_elapsed,decode_coverage_verified=False,
    hashes={n:_af_hash.sha256((_af_root/n).read_bytes()).hexdigest() for n in ['submission.csv','parent943.csv','author929.csv','author_control_receipt.json']})
(_af_root/'author_fusion_receipt.json').write_text(_af_json.dumps(_af_receipt,indent=2))
print('AUTHOR FUSION COMPLETE',_af_receipt)
""")
    nb = copy.deepcopy(parent)
    nb['cells'] = [begin]+copy.deepcopy(author['cells'])+[save]+copy.deepcopy(parent['cells'])+[finish]
    transform = TransformerManager()
    for i,c in enumerate(nb['cells']):
        c['id']=f'author-fusion-{i}'
        if c['cell_type']=='code':
            c.update(outputs=[],execution_count=None)
            ast.parse(transform.transform_cell(''.join(c['source'])))
    codes=lambda cells:[''.join(c['source']) for c in cells if c['cell_type']=='code']
    assert codes(nb['cells'][1:1+len(author['cells'])])==codes(author['cells'])
    assert codes(nb['cells'][len(author['cells'])+2:-1])==codes(parent['cells'])
    nb['nbformat_minor']=5
    nbformat.validate(nbformat.from_dict(nb))
    meta=json.loads((PARENT/'kernel-metadata.json').read_text())
    author_meta=json.loads((HERE/'author-control/kernel-metadata.json').read_text())
    for key in ['dataset_sources','model_sources','kernel_sources','competition_sources']:
        meta[key]=list(dict.fromkeys(meta.get(key,[])+author_meta.get(key,[])))
    meta.update(id='easoncyy/rsna-cnx-author-fusion30',title='RSNA Author Reader Fixed 30',code_file='fusion.ipynb',is_private=True)
    folder=HERE/'author-fusion30';folder.mkdir(exist_ok=True)
    raw=json.dumps(nb,ensure_ascii=False,indent=1).encode()
    (folder/'fusion.ipynb').write_bytes(raw)
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    receipt=dict(status='BUILT_NOT_RUN',source_sha256=sha(raw),parent_sha256=sha(parent_raw),reader_sha256=sha(author_raw),
                 original_code_sequences_preserved=True,reader_weight=0.3,decode_coverage_verified=False)
    (HERE/'author_fusion_build.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))

if __name__=='__main__': build()
