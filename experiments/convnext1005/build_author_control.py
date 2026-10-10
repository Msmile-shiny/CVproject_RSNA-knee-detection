"""Minimal author-v5 control: switch RUN_STACK only, plus read-only verification."""
import copy
import hashlib
import json
from pathlib import Path
import nbformat
from IPython.core.inputtransformer2 import TransformerManager
import ast

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    raw=(HERE/'upstream/author-v5.ipynb').read_bytes();original=json.loads(raw);nb=copy.deepcopy(original)
    switched=0
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        code=''.join(cell['source'])
        if '\nRUN_STACK = True\n' in code:
            code=code.replace('\nRUN_STACK = True\n','\nRUN_STACK = False\n');switched+=1
        cell.update(source=code,outputs=[],execution_count=None)
    assert switched==1
    before='''import hashlib as _audit_hash, json as _audit_json, time as _audit_time
from pathlib import Path as _AuditPath
_audit_started=_audit_time.monotonic()
_audit_expected={
'cnxt_v0_fold0.pt':'0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80',
'cnxt_v0_fold1.pt':'61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb',
'cnxt_v0_fold2.pt':'f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6'}
_audit_weights={}
for name,expected in _audit_expected.items():
    found=[p for pattern in ['*/'+name,'*/*/'+name,'*/*/*/'+name] for p in _AuditPath('/kaggle/input').glob(pattern)]
    assert found, name
    for path in found:
        with path.open('rb') as f: actual=_audit_hash.file_digest(f,'sha256').hexdigest()
        assert actual==expected,name
    _audit_weights[name]=expected
'''
    after='''import pandas as _audit_pd, numpy as _audit_np
assert RUN_STACK is False
assert len(_o_ck)==3
_audit_frame=_audit_pd.read_csv(_o_pub,dtype={'StudyInstanceUID':str},float_precision='round_trip')
_audit_sample=_audit_pd.read_csv(_o_data+'/sample_submission.csv',dtype={'StudyInstanceUID':str})
assert _audit_frame.columns.tolist()==_audit_sample.columns.tolist()
assert _audit_frame.StudyInstanceUID.tolist()==_audit_sample.StudyInstanceUID.tolist()
assert _audit_frame.StudyInstanceUID.is_unique and len(_audit_frame)>0
_audit_v=_audit_frame.iloc[:,1:].to_numpy()
assert _audit_v.shape==(len(_audit_frame),12) and _audit_np.isfinite(_audit_v).all() and (_audit_v>=0).all() and (_audit_v<=1).all()
_audit_receipt=dict(status='COMPLETE',parent_used=False,author_version=5,study_count=len(_audit_frame),model_count=len(_o_ck),
    checkpoint_hashes=_audit_weights,elapsed_seconds=_audit_time.monotonic()-_audit_started,
    submission_sha256=_audit_hash.sha256(_AuditPath(_o_pub).read_bytes()).hexdigest(),
    raw_sha256=_audit_hash.sha256(_AuditPath('/kaggle/working/_own.csv').read_bytes()).hexdigest(),
    source_hashes={p.name:_audit_hash.sha256(p.read_bytes()).hexdigest() for p in _AuditPath('/kaggle/working/own_src').glob('*.py')},
    decode_coverage_verified=False,missing_series_policy='unaltered upstream',score_verified=False)
_AuditPath('/kaggle/working/author_control_receipt.json').write_text(_audit_json.dumps(_audit_receipt,indent=2))
print('AUTHOR CONTROL COMPLETE',_audit_receipt)
'''
    nb['cells'].insert(0,nbformat.v4.new_code_cell(before))
    nb['cells'].append(nbformat.v4.new_code_cell(after))
    for i,cell in enumerate(nb['cells']):cell.setdefault('id',f'author-control-{i}')
    nb['nbformat_minor']=5
    transformer=TransformerManager()
    for cell in nb['cells']:
        if cell['cell_type']=='code':ast.parse(transformer.transform_cell(''.join(cell['source'])))
    nbformat.validate(nbformat.from_dict(nb))
    source_map={}
    for cell in original['cells']:
        text=''.join(cell['source'])
        if text.startswith('%%writefile /kaggle/working/own_src/'):
            name=text.splitlines()[0].split('/')[-1]
            # IPython writefile emits the supplied body with a final newline.
            body=text.split('\n',1)[1]
            if not body.endswith('\n'):body+='\n'
            source_map[name]=hashlib.sha256(body.encode()).hexdigest()
    folder=HERE/'author-control';folder.mkdir(exist_ok=True)
    payload=json.dumps(nb,ensure_ascii=False,indent=1).encode();(folder/'control.ipynb').write_bytes(payload)
    meta=json.loads((HERE/'upstream/author-v5-metadata.json').read_text());meta.pop('id_no',None)
    meta.update(id='easoncyy/rsna-cnx-author-control',title='RSNA CNX Author Control',code_file='control.ipynb',is_private=True)
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (HERE/'author_control_build.json').write_text(json.dumps(dict(status='BUILT_NOT_RUN',upstream_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(payload).hexdigest(),expected_sources=source_map,author_version=5,changed_author_cells=1),indent=2))
    print('Author cells preserved except RUN_STACK=False; independent input/output checks appended')
