"""Pinned R6 interpretation of the newly published recipe, strict complete members."""
import ast,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REF='easoncyy/rsna-r6-published-v2-strict'
sha=lambda b:hashlib.sha256(b).hexdigest()

def build():
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    assert not (HERE/'launch_attempt.json').exists()
    meta=json.loads((HERE/'new/kernel-metadata.json').read_text())
    raw=(HERE/'new'/meta['code_file']).read_bytes()
    nb=nbformat.reads(raw.decode(),as_version=4)
    guard='''import time, hashlib, json
from pathlib import Path
_strict_start=time.monotonic()
_strict_paths=list(Path('/kaggle/input').rglob('rsna-knee-r6-frozen-fe055406-w1/manifest.json'))
assert len(_strict_paths)==1, _strict_paths
assert hashlib.sha256(_strict_paths[0].read_bytes()).hexdigest()=='6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648'
_strict_manifest=json.loads(_strict_paths[0].read_text())
assert _strict_manifest['bundle_id']=='fe055406fad6a12f'
for _name,_digest in _strict_manifest['sha256'].items():
    assert hashlib.sha256((_strict_paths[0].parent/_name).read_bytes()).hexdigest()==_digest, _name
'''
    s=nb.cells[6].source
    assert s.count("C_SLUG = 'rsna-knee-r6-frozen-fe055406'")==1
    nb.cells[6].source=s.replace("C_SLUG = 'rsna-knee-r6-frozen-fe055406'","C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'")
    s=nb.cells[9].source
    marker='        _o_final.to_csv(_o_pub, index=False)'
    assert s.count(marker)==1
    s=s.replace(marker,marker+'\n        _strict_reader_success=True')
    s='_strict_reader_success=False\n'+s
    nb.cells[9].source=s
    audit='''assert _strict_reader_success, 'Reader failed or skipped; refuse degraded candidate'
assert not r384_fail.any() and not r224_fail.any() and not our_fail.any()
assert A224==.35 and FLIP_TTA and len(_o_ck)==3 and OWN_MODE=='end'
assert W_RAPTOR_C=={'Medial OA':.5,'Lateral OA':.15,'PF OA':.65,'Synovitis':.85,"Baker's":.30}
assert manifest['bundle_id']=='fe055406fad6a12f'
_strict_expected=pd.DataFrame(index=raw.index)
for _c in labels:
    _w=W_RAPTOR_C.get(_c,.5)
    _strict_expected[_c]=_w*(.65*raw[_c].rank(pct=True)+.35*raw224[_c].rank(pct=True))+(1-_w)*ours[_c].rank(pct=True)
_strict_parent=pd.read_csv(WORK/'_public_stack.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[test_ids,labels]
np.testing.assert_allclose(_strict_parent.values,_strict_expected.values.astype(np.float32),rtol=1e-6,atol=1e-7)
_strict_own=pd.read_csv(WORK/'_own.csv',dtype={'StudyInstanceUID':str})
assert _strict_own.StudyInstanceUID.is_unique and set(_strict_own.StudyInstanceUID)==set(test_ids)
_strict_own=_strict_own.set_index('StudyInstanceUID').loc[test_ids,labels]
assert np.isfinite(_strict_own.values).all() and ((_strict_own.values>=0)&(_strict_own.values<=1)).all()
_strict_expected=_strict_parent.rank(pct=True)
for _c in labels:
    _w=OWN_W_PER_TARGET[_c]
    _strict_expected[_c]=(1-_w)*_strict_expected[_c]+_w*_strict_own[_c].rank(pct=True)
_strict_expected=_strict_expected.rank(pct=True)
_strict_final=pd.read_csv(WORK/'submission.csv',dtype={'StudyInstanceUID':str})
assert _strict_final.StudyInstanceUID.is_unique and _strict_final.StudyInstanceUID.tolist()==test_ids
np.testing.assert_allclose(_strict_final[labels].values,_strict_expected.values,rtol=1e-7,atol=1e-8)
assert time.monotonic()-_strict_start<8.8*3600
_strict_files=['submission.csv','_public_stack.csv','_own.csv','raptor_raw.csv','raptor224_raw.csv','ours_sub.csv','blend_receipt.json']
_strict_receipt=dict(status='PASS',rows=len(test_ids),elapsed_seconds=time.monotonic()-_strict_start,bundle=manifest['bundle_id'],raptor_weights=W_RAPTOR_C,reader_weights=OWN_W_PER_TARGET,decode_coverage_verified=False,hashes={n:hashlib.sha256((WORK/n).read_bytes()).hexdigest() for n in _strict_files},reader_checkpoints={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in _o_ck})
(WORK/'strict_receipt.json').write_text(json.dumps(_strict_receipt,indent=2))
print('STRICT_COMPLETE',_strict_receipt)
'''
    nb.cells=[nbformat.v4.new_code_cell(guard)]+nb.cells+[nbformat.v4.new_code_cell(audit)]
    for c in nb.cells:
        if c.cell_type=='code':
            c.outputs=[];c.execution_count=None
            ast.parse(TransformerManager().transform_cell(c.source))
    nb.nbformat_minor=5
    for i,c in enumerate(nb.cells):c['id']=f'strict-{i}'
    nbformat.validate(nb)
    out=HERE/'candidate';out.mkdir(exist_ok=True)
    nbformat.write(nb,out/'candidate.ipynb')
    meta.pop('id_no',None)
    meta.update(id=REF,title='RSNA R6 Published V2 Strict',code_file='candidate.ipynb',is_private=True,kernel_sources=[])
    meta['dataset_sources']=[x.replace('heliosli/rsna-knee-r6-frozen-fe055406','heliosli/rsna-knee-r6-frozen-fe055406-w1') for x in meta['dataset_sources']]
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    record=dict(status='BUILT_NOT_RUN',upstream_sha256=sha(raw),source_sha256=sha((out/'candidate.ipynb').read_bytes()),protected_submission=57261794,protected_score=.958,changes=['Pin actual R6 w1 assets; upstream mutable dataset may contain R7','Require reader and all base members; no fallback accepted','Audit fixed formula; remove unused kernel-output dependency'])
    (HERE/'build.json').write_text(json.dumps(record,indent=2));print(record)

def launch():
    sys.path.insert(0,str(HERE.parent/'convnext1005'))
    from run_scale_probe import prepare_dns
    from kaggle.api.kaggle_api_extended import KaggleApi
    assert not (HERE/'launch_attempt.json').exists()
    record=json.loads((HERE/'build.json').read_text())
    assert sha((HERE/'candidate/candidate.ipynb').read_bytes())==record['source_sha256']
    prepare_dns();a=KaggleApi();a.authenticate()
    assert not any(k.ref==REF for k in a.kernels_list(mine=True,search='rsna-r6-published-v2-strict',page_size=100))
    (HERE/'launch_attempt.json').write_text(json.dumps(record,indent=2))
    r=a.kernels_push(str(HERE/'candidate'),acc='NvidiaTeslaT4').to_dict()
    (HERE/'launch.json').write_text(json.dumps(dict(build=record,response=r),indent=2));print(r)
    assert not r.get('invalidDatasetSources'),r

if __name__=='__main__':
    build() if sys.argv[1]=='build' else launch()
