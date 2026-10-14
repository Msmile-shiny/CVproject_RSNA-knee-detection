"""One fixed published three-model candidate, with no degraded submission."""
import argparse, ast, hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REF='easoncyy/rsna-r7-full-three-model'
sha=lambda b:hashlib.sha256(b).hexdigest()

def build():
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    assert not (HERE/'launch_attempt.json').exists()
    raw=next((HERE/'threeway').glob('*.ipynb')).read_bytes();nb=json.loads(raw)
    prior=json.loads((HERE.parent/'r71013/candidate/r7.ipynb').read_bytes())
    guard=''.join(prior['cells'][0]['source'])
    # Export the actual 224 predictions before ranking; reject silent all-neutral studies.
    cell=''.join(nb['cells'][9]['source'])
    marker='    for d in devs:\n        for m, _ in models[d]: del m'
    assert cell.count(marker)==1
    extra="""    _raw224 = arm_probs[0].astype(np.float32)
    assert np.isfinite(_raw224).all() and not (_raw224 == .5).all(axis=1).any(), '224 failed studies: stop'
    _rawframe = pd.DataFrame(_raw224, columns=LAB)
    _rawframe.insert(0, 'StudyInstanceUID', test_ids)
    _rawframe.to_csv('/kaggle/working/raptor224_raw.csv', index=False)
"""
    nb['cells'][9]['source']=cell.replace(marker,extra+marker)
    audit="""assert R224_OK and r224 is not None, 'Missing third member; refusing baseline-only output'
assert A224 == .35 and W_RAPTOR == .5 and W_OURS == .5
assert not rap_fail.any() and not our_fail.any(), 'Incomplete member coverage'
assert all(w == .5 for w in W_RAPTOR_C.values())
_r224raw=pd.read_csv(WORK/'raptor224_raw.csv',dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID').loc[test_ids,labels]
assert _r224raw.index.is_unique and np.isfinite(_r224raw.values).all()
assert ((_r224raw.values>=0)&(_r224raw.values<=1)).all()
assert not (_r224raw.values==.5).all(axis=1).any()
_expected=.325*raw.rank(pct=True)+.175*r224.rank(pct=True)+.5*ours.rank(pct=True)
np.testing.assert_allclose(sub[labels].values,_expected.values.astype(np.float32),rtol=1e-7,atol=1e-8)
_elapsed=_pub_time.monotonic()-_pub_started
assert 0<_elapsed<8.8*3600
_names=['submission.csv','raptor_raw.csv','raptor_sub.csv','raptor224_sub.csv','raptor224_raw.csv','ours_sub.csv','blend_receipt.json','submission_receipt.json']
_audit=dict(status='PASS',complete_three_members=True,elapsed_seconds=_elapsed,rows=len(test_ids),weights=[.325,.175,.5],decode_coverage_verified=False,hashes={n:_pub_hash.sha256((WORK/n).read_bytes()).hexdigest() for n in _names})
(WORK/'threeway_receipt.json').write_text(json.dumps(_audit,indent=2))
print('THREEWAY_VERIFIED',_audit)
"""
    nb['cells']=[nbformat.v4.new_code_cell(guard)]+nb['cells']+[nbformat.v4.new_code_cell(audit)]
    for i,c in enumerate(nb['cells']):
        c['id']=f'threeway-{i}'
        if c['cell_type']=='code':
            c.update(outputs=[],execution_count=None)
            ast.parse(TransformerManager().transform_cell(''.join(c['source'])))
    nb['nbformat_minor']=5;nbformat.validate(nbformat.from_dict(nb))
    out=HERE/'candidate';out.mkdir(exist_ok=True)
    payload=json.dumps(nb,ensure_ascii=False,indent=1).encode();(out/'threeway.ipynb').write_bytes(payload)
    meta=json.loads((HERE/'threeway/kernel-metadata.json').read_text());meta.pop('id_no',None)
    meta.update(id=REF,title='RSNA R7 Full Three Model',code_file='threeway.ipynb',is_private=True,enable_internet=False)
    assert all(meta['dataset_sources'])
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    record=dict(upstream=meta.get('title'),upstream_ref='varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956',upstream_sha256=sha(raw),source_sha256=sha(payload),protected_submission=57144024,weights=[.325,.175,.5],changes='224 raw export and strict complete-member checks; prediction formula unchanged')
    (HERE/'build.json').write_text(json.dumps(record,indent=2));print(record)

def launch():
    sys.path.insert(0,str(HERE.parent/'convnext1005'))
    from run_scale_probe import prepare_dns
    from kaggle.api.kaggle_api_extended import KaggleApi
    assert not (HERE/'launch_attempt.json').exists()
    built=json.loads((HERE/'build.json').read_text())
    assert sha((HERE/'candidate/threeway.ipynb').read_bytes())==built['source_sha256']
    prepare_dns();a=KaggleApi();a.authenticate()
    assert not any(k.ref==REF for k in a.kernels_list(mine=True,search='rsna-r7-full-three-model',page_size=100))
    (HERE/'launch_attempt.json').write_text(json.dumps(built,indent=2))
    r=a.kernels_push(str(HERE/'candidate'),acc='NvidiaTeslaT4').to_dict()
    (HERE/'launch.json').write_text(json.dumps(dict(build=built,response=r),indent=2));print(r)
    assert not r.get('invalidDatasetSources'),r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','launch']);a=p.parse_args()
    build() if a.action=='build' else launch()
