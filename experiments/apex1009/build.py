import ast
import copy
import hashlib
import json
from pathlib import Path
import nbformat
from IPython.core.inputtransformer2 import TransformerManager

HERE=Path(__file__).resolve().parent
def reader_source():
    nb=json.loads(next((HERE/'upstream').glob('*.ipynb')).read_bytes())
    src=''.join(nb['cells'][8]['source']).split('\n',1)[1]
    original=src
    start=src.index('    dl0 = ');end=src.index('        x0, mask0, pos0 =',start)
    src=src[:start]+'''    dl = torch.utils.data.DataLoader(PairedDicomStudyDataset(studies, table, root, k), batch_size=bs,
                                     num_workers=workers, pin_memory=True)
    out = []
    for (x0, mask0, pos0), (x1, mask1, pos1) in dl:
'''+src[end:]
    src=src.replace('\ndef load_models(', '\n'+(HERE/'paired_dataset.py').read_text(encoding='utf8')+'\n\ndef load_models(',1)
    ast.parse(src)
    return original,src

def build():
    path=next((HERE/'upstream').glob('*.ipynb'));raw=path.read_bytes();nb=json.loads(raw)
    original,src=reader_source()
    nb['cells'][8]['source']='%%writefile /kaggle/working/own_src/infer.py\n'+src
    run=''.join(nb['cells'][9]['source'])
    # A file restored by a catch block must not masquerade as successful full fusion.
    run="_apex_full_success = False\n"+run
    marker='        tier3_df.to_csv(_o_pub, index=False)'
    assert run.count(marker)==1
    run=run.replace(marker,marker+'\n        _apex_full_success = True')
    nb['cells'][9]['source']=run
    control=json.loads((HERE.parent/'convnext1005/author-control/control.ipynb').read_bytes())
    nb['cells'].insert(0,copy.deepcopy(control['cells'][0]))
    audit='''assert _apex_full_success, 'Full Apex failed; refusing silent fallback submission'
assert len(_o_ck)==3, 'Unexpected reader checkpoint set'
import hashlib as _h, json as _j
from pathlib import Path as _P
sample=pd.read_csv(_o_data+'/sample_submission.csv',dtype={'StudyInstanceUID':str})
assert sub.columns.tolist()==sample.columns.tolist()
assert sub.StudyInstanceUID.is_unique and sub.StudyInstanceUID.tolist()==sample.StudyInstanceUID.tolist()
receipt=dict(status='COMPLETE',full_fusion=True,reader_models=3,decode_coverage_verified=False,
    elapsed_seconds=_audit_time.monotonic()-_audit_started,study_count=len(sub),reader_weights=_audit_weights,
    swa_sha256=digest,reader_source_sha256=_h.sha256(_P('/kaggle/working/own_src/infer.py').read_bytes()).hexdigest(),
    routing=TARGET_WEIGHTS_V5,hashes={n:_h.sha256(_P('/kaggle/working',n).read_bytes()).hexdigest() for n in ['submission.csv','_sota_0949.csv','_own.csv']})
assert receipt['elapsed_seconds']<8.5*3600
_P('/kaggle/working/apex_receipt.json').write_text(_j.dumps(receipt,indent=2))
print('APEX AUDIT',receipt)
'''
    nb['cells'].append(nbformat.v4.new_code_cell(audit))
    transform=TransformerManager()
    for i,c in enumerate(nb['cells']):
        c['id']=f'apex-{i}'
        if c['cell_type']=='code':
            c.update(outputs=[],execution_count=None)
            ast.parse(transform.transform_cell(''.join(c['source'])))
    nb['nbformat_minor']=5;nbformat.validate(nbformat.from_dict(nb))
    folder=HERE/'candidate';folder.mkdir(exist_ok=True)
    payload=json.dumps(nb,ensure_ascii=False,indent=1).encode()
    (folder/'apex.ipynb').write_bytes(payload)
    meta=json.loads((HERE/'upstream/kernel-metadata.json').read_text());meta.pop('id_no',None)
    meta.update(id='easoncyy/rsna-apex-paired-reader',title='RSNA Apex Paired Reader',code_file='apex.ipynb',is_private=True)
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    sha=lambda b:hashlib.sha256(b).hexdigest()
    receipt=dict(status='BUILT_NOT_RUN',upstream_version=3,upstream_sha256=sha(raw),source_sha256=sha(payload),reader_sha256=sha((src.rstrip('\n')+'\n').encode()),prediction_recipe_changed=False,decode_coverage_verified=False)
    (HERE/'build.json').write_text(json.dumps(receipt,indent=2));print(receipt)

if __name__=='__main__':build()
