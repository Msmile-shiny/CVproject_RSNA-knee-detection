"""Build a 24-epoch mean-pooling control with audited report evidence."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'stage6-label-assets/stage6c-upload'


def build():
    reference=ROOT/'experiments/stage6/reference_v3/run_receipt.json'
    base_receipt=json.loads(reference.read_text())
    label_receipt_path=BASE/'stage6c_label_receipt.json'
    label_receipt=json.loads(label_receipt_path.read_text())
    assert base_receipt['status']=='PILOT_COMPLETE' and base_receipt['completed_epochs']==12
    assert base_receipt['labels_sha256']==label_receipt['source_v5_sha256']
    assert hashlib.sha256((BASE/'v5_labels_stage6c.csv').read_bytes()).hexdigest()==label_receipt['output_sha256']
    original=json.loads((ROOT/'experiments/stage6/notebook/stage6a.ipynb').read_text(encoding='utf-8'))
    sources=[''.join(c['source']) for c in original['cells']]
    assert hashlib.sha256(''.join(sources[2:]).encode()).hexdigest()==base_receipt['config']['implementation_sha256']
    runtime=sources[-1]
    replacements={
        "if not path.is_file() and S6['resume']:":"if not path.is_file() and S6['cache_input']:",
        "path=Path(S6['resume'])/'pixel_cache'/filename":"path=Path(S6['cache_input'])/'pixel_cache'/filename",
        "if resume and (resume/'pixel_cache'/filename).is_file():":"if (Path(S6['cache_input'])/'pixel_cache'/filename).is_file():",
        "with np.load(resume/'pixel_cache'/filename) as z:":"with np.load(Path(S6['cache_input'])/'pixel_cache'/filename) as z:",
        "asset(S6['labels'], 'v5_labels.csv')":"asset(S6['labels'], 'v5_labels_stage6c.csv')",
        base_receipt['labels_sha256']:label_receipt['output_sha256'],
    }
    for old,new in replacements.items():
        assert runtime.count(old)==1,old
        runtime=runtime.replace(old,new)
    sources[-1]=runtime
    implementation=hashlib.sha256(''.join(sources[2:]).encode()).hexdigest()
    config=dict(base_receipt['config'],epochs=24,minutes=240,resume='',cache_input='',
                implementation_sha256=implementation)
    sources[0]='# Stage 6C: Report evidence supervision control\nOnly target probabilities differ from the 24-epoch mean-pooling reference. Gold development only.'
    sources[1]='S6 = '+repr(config)+f'''
import hashlib,json
from pathlib import Path
ref_candidates=[p for p in Path('/kaggle/input').rglob('run_receipt.json')
                if hashlib.sha256(p.read_bytes()).hexdigest()=={hashlib.sha256(reference.read_bytes()).hexdigest()!r}]
label_candidates=[p for p in Path('/kaggle/input').rglob('stage6c_label_receipt.json')
                  if hashlib.sha256(p.read_bytes()).hexdigest()=={hashlib.sha256(label_receipt_path.read_bytes()).hexdigest()!r}]
assert len(ref_candidates)==1 and len(label_candidates)==1, 'Mount exact Stage6A cache and Stage6C label dataset'
S6['cache_input']=str(ref_candidates[0].parent)
S6['labels']=str(label_candidates[0].parent/'v5_labels_stage6c.csv')
assert (ref_candidates[0].parent/'pixel_cache').is_dir()
assert hashlib.sha256(Path(S6['labels']).read_bytes()).hexdigest()=={label_receipt['output_sha256']!r}
assert not S6['resume'] and S6['pooling']=='mean'
print('Stage6C source-locked labels and cache verified')
'''
    for cell,source in zip(original['cells'],sources):
        cell['source']=source.splitlines(True)
        if cell['cell_type']=='code':ast.parse(source);cell['outputs']=[];cell['execution_count']=None
    dest=ROOT/'experiments/stage6/supervision';dest.mkdir(parents=True,exist_ok=True)
    (dest/'supervision.ipynb').write_text(json.dumps(original,indent=1),encoding='utf-8')
    meta=json.loads((ROOT/'experiments/stage6/notebook/kernel-metadata.json').read_text())
    meta.update(id='easoncyy/rsna-stage6c-report-supervision',title='RSNA Stage6C Report Supervision',code_file='supervision.ipynb')
    meta['dataset_sources']=['easoncyy/rsna-knee-stage6c-labels']
    meta['kernel_sources'].append('easoncyy/rsna-stage6a-resnet34-reference')
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(dest)


if __name__=='__main__':build()
