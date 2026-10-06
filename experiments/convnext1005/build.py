"""Build a fixed public three-fold ConvNeXt addition to the protected 0.943 parent."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'experiments/sprint0930/fourway-public'
UPSTREAM = HERE / 'upstream/public-reader.ipynb'
PARENT_SHA = 'd0165cf3d144b6be4e05ece77f80b233c27393cdbae67cb06c0d03b96f178505'
ASSET = 'goodpjw2008/rsna-knee-2-5d-convnext-reader'

def sha(data): return hashlib.sha256(data).hexdigest()

def build():
    upstream_raw = UPSTREAM.read_bytes()
    upstream = json.loads(upstream_raw)
    sources = {}
    for i, name in [(34,'preprocess.py'),(35,'knee.py'),(36,'infer.py')]:
        text=''.join(upstream['cells'][i]['source'])
        assert text.splitlines()[0].endswith('/'+name)
        sources[name]=text.split('\n',1)[1]
    # Preserve pixel operations; fail instead of silently skipping corrupt selected series.
    text=sources['infer.py']
    old='''            try:
                vol, _ = load_series(f"{self.root}/{st}/{cands[0]}")
            except Exception:
                continue
            n = len(vol)
            if n < 3:
                continue'''
    new='''            vol, info = load_series(f"{self.root}/{st}/{cands[0]}")
            if info.get("n_bad", 0):
                raise RuntimeError(f"DICOM decoding failed: {st}/{cands[0]}: {info}")
            n = len(vol)
            if n < 3:
                raise RuntimeError(f"Selected series has fewer than 3 valid slices: {st}/{cands[0]}")'''
    assert text.count(old)==1
    text=text.replace(old,new)
    text=text.replace('weights_only=False','weights_only=True')
    text=text.replace('m.load_state_dict(ck["model"])','m.load_state_dict(ck["model"], strict=True)')
    text=text.replace('''        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos)''','''        if not mask.any():
            raise RuntimeError(f"Study has no usable series: {st}")
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos)''')
    sources['infer.py']=text
    for name, text in sources.items():
        ast.parse(text)
        (HERE/name).write_text(text,encoding='utf-8')
    worker=(HERE/'worker_tail.py').read_text(encoding='utf-8')
    ast.parse(worker)
    startup='''import os as _c3_os, subprocess as _c3_sp, sys as _c3_sys, time as _c3_time
import threading as _c3_threading
from pathlib import Path as _C3Path
_c3_started = _c3_time.monotonic()
_c3_done = _c3_threading.Event()
def _c3_deadline():
    if not _c3_done.wait(8.5*3600):
        print('FATAL: total runtime exceeded 8.5h', flush=True)
        _c3_os._exit(124)
_c3_threading.Thread(target=_c3_deadline,daemon=True).start()
_c3_src=_C3Path('/kaggle/working/cnx3_source')
_c3_src.mkdir(exist_ok=True)
for _c3_name, _c3_code in __SOURCES__.items():
    (_c3_src/_c3_name).write_text(_c3_code,encoding='utf-8')
(_c3_src/'worker.py').write_text(__WORKER__,encoding='utf-8')
_c3_sp.run([_c3_sys.executable,'-u',str(_c3_src/'worker.py')],check=True,timeout=2*3600)
'''.replace('__SOURCES__',repr(sources)).replace('__WORKER__',repr(worker))
    finish=(HERE/'finish_tail.py').read_text(encoding='utf-8').replace('__PARENT_SHA__',PARENT_SHA).replace('__SOURCE_SHA__',sha(upstream_raw))
    raw=(PARENT/'sprint-fourway.ipynb').read_bytes()
    assert sha(raw)==PARENT_SHA
    nb=copy.deepcopy(json.loads(raw));nb['nbformat_minor']=5
    for c in nb['cells']:
        if c['cell_type']=='code': c.update(outputs=[],execution_count=None)
    for source, ident, where in [(startup,'cnx3-worker',0),(finish,'cnx3-overlay',len(nb['cells'])+1)]:
        ast.parse(source)
        nb['cells'].insert(where,dict(cell_type='code',id=ident,metadata={},source=source.splitlines(True),outputs=[],execution_count=None))
    for i,c in enumerate(nb['cells']): c.setdefault('id',f'cnx3-{i}')
    import nbformat
    nbformat.validate(nbformat.from_dict(nb))
    meta=json.loads((PARENT/'kernel-metadata.json').read_text())
    meta.update(id='easoncyy/rsna-sprint-three-fold-convnext-30',title='RSNA Sprint Three-fold ConvNeXt 30',code_file='cnx3fold30.ipynb')
    meta['dataset_sources'].append(ASSET)
    out=HERE/'cnx3fold30';out.mkdir(exist_ok=True)
    payload=json.dumps(nb,ensure_ascii=False,indent=1).encode('utf-8')
    (out/meta['code_file']).write_bytes(payload)
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    receipt=dict(status='BUILT_NOT_RUN',parent_sha256=PARENT_SHA,notebook_sha256=sha(payload),upstream_sha256=sha(upstream_raw),reader_weight=.3,folds=[0,1,2],new_dependencies=[ASSET],sources={name:sha(text.encode()) for name,text in sources.items()})
    (HERE/'build_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': build()
