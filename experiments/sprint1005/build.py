"""Build only. Exact parent code plus isolated specialist and fixed rank overlay."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'experiments/sprint0930/fourway-public'
PARENT_SHA = 'd0165cf3d144b6be4e05ece77f80b233c27393cdbae67cb06c0d03b96f178505'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def code_cell(source, name):
    ast.parse(source)
    return dict(cell_type='code', id=name, metadata={}, source=source.splitlines(True), outputs=[], execution_count=None)


def build():
    raw = (PARENT / 'sprint-fourway.ipynb').read_bytes()
    assert sha(raw) == PARENT_SHA, 'Protected parent changed'
    parent = json.loads(raw)
    author = (HERE / 'author_preprocess_source.py').read_text(encoding='utf-8')
    assert author.endswith('run_dinov2()\n')
    definitions = author[:-len('run_dinov2()\n')]
    manifest = json.loads((ROOT / 'experiments/sprint1002/asset_manifests/dino/manifest.json').read_text(encoding='utf-8'))
    groups = {m['pixel_group'] for m in manifest['members']}
    assert len(groups) == 1
    cfg = json.loads(groups.pop())
    tail = (HERE / 'specialist_tail.py').read_text(encoding='utf-8').replace('__PIXEL_CONFIG__', repr(cfg)).replace('__SOURCE_SHA__', sha(author.encode()))
    script = definitions + '\n' + tail
    ast.parse(script)
    startup = '''import time as _m10_time, threading as _m10_threading, os as _m10_os
import subprocess as _m10_subprocess, sys as _m10_sys
from pathlib import Path as _M10Path
_m10_started = _m10_time.monotonic()
_m10_done = _m10_threading.Event()
def _m10_deadline():
    if not _m10_done.wait(8.5 * 3600):
        print('FATAL: 8.5 hour total runtime deadline exceeded', flush=True)
        _m10_os._exit(124)
_m10_threading.Thread(target=_m10_deadline, daemon=True).start()
_m10_script = _M10Path('/kaggle/working/meniscus10_worker.py')
_m10_script.write_text(__SCRIPT__, encoding='utf-8')
_m10_subprocess.run([_m10_sys.executable, '-u', str(_m10_script)], check=True, timeout=2*3600)
'''.replace('__SCRIPT__', repr(script))
    overlay_source = (HERE / 'overlay.py').read_text(encoding='utf-8')
    finish = '''
import hashlib as _m10_hash, json as _m10_json
_m10_root = _M10Path('/kaggle/working')
assert _sprint_receipt['ready_for_scoring'], 'Parent failed its own readiness gate'
_m10_parent = _m10_root / 'parent943_replay.csv'
assert not _m10_parent.exists()
_m10_parent.write_bytes((_m10_root / 'submission.csv').read_bytes())
_m10_namespace = {}
exec(__OVERLAY__, _m10_namespace)
_m10_output = _m10_namespace['overlay'](_m10_parent.read_text(), (_m10_root / 'meniscus10/public0033_bag_raw.csv').read_text())
(_m10_root / 'submission.csv').write_text(_m10_output, encoding='utf-8')
_m10_files = ['submission.csv', 'parent943_replay.csv', 'meniscus10/public0033_bag_raw.csv', 'meniscus10/public0033_cached_inference_receipt.json', 'meniscus10/specialist_receipt.json']
_m10_receipt = dict(status='COMPLETE', ready_for_scoring=True, parent_submission=56700487,
    parent_public_score=.943, parent_sha256='__PARENT_SHA__', specialist_script_sha256='__SCRIPT_SHA__',
    overlay_sha256='__OVERLAY_SHA__', changed_labels=['Medial Meniscus', 'Lateral Meniscus'],
    specialist_weight=.1, study_count=len(_RSNA_TEST_IDS), elapsed_seconds=_m10_time.monotonic()-_m10_started,
    hashes={f:_m10_hash.sha256((_m10_root/f).read_bytes()).hexdigest() for f in _m10_files},
    score=None, diagnostics_are_not_validation=True)
assert _m10_receipt['elapsed_seconds'] < 8.5*3600
(_m10_root / 'meniscus10_receipt.json').write_text(_m10_json.dumps(_m10_receipt, indent=2))
_m10_done.set()
print('MENISCUS10 COMPLETE', _m10_receipt)
'''.replace('__OVERLAY__', repr(overlay_source)).replace('__PARENT_SHA__', PARENT_SHA).replace('__SCRIPT_SHA__', sha(script.encode())).replace('__OVERLAY_SHA__', sha(overlay_source.encode()))
    notebook = copy.deepcopy(parent)
    notebook['nbformat_minor'] = 5
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell.update(outputs=[], execution_count=None)
    notebook['cells'].insert(0, code_cell(startup, 'meniscus10-isolated-specialist'))
    notebook['cells'].append(code_cell(finish, 'meniscus10-fixed-overlay'))
    for index, cell in enumerate(notebook['cells']):
        cell.setdefault('id', f'm10-{index}')
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    meta = json.loads((PARENT / 'kernel-metadata.json').read_text(encoding='utf-8'))
    meta.update(id='easoncyy/rsna-sprint-meniscus10', title='RSNA Sprint Meniscus10', code_file='meniscus10.ipynb')
    asset = 'renta0426/rsna-knee-public0033-meniscus-bag-v1'
    assert asset not in meta['dataset_sources']
    meta['dataset_sources'].append(asset)
    assert meta['enable_gpu'] and not meta['enable_internet']
    out = HERE / 'meniscus10'
    out.mkdir(exist_ok=True)
    payload = json.dumps(notebook, ensure_ascii=False, indent=1).encode('utf-8')
    (out / 'meniscus10.ipynb').write_bytes(payload)
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    receipt = dict(status='BUILT_NOT_RUN', parent_sha256=PARENT_SHA, notebook_sha256=sha(payload),
                   specialist_script_sha256=sha(script.encode()), overlay_sha256=sha(overlay_source.encode()),
                   author_source_sha256=sha(author.encode()), changed_labels=['Medial Meniscus', 'Lateral Meniscus'],
                   specialist_weight=.1, new_dependencies=[asset], full_batch_gpu_memory_verified=False)
    (HERE / 'build_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    build()
