"""Public four-reader base + a preregistered 10% public ConvNeXt member."""
import ast
import hashlib
import json
from pathlib import Path
from build_public943 import build as build_parent

ROOT = Path(__file__).resolve().parent

PRE = r'''# Run the independent member FIRST in a child process; free its GPU before the parent.
import os as _cx_os, sys as _cx_sys, json as _cx_json, hashlib as _cx_hash, subprocess as _cx_sp
from pathlib import Path as _CxPath
def _cx_sha(path):
    h = _cx_hash.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()
def _cx_find(pattern, digest):
    for p in sorted(_CxPath('/kaggle/input').rglob(pattern)):
        if p.is_file() and _cx_sha(p) == digest: return p
    raise FileNotFoundError(f'Cannot find verified asset: {pattern}')
_cx_manifest_path = _cx_find('btkr_cnx_montage448_manifest.json', '3fa478207ab2e20d3c51999ce307aae78e43c4f34760c0e1ef8fe4ccb513e9f2')
_cx_art = _cx_manifest_path.parent
_cx_manifest = _cx_json.loads(_cx_manifest_path.read_text())
for _cx_name, _cx_info in _cx_manifest['runtime_files'].items():
    assert _cx_sha(_cx_art/_cx_name) == _cx_info['sha256'], _cx_name
_cx_checkpoint = _cx_manifest['checkpoint']
assert _cx_sha(_cx_art/_cx_checkpoint['file']) == _cx_checkpoint['sha256']
_cx_timm = _cx_find('timm-1.0.22-py3-none-any.whl', '888981753e65cbaacfc07494370138b1700a27b1f0af587f4f9b47bc024161d0')
_cx_cv = _cx_find('opencv_python_headless-4.12.0.88-*.whl', '236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9')
_cx_env = _CxPath('/kaggle/working/_sprint_cnx_env')
_cx_sp.run([_cx_sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--quiet', '--target', str(_cx_env), str(_cx_timm), str(_cx_cv)], check=True)
_cx_program = '\n'.join([
    'import sys',
    f'sys.path[:0] = {[str(_cx_art),str(_cx_env)]!r}',
    'from pathlib import Path',
    'import torch, cv2, timm',
    "assert torch.cuda.is_available() and cv2.__version__ == '4.12.0' and timm.__version__ == '1.0.22'",
    'import btkr_cnx_montage448_optional as rt',
    f"rt.run(Path({str(_cx_art/_cx_checkpoint['file'])!r}), Path('/kaggle/working/cnx_member.csv'), expected_checkpoint_sha256={_cx_checkpoint['sha256']!r}, workers=4, micro=4)",
])
_cx_child_env = dict(_cx_os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
_cx_sp.run([_cx_sys.executable, '-c', _cx_program], env=_cx_child_env, check=True)
print('ConvNeXt member complete; GPU child exited before public base starts', flush=True)
'''

POST = r'''import shutil as _cnx_shutil
from pathlib import Path as _CnxPath
import json as _cnx_json, hashlib as _cnx_hash, os as _cnx_os
_cnx_root = _CnxPath('/kaggle/working')
_cnx_base_receipt = _cnx_json.loads((_cnx_root/'sprint_fourway_receipt.json').read_text())
if not _cnx_base_receipt['all_four_readers_present']:
    raise RuntimeError('Do not evaluate CNX against a degraded public base')
_cnx_parent = pd.read_csv(_cnx_root/'submission.csv', dtype={'StudyInstanceUID': str})
_cnx_member = pd.read_csv(_cnx_root/'cnx_member.csv', dtype={'StudyInstanceUID': str})
_cnx_blend = rank_blend(_cnx_parent, _cnx_member, weight=0.10)
_cnx_shutil.copyfile(_cnx_root/'submission.csv', _cnx_root/'submission_fourway_parent.csv')
_cnx_tmp = _cnx_root/'submission_cnx10.tmp'
_cnx_blend.to_csv(_cnx_tmp, index=False)
_cnx_os.replace(_cnx_tmp, _cnx_root/'submission.csv')
_cnx_receipt = {
    'status': 'COMPLETE', 'parent': _cnx_base_receipt,
    'member': 'mattiaangeli/rsna-knee-cnx-m448-f0-public',
    'member_checkpoint_sha256': _cx_checkpoint['sha256'],
    'member_weight': 0.10, 'weight_selection': 'fixed before any scoring; no per-class search',
    'own_public_score': None, 'studies': len(_cnx_blend),
    'submission_sha256': _cnx_hash.sha256((_cnx_root/'submission.csv').read_bytes()).hexdigest(),
}
(_cnx_root/'sprint_cnx10_receipt.json').write_text(_cnx_json.dumps(_cnx_receipt, indent=2))
print('SPRINT CNX10 COMPLETE', _cnx_receipt)
'''


def cell(source):
    ast.parse(source)
    return {'cell_type':'code', 'metadata':{}, 'source':source.splitlines(True), 'outputs':[], 'execution_count':None}


def build():
    build_parent()
    base = ROOT/'fourway-public'
    n = json.loads((base/'sprint-fourway.ipynb').read_text(encoding='utf-8'))
    n['cells'].insert(1, cell(PRE))
    core = (ROOT/'rank_blend.py').read_text(encoding='utf-8')
    n['cells'].append(cell(core+'\n'+POST))
    meta = json.loads((base/'kernel-metadata.json').read_text(encoding='utf-8'))
    meta.update(id='easoncyy/rsna-sprint-cnx10', title='RSNA Sprint CNX10', code_file='sprint-cnx10.ipynb')
    meta['dataset_sources'].append('mattiaangeli/rsna-knee-cnx-m448-f0-public')
    out = ROOT/'cnx10'; out.mkdir(exist_ok=True)
    (out/meta['code_file']).write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Built CNX10: parent unchanged, 10% fixed global member weight, no network inference')


if __name__ == '__main__':
    build()
