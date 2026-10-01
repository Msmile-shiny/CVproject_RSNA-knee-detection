"""Build an audited copy of upstream Fast 2xT4 v5; preserve inference arithmetic."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'fast-locked/source.ipynb'
SHA = 'b02169d71b873985d5f4272beb191dbce73377150605ed111f87830c4d571c86'

def code(text):
    return dict(cell_type='code', metadata={}, source=text.splitlines(True), outputs=[], execution_count=None)

def replace_once(src, old, new):
    assert src.count(old) == 1, (old[:90], src.count(old))
    return src.replace(old, new)

def build():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    nb = json.loads(SOURCE.read_text(encoding='utf-8'))
    audit = json.loads((ROOT / 'dependency_audit.json').read_text())
    assert len(audit) == 14 and all(x['accessible'] and x['files'] for x in audit)
    expected = []
    for asset in audit:
        if asset['kind'] == 'dataset':
            required = [f for f in asset['files'] if f['name'].endswith(('.pt', '.pth', '.h5', '.whl', 'manifest.json'))]
            expected.append(dict(ref=asset['ref'], version=asset['version'], files=required))
    preflight = '''import os, json, time, hashlib
from pathlib import Path
ANCHOR_STARTED = time.time()
os.environ['RSNA_PRESET'] = 'probe22'
os.environ.setdefault('CUDNN_CONV_WSCAP_DBG', '1024')
ANCHOR_EXPECTED = __EXPECTED__
ANCHOR_ASSETS = []
for item in ANCHOR_EXPECTED:
    owner, slug = item['ref'].split('/')
    roots = [Path('/kaggle/input') / slug, Path('/kaggle/input/datasets') / owner / slug]
    root = next((p for p in roots if p.is_dir()), None)
    if root is None:
        raise FileNotFoundError(f"Required input missing: {item['ref']}; checked {roots}")
    for entry in item['files']:
        file = root / entry['name']
        if not file.is_file() or file.stat().st_size != entry['totalBytes']:
            raise RuntimeError(f'Asset missing or changed: {file}')
        ANCHOR_ASSETS.append({'path': str(file), 'bytes': file.stat().st_size,
                              'audited_dataset_version': item['version']})
Path('/kaggle/working/anchor941_preflight.json').write_text(json.dumps({
    'upstream': 'jiweiliu/rsna-knee-fast-2xt4-inference', 'upstream_version': 5,
    'source_sha256': '__SHA__', 'preset': 'probe22', 'assets': ANCHOR_ASSETS,
    'note': 'File sizes checked; upstream cryptographic checks run during inference.'
}, indent=2))
print(f'[anchor941] preflight passed: {len(ANCHOR_ASSETS)} required assets')
'''.replace('__EXPECTED__', repr(expected)).replace('__SHA__', SHA)
    changes = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        old_src = src
        if i == 24:
            src = replace_once(src, "    members = man['members']", "    members = man['members']\n    assert len(members) == 20, f'Expected 20 DINO members, got {len(members)}'")
            src = replace_once(src, "    if len(public_frontier_members) == len(members):", "    assert len(public_frontier_members) == len(members), 'DINO member missing; partial output rejected'\n    if len(public_frontier_members) == len(members):")
        elif i == 28:
            src = replace_once(src, "    pkg = find_weights()", "    pkg = find_weights()\n    assert pkg is not None, 'Missing pretrained package; training fallback disabled'")
            src = replace_once(src, '        except Exception as public_frontier_error:\n', '        except Exception as public_frontier_error:\n            raise RuntimeError("DINO frontier promotion failed") from public_frontier_error\n')
        elif i == 29:
            src = "main()\nlog('done')\n"
        elif i == 34:
            src = replace_once(src, '            except Exception as e:\n', '            except Exception as e:\n                raise RuntimeError("A5 study preparation failed") from e\n')
        elif i == 35:
            src += "\n_a5_sub.to_csv('/kaggle/working/anchor941_after_a5.csv', index=False)\n"
        elif i == 37:
            src += "\n_rad_pd.read_csv('/kaggle/working/submission.csv').to_csv('/kaggle/working/anchor941_after_rad.csv', index=False)\n"
        elif i == 39:
            assert src.count('            except Exception as error:\n') == 2
            src = src.replace('            except Exception as error:\n', '            except Exception as error:\n                raise RuntimeError("Raptor study inference failed") from error\n')
            src = replace_once(src, 'except Exception as _coat_err:\n', 'except Exception as _coat_err:\n    raise RuntimeError("Required residual CoAtNet failed; partial recipe rejected") from _coat_err\n')
        ast.parse(src)
        if src != old_src:
            changes.append(dict(upstream_cell=i, original_sha256=hashlib.sha256(old_src.encode()).hexdigest(),
                                patched_sha256=hashlib.sha256(src.encode()).hexdigest()))
        cell.update(source=src.splitlines(True), outputs=[], execution_count=None)
    final = '''import json, time
from pathlib import Path
import numpy as np
import pandas as pd
_anchor_labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
                 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
_anchor_root = Path('/kaggle/working')
_anchor_frame = pd.read_csv(_anchor_root/'submission.csv', dtype={'StudyInstanceUID': str})
_anchor_test = pd.read_csv(COMP/'test.csv', dtype={'StudyInstanceUID': str})
assert _anchor_frame.columns.tolist() == ['StudyInstanceUID', *_anchor_labels]
assert not _anchor_frame.StudyInstanceUID.duplicated().any()
assert _anchor_frame.StudyInstanceUID.tolist() == _anchor_test.StudyInstanceUID.tolist()
assert np.isfinite(_anchor_frame[_anchor_labels].to_numpy()).all()
assert _anchor_frame[_anchor_labels].ge(0).all().all() and _anchor_frame[_anchor_labels].le(1).all().all()
_anchor_receipts = {}
for name in ['v50_v2_repro_receipt.json', '_coat_arm_receipt.json',
             '_coat_raptor_blend_receipt.json', 'probe22_outer_routing_receipt.json']:
    _anchor_receipts[name] = json.loads((_anchor_root/name).read_text())
assert _anchor_receipts['_coat_arm_receipt.json']['models'] == 3
assert _anchor_receipts['_coat_arm_receipt.json']['fallback_studies'] == 0
assert _anchor_receipts['probe22_outer_routing_receipt.json']['status'] == 'passed'
(_anchor_root/'anchor941_run_receipt.json').write_text(json.dumps({
    'complete': True, 'upstream_version': 5, 'upstream_sha256': '__SHA__',
    'preset': 'probe22', 'studies': len(_anchor_frame),
    'elapsed_seconds': time.time()-ANCHOR_STARTED, 'receipts': _anchor_receipts,
    'limitation': 'Visible test completion does not establish hidden-test score or runtime.'
}, indent=2))
print('[anchor941] COMPLETE:', len(_anchor_frame), 'studies; all required branches passed')
'''.replace('__SHA__', SHA)
    ast.parse(preflight)
    ast.parse(final)
    nb['cells'].insert(1, code(preflight))
    nb['cells'].append(code(final))
    for key in ('widgets', 'papermill'):
        nb['metadata'].pop(key, None)
    nb['metadata']['anchor941'] = dict(upstream_version=5, source_sha256=SHA, strict=True)
    out = ROOT/'notebook'
    out.mkdir(exist_ok=True)
    (out/'anchor941.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    metadata = json.loads((ROOT/'latest-fast/kernel-metadata.json').read_text())
    metadata.pop('id_no', None)
    metadata.update(id='easoncyy/rsna-anchor941-audited', title='RSNA Anchor941 Audited',
                    code_file='anchor941.ipynb', is_private=True, enable_internet=False)
    (out/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (ROOT/'build_receipt.json').write_text(json.dumps(dict(upstream_sha256=SHA,
        upstream_version=5, changes=changes, notebook_sha256=hashlib.sha256((out/'anchor941.ipynb').read_bytes()).hexdigest(),
        note='Guards and diagnostic exports only; preset, models, preprocessing and fusion unchanged.'), indent=2))
    print('Built', out, '; compiled all code cells; audited', len(changes), 'modified cells')

if __name__ == '__main__':
    build()
