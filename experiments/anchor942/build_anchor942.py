"""Build a strict, auditable copy of the public 0.942 Speedy recipe."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'upstream/rsna-knee-0942-restructured.ipynb'
META = ROOT/'upstream/kernel-metadata.json'
OUT = ROOT/'notebook'

def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new)

def build():
    raw = SOURCE.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    nb = json.loads(raw)
    changed = []
    for index, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        original = src
        if index == 45:
            src = replace_once(src,
                "        if int(_coat_receipt.get('fallback_studies', 0) or 0):\n            rsna_event('coat_resgated_fallback_studies', count=int(_coat_receipt['fallback_studies']), failures=list(_coat_receipt.get('failures', []))[:20])",
                "        if int(_coat_receipt.get('fallback_studies', 0) or 0):\n            raise RuntimeError(f\"Residual CoAt fallback studies: {_coat_receipt.get('failures', [])[:20]}\")")
            src = replace_once(src,
                "    except Exception as _coat_exc:\n        rsna_event('coat_resgated_child_failed', error=f'{type(_coat_exc).__name__}: {str(_coat_exc)[:1500]}')\n        print(f'[coat-arm] residual CoAt child FAILED; continuing without it (flagged): {type(_coat_exc).__name__}', flush=True)\n        _coat_resgated_ok = False",
                "    except Exception as _coat_exc:\n        raise RuntimeError('Required residual CoAt child failed') from _coat_exc")
            src = replace_once(src,
                "    except Exception as _d4_exc:\n        rsna_event('coat_d4_child_failed', error=f'{type(_d4_exc).__name__}: {str(_d4_exc)[:1500]}')\n        print(f'[coat-arm] D4 CoAt child FAILED; continuing without it (flagged): {type(_d4_exc).__name__}', flush=True)\n        _coat_d4_ok = False",
                "    except Exception as _d4_exc:\n        raise RuntimeError('Required D4 CoAt child failed') from _d4_exc")
            src = replace_once(src,
                "        except Exception as _exc:\n            rsna_event('coat_resgated_output_rejected', error=f'{type(_exc).__name__}: {str(_exc)[:500]}')",
                "        except Exception as _exc:\n            raise RuntimeError('Residual CoAt output rejected') from _exc")
            src = replace_once(src,
                "        except Exception as _exc:\n            rsna_event('coat_d4_output_rejected', error=f'{type(_exc).__name__}: {str(_exc)[:500]}')",
                "        except Exception as _exc:\n            raise RuntimeError('D4 CoAt output rejected') from _exc")
            src = replace_once(src, "    if not _members:\n", "    if len(_members) != 2:\n        raise RuntimeError(f'Expected both CoAt children, got {[m for m, _ in _members]}')\n    if not _members:\n")
        ast.parse(src)
        if src != original:
            changed.append({'cell': index, 'before': hashlib.sha256(original.encode()).hexdigest(),
                            'after': hashlib.sha256(src.encode()).hexdigest()})
        cell['source'] = src.splitlines(True)
        cell['outputs'] = []
        cell['execution_count'] = None
    final = '''import json as _final_json
from pathlib import Path as _FinalPath
import numpy as _final_np
import pandas as _final_pd
_final_root = _FinalPath('/kaggle/working')
_final_labels = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
_final_sub = _final_pd.read_csv(_final_root/'submission.csv', dtype={'StudyInstanceUID':str})
assert _final_sub.columns.tolist() == ['StudyInstanceUID', *_final_labels]
assert _final_sub.StudyInstanceUID.tolist() == list(map(str, _RSNA_TEST_IDS))
assert not _final_sub.StudyInstanceUID.duplicated().any()
assert _final_np.isfinite(_final_sub[_final_labels].to_numpy()).all()
assert _final_sub[_final_labels].ge(0).all().all() and _final_sub[_final_labels].le(1).all().all()
_coat = _final_json.loads((_final_root/'_coat_arm_receipt.json').read_text())
_family = _final_json.loads((_final_root/'_coat_raptor_blend_receipt.json').read_text())
assert int(_coat['fallback_studies']) == 0
assert int(_coat['models']) == 3
assert set(_family['members']) == {'resgated_top3','d4_swa3'}
(_final_root/'anchor942_run_receipt.json').write_text(_final_json.dumps({
  'complete': True, 'source_sha256': '__SHA__', 'preset': 'speedy',
  'studies': len(_final_sub), 'coat_family_members': _family['members'],
  'residual_fallback_studies': _coat['fallback_studies'],
  'limitation': 'Public recipe reproduction; visible test is a runtime gate, not validation.'
}, indent=2))
print('[anchor942] COMPLETE: full Speedy recipe, both CoAt children, no residual fallback')
'''.replace('__SHA__', source_sha)
    ast.parse(final)
    nb['cells'].append({'cell_type':'code','metadata':{},'source':final.splitlines(True),'outputs':[],'execution_count':None})
    nb['metadata'].pop('widgets',None)
    nb['metadata']['anchor942']={'upstream':'maverickss26/rsna-knee-0942-restructured','source_sha256':source_sha,'strict':True}
    OUT.mkdir(exist_ok=True)
    notebook = OUT/'anchor942.ipynb'
    notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads(META.read_text())
    meta.pop('id_no',None)
    meta.update(id='easoncyy/rsna-anchor942-audited',title='RSNA Anchor942 Audited',code_file='anchor942.ipynb',is_private=True,enable_internet=False)
    (OUT/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (ROOT/'anchor942_build_receipt.json').write_text(json.dumps({'source_sha256':source_sha,'changes':changed,
        'notebook_sha256':hashlib.sha256(notebook.read_bytes()).hexdigest(),
        'recipe_changes':'none; strict failure and final receipt checks only'},indent=2))
    print('Built strict Anchor942; changed cells:', [x['cell'] for x in changed])

if __name__=='__main__': build()
