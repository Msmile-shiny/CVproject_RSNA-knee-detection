"""Change one branch's slice sampling density; preserve span and model arithmetic."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'notebook/anchor941.ipynb'
OUTPUT = ROOT/'native64-notebook'
CHANGE = '''
# Experiment: native384-v8 sampling density, 44 -> 64 actual slices.
# Span stays (0.06, 0.94); traverse all D-2 windows in either configuration.
assert _KE_NS['ARMS'][3]['name'] == 'native384-v8'
assert sum(slot[2] for slot in _KE_NS['ARMS'][3]['slots']) == 44
assert _KE_NS['ARMS'][3]['k_eval'] == 42
assert tuple(_KE_NS['ARMS'][3]['span']) == (0.06, 0.94)
_KE_NS['ARMS'][3] = dict(_KE_NS['ARMS'][3],
    slots=[('Sagittal', 1, 18), ('Sagittal', 0, 14),
           ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)],
    k_eval=62)
print('[native64] native384-v8: 64 slices, 62 windows; original span and blend retained', flush=True)
'''

def validate_contract(source, patched):
    """Extract constant upstream configuration without importing/executing its runtime."""
    import numpy as np
    tree = ast.parse(source)
    embedded = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == '_KE_SRC' for t in n.targets))
    nodes = []
    for n in ast.parse(embedded).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('_SLOTS44', '_SLOTS64', 'ARMS') for t in n.targets):
            nodes.append(n)
        if isinstance(n, ast.FunctionDef) and n.name == '_eval_centers':
            nodes.append(n)
    env = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'extracted_configuration', 'exec'), env)
    before = json.loads(json.dumps(env['ARMS']))
    centers = env['_eval_centers']
    assert len(set(centers(np.ones(44), 44, 62))) == 42, '42->62 alone is not new coverage'
    exec(CHANGE, {'_KE_NS': env})
    after = json.loads(json.dumps(env['ARMS']))
    assert before[:3] == after[:3]
    assert {k:v for k,v in before[3].items() if k not in ('slots','k_eval')} == {
        k:v for k,v in after[3].items() if k not in ('slots','k_eval')}
    assert len(set(centers(np.ones(64), 64, 62))) == 62
    assert source == patched.replace(CHANGE, '', 1), 'Unexpected source change'
    return dict(before=before, after=after, old_unique_windows=42, new_unique_windows=62)

def build():
    nb = json.loads(BASE.read_text(encoding='utf-8'))
    needle = "exec(compile(_KE_SRC, '<raptor>', 'exec'), _KE_NS)\n"
    matches = [cell for cell in nb['cells'] if cell['cell_type']=='code' and needle in ''.join(cell['source'])]
    assert len(matches) == 1
    target = matches[0]
    original = ''.join(target['source'])
    assert original.count(needle) == 1
    patched = original.replace(needle, needle+CHANGE)
    contract = validate_contract(original, patched)
    target['source'] = patched.splitlines(True)
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))
            cell.update(outputs=[], execution_count=None)
    nb['cells'].append(dict(cell_type='code',metadata={},outputs=[],execution_count=None,source=[
        "_density_receipt = json.loads((_anchor_root/'anchor941_run_receipt.json').read_text())\n",
        "_density_receipt['experiment'] = 'native384-v8: 44 to 64 slices; original span, weights, fusion'\n",
        "assert _KE_NS['ARMS'][3]['k_eval'] == 62\n",
        "_density_receipt['raptor_configuration'] = _KE_NS['ARMS']\n",
        "(_anchor_root/'native64_run_receipt.json').write_text(json.dumps(_density_receipt, indent=2))\n"
    ]))
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'native64.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    meta=json.loads((ROOT/'notebook/kernel-metadata.json').read_text())
    meta.update(id='easoncyy/rsna-anchor941-native64',title='RSNA Anchor941 Native64',code_file='native64.ipynb')
    (OUTPUT/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (ROOT/'native64_build_receipt.json').write_text(json.dumps(dict(
        baseline_submission=56290048,baseline_public=0.941,
        baseline_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256((OUTPUT/'native64.ipynb').read_bytes()).hexdigest(),
        hypothesis='Denser sampling in native384-v8 may recover small findings; no guaranteed gain.',
        contract=contract),indent=2))
    print('PASS: other three branches unchanged; span, crop, models, routing unchanged; unique windows 42 -> 62.')

if __name__=='__main__':
    build()
