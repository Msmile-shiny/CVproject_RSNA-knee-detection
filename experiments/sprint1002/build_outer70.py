"""Build one preregistered inference ablation locally; never launches Kaggle."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'sprint0930/fourway-public'
PARENT_SHA = 'd0165cf3d144b6be4e05ece77f80b233c27393cdbae67cb06c0d03b96f178505'
OLD = '_coatnet_weight = {label: 0.60 for label in _blend_labels}'
NEW = '_coatnet_weight = {label: 0.70 for label in _blend_labels}'
CHANGED = ['MCL', 'Medial OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion']

AUDIT = r'''# Reconstruct the 0.943 recipe from this SAME inference pass; no extra forward passes.
import json as _o70_json, hashlib as _o70_hash
from pathlib import Path as _O70Path
_o70_root = _O70Path('/kaggle/working')
if not _sprint_receipt['ready_for_scoring']:
    raise RuntimeError('Inspect parent runtime issues before scoring outer70')
_o70_special = {'ACL': .75, 'Medial Meniscus': .80, 'Lateral Meniscus': 1., 'Lateral OA': .75, 'Fracture': .75}
_o70_baseline = _blend_transformer.copy()
for _o70_label in _blend_labels:
    _o70_w = _o70_special.get(_o70_label, .60)
    _o70_baseline[_o70_label] = (1-_o70_w)*_blend_tr[_o70_label] + _o70_w*_blend_cr[_o70_label]
_o70_baseline[_blend_labels] = _o70_baseline[_blend_labels].rank(method='average', pct=True)
_o70_actual = rsna_frame(pd.read_csv(_o70_root/'submission.csv', dtype={'StudyInstanceUID':str}), _RSNA_TEST_IDS, _RSNA_LABELS, 'OUTER70')
_o70_baseline = rsna_frame(_o70_baseline, _RSNA_TEST_IDS, _RSNA_LABELS, 'BASELINE60_REPLAY')
for _o70_label in _o70_special:
    if not _ke_np.allclose(_o70_baseline[_o70_label], _o70_actual[_o70_label], rtol=0, atol=1e-12):
        raise RuntimeError('Unchanged class drift: '+_o70_label)
_o70_baseline.to_csv(_o70_root/'baseline60_replay.csv', index=False)
_o70_transformer = _blend_transformer.copy()
_o70_transformer[_blend_labels] = _blend_tr
_o70_transformer.to_csv(_o70_root/'transformer_branch_rank.csv', index=False)
_blend_coatnet.to_csv(_o70_root/'coat_raptor_branch.csv', index=False)
_o70_files = ['submission.csv', 'baseline60_replay.csv', 'transformer_branch_rank.csv', 'coat_raptor_branch.csv']
_o70_receipt = {
    'status': 'COMPLETE', 'ready_for_scoring': True,
    'candidate': 'outer70', 'parent_submission': 56700487, 'parent_public_score': .943,
    'parent_notebook_sha256': '__PARENT_SHA__',
    'candidate_core_sha256': '__CORE_SHA__',
    'elapsed_seconds': time.time()-T0,
    'changed_labels': __CHANGED__, 'default_weight_before': .60, 'default_weight_after': .70,
    'unchanged_special_weights': _o70_special, 'own_public_score': None,
    'study_count': len(_o70_actual),
    'diagnostics_are_not_validation': True,
    'hashes': {f:_o70_hash.sha256((_o70_root/f).read_bytes()).hexdigest() for f in _o70_files},
}
(_o70_root/'outer70_receipt.json').write_text(_o70_json.dumps(_o70_receipt, indent=2))
print('OUTER70 RECEIPT', _o70_receipt)
'''


def code_cells(notebook):
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']


def make_candidate(parent):
    n = copy.deepcopy(parent)
    hits = 0
    for c in n['cells']:
        if c['cell_type'] == 'code':
            s = ''.join(c['source'])
            hits += s.count(OLD)
            c['source'] = s.replace(OLD, NEW).splitlines(True)
            c.update(outputs=[], execution_count=None)
    if hits != 1:
        raise ValueError(f'Expected exactly one outer-weight assignment, got {hits}')
    # Inherited receipt describes a reproduction; make its change description accurate.
    c = n['cells'][-1]
    s = ''.join(c['source']).replace("'inference_changes': 'none; post-run audit only'", "'inference_changes': 'outer default weight 0.60 to 0.70; five overrides retained'")
    c['source'] = s.splitlines(True)
    n['cells'][0]['source'] = [
        '# Outer70: one-variable ablation of our confirmed 0.943 parent\n',
        'Seven default classes: CoAt/Raptor weight 0.60 → 0.70. Five overrides, all models and preprocessing retained.\n',
        'Candidate score is UNKNOWN. No training. Run only after GPU quota recovers.\n',
    ]
    core_sha = hashlib.sha256(json.dumps(code_cells(n), ensure_ascii=False).encode('utf-8')).hexdigest()
    audit = AUDIT.replace('__PARENT_SHA__', PARENT_SHA).replace('__CHANGED__', repr(CHANGED)).replace('__CORE_SHA__', core_sha)
    n['cells'].append({'cell_type':'code', 'metadata':{}, 'source':audit.splitlines(True), 'outputs':[], 'execution_count':None})
    for source in code_cells(n):
        ast.parse(source)
    return n


def build():
    raw = (PARENT/'sprint-fourway.ipynb').read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise RuntimeError('Confirmed parent artifact changed; review before building')
    parent = json.loads(raw)
    n = make_candidate(parent)
    meta = json.loads((PARENT/'kernel-metadata.json').read_text())
    for key in ('dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources'):
        if not meta.get(key) or not all(meta[key]):
            raise ValueError('Empty dependency: '+key)
    meta.update(id='easoncyy/rsna-sprint-outer70', title='RSNA Sprint Outer70', code_file='outer70.ipynb')
    assert meta['enable_gpu'] and not meta['enable_internet']
    out = ROOT/'outer70'
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(n, ensure_ascii=False, indent=1).encode('utf-8')
    (out/'outer70.ipynb').write_bytes(payload)
    (out/'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    report = {
        'status':'BUILT_NOT_RUN', 'gpu_launched':False, 'candidate_public_score':None,
        'parent_sha256':PARENT_SHA, 'notebook_sha256':hashlib.sha256(payload).hexdigest(),
        'candidate_core_sha256':hashlib.sha256(json.dumps(code_cells(n)[:-1], ensure_ascii=False).encode('utf-8')).hexdigest(),
        'parent_code_cells':len(code_cells(parent)), 'candidate_code_cells':len(code_cells(n)),
        'changed_labels':CHANGED, 'new_dependencies':[],
        'dependency_audit':'remote_audit.json and asset_check_receipt.json contain separate timestamped evidence',
    }
    (ROOT/'build_receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    build()
