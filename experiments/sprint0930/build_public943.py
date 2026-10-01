"""Reproduce the fully public four-reader pipeline; do not claim its title score."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "upstream/yama946/rsna-knee-d4-public0946.ipynb"
OUT = ROOT / "fourway-public"


def build():
    raw = SOURCE.read_bytes()
    notebook = json.loads(raw)
    original_code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            ast.parse(source)
            original_code.append(source)
            cell.update(outputs=[], execution_count=None)
    receipt = '''# Audit only: do not modify upstream predictions or fallback behavior.
import json as _sprint_json
from pathlib import Path as _SprintPath
_sprint_root = _SprintPath('/kaggle/working')
_sprint_audit = _sprint_json.loads((_sprint_root/'btkd_v559_complete.json').read_text())
_sprint_family = _sprint_json.loads((_sprint_root/'_coat_raptor_blend_receipt.json').read_text())
_sprint_expected = {'resgated_top3', 'global96_top3', 'd4_swa3', 'repairv1_top3'}
_sprint_events = _sprint_audit.get('events', [])
_sprint_issues = [e for e in _sprint_events if any(k in e.get('kind', '') for k in ('failed', 'fallback', 'unavailable', 'rejected', 'incomplete'))]
_sprint_full = set(_sprint_family.get('members', [])) == _sprint_expected
_sprint_receipt = {
    'status': 'COMPLETE', 'source_sha256': '__SHA__',
    'inference_changes': 'none; post-run audit only',
    'upstream_reported_public_score': 0.943, 'own_public_score': None,
    'family_members': _sprint_family.get('members', []),
    'family_reduction': _sprint_family.get('family_reduction'),
    'all_four_readers_present': _sprint_full,
    'issues': _sprint_issues,
    'ready_for_scoring': _sprint_full and not _sprint_issues,
    'study_count': _sprint_family.get('study_count'),
    'submission_sha256': _sprint_audit['submission_sha256'],
    'elapsed_seconds': _sprint_audit.get('elapsed_seconds'),
}
(_sprint_root/'sprint_fourway_receipt.json').write_text(_sprint_json.dumps(_sprint_receipt, indent=2))
print('SPRINT FOURWAY RECEIPT:', _sprint_receipt)
'''.replace('__SHA__', hashlib.sha256(raw).hexdigest())
    ast.parse(receipt)
    notebook['cells'].insert(0, {
        'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# RSNA September 30 sprint — public four-reader reproduction\n',
            'Upstream code and attribution are retained unchanged. The upstream author corrected the title: **0.943, not 0.946**. Our score is pending.\n',
            'Adds Global96 and Repair-v1 to the old two-reader CoAt family; this is a reproducibility step, not an original-model claim.\n',
        ],
    })
    notebook['cells'].append({'cell_type': 'code', 'metadata': {}, 'source': receipt.splitlines(True), 'outputs': [], 'execution_count': None})
    notebook['metadata'].pop('widgets', None)
    assert original_code == [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'][:-1]
    meta = json.loads((SOURCE.parent/'kernel-metadata.json').read_text(encoding='utf-8'))
    for field in ('dataset_sources', 'kernel_sources', 'model_sources'):
        assert all(meta.get(field, [])), f'Hidden or empty dependency: {field}'
    meta.pop('id_no', None)
    meta.update(id='easoncyy/rsna-sprint-public-fourway', title='RSNA Sprint Public Fourway', code_file='sprint-fourway.ipynb', is_private=True, enable_internet=False)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(notebook, ensure_ascii=False, indent=1)
    (OUT/meta['code_file']).write_text(payload, encoding='utf-8')
    (OUT/'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    build_receipt = {'upstream': 'yamadan96/rsna-knee-d4-public0946', 'source_sha256': hashlib.sha256(raw).hexdigest(), 'notebook_sha256': hashlib.sha256(payload.encode('utf-8')).hexdigest(), 'original_code_cells': len(original_code), 'original_code_unchanged': True, 'upstream_claim': 0.943, 'own_score': None, 'private_dependencies': False}
    (ROOT/'fourway_build_receipt.json').write_text(json.dumps(build_receipt, indent=2), encoding='utf-8')
    print(json.dumps(build_receipt, indent=2))


if __name__ == '__main__':
    build()
