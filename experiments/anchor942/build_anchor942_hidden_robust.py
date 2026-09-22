"""Build the hidden-test robust Anchor942 reproduction.

The public recipe's inference and fallback behavior are left unchanged.  The
previous strict build converted recoverable CoAt failures into fatal errors;
that is useful for a three-study audit, but unsafe on a larger hidden cohort.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "upstream" / "rsna-knee-0942-restructured.ipynb"
META = ROOT / "upstream" / "kernel-metadata.json"
OUT = ROOT / "hidden-robust-notebook"


def build():
    raw = SOURCE.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    notebook = json.loads(raw)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            ast.parse(source)
            cell["outputs"] = []
            cell["execution_count"] = None

    receipt_cell = '''import json as _hr_json
from pathlib import Path as _HrPath
import numpy as _hr_np
import pandas as _hr_pd
_hr_root = _HrPath('/kaggle/working')
_hr_labels = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
_hr_sub = _hr_pd.read_csv(_hr_root/'submission.csv', dtype={'StudyInstanceUID':str})
assert _hr_sub.columns.tolist() == ['StudyInstanceUID', *_hr_labels]
assert _hr_sub.StudyInstanceUID.tolist() == list(map(str, _RSNA_TEST_IDS))
assert not _hr_sub.StudyInstanceUID.duplicated().any()
_hr_values = _hr_sub[_hr_labels].to_numpy(_hr_np.float64)
assert _hr_np.isfinite(_hr_values).all()
assert (_hr_values >= 0).all() and (_hr_values <= 1).all()

def _hr_load(name):
    path = _hr_root/name
    if not path.is_file():
        return {'missing': True}
    try:
        return _hr_json.loads(path.read_text())
    except Exception as exc:
        return {'unreadable': f'{type(exc).__name__}: {exc}'}

_hr_coat = _hr_load('_coat_arm_receipt.json')
_hr_family = _hr_load('_coat_raptor_blend_receipt.json')
_hr_members = list(_hr_family.get('members', []))
_hr_receipt = {
  'complete': True,
  'source_sha256': '__SOURCE_SHA__',
  'preset': 'speedy',
  'studies': len(_hr_sub),
  'hidden_robust': True,
  'inference_recipe_changes': 'none',
  'coat_family_members': _hr_members,
  'residual_fallback_studies': int(_hr_coat.get('fallback_studies', 0) or 0),
  'degraded': bool(_hr_coat.get('fallback_studies', 0)) or len(_hr_members) != 2,
  'note': 'Recoverable public-recipe fallbacks are recorded rather than promoted to hidden-fatal exceptions.'
}
(_hr_root/'anchor942_hidden_robust_receipt.json').write_text(_hr_json.dumps(_hr_receipt, indent=2))
print('[anchor942-hidden-robust] COMPLETE', _hr_receipt)
'''.replace('__SOURCE_SHA__', source_sha)
    ast.parse(receipt_cell)
    notebook["cells"].append({
        "cell_type": "code", "metadata": {}, "source": receipt_cell.splitlines(True),
        "outputs": [], "execution_count": None,
    })
    notebook["metadata"].pop("widgets", None)
    notebook["metadata"]["anchor942_hidden_robust"] = {
        "upstream": "maverickss26/rsna-knee-0942-restructured",
        "source_sha256": source_sha,
        "inference_recipe_changes": "none",
        "reason": "preserve recoverable fallbacks on larger hidden cohorts",
    }

    OUT.mkdir(exist_ok=True)
    notebook_path = OUT / "anchor942-hidden-robust.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    metadata = json.loads(META.read_text(encoding="utf-8"))
    metadata.pop("id_no", None)
    metadata.update(
        id="easoncyy/rsna-anchor942-hidden-robust",
        title="RSNA Anchor942 Hidden Robust",
        code_file=notebook_path.name,
        is_private=True,
        enable_internet=False,
    )
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (ROOT / "anchor942_hidden_robust_build_receipt.json").write_text(json.dumps({
        "source_sha256": source_sha,
        "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "inference_recipe_changes": "none",
        "strict_hidden_fatal_patches": 0,
        "added_behavior": "post-run schema and degradation receipt only",
    }, indent=2), encoding="utf-8")
    print("Built hidden-robust Anchor942 from untouched upstream inference recipe")


if __name__ == "__main__":
    build()
