"""Build the single-variable RadImageNet Fracture re-admission ablation.

Run only after the audited Anchor942 submission reproduces at least 0.942.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "notebook" / "anchor942.ipynb"
OUT = ROOT / "fracture-notebook"

OLD = "_RAD_EXCLUDE = (\"Baker's\", 'Fracture')"
NEW = "_RAD_EXCLUDE = (\"Baker's\",)"


def build():
    raw = SOURCE.read_bytes()
    nb = json.loads(raw)
    matches = []
    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if OLD in source:
            assert source.count(OLD) == 1
            source = source.replace(OLD, NEW)
            ast.parse(source)
            cell["source"] = source.splitlines(True)
            matches.append(index)
    assert matches == [43], f"unexpected Rad exclusion matches: {matches}"

    receipt_cell = '''import json as _fx_json
from pathlib import Path as _FxPath
_fx_root = _FxPath('/kaggle/working')
_fx_parent = _fx_json.loads((_fx_root/'anchor942_run_receipt.json').read_text())
assert _fx_parent['complete'] is True
(_fx_root/'anchor942_fracture_ablation_receipt.json').write_text(_fx_json.dumps({
  'complete': True,
  'parent': 'audited Anchor942 Speedy recipe',
  'only_recipe_change': "_RAD_EXCLUDE: (Baker's, Fracture) -> (Baker's,)",
  'interpretation': 'Fracture is re-admitted to the RadImageNet blend; Baker remains excluded.'
}, indent=2))
print('[anchor942-fracture] COMPLETE: one-variable Rad Fracture re-admission')
'''
    ast.parse(receipt_cell)
    nb["cells"].append({
        "cell_type": "code", "metadata": {}, "source": receipt_cell.splitlines(True),
        "outputs": [], "execution_count": None,
    })
    nb["metadata"]["anchor942_fracture_ablation"] = {
        "parent_sha256": hashlib.sha256(raw).hexdigest(),
        "only_recipe_change": OLD + " -> " + NEW,
    }

    OUT.mkdir(exist_ok=True)
    notebook = OUT / "anchor942-fracture.ipynb"
    notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    metadata = json.loads((ROOT / "notebook" / "kernel-metadata.json").read_text())
    metadata.update(
        id="easoncyy/rsna-anchor942-fracture-ablation",
        title="RSNA Anchor942 Fracture Ablation",
        code_file=notebook.name,
        is_private=True,
        enable_internet=False,
    )
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (ROOT / "fracture_ablation_build_receipt.json").write_text(json.dumps({
        "parent_sha256": hashlib.sha256(raw).hexdigest(),
        "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
        "changed_cell": matches[0],
        "only_recipe_change": OLD + " -> " + NEW,
        "promotion_gate": "Run only if submission 56446116 scores >= 0.942",
    }, indent=2), encoding="utf-8")
    print(f"Built Fracture ablation; changed cell: {matches[0]}")


if __name__ == "__main__":
    build()
