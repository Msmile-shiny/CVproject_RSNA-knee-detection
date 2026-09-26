"""Build a private offline CPU audit, using the exact existing slot matcher."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def build():
    config=ast.parse((ROOT/'cells_v5/03_config.py').read_text(encoding='utf-8'))
    slots=next(n for n in config.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SLOTS' for t in n.targets))
    matcher=ast.parse((ROOT/'cells_v5/04_slot_matching.py').read_text(encoding='utf-8'))
    function=next(n for n in matcher.body if isinstance(n,ast.FunctionDef) and n.name=='match_slots_for_study')
    source=ast.unparse(slots)+'\n\n'+ast.unparse(function)+'\n\n'+(ROOT/'stage6_series_audit.py').read_text(encoding='utf-8')
    ast.parse(source)
    dest=ROOT.parent/'experiments/stage6/series-audit';dest.mkdir(parents=True,exist_ok=True)
    (dest/'audit.py').write_text(source,encoding='utf-8')
    metadata=dict(id='easoncyy/rsna-stage6-series-input-audit',title='RSNA Stage6 Series Input Audit',
        code_file='audit.py',language='python',kernel_type='script',is_private=True,
        enable_gpu=False,enable_tpu=False,enable_internet=False,
        competition_sources=['rsna-knee-abnormality-detection'],dataset_sources=[],kernel_sources=[],model_sources=[])
    (dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(dest)


if __name__=='__main__': build()
