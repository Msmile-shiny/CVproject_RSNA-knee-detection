"""Build an offline Stage 6A smoke notebook; no hidden-test submission."""
import ast
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent / 'experiments/stage6/notebook'


def build(smoke_receipt=None):
    DEST.mkdir(parents=True, exist_ok=True)
    sources = [
        '# Stage 6A: ResNet-34 end-to-end reference\nSmoke first; no submission.csv. Gold is development-only. Edit paths in the next cell. Do not infer score from a smoke run.',
        '''S6 = dict(competition='/kaggle/input/competitions/rsna-knee-abnormality-detection',
    weights='', labels='', output='/kaggle/working', resume='',
    size=224, windows=4, batch_size=1, fold=0, seed=42,
    pooling='mean', backbone_lr=1e-5, head_lr=3e-4,
    smoke=True, epochs=1, minutes=90)
''',
        (ROOT / 'cells_v5/03_config.py').read_text(encoding='utf-8').split('# ---- Anatomical Priors')[0],
        'import re\nimport numpy as np\nimport pandas as pd\nIS_MAIN=True\n' + (ROOT / 'cells_v5/04_slot_matching.py').read_text(encoding='utf-8'),
        (ROOT/'stage6_resnet_core.py').read_text(encoding='utf-8'),
        (ROOT/'stage6_resnet_runtime.py').read_text(encoding='utf-8'),
    ]
    implementation=hashlib.sha256(''.join(sources[2:]).encode()).hexdigest()
    sources[1]+=f'\nS6["implementation_sha256"] = {implementation!r}\n'
    if smoke_receipt:
        proof=json.loads(Path(smoke_receipt).read_text())
        assert proof['status']=='SMOKE_COMPLETE', 'Cloud smoke has not completed'
        assert proof['device']=='cuda' and proof['first_backbone_gradient_norm']>0
        assert proof['config']['implementation_sha256']==implementation, 'Smoke tested different source'
        assert proof['config']['size']==224 and proof['config']['windows']==4
        sources[1]=sources[1].replace('smoke=True, epochs=1, minutes=90','smoke=False, epochs=12, minutes=480')
    cells=[]
    for i, source in enumerate(sources):
        cell=dict(cell_type='markdown' if i==0 else 'code', id=f's6-{i}', metadata={}, source=source.splitlines(True))
        if i:
            ast.parse(source); cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    (DEST/'stage6a.ipynb').write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}),indent=1),encoding='utf-8')
    metadata=dict(id='easoncyy/rsna-stage6a-resnet34-reference',title='RSNA Stage6A ResNet34 Reference',
        code_file='stage6a.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_internet=False,enable_tpu=False,
        dataset_sources=['easoncyy/rsna-knee-v5-labels'],
        competition_sources=['rsna-knee-abnormality-detection'],
        kernel_sources=['easoncyy/rsna-stage6-official-assets'],model_sources=[])
    (DEST/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(DEST)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot-after-smoke',type=Path,help='Cloud run_receipt.json required to generate full pilot')
    build(parser.parse_args().pilot_after_smoke)
