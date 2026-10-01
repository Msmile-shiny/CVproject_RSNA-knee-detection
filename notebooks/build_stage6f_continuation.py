"""Fixed 12-to-24 epoch resolution control, with verified full-cache reuse."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(receipt_path):
    receipt = json.loads(receipt_path.read_text())
    reference = json.loads((ROOT/'experiments/stage6/reference_v3/run_receipt.json').read_text())
    assert receipt['status'] == 'PILOT_COMPLETE' and receipt['completed_epochs'] == 12
    assert receipt['device'] == 'cuda' and receipt['first_backbone_gradient_norm'] > 0
    for key in ('labels_sha256', 'weights_sha256', 'train_metadata_sha256',
                'series_metadata_sha256', 'group_unit', 'train_count', 'val_count', 'gold_count'):
        assert receipt[key] == reference[key], key
    for key in ('windows', 'batch_size', 'fold', 'seed', 'pooling', 'backbone_lr',
                'head_lr', 'smoke', 'epochs', 'implementation_sha256'):
        assert receipt['config'][key] == reference['config'][key], key
    assert receipt['config']['size'] == 288
    notebook = json.loads((ROOT/'experiments/stage6/resolution/pilot/resolution.ipynb').read_text())
    sources = [''.join(c['source']) for c in notebook['cells']]
    assert hashlib.sha256(''.join(sources[2:]).encode()).hexdigest() == receipt['config']['implementation_sha256']
    config = dict(receipt['config'], epochs=24, minutes=240, resume='')
    expected = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    sources[0] = '# Stage 6F: fixed continuation from epoch 12 to 24\n288px, 4 windows, original mean pooling.'
    sources[1] = 'S6 = ' + repr(config) + '\n' + f'''
import hashlib
from pathlib import Path
import torch
matches = [p for p in Path('/kaggle/input').rglob('run_receipt.json')
           if hashlib.sha256(p.read_bytes()).hexdigest() == {expected!r}]
assert len(matches) == 1, 'Mount the exact complete Stage 6F pilot output'
resume_root = matches[0].parent
assert len(list((resume_root/'pixel_cache').glob('*.npz'))) == 4407, 'Full cache required'
saved = torch.load(resume_root/'last.pt', map_location='cpu', weights_only=False)
assert saved['epoch'] == 12 and len(saved['history']) == 12
for key in ('size','windows','pooling','fold','seed','implementation_sha256'):
    assert saved['config'][key] == S6[key], key
del saved
S6['resume'] = str(resume_root)
print('Verified 4407 cached studies; restoring epoch 12 for a fixed epoch-24 evaluation')
'''
    for cell, source in zip(notebook['cells'], sources):
        cell['source'] = source.splitlines(True)
        if cell['cell_type'] == 'code':
            ast.parse(source)
            cell['outputs'] = []
            cell['execution_count'] = None
    dest = ROOT/'experiments/stage6/resolution/continuation'
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'resolution.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    metadata = json.loads((ROOT/'experiments/stage6/resolution/pilot/kernel-metadata.json').read_text())
    metadata.update(id='easoncyy/rsna-stage6f-resolution-continuation', title='RSNA Stage6F Resolution Continuation')
    metadata['kernel_sources'] = list(dict.fromkeys(metadata['kernel_sources'] + ['easoncyy/rsna-stage6f-resolution-pilot']))
    (dest/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    build(parser.parse_args().receipt)
