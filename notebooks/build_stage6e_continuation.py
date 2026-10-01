"""Continue completed 6E to 24 epochs using both immutable cache outputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(completed, paused):
    a, b = [json.loads(p.read_text()) for p in (completed, paused)]
    assert a['status'] == 'PILOT_COMPLETE' and a['completed_epochs'] == 12
    assert b['status'] == 'PAUSED_PREPROCESSING'
    assert a['config']['windows'] == 8 and a['config']['size'] == 224
    assert a['device'] == 'cuda' and a['first_backbone_gradient_norm'] > 0
    for key in ('labels_sha256', 'weights_sha256', 'train_metadata_sha256',
                'series_metadata_sha256', 'train_count', 'val_count', 'gold_count'):
        assert a[key] == b[key], key
    for key in ('size', 'windows', 'batch_size', 'fold', 'seed', 'pooling',
                'backbone_lr', 'head_lr', 'smoke', 'implementation_sha256'):
        assert a['config'][key] == b['config'][key], key
    notebook = json.loads((ROOT/'experiments/stage6/coverage/resume/coverage.ipynb').read_text())
    sources = [''.join(c['source']) for c in notebook['cells']]
    assert hashlib.sha256(''.join(sources[2:]).encode()).hexdigest() == a['config']['implementation_sha256']
    cfg = dict(a['config'], epochs=24, minutes=480, resume='')
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (completed, paused)]
    sources[0] = '# Stage 6E continuation: 12 to 24 epochs\nReuse both cache outputs; unchanged training implementation.'
    sources[1] = 'S6 = ' + repr(cfg) + '\n' + f'''
import hashlib
import json
from pathlib import Path
import torch
import tempfile
receipts = list(Path('/kaggle/input').rglob('run_receipt.json'))
roots = []
for expected in {hashes!r}:
    matches = [p for p in receipts if hashlib.sha256(p.read_bytes()).hexdigest() == expected]
    assert len(matches) == 1, 'Mount both exact Stage 6E outputs'
    roots.append(matches[0].parent)
checkpoint_root, preprocessing_root = roots
saved = torch.load(checkpoint_root/'last.pt', map_location='cpu', weights_only=False)
assert saved['epoch'] == 12 and len(saved['history']) == 12
for key in ('size','windows','pooling','fold','seed','implementation_sha256'):
    assert saved['config'][key] == S6[key], key
del saved
union = Path(tempfile.mkdtemp(prefix='stage6e-resume-'))
(union/'pixel_cache').mkdir()
names = set()
counts = []
for root in roots:
    files = list((root/'pixel_cache').glob('*.npz'))
    counts.append(len(files))
    for source in files:
        assert source.name not in names, 'Overlapping cache entries need explicit reconciliation'
        names.add(source.name)
        (union/'pixel_cache'/source.name).symlink_to(source)
assert len(names) == 4407, ('Incomplete cache union', counts)
for name in ('last.pt','run_receipt.json'):
    (union/name).symlink_to(checkpoint_root/name)
S6['resume'] = str(union)
print('Verified complete cache union:', counts, 'total:', len(names), 'resume epoch: 12')
'''
    for cell, source in zip(notebook['cells'], sources):
        cell['source'] = source.splitlines(True)
        if cell['cell_type'] == 'code':
            ast.parse(source)
            cell['outputs'] = []
            cell['execution_count'] = None
    dest = ROOT/'experiments/stage6/coverage/continuation'
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'coverage.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    meta = json.loads((ROOT/'experiments/stage6/coverage/resume/kernel-metadata.json').read_text())
    meta.update(id='easoncyy/rsna-stage6e-coverage-continuation',
                title='RSNA Stage6E Coverage Continuation', code_file='coverage.ipynb')
    meta['kernel_sources'] = list(dict.fromkeys(meta['kernel_sources'] +
        ['easoncyy/rsna-stage6e-coverage-pilot', 'easoncyy/rsna-stage6e-coverage-resume']))
    (dest/'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('completed', type=Path)
    parser.add_argument('paused', type=Path)
    args = parser.parse_args()
    build(args.completed, args.paused)
