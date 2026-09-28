"""Build the 4-to-8 window Stage 6 coverage control from the locked reference."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'experiments/stage6'


def build(smoke_receipt=None, variant='coverage', resume_receipt=None, resume_source=None):
    assert variant in ('coverage', 'resolution')
    assert not (smoke_receipt and resume_receipt), 'Select launch or recovery mode'
    assert (resume_receipt is None) == (resume_source is None), 'Recovery needs the exact Kaggle output source'
    reference = json.loads((ROOT / 'experiments/stage6/reference_v3/run_receipt.json').read_text())
    assert reference['status'] == 'PILOT_COMPLETE' and reference['completed_epochs'] == 12
    notebook = json.loads((ROOT / 'experiments/stage6/notebook/stage6a.ipynb').read_text(encoding='utf-8'))
    sources = [''.join(cell['source']) for cell in notebook['cells']]
    implementation = hashlib.sha256(''.join(sources[2:]).encode()).hexdigest()
    assert implementation == reference['config']['implementation_sha256']
    assert reference['config']['windows'] == 4 and reference['config']['pooling'] == 'mean'
    change = {'windows': 8} if variant == 'coverage' else {'size': 288}
    is_smoke = smoke_receipt is None and resume_receipt is None
    config = dict(reference['config'], **change, smoke=is_smoke, epochs=1 if is_smoke else 12,
                  minutes=90 if is_smoke else 480, resume='')
    if smoke_receipt is not None:
        proof = json.loads(Path(smoke_receipt).read_text())
        assert proof['status'] == 'SMOKE_COMPLETE' and proof['completed_epochs'] == 1
        assert proof['device'] == 'cuda' and proof['first_backbone_gradient_norm'] > 0
        for key in ('implementation_sha256', 'size', 'windows', 'batch_size', 'fold', 'seed',
                    'pooling', 'backbone_lr', 'head_lr'):
            assert proof['config'][key] == config[key], key
        assert proof['labels_sha256'] == reference['labels_sha256']
        assert proof['weights_sha256'] == reference['weights_sha256']
    if resume_receipt is not None:
        prior = json.loads(Path(resume_receipt).read_text())
        assert prior['status'] in ('PAUSED_PREPROCESSING', 'PAUSED_TRAINING')
        assert prior['labels_sha256'] == reference['labels_sha256']
        assert prior['weights_sha256'] == reference['weights_sha256']
        for key in ('implementation_sha256', 'size', 'windows', 'batch_size', 'fold', 'seed',
                    'pooling', 'backbone_lr', 'head_lr', 'smoke', 'epochs'):
            assert prior['config'][key] == config[key], key
        assert 0 <= prior.get('completed_epochs', 0) < 12
    description = ('Eight positions per slot instead of four' if variant == 'coverage'
                   else '288 pixels instead of 224, keeping four positions per slot')
    sources[0] = (f'# Stage 6 {variant} input control\n'
                  f'{description}. Gold is development-only; no leaderboard submission.')
    sources[1] = 'S6 = ' + repr(config) + '\n'
    if smoke_receipt is not None:
        cache_files = list((Path(smoke_receipt).parent / 'pixel_cache').glob('*.npz'))
        assert len(cache_files) == 16, 'Use the complete 16-study smoke output to estimate disk needs'
        projected = int(sum(path.stat().st_size for path in cache_files) / len(cache_files) * 4407)
        sources[1] += f'''
import shutil
_free = shutil.disk_usage('/kaggle/working').free
_projected = {projected}
print('Preflight disk free bytes:', _free, 'projected 4407-study cache bytes:', _projected)
assert _free > _projected + 1_000_000_000, 'Insufficient free disk for projected pixel cache plus checkpoint'
'''
    if resume_receipt is not None:
        receipt_sha = hashlib.sha256(Path(resume_receipt).read_bytes()).hexdigest()
        sources[1] += f'''
import hashlib
from pathlib import Path
_matches = [p for p in Path('/kaggle/input').rglob('run_receipt.json')
            if hashlib.sha256(p.read_bytes()).hexdigest() == {receipt_sha!r}]
assert len(_matches) == 1, 'Mount the exact paused Stage 6 output'
_resume_root = _matches[0].parent
assert (_resume_root / 'pixel_cache').is_dir(), 'Prior pixel cache missing'
if {prior['status'] == 'PAUSED_TRAINING'!r}:
    assert (_resume_root / 'last.pt').is_file(), 'Training checkpoint missing'
S6['resume'] = str(_resume_root)
print('Verified Stage 6 resume source:', _resume_root)
'''
    for cell, source in zip(notebook['cells'], sources):
        cell['source'] = source.splitlines(True)
        if cell['cell_type'] == 'code':
            ast.parse(source)
            cell['outputs'] = []
            cell['execution_count'] = None
    suffix = 'resume' if resume_receipt is not None else ('smoke' if is_smoke else 'pilot')
    dest = DEST / variant / suffix
    dest.mkdir(parents=True, exist_ok=True)
    notebook_name = f'{variant}.ipynb'
    (dest / notebook_name).write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    metadata = json.loads((ROOT / 'experiments/stage6/notebook/kernel-metadata.json').read_text())
    stage = '6e' if variant == 'coverage' else '6f'
    metadata.update(id=f'easoncyy/rsna-stage{stage}-{variant}-{suffix}',
                    title=f'RSNA Stage{stage.upper()} {variant.title()} {suffix.title()}', code_file=notebook_name)
    if resume_source is not None:
        metadata['kernel_sources'].append(resume_source)
    (dest / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot-after-smoke', type=Path)
    parser.add_argument('--resume-after-paused', type=Path)
    parser.add_argument('--resume-source')
    parser.add_argument('--variant', choices=('coverage', 'resolution'), default='coverage')
    args = parser.parse_args()
    build(args.pilot_after_smoke, args.variant, args.resume_after_paused, args.resume_source)
