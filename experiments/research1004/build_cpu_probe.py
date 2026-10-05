"""Build a CPU-only, non-submission gate from the author's recovered public source."""
import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'cpu-probe'
SOURCE = BASE / 'results/research1005/meniscus/rsna-knee-0-937-weak-label-dinov2-meniscus-resid.ipynb'


def build():
    source_bytes = SOURCE.read_bytes()
    original = json.loads(source_bytes)
    code = ''.join(original['cells'][2]['source'])
    assert code.endswith('run_dinov2()\n')
    code = code[:-len('run_dinov2()\n')]
    # Keep original definitions and preprocessing, but never run its ensemble.
    assert not any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                   for n in ast.parse(code).body)
    manifest = json.loads((BASE / 'experiments/sprint1002/asset_manifests/dino/manifest.json').read_text())
    configs = {m['pixel_group'] for m in manifest['members']}
    assert len(configs) == 1
    config = json.loads(configs.pop())
    probe = '''
import importlib.util
import sys
assert torch.cuda.device_count() == 0, 'CPU gate must not allocate GPU'
cfg = CONFIG_PLACEHOLDER
adopt_config_globals(cfg)
test = pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})
assert 2 <= len(test) <= 10, 'Visible-only engineering probe; not a competition submission'
series = pd.read_csv(ROOT / 'test_series.csv', dtype=str)
plane_map = dict(zip(series['SeriesInstanceUID'], series['Anatomical_Plane']))
headers = annotate(walk('test_series'))
studies, cache, mask = build_cache(pick_slots(headers, plane_map), plane_map, lat_of(headers), 'cpu_probe')
assert set(studies) == set(test.StudyInstanceUID)
assert cache.shape == (len(studies), 6, 12, 336, 336)
assert cache.dtype == np.uint8 and mask.dtype == np.float32
assert np.all(mask.sum(axis=1) > 0), 'Empty study'
roots = [p.parent for p in Path('/kaggle/input').rglob('bundle_manifest.json')
         if json.loads(p.read_text()).get('schema_version') == 'public0033_meniscus10_bundle_v1']
assert len(roots) == 1
bundle = roots[0]
runtime_path = bundle / 'public0033_runtime.py'
assert hashlib.sha256(runtime_path.read_bytes()).hexdigest() == '9541f82a993d7dc2942ca2a5112e110471285f8d80701846079a3dc708761683'
spec = importlib.util.spec_from_file_location('public0033_runtime', runtime_path)
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)
runtime.verify_bundle(bundle)
receipt = {'status': 'PREPROCESS_AND_ASSETS_PASS', 'gpu_used': False,
           'source_sha256': 'SOURCE_SHA_PLACEHOLDER',
           'studies': studies, 'shape': list(cache.shape), 'slots_present': mask.sum(1).tolist(),
           'cache_sha256': hashlib.sha256(cache.tobytes()).hexdigest(),
           'pixel_config': cfg, 'elapsed_seconds': time.time() - T0,
           'performance_verified': False}
Path('/kaggle/working/cpu_probe_receipt.json').write_text(json.dumps(receipt, indent=2))
print(receipt, flush=True)
torch.set_num_threads(4)
receipt['model_preflight'] = dict(runtime.run_cpu_one_study_preflight(bundle))
receipt['status'] = 'CPU_GATE_PASS'
receipt['elapsed_seconds'] = time.time() - T0
Path('/kaggle/working/cpu_probe_receipt.json').write_text(json.dumps(receipt, indent=2))
print(receipt, flush=True)
'''.replace('CONFIG_PLACEHOLDER', repr(config)).replace('SOURCE_SHA_PLACEHOLDER', hashlib.sha256(source_bytes).hexdigest())
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# CPU engineering gate only\nSource: renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid. Original author preprocessing retained. No score or submission produced.']}]
    for text in (code, probe):
        ast.parse(text)
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(True), 'outputs': [], 'execution_count': None})
    OUT.mkdir(parents=True, exist_ok=True)
    for i, cell in enumerate(cells):
        cell['id'] = f'cpu-gate-{i}'
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    (OUT / 'probe.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    meta = dict(id='easoncyy/rsna-meniscus-cpu-gate', title='RSNA Meniscus CPU Gate', code_file='probe.ipynb', language='python', kernel_type='notebook', is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=False, dataset_sources=['renta0426/rsna-knee-public0033-meniscus-bag-v1'], competition_sources=['rsna-knee-abnormality-detection'], kernel_sources=[], model_sources=[])
    (OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('BUILT CPU ONLY: no training, no submission, no GPU')


if __name__ == '__main__':
    build()
