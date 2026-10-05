"""Read-only source checks; never import downloaded community code or start GPU jobs."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'meniscus-source'


def audit():
    manifest = json.loads((ROOT / 'bundle_manifest.json').read_text(encoding='utf-8'))
    results = {}
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Unexpected non-root artifact name')
        path = ROOT / name
        if not path.is_file():
            results[name] = 'NOT_DOWNLOADED'
            continue
        raw = path.read_bytes()
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError(f'Artifact mismatch: {name}')
        results[name] = 'HASH_AND_SIZE_PASS'
    ast.parse((ROOT / 'public0033_runtime.py').read_text(encoding='utf-8'))
    return {'source_audit': results, 'runtime_syntax': 'PASS',
            'weight_bytes_verified': results.get('checkpoint_epoch30.safetensors') == 'HASH_AND_SIZE_PASS',
            'gpu_executed': False, 'performance_verified': False}


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
