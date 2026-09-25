"""Validate the files that must be uploaded for the offline Kaggle Phase 4A run."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

EXPECTED_LABEL_SHA = 'c13adffaabf4f8e518abb038282bb1aa09baac7652a9165e030710c457d0be6a'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_containers(obj, name='root'):
    found = []
    if isinstance(obj, dict):
        tensors = {str(k): v for k, v in obj.items() if torch.is_tensor(v)}
        if tensors:
            found.append({'path': name, 'tensors': len(tensors),
                          'parameters': sum(v.numel() for v in tensors.values())})
        for key, value in obj.items():
            if isinstance(value, dict):
                found.extend(tensor_containers(value, name + '.' + str(key)))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True,
                        help='Downloaded official OrthoFoudation-L.pth Git-LFS binary')
    parser.add_argument('--labels', type=Path, default=Path('data/processed/v5_labels.csv'))
    parser.add_argument('--dinov3-source', type=Path,
                        help='Clone of the official facebookresearch/dinov3 repository')
    parser.add_argument('--manifest', type=Path, default=Path('phase4a_asset_manifest.json'))
    args = parser.parse_args()
    if not args.weights.is_file() or args.weights.stat().st_size < 100_000_000:
        raise SystemExit('Weight is absent/too small; it is probably a Git-LFS pointer, not the binary.')
    if sha256(args.labels) != EXPECTED_LABEL_SHA:
        raise SystemExit('v5_labels.csv does not match the frozen historical supervision asset.')
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    containers = tensor_containers(checkpoint)
    if not containers:
        raise SystemExit('Checkpoint contains no tensor dictionary.')
    source_root = args.dinov3_source or (args.weights.parent / 'dinov3')
    if not (source_root / 'dinov3/hub/backbones.py').is_file():
        raise SystemExit('Official DINOv3 source is absent; pass --dinov3-source.')
    sys.path.insert(0, str(source_root.resolve()))
    from dinov3.hub.backbones import dinov3_vitl16
    backbone = dinov3_vitl16(pretrained=False)
    if not all(k.startswith('backbone.') for k in checkpoint):
        raise SystemExit('Checkpoint keys do not use the expected backbone. prefix.')
    state = {k.removeprefix('backbone.'): value for k, value in checkpoint.items()}
    backbone.load_state_dict(state, strict=True)
    manifest = {
        'weights_name': args.weights.name,
        'weights_bytes': args.weights.stat().st_size,
        'weights_sha256': sha256(args.weights),
        'labels_name': args.labels.name,
        'labels_sha256': EXPECTED_LABEL_SHA,
        'tensor_containers': containers,
        'architecture': 'official facebookresearch/dinov3 dinov3_vitl16',
        'architecture_tensors': len(backbone.state_dict()),
        'strict_load': True,
        'kaggle_mounts': ['OrthoFoudation-L.pth', 'dinov3/hub/backbones.py'],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(args.manifest.resolve())


if __name__ == '__main__':
    main()
