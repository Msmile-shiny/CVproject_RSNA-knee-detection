"""Super-ensemble 复刻包离线校验 (CPU, 不跑推理)。

检查:
  A. push 包 notebook 与 reference_code 原文逐字一致 (源 sha256)
  B. 6 cells 结构 + 所有 code cell 可编译
  C. 硬编码输入路径 (ROOT/ASSET/DINO/CKPT/B3) 与 kernel-metadata 一致性
  D. 无网络依赖 (urllib/requests/wget) / 无私有依赖 import
  E. ASSET 内期望文件清单 (人工已核 API, 此处只列出来对照)

用法:
  PYTHONIOENCODING=utf-8 python scripts/verify_super_ensemble_package.py
"""

import hashlib
import json
import re
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
REF = PROJ / 'reference_code' / 'rsna-knee-super-ensemble.ipynb'
PKG = PROJ / 'notebooks' / 'kernel_push_super'
REPLICA = PKG / 'kaggle_super_ensemble_replica.ipynb'
META = json.loads((PKG / 'kernel-metadata.json').read_text(encoding='utf-8'))


def _src_sha(nb):
    return hashlib.sha256(
        '\n'.join(''.join(c['source']) for c in nb['cells']).encode()).hexdigest()


def _load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ref, rep = _load(REF), _load(REPLICA)

    # A. 逐字一致
    assert _src_sha(ref) == _src_sha(rep), 'replica source differs from reference'
    assert ref['cells'] == rep['cells'], 'cell structures differ'
    print('A OK: replica verbatim (source sha256 identical)')

    # B. 结构 + 编译
    cells = rep['cells']
    assert len(cells) == 6, f'expected 6 cells, got {len(cells)}'
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), f'<super-cell-{i}>', 'exec')
    md = ''.join(cells[0]['source'])
    assert 'Rank Ensemble' in md, 'markdown title mismatch'
    print('B OK: 6 cells, all code cells compile')

    # C. 输入路径 vs metadata
    src = '\n'.join(''.join(c['source']) for c in cells)
    assert "ASSET = Path('/kaggle/input/datasets/tonylica/rsna-knee-bend-dinov3-0917-repro-assets')" in src
    assert "ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')" in src
    assert "DINO = Path('/kaggle/input/models/metaresearch/dinov2/pytorch/small/1')" in src
    assert "CKPT = ASSET / 'knee-mri-fold-weights'" in src
    assert "/kaggle/input/rsna-knee-b3-v47-folds-0-3" in src
    ds = set(META['dataset_sources'])
    assert 'tonylica/rsna-knee-bend-dinov3-0917-repro-assets' in ds
    assert META['competition_sources'] == ['rsna-knee-abnormality-detection']
    assert 'metaresearch/dinov2/PyTorch/small/1' in META['model_sources']
    assert 'metaresearch/dinov2/PyTorch/base/1' in META['model_sources']
    assert len(ds) == 12, f'expected 12 datasets, got {len(ds)}'
    assert 'prvsiyan/rsna-knee-b3-v47-public-deployment' not in ds, \
        'B3 数据集不应在默认 metadata (评分配置 = 12 数据集无 B3); 实验时网页加挂'
    print('C OK: hardcoded paths covered by metadata; 12 datasets, no B3 (faithful)')

    # D. 依赖审计
    for bad in ('urllib', 'requests.', 'wget', 'gdown'):
        assert bad not in src, f'unexpected network marker: {bad}'
    toplevel = set()
    for line in re.findall(r'^\s*import ([^\n]+)$', src, re.M):
        for t in line.split(','):
            t = t.strip().split(' as ')[0].split('.')[0]
            if t:
                toplevel.add(t)
    for line in re.findall(r'^\s*from (\S+) import', src, re.M):
        toplevel.add(line.split('.')[0])
    std_ok = {'__future__', 'numpy', 'pandas', 'pydicom', 'torch', 'torchvision',
              'cv2', 'timm', 'transformers', 'scipy', 'sklearn', 'matplotlib',
              'concurrent', 'pathlib', 'hashlib', 'json', 're', 'time',
              'traceback', 'threading', 'os', 'gc', 'warnings', 'contextlib',
              'subprocess', 'math', 'collections', 'itertools', 'functools',
              'typing', 'sys'}
    unknown = {t for t in toplevel if t not in std_ok}
    assert not unknown, f'unexpected imports: {sorted(unknown)}'
    print('D OK: no network deps; imports all standard-image libraries')

    # E. ASSET 期望文件清单 (对照 tonylica 数据集 API 文件列表核过)
    expect = [
        'knee-mri-fold-weights/m_f0.pt ... m_f4.pt   (DINOv3 5 folds)',
        'rsna-knee-weights/* (20 DINOv2 ckpts + manifest.json)',
        'resnet-50-radimagenet-marwan/ResNet50.pt   (rad 编码器, sha 校验)',
        'rsna-knee-e9-radimagenet-heads-v15/v52_radimagenet_heads.pt  (5 参考头, sha 校验)',
        'kernel-sources/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt  (5 E13 头, sha 校验)',
    ]
    print('E ASSET expected files (verified against Kaggle API 2026-08-19):')
    for e in expect:
        print('   -', e)

    print('\nALL VERIFY CHECKS PASSED')


if __name__ == '__main__':
    main()
