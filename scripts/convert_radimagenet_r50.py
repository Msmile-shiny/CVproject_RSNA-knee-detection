"""Convert official RadImageNet ResNet50 Keras .h5 (notop) → torchvision resnet50 state_dict.

来源: Kaggle Dataset ipythonx/notop-wg-radimagenet (官方 Google Drive 镜像,
RadImageNet-ResNet50_notop.h5)。Keras 命名: conv{s}_block{j}_{k}_conv/_bn
(k=0 = shortcut, k=1..3 = bottleneck convs), conv1_conv/conv1_bn。
输出: 'backbone.{0..7}.*' 前缀 state_dict, 与 torchvision resnet50 (fc→Identity) 严格对应。

bias 吸收: Keras 的 conv 带非零 bias (conv1 max abs 0.067), torchvision 的 conv
为 bias=False → 将 bias 吸收进 BN running_mean (μ' = μ − b), 冻结编码器仅 eval
推理时数学严格等价 (见 tmp_verify_absorb 数值验证)。

交叉验证注记 (2026-08-15): HF Lab-Rasool/RadImageNet 的 ResNet50.pt 经核验
与官方 h5 不同源 (conv1 展平相关仅 0.03, 值域差 ~3x, 无任何轴排列可匹配,
bn1 gamma 亦不同) → 不再与其交叉验证, 官方 h5 为唯一权威来源。
"""

import io
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision

STAGE_BLOCKS = {2: 3, 3: 4, 4: 6, 5: 3}  # Keras ResNet50 per-stage block counts


def load_keras_h5(h5_path):
    """h5_path: .h5 文件或包含 RadImageNet-ResNet50_notop.h5 的 .zip"""
    p = Path(h5_path)
    if p.suffix == '.zip':
        with zipfile.ZipFile(p) as z:
            data = z.read('RadImageNet-ResNet50_notop.h5')
        return h5py.File(io.BytesIO(data), 'r')
    return h5py.File(p, 'r')


def keras_weights(h5):
    """返回 {layer_name: {tensor_name: np.ndarray}} (key 如 conv2_block1_1_conv)。"""
    out = {}
    root = h5['model_weights']
    for layer in root:
        grp = root[layer]
        if isinstance(grp, h5py.Group):   # 一层同名嵌套 (model_weights/conv1_conv/conv1_conv)
            try:
                grp = grp[layer]
            except KeyError:
                continue                 # 无权重层 (如 conv1_pad/pool)
        for tensor in grp:
            out.setdefault(layer, {})[tensor.split(':')[0]] = np.array(grp[tensor])
    return out


def conv_to_torch(w):
    """Keras [kh,kw,cin,cout] → torch [cout,cin,kh,kw]"""
    return torch.from_numpy(w.transpose(3, 2, 0, 1).copy())


def convert(h5_path, out_path, verify_pt=None):
    h = load_keras_h5(h5_path)
    w = keras_weights(h)

    # ---- 结构断言: 各 stage block 数与标准 ResNet50 一致 ----
    for s, n_blocks in STAGE_BLOCKS.items():
        found = sorted({int(k.split('_')[1].replace('block', ''))
                        for k in w if k.startswith(f'conv{s}_')})
        assert found == list(range(1, n_blocks + 1)), \
            f'stage {s}: blocks {found} != 1..{n_blocks}'

    sd = {}
    # conv1 / bn1 (bias 先记录, 之后吸收进 BN running_mean)
    sd['conv1.weight'] = conv_to_torch(w['conv1_conv']['kernel'])
    sd['conv1.bias'] = torch.from_numpy(w['conv1_conv']['bias'].copy())
    for tname, kname in [('gamma', 'weight'), ('beta', 'bias'),
                         ('moving_mean', 'running_mean'),
                         ('moving_variance', 'running_var')]:
        sd[f'bn1.{kname}'] = torch.from_numpy(w['conv1_bn'][tname].copy())

    # layers: stage s (keras conv2..conv5) → torchvision layer idx (0..3)
    #   conv2(3块)→layer1, conv3(4块)→layer2, conv4(6块)→layer3, conv5(3块)→layer4
    for s, n_blocks in STAGE_BLOCKS.items():
        layer_idx = s - 1
        for j in range(1, n_blocks + 1):
            pre = f'layer{layer_idx}.{j - 1}'
            for k in (1, 2, 3):
                base = f'conv{s}_block{j}_{k}'
                sd[f'{pre}.conv{k}.weight'] = conv_to_torch(w[f'{base}_conv']['kernel'])
                sd[f'{pre}.conv{k}.bias'] = torch.from_numpy(w[f'{base}_conv']['bias'].copy())
                for tname, kname in [('gamma', 'weight'), ('beta', 'bias'),
                                     ('moving_mean', 'running_mean'),
                                     ('moving_variance', 'running_var')]:
                    sd[f'{pre}.bn{k}.{kname}'] = torch.from_numpy(
                        w[f'{base}_bn'][tname].copy())
            if f'conv{s}_block{j}_0_conv' in w:  # stride-2 块有 shortcut
                base = f'conv{s}_block{j}_0'
                sd[f'{pre}.downsample.0.weight'] = conv_to_torch(w[f'{base}_conv']['kernel'])
                sd[f'{pre}.downsample.0.bias'] = torch.from_numpy(w[f'{base}_conv']['bias'].copy())
                for tname, kname in [('gamma', 'weight'), ('beta', 'bias'),
                                     ('moving_mean', 'running_mean'),
                                     ('moving_variance', 'running_var')]:
                    sd[f'{pre}.downsample.1.{kname}'] = torch.from_numpy(
                        w[f'{base}_bn'][tname].copy())
    h.close()

    assert len(sd) == 318, f'expected 318 tensors (with biases, no num_batches_tracked), got {len(sd)}'

    # ---- ★ conv bias 吸收进 BN running_mean (推理期数学等价, 输出 318 键无偏置) ----
    #   Keras: y = BN(conv(x) + b) = γ·(conv(x) + b − μ)/σ + β
    #   = γ·(conv(x) − (μ − b))/σ + β  → torch 无偏置 conv + BN running_mean' = μ − b
    #   冻结编码器仅 eval 推理 → 严格等价; HF 版直接丢弃 bias 未补偿 = 有损
    absorbed = 0
    for k in [k for k in sd if k.endswith('.bias')
              and (k == 'conv1.bias' or '.conv' in k or '.downsample.0.bias' in k)]:
        bn_mean_key = None
        parts = k.split('.')
        if parts[0] == 'conv1':
            bn_mean_key = 'bn1.running_mean'
        elif parts[0].startswith('layer'):
            block = parts[1]
            if 'downsample' in parts:
                bn_mean_key = f'layer{parts[0][5:]}.{block}.downsample.1.running_mean'
            else:
                bn_mean_key = f'layer{parts[0][5:]}.{block}.bn{parts[2][4:]}.running_mean'
        assert bn_mean_key in sd, f'no BN for {k}'
        sd[bn_mean_key] = sd[bn_mean_key] - sd.pop(k)
        absorbed += 1
    # 补 torchvision BN 的 num_batches_tracked (Keras 无此 buffer; 冻结 eval 用不到)
    for k in [k for k in sd if k.endswith('.running_var')]:
        sd[k.replace('running_var', 'num_batches_tracked')] = torch.zeros(1, dtype=torch.long)
    assert len(sd) == 318, f'expected 318 tensors after absorption, got {len(sd)}'
    print(f'bias absorbed into BN running_mean: {absorbed} conv layers (exact for eval mode)')

    # ---- 结构验证: strict load 进 torchvision resnet50 (去 fc) ----
    r50 = torchvision.models.resnet50(weights=None)
    r50.fc = torch.nn.Identity()
    r50.load_state_dict(sd, strict=True)
    print(f'strict load OK ({len(sd)} tensors) → torchvision resnet50 (fc removed)')

    # ---- 保存格式: 'backbone.{child_idx}.*' 前缀 (与 HF Lab-Rasool 版同构) ----
    #   torchvision resnet50 children: 0=conv1 1=bn1 2=relu 3=maxpool 4..7=layer1..4
    def to_backbone_keys(d):
        out = {}
        for k, v in d.items():
            if k.startswith('conv1.'):
                out['backbone.0.' + k[len('conv1.'):]] = v
            elif k.startswith('bn1.'):
                out['backbone.1.' + k[len('bn1.'):]] = v
            elif k.startswith('layer'):
                li = int(k[len('layer'):].split('.')[0])
                out[f'backbone.{3 + li}.' + k[len(f'layer{li}.'):]] = v
            else:
                raise KeyError(k)
        return out

    sd = to_backbone_keys(sd)

    # ---- verify_pt: 仅统计报告 (HF 版已确认与官方 h5 不同源, 不再断言) ----
    if verify_pt:
        ref = torch.load(verify_pt, map_location='cpu', weights_only=True)
        if 'state_dict' in ref:
            ref = ref['state_dict']
        if set(sd) == set(ref):
            n_same = sum(1 for k in sd
                         if torch.equal(sd[k].to(torch.float32), ref[k].to(torch.float32)))
            print(f'verify_pt (report-only): key sets match, {n_same}/{len(sd)} '
                  f'tensors bitwise equal — 已知不同源, 不等属预期, 以官方 h5 为准')
        else:
            print('verify_pt: key set mismatch (结构不同), 忽略')

    torch.save(sd, out_path)
    print(f'saved: {out_path} ({Path(out_path).stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    h5_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        h5_path.parent / 'radimagenet_resnet50_notop.pt'
    verify_pt = sys.argv[3] if len(sys.argv) > 3 else None
    convert(h5_path, out_path, verify_pt)
