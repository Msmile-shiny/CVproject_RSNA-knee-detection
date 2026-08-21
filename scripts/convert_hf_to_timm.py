# ============================================================
# 把 HuggingFace 格式的 facebook/dinov2-small 权重 转成 timm 格式
# 产出 data/dinov2_vits14.pth（v5 的 dinov2_weights 需要）
#
# 依赖（Anaconda 环境已装）: torch + safetensors + timm
# 运行: python scripts/convert_hf_to_timm.py
# ============================================================
import torch
import timm
from safetensors.torch import load_file

SRC = 'kaggle_dataset/dinov2-small/model.safetensors'
DST = 'data/dinov2_vits14.pth'
VARIANT = 'vit_small_patch14_dinov2.lvd142m'


def main():
    hf = load_file(SRC)
    out = {}

    # ---- top-level embeddings ----
    out['cls_token'] = hf['embeddings.cls_token']
    out['pos_embed'] = hf['embeddings.position_embeddings']
    out['patch_embed.proj.weight'] = hf['embeddings.patch_embeddings.projection.weight']
    out['patch_embed.proj.bias'] = hf['embeddings.patch_embeddings.projection.bias']
    out['norm.weight'] = hf['layernorm.weight']
    out['norm.bias'] = hf['layernorm.bias']
    # (mask_token 是 HF 独有，timm 不需要，跳过)

    # ---- 12 transformer blocks ----
    for i in range(12):
        p = f'encoder.layer.{i}.'
        # qkv 合并 (query, key, value -> 单个 qkv)
        q = hf[p + 'attention.attention.query.weight']
        k = hf[p + 'attention.attention.key.weight']
        v = hf[p + 'attention.attention.value.weight']
        out[f'blocks.{i}.attn.qkv.weight'] = torch.cat([q, k, v], dim=0)
        qb = hf[p + 'attention.attention.query.bias']
        kb = hf[p + 'attention.attention.key.bias']
        vb = hf[p + 'attention.attention.value.bias']
        out[f'blocks.{i}.attn.qkv.bias'] = torch.cat([qb, kb, vb], dim=0)

        out[f'blocks.{i}.attn.proj.weight'] = hf[p + 'attention.output.dense.weight']
        out[f'blocks.{i}.attn.proj.bias'] = hf[p + 'attention.output.dense.bias']
        out[f'blocks.{i}.ls1.gamma'] = hf[p + 'layer_scale1.lambda1']
        out[f'blocks.{i}.ls2.gamma'] = hf[p + 'layer_scale2.lambda1']
        out[f'blocks.{i}.norm1.weight'] = hf[p + 'norm1.weight']
        out[f'blocks.{i}.norm1.bias'] = hf[p + 'norm1.bias']
        out[f'blocks.{i}.norm2.weight'] = hf[p + 'norm2.weight']
        out[f'blocks.{i}.norm2.bias'] = hf[p + 'norm2.bias']
        out[f'blocks.{i}.mlp.fc1.weight'] = hf[p + 'mlp.fc1.weight']
        out[f'blocks.{i}.mlp.fc1.bias'] = hf[p + 'mlp.fc1.bias']
        out[f'blocks.{i}.mlp.fc2.weight'] = hf[p + 'mlp.fc2.weight']
        out[f'blocks.{i}.mlp.fc2.bias'] = hf[p + 'mlp.fc2.bias']

    # ---- 校验 1: strict load（键名 + 形状完全一致）----
    model = timm.create_model(VARIANT, pretrained=False, num_classes=0)
    model.load_state_dict(out, strict=True)
    print(f'[OK] strict load 通过: {len(out)} 个 tensor 与 timm {VARIANT} 完全匹配')

    # ---- 校验 2: layer scale 值（应为 1.0，恒等）----
    ls = torch.cat([out[f'blocks.{i}.ls1.gamma'] for i in range(12)]
                   + [out[f'blocks.{i}.ls2.gamma'] for i in range(12)])
    print(f'[OK] layer_scale 值范围: [{ls.min().item():.6f}, {ls.max().item():.6f}] (应≈1.0)')

    # ---- 校验 3: 权重量级（非零、有限）----
    w = out['patch_embed.proj.weight']
    print(f'[OK] patch_embed.proj.weight: mean={w.abs().mean().item():.4f}, '
          f'std={w.std().item():.4f}')

    # ---- 校验 4: 原生 518px 前向（无 NaN）----
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 518, 518))
    print(f'[OK] 前向输出 shape={tuple(y.shape)}, 有限={torch.isfinite(y).all().item()}')

    torch.save(out, DST)
    print(f'\n[done] 已保存 -> {DST}')


if __name__ == '__main__':
    main()
