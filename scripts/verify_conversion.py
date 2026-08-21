# ============================================================
# 数值验证: HF dinov2-small vs 转换后的 timm 权重，输出是否一致
# 若 max abs diff ~ 1e-6 则转换完全正确
# ============================================================
import torch
import timm
from transformers import AutoModel

HF_DIR = 'kaggle_dataset/dinov2-small'
TMM_PTH = 'data/dinov2_vits14.pth'

torch.manual_seed(0)
x = torch.randn(1, 3, 518, 518)   # 原生 518px，无 pos_embed 插值

# ---- HF ----
hf_model = AutoModel.from_pretrained(HF_DIR, local_files_only=True)
hf_model.eval()
with torch.no_grad():
    hf_out = hf_model(pixel_values=x)
print('HF output keys:', list(hf_out.keys()) if hasattr(hf_out, 'keys') else type(hf_out))
hf_seq = hf_out.last_hidden_state  # (1, 1370, 384)，已过最终 layernorm

# ---- timm (转换后权重) ----
timm_model = timm.create_model('vit_small_patch14_dinov2.lvd142m',
                               pretrained=False, num_classes=0)
sd = torch.load(TMM_PTH, map_location='cpu', weights_only=True)
timm_model.load_state_dict(sd, strict=True)
timm_model.eval()
with torch.no_grad():
    timm_seq = timm_model.forward_features(x)  # (1, 1370, 384)，已过最终 norm

print('hf seq shape  :', tuple(hf_seq.shape))
print('timm seq shape:', tuple(timm_seq.shape))

diff = (hf_seq - timm_seq).abs()
print('\nmax abs diff  :', diff.max().item())
print('mean abs diff :', diff.mean().item())
print('CLS max diff  :', (hf_seq[:, 0] - timm_seq[:, 0]).abs().max().item())
print('norm eps: hf =', hf_model.config.layer_norm_eps,
      '| timm =', timm_model.norm.eps)

ok = diff.max().item() < 1e-3
print('\nRESULT:', 'PASS (转换正确)' if ok else 'FAIL (有差异，需排查)')
