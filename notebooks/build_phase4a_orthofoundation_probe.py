"""Build a controlled OrthoFoundation-L frozen-backbone probe for Kaggle."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'kaggle_train_phase4a_orthofoundation_probe.ipynb'
LABEL_SHA = 'c13adffaabf4f8e518abb038282bb1aa09baac7652a9165e030710c457d0be6a'


def cell(kind, source):
    result = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(True), 'id': f'p4a-{cell.counter:02d}'}
    cell.counter += 1
    if kind == 'code':
        compile(source, result['id'], 'exec')
        result.update(execution_count=None, outputs=[])
    return result
cell.counter = 0


ORTHO_BUILD = r'''
# Phase 4A: frozen OrthoFoundation-L backbone. Fail closed on incompatible assets.
if CFG['dinov2_variant'] not in timm.list_models():
    raise RuntimeError(f"Installed timm {timm.__version__} lacks {CFG['dinov2_variant']}; "
                       "use a Kaggle image with timm>=1.0.20 or mount an offline wheel")

def _find_file(name):
    hits = list(Path('/kaggle/input').rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f'Expected exactly one {name} below /kaggle/input; found {hits}')
    return hits[0]

weights_path = _find_file('OrthoFoudation-L.pth')
if weights_path.stat().st_size < 100_000_000:
    raise RuntimeError('OrthoFoundation file is too small; upload the Git-LFS binary, not its pointer')

if IS_MAIN: print('Creating timm DINOv3-L architecture for OrthoFoundation...')
backbone = timm.create_model(CFG['dinov2_variant'], pretrained=False, num_classes=0,
                             img_size=CFG['image_size'])
raw = torch.load(weights_path, map_location='cpu', weights_only=True)

def _tensor_dicts(obj, path='root'):
    out = []
    if isinstance(obj, dict):
        tensors = {str(k): v for k, v in obj.items() if torch.is_tensor(v)}
        if tensors:
            out.append((path, tensors))
        for k, v in obj.items():
            if isinstance(v, dict):
                out.extend(_tensor_dicts(v, path + '.' + str(k)))
    return out

target = backbone.state_dict()
anchors = ('patch_embed.proj.weight', 'cls_token', 'blocks.0.norm1.weight')
best = None
for container, source in _tensor_dicts(raw):
    prefixes = {''}
    for key in source:
        for anchor in anchors:
            if key.endswith(anchor):
                prefixes.add(key[:-len(anchor)])
    for prefix in prefixes:
        mapped = {k[len(prefix):]: v for k, v in source.items() if k.startswith(prefix)}
        matched = {k: v for k, v in mapped.items() if k in target and v.shape == target[k].shape}
        numel = sum(target[k].numel() for k in matched)
        candidate = (numel, len(matched), container, prefix, matched)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

total_numel = sum(v.numel() for v in target.values())
coverage = best[0] / total_numel
audit = {'checkpoint': str(weights_path), 'checkpoint_bytes': weights_path.stat().st_size,
         'container': best[2], 'prefix': best[3], 'matched_tensors': best[1],
         'target_tensors': len(target), 'parameter_coverage': coverage}
print('OrthoFoundation load audit:', audit)
Path(CFG['output_dir']).mkdir(parents=True, exist_ok=True)
(Path(CFG['output_dir']) / 'phase4a_weight_audit.json').write_text(json.dumps(audit, indent=2))
if coverage != 1.0 or best[1] != len(target):
    raise RuntimeError(f'OrthoFoundation checkpoint is not an exact architecture match: '
                       f'coverage={coverage:.3%}, tensors={best[1]}/{len(target)}. '
                       'Do not train with partially initialized medical weights.')
missing, unexpected = backbone.load_state_dict(best[4], strict=False)
audit['missing'] = missing
audit['unexpected'] = unexpected
del raw
gc.collect()

model = MultiViewModel(
    dinov2_model=backbone, n_slots=N_SLOT, cls_dim=CFG['cls_dim'],
    n_classes=CFG['num_classes'], slot_hidden=CFG['slot_hidden'],
    dropout=CFG['dropout'], unfreeze_layers=0).to(DEVICE)
if N_GPUS > 1:
    model = nn.DataParallel(model)

head_params = [p for p in model.parameters() if p.requires_grad]
assert head_params and all(not p.requires_grad for p in (model.module if N_GPUS > 1 else model).dinov2.parameters())
optimizer = torch.optim.AdamW(head_params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
criterion = WeightedSoftBCELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=CFG['lr_t0'], T_mult=CFG['lr_t_mult'], eta_min=CFG['lr_eta_min'])
scaler = torch.amp.GradScaler('cuda') if CFG['mixed_precision'] else None
ema = EMAModel(model.module if N_GPUS > 1 else model, decay=CFG['ema_decay'])
(Path(CFG['output_dir']) / 'phase4a_weight_audit.json').write_text(json.dumps(audit, indent=2))
print(f'Phase 4A model ready: frozen OrthoFoundation-L, trainable head={sum(p.numel() for p in head_params)/1e6:.2f}M')
'''


def main():
    assert hashlib.sha256((ROOT.parent / 'data/processed/v5_labels.csv').read_bytes()).hexdigest() == LABEL_SHA
    cells = [cell('markdown', '''# Phase 4A — OrthoFoundation-L controlled probe

Independent member screen: knee-MRI-specific DINOv3-L initialization, historical v5 supervision, frozen backbone. The 9-slice/130-mm input remains fixed so this run isolates representation transfer. Mount competition data, `rsna-knee-v5-labels`, the public OrthoFoundation weight dataset, and an offline copy of the official `facebookresearch/dinov3` source tree. T4 x2.

This is a screening experiment. Dense 64–96-slice MIL is Phase 4B only if this backbone adds signal.
''')]
    # Load and validate the large foundation model before the ~80-minute DICOM cache.
    # Definitions in cells 12/13 do not depend on the cache or dataloaders.
    def phase4_order(path):
        number = int(path.name[:2])
        return {12: 10, 13: 11, 10: 12, 11: 13}.get(number, number)

    for path in sorted((ROOT / 'cells_v5').iterdir(), key=phase4_order):
        if path.suffix not in ('.py', '.md'):
            continue
        source = path.read_text(encoding='utf-8')
        if path.name == '03_config.py':
            source = source.replace("'image_size': 288,", "'image_size': 256,")
            source = source.replace("'dinov2_variant': 'vit_small_patch14_dinov2.lvd142m',", "'dinov2_variant': 'vit_large_patch16_dinov3',")
            source = source.replace("'cls_dim': 384,", "'cls_dim': 1024,")
            source = source.replace("'feature_dim': 1152,", "'feature_dim': 3072,")
            source = source.replace("'unfreeze_layers': 6,", "'unfreeze_layers': 0,")
            source = source.replace("'batch_size': 6,", "'batch_size': 2,")
            source = source.replace("'grad_accum_steps': 2,", "'grad_accum_steps': 6,")
            source = source.replace("'epochs': 30,", "'epochs': 20,")
            source += "\nCFG.update(experiment_name='phase4a_orthofoundation_probe', tta_jitter=False)\n"
        elif path.name == '06_model.py':
            source = source.replace(
                "        n_blocks = len(self.dinov2.blocks)\n        if unfreeze_layers > 0:\n            for p in self.dinov2.parameters():\n                p.requires_grad = False",
                "        n_blocks = len(self.dinov2.blocks)\n        # Always freeze first; selectively reopen only the requested final blocks.\n        for p in self.dinov2.parameters():\n            p.requires_grad = False\n        if unfreeze_layers > 0:")
            source = source.replace(
                "        patches = features[:, 1:, :]",
                "        if isinstance(features, dict):\n            cls = features['x_norm_clstoken']\n            patches = features['x_norm_patchtokens']\n        else:\n            cls = features[:, 0, :]\n            n_prefix = int(getattr(self.dinov2, 'num_prefix_tokens', 1))\n            patches = features[:, n_prefix:, :]\n        if patches.shape[1] == 0:\n            raise RuntimeError('Backbone returned no image patch tokens')")
            source = source.replace("        cls = features[:, 0, :]\n        if isinstance(features, dict):", "        if isinstance(features, dict):")
        elif path.name == '09_load_data.py':
            marker = 'fused_df = pd.read_csv(v5_label_file)'
            guard = f"import hashlib\nassert hashlib.sha256(v5_label_file.read_bytes()).hexdigest() == {LABEL_SHA!r}, 'Wrong historical v5 labels'\n"
            source = source.replace(marker, guard + marker)
        elif path.name == '13_build_model.py':
            source = (ROOT / 'phase4a_official_build.py').read_text(encoding='utf-8')
        elif path.name == '15_training_loop.py':
            source = source.replace('288px / Physical crop', '256px / Physical crop')
        elif path.name == '17_submission.py':
            source = source.replace(
                "infer_backbone = timm.create_model(\n    CFG['dinov2_variant'], pretrained=False, num_classes=0, img_size=CFG['image_size'])",
                "infer_backbone = build_official_dinov3_backbone(load_medical_weights=False)")
            source += "\n_phase4a = {'experiment': CFG['experiment_name'], 'gold_macro_auc': gold_macro, 'best_epoch': best_epoch, 'weight_audit': audit, 'label_sha256': '" + LABEL_SHA + "'}\n(output_dir / 'phase4a_manifest.json').write_text(json.dumps(_phase4a, indent=2))\n"
        cells.append(cell('code' if path.suffix == '.py' else 'markdown', source))
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': cells}
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
