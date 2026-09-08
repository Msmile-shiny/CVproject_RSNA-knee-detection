import json
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


class TestPhase4A(unittest.TestCase):
    def test_generated_contract(self):
        nb = json.loads((ROOT / 'notebooks/kaggle_train_phase4a_orthofoundation_probe.ipynb').read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), c['id'], 'exec')
        self.assertIn("'dinov2_variant': 'vit_large_patch16_dinov3'", code)
        self.assertIn("'cls_dim': 1024", code)
        self.assertIn("'unfreeze_layers': 0", code)
        self.assertIn('backbone.load_state_dict(state, strict=True)', code)
        self.assertIn("tta_jitter=False", code)
        self.assertIn("features['x_norm_patchtokens']", code)
        freeze = code.index('for p in self.dinov2.parameters():')
        conditional = code.index('if unfreeze_layers > 0:', freeze)
        self.assertLess(freeze, conditional)
        self.assertIn('weights_only=True', code)
        self.assertIn('build_official_dinov3_backbone(load_medical_weights=False)', code)

    def test_prefix_selection_algorithm(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        target = model.state_dict()
        source = {'teacher.backbone.' + k: v.clone() for k, v in target.items()}
        prefix = 'teacher.backbone.'
        mapped = {k[len(prefix):]: v for k, v in source.items() if k.startswith(prefix)}
        matched = {k: v for k, v in mapped.items() if k in target and v.shape == target[k].shape}
        self.assertEqual(set(matched), set(target))
        model.load_state_dict(matched, strict=True)


if __name__ == '__main__':
    unittest.main()
