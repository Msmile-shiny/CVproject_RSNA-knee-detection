import json
import unittest
from pathlib import Path

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
        self.assertIn('import json', code)

        sources = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        model_build = next(i for i, source in enumerate(sources) if 'Creating official DINOv3-L' in source)
        cache_build = next(i for i, source in enumerate(sources) if 'Studies to cache:' in source)
        self.assertLess(model_build, cache_build)

    def test_center_repeat_changes_only_declared_input_contract(self):
        base = json.loads((ROOT / 'notebooks/kaggle_train_phase4a_orthofoundation_probe.ipynb').read_text(encoding='utf-8'))
        aligned = json.loads((ROOT / 'notebooks/kaggle_train_phase4a2_orthofoundation_center_repeat.ipynb').read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in aligned['cells'] if c['cell_type'] == 'code')
        self.assertIn("channel_mode='center_repeat'", code)
        repeat = code.index("x = x[:, 1:2].expand(-1, 3, -1, -1)")
        normalize = code.index("x = x.float().div_(255.0)", repeat)
        self.assertLess(repeat, normalize)
        self.assertIn("phase4a2_manifest.json", code)
        self.assertIn('backbone.load_state_dict(state, strict=True)', code)
        for cell in aligned['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
        self.assertEqual(len(base['cells']), len(aligned['cells']))

if __name__ == '__main__':
    unittest.main()
