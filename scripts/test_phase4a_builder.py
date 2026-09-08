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

if __name__ == '__main__':
    unittest.main()
