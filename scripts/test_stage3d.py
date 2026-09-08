"""CPU tests for generated notebook data joins, ranking gradients and training."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


class Stage3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads((ROOT / 'notebooks/kaggle_train_v5_stage3d_rank.ipynb').read_text(encoding='utf-8'))
        cls.sources = [''.join(c['source']) for c in cls.nb['cells'] if c['cell_type'] == 'code']
        cls.env = {'torch': torch, 'nn': torch.nn, 'F': F, 'IS_MAIN': False}
        exec(next(s for s in cls.sources if 'class WeightedSoftBCELoss' in s), cls.env)

    def test_notebook_control_only_lambda_differs(self):
        control = json.loads((ROOT / 'notebooks/kaggle_train_v5_stage3d_control.ipynb').read_text(encoding='utf-8'))
        for a, b in zip(self.nb['cells'][1:], control['cells'][1:]):
            sa = ''.join(a['source']).replace('stage3d_rank', 'stage3d_control').replace('rank_lambda=0.05', 'rank_lambda=0.0')
            self.assertEqual(sa, ''.join(b['source']))
        for source in self.sources:
            compile(source, 'generated', 'exec')

    def test_direction_mask_and_empty_pairs(self):
        fn = self.env['trusted_pair_rank_loss']
        states = torch.tensor([[1], [-1]], dtype=torch.int8)
        logits = torch.zeros(2, 1, requires_grad=True)
        loss, count = fn(logits, states, torch.ones(2, 1))
        loss.backward()
        self.assertEqual(count, [1])
        self.assertLess(logits.grad[0].item(), 0)
        self.assertGreater(logits.grad[1].item(), 0)
        good, _ = fn(torch.tensor([[2.], [-2.]]), states, torch.ones(2, 1))
        self.assertLess(good.item(), loss.item())
        empty, count = fn(logits, states, torch.zeros(2, 1))
        self.assertEqual(empty.item(), 0)
        self.assertEqual(count, [0])
        empty.backward()

    def test_preflight_and_original_supervision(self):
        config_source = next(s for s in self.sources if 'TARGET_COLUMNS =' in s)
        # Get the original config definitions without Kaggle path resolution.
        env = {'Path': Path, 'pd': pd, 'np': np, 'IS_MAIN': False,
               'torch': torch, 'random': __import__('random'), 'os': __import__('os')}
        # Constants are defined by the checked-in config, with logging globals.
        env.update(N_GPUS=0, DEVICE='cpu')
        exec(config_source.split('# Resolve documented')[0], env)
        with tempfile.TemporaryDirectory() as tmp:
            env['CFG'].update(comp_input=str(ROOT / 'data/metadata'),
                              label_input=str(ROOT / 'data/processed'),
                              trust_input=str(ROOT / 'data/processed/stage3d_trust'), output_dir=tmp)
            exec(next(s for s in self.sources if 'train_meta = pd.read_csv' in s), env)
            old = pd.read_csv(ROOT / 'data/processed/v5_labels.csv').set_index('StudyInstanceUID')
            train = env['train_labels']
            self.assertEqual(len(train), 4349)
            self.assertFalse(set(train.index) & set(env['val_labels'].index))
            cols = env['PROB_COLS'] + env['WEIGHT_COLS'] + env['MASK_COLS']
            np.testing.assert_allclose(train[cols], old.loc[train.index, cols], rtol=1e-6)
            self.assertEqual(len(env['_active_classes']), 8)
            self.assertTrue((train['rank_Synovitis'] == 0).all())

    def test_training_cpu_and_zero_lambda_gradient(self):
        env = dict(self.env, np=np, json=json, Path=Path, DEVICE='cpu', TARGET_COLUMNS=['a', 'b'])
        exec(next(s for s in self.sources if 'def train_epoch' in s), env)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(3, 2)
            def forward(self, x, mask):
                return self.layer(x)
        model = Toy()
        criterion = env['WeightedSoftBCELoss']()
        batch = {'slots': torch.randn(6, 3), 'mask': torch.ones(6, 1),
                 'prob_targets': torch.rand(6, 2), 'weights': torch.ones(6, 2),
                 'soft_masks': torch.ones(6, 2), 'rank_states': torch.tensor([[1, 0], [-1, 0]] * 3)}
        with tempfile.TemporaryDirectory() as tmp:
            env['CFG'] = dict(grad_accum_steps=2, rank_margin=.1, rank_lambda=.05, grad_clip=1., output_dir=tmp)
            before = model.layer.weight.detach().clone()
            loss = env['train_epoch'](model, [batch, batch], torch.optim.SGD(model.parameters(), lr=.1), criterion, None, 1)
            self.assertTrue(np.isfinite(loss))
            self.assertFalse(torch.equal(before, model.layer.weight))
            record = json.loads((Path(tmp) / 'ranking_history.jsonl').read_text())
            self.assertEqual(record['pairs_by_class'], {'a': 18, 'b': 0})
        logits = torch.randn(6, 2, requires_grad=True)
        bce = criterion(logits, batch['prob_targets'], batch['weights'], batch['soft_masks'])
        rank, _ = env['trusted_pair_rank_loss'](logits, batch['rank_states'], batch['soft_masks'])
        ga = torch.autograd.grad(bce, logits, retain_graph=True)[0]
        gb = torch.autograd.grad(bce + 0 * rank, logits)[0]
        torch.testing.assert_close(ga, gb, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
