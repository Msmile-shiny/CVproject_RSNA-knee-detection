"""CPU checks for the candidate pooling; no experiment launch or data needed."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
from stage6_resnet_core import KneeResNet, masked_top_half


class TopHalfTests(unittest.TestCase):
    def test_ragged_mask_and_gradient(self):
        x = torch.tensor([[[1.], [3.], [5.], [100.]], [[2.], [100.], [100.], [100.]]], requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
        result = masked_top_half(x, mask)
        torch.testing.assert_close(result, torch.tensor([[4.], [2.]]))
        result.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(x.grad[~mask].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            masked_top_half(x, torch.zeros_like(mask))

    def test_single_window_matches_mean_and_checkpoint_compatible(self):
        torch.set_num_threads(2)
        mean = KneeResNet().eval()
        candidate = KneeResNet(pooling='top_half').eval()
        candidate.load_state_dict(mean.state_dict(), strict=True)
        images = torch.randn(1, 2, 3, 32, 32)
        valid = torch.tensor([[True, False]])
        slots = torch.tensor([[0, 1]])
        with torch.no_grad():
            torch.testing.assert_close(mean(images, valid, slots), candidate(images, valid, slots))
        candidate(images, valid, slots).sum().backward()
        self.assertGreater(candidate.encoder.conv1.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
