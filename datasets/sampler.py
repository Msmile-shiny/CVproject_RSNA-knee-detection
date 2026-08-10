"""Gold/Pseudo Balanced Batch Sampler.

每个 batch 固定 gold:pseudo 比例, gold 样本被循环过采样,
pseudo 样本逐 epoch 随机打乱.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class GoldPseudoSampler(Sampler):
    """Fixed-ratio batch sampler for gold + pseudo label training.

    每个 batch 中 gold 样本占固定比例 (默认 1/8), 剩余为 pseudo 样本.
    gold 少量时会循环采样, pseudo 按正常顺序遍历.

    Args:
        gold_indices: gold 样本在 dataset 中的索引列表
        pseudo_indices: pseudo 样本在 dataset 中的索引列表
        batch_size: 每 batch 总样本数
        gold_ratio: gold 样本占比 (默认 0.125 → bs=8 时每 batch 1 条 gold)
        shuffle: 是否打乱 pseudo 顺序
        drop_last: 是否丢弃最后不完整的 batch
    """

    def __init__(
        self,
        gold_indices: list[int] | np.ndarray,
        pseudo_indices: list[int] | np.ndarray,
        batch_size: int = 8,
        gold_ratio: float = 0.125,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if not gold_indices:
            raise ValueError("gold_indices 不能为空")
        if not pseudo_indices:
            raise ValueError("pseudo_indices 不能为空")

        self.gold_indices = np.asarray(gold_indices, dtype=np.int64)
        self.pseudo_indices = np.asarray(pseudo_indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        # 每 batch 中 gold 数量
        self.gold_per_batch = max(1, int(batch_size * gold_ratio))
        self.pseudo_per_batch = batch_size - self.gold_per_batch

        # 总 batch 数
        n_full = len(self.pseudo_indices) // self.pseudo_per_batch
        remainder = len(self.pseudo_indices) % self.pseudo_per_batch
        self._n_batches = n_full + (1 if remainder > 0 and not drop_last else 0)

    def __iter__(self):
        pseudo_order = np.random.permutation(self.pseudo_indices) if self.shuffle \
            else self.pseudo_indices.copy()
        gold_order = np.random.permutation(self.gold_indices) if self.shuffle \
            else self.gold_indices.copy()

        gold_ptr = 0
        pseudo_ptr = 0
        rng = np.random.default_rng()

        for batch_idx in range(self._n_batches):
            batch = []

            # gold: 循环采样 (少量 gold 会重复出现)
            for _ in range(self.gold_per_batch):
                batch.append(int(gold_order[gold_ptr % len(gold_order)]))
                gold_ptr += 1
                if gold_ptr >= len(gold_order):
                    gold_order = rng.permutation(self.gold_indices)
                    gold_ptr = 0

            # pseudo: 顺序遍历 (不打乱 gold 的相对顺序)
            for _ in range(self.pseudo_per_batch):
                if pseudo_ptr >= len(pseudo_order):
                    break
                batch.append(int(pseudo_order[pseudo_ptr]))
                pseudo_ptr += 1

            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        return self._n_batches

    @property
    def effective_gold_ratio(self) -> float:
        """实际 gold 占比."""
        return self.gold_per_batch / self.batch_size
