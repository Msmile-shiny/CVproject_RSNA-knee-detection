# ============================================================
# v6a: MultiViewDataset — 6-slot clinical MRI (224px, RadImageNet R50)
# ============================================================

class MultiViewDataset(Dataset):
    """Multi-view knee MRI dataset with 6 clinical slots.

    训练：随机 3-slice 窗口 + 数据增强标志
    验证/测试：固定中间窗口
    """

    def __init__(self, study_uids, slot_map, cache, mask_array,
                 labels_df, study_index, is_train=True):
        self.study_uids = list(study_uids)
        self.slot_map = slot_map
        self.cache = cache
        self.mask_array = mask_array
        self.labels_df = labels_df
        self.study_index = study_index
        self.is_train = is_train

        # Filter to cached studies
        valid_uids = [u for u in self.study_uids if u in self.study_index]
        if IS_MAIN and len(valid_uids) < len(self.study_uids):
            print(f'[Dataset] {len(self.study_uids) - len(valid_uids)} studies skipped (not in cache)')
        self.study_uids = valid_uids

    def __len__(self):
        return len(self.study_uids)

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        row_idx = self.study_index[uid]

        slots = torch.from_numpy(self.cache[row_idx].copy())  # [6, 9, H, W]
        mask = torch.from_numpy(self.mask_array[row_idx].copy())  # [6]

        n_slices = slots.shape[1]  # 9
        if self.is_train:
            max_start = n_slices - CFG['group_size']
            start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
            window = slots[:, start:start + CFG['group_size']]  # [6, 3, H, W]
        else:
            # ★ 返回全部9切片用于7窗口TTA
            window = slots  # [6, 9, H, W]

        label_row = self.labels_df.loc[uid]

        if self.is_train:
            probs = torch.tensor(
                [float(label_row.get(c, 0.5)) for c in PROB_COLS], dtype=torch.float32)
            weights = torch.tensor(
                [float(label_row.get(c, 0.1)) for c in WEIGHT_COLS], dtype=torch.float32)
            soft_masks = torch.tensor(
                [float(label_row.get(c, 0.0)) for c in MASK_COLS], dtype=torch.float32)
            return {
                'slots': window, 'mask': mask,
                'prob_targets': probs, 'weights': weights, 'soft_masks': soft_masks,
                'study_uid': uid,
            }
        else:
            labels = torch.tensor(
                [float(label_row.get(c, 0.0)) for c in TARGET_COLUMNS], dtype=torch.float32)
            val_masks = torch.tensor(
                [float(label_row.get(c, 0.0)) for c in MASK_COLS], dtype=torch.float32)
            return {
                'slots': window, 'mask': mask,
                'labels': labels, 'val_masks': val_masks,
                'study_uid': uid,
            }

print('Dataset v4 ready.')
