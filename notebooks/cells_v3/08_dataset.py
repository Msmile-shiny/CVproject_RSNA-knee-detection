# ============================================================
# v3: MultiViewDataset — 6-slot clinical MRI
# ============================================================

class MultiViewDataset(Dataset):
    """Multi-view knee MRI dataset with 6 clinical slots.

    Each study returns:
      - slots: [6, 3, 224, 224] uint8 tensor (3 adjacent slices as RGB-like channels)
      - mask: [6] float32 (1=slot present, 0=missing)
      - labels or soft_label info depending on is_train and label_type

    Training data augmentation:
      - Random window selection from cached 9 slices
    Validation:
      - Fixed middle window
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

        # Filter: keep only studies that are in the cache
        valid_uids = []
        skipped = 0
        for uid in self.study_uids:
            if uid in self.study_index:
                valid_uids.append(uid)
            else:
                skipped += 1
        self.study_uids = valid_uids
        if IS_MAIN and skipped:
            print(f'[{type(self).__name__}] {skipped} studies skipped (not in cache)')

    def __len__(self):
        return len(self.study_uids)

    def __getitem__(self, idx):
        uid = self.study_uids[idx]
        row_idx = self.study_index[uid]

        # Load from cache
        slots = torch.from_numpy(self.cache[row_idx].copy())  # [6, 9, 224, 224]
        mask = torch.from_numpy(self.mask_array[row_idx].copy())  # [6]

        # Select 3-slice window
        n_slices = slots.shape[1]  # 9
        if self.is_train:
            max_start = n_slices - 3
            start = torch.randint(0, max_start + 1, (1,)).item() if max_start > 0 else 0
        else:
            start = (n_slices - 3) // 2  # middle window

        window = slots[:, start:start+3]  # [6, 3, 224, 224]

        # Labels
        label_row = self.labels_df.loc[uid]

        if self.is_train:
            # Unified soft-label format: gold studies have prob=hard_label, weight=1, mask=1
            probs = torch.tensor(
                [float(label_row.get(c, 0.5)) for c in PROB_COLS], dtype=torch.float32)
            weights = torch.tensor(
                [float(label_row.get(c, 0.1)) for c in WEIGHT_COLS], dtype=torch.float32)
            soft_masks = torch.tensor(
                [float(label_row.get(c, 0.0)) for c in MASK_COLS], dtype=torch.float32)
            return {
                'slots': window,
                'mask': mask,
                'prob_targets': probs,
                'weights': weights,
                'soft_masks': soft_masks,
                'study_uid': uid,
            }
        else:
            # Validation: hard labels + masks for partial-label support
            labels = torch.tensor(
                [float(label_row.get(c, 0.0)) for c in TARGET_COLUMNS], dtype=torch.float32)
            val_masks = torch.tensor(
                [float(label_row.get(c, 0.0)) for c in MASK_COLS], dtype=torch.float32)
            return {
                'slots': window,
                'mask': mask,
                'labels': labels,
                'val_masks': val_masks,
                'study_uid': uid,
            }
