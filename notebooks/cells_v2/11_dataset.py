# ============================================================
# 2.5D Dataset -- v2: returns soft prob/weight/mask alongside images
# ============================================================

class Knee25DSoftLabelDataset(Dataset):
    """2.5D knee MRI dataset with soft pseudo-labels.

    v2 changes from v1:
      - 'labels' DataFrame stores prob_*/weight_*/mask_* instead of hard 0/1
      - __getitem__ returns prob_targets, weights, masks for training
      - No confidence_filter -- uses per-class weights instead of row-level filtering
      - Validation mode returns gold hard labels (for AUC/F1 scoring)
    """

    def __init__(self, series_df, labels_df, dicom_root, image_size=392,
                 slice_count=5, is_train=True, center_stride=3,
                 volume_cache=None, use_soft_labels=True):
        self.dicom_root = Path(dicom_root)
        self.image_size = image_size
        self.slice_count = slice_count
        self.is_train = is_train
        self.half_window = slice_count // 2
        self.volume_cache = volume_cache
        self.use_soft_labels = use_soft_labels and is_train

        # Filter Sagittal T2 FS series
        df = series_df.copy()
        df = df[df['Anatomical_Plane'] == 'Sagittal']
        if 'Fluid_Sensitive' in df.columns: df = df[df['Fluid_Sensitive'] == 1]
        if 'Fat_Suppression' in df.columns: df = df[df['Fat_Suppression'] == 1]

        self.label_df = labels_df
        self.samples = []
        skipped = 0
        for (study_uid, series_uid), grp in df.groupby(['StudyInstanceUID', 'SeriesInstanceUID']):
            if study_uid not in self.label_df.index:
                skipped += 1; continue
            plane = grp.iloc[0]['Anatomical_Plane']
            dicom_dir = self.dicom_root / study_uid / series_uid
            dicom_dir_str = str(dicom_dir)

            if self.volume_cache is not None and dicom_dir_str not in self.volume_cache:
                skipped += 1; continue
            if self.volume_cache is None:
                if not dicom_dir.exists():
                    skipped += 1; continue
                dcm_files = list(dicom_dir.glob('*.dcm'))
                if not dcm_files: dcm_files = list(dicom_dir.glob('*'))
                n_slices = len(dcm_files)
            else:
                n_slices = self.volume_cache[dicom_dir_str].shape[0]

            if n_slices < 3:
                skipped += 1; continue

            for center_idx in range(0, n_slices, center_stride):
                self.samples.append({
                    'study_uid': study_uid, 'series_uid': series_uid,
                    'dicom_dir': dicom_dir_str, 'plane': plane,
                    'center_idx': center_idx, 'n_slices': n_slices,
                })

        if IS_MAIN and skipped:
            print(f'[{type(self).__name__}] {skipped} series skipped')

    def __len__(self): return len(self.samples)

    def _read_volume(self, sample):
        if self.volume_cache is not None:
            volume = self.volume_cache[sample['dicom_dir']]
            is_uint8 = volume.dtype == np.uint8
        else:
            is_uint8 = False
            try:
                volume = read_dicom_series(sample['dicom_dir'], plane=sample['plane'],
                                           image_size=self.image_size)
            except:
                return None, False

        center = sample['center_idx']
        n_total = volume.shape[0]
        half = self.half_window
        indices = [max(0, min(n_total-1, center+o)) for o in range(-half, half+1)]
        stack = volume[indices]

        if is_uint8:
            stack = torch.from_numpy(stack.copy()).float().div_(255.0)
        else:
            stack = torch.from_numpy(stack.copy())
        return stack, True

    def __getitem__(self, idx):
        sample = self.samples[idx]
        study_uid = sample['study_uid']

        image, ok = self._read_volume(sample)
        if not ok:
            image = torch.zeros(self.slice_count, self.image_size, self.image_size)

        label_row = self.label_df.loc[study_uid]

        if self.use_soft_labels:
            # v2: Return soft prob targets + weights + masks
            probs = torch.tensor([float(label_row.get(c, 0.5)) for c in PROB_COLS], dtype=torch.float32)
            weights = torch.tensor([float(label_row.get(c, 0.1)) for c in WEIGHT_COLS], dtype=torch.float32)
            masks = torch.tensor([float(label_row.get(c, 0.0)) for c in MASK_COLS], dtype=torch.float32)
            return {
                'image': image,
                'prob_targets': probs,
                'weights': weights,
                'masks': masks,
                'study_uid': study_uid,
                'plane': sample['plane'],
            }
        else:
            # Validation: return hard labels for scoring
            labels = torch.tensor([float(label_row.get(c, 0.0)) for c in TARGET_COLUMNS], dtype=torch.float32)
            return {
                'image': image,
                'labels': labels,
                'study_uid': study_uid,
                'plane': sample['plane'],
            }
