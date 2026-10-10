class PairedDicomStudyDataset(torch.utils.data.Dataset):
    """Decode each selected series once; reproduce both original offset views."""
    def __init__(self, studies, table, root, k):
        self.studies, self.table, self.root, self.k = list(studies), table, root, k

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        x = np.zeros((2, N_SLOTS, self.k, 3, SIZE, SIZE), np.uint8)
        mask = np.zeros((2, N_SLOTS), bool)
        pos = np.zeros((2, N_SLOTS, self.k), np.float32)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            try:
                vol, _ = load_series(f'{self.root}/{st}/{cands[0]}')
            except Exception:
                continue
            n = len(vol)
            if n < 3:
                continue
            centres = window_centres(n, self.k, False)
            for offset in (0, 1):
                c = centres if offset == 0 else np.clip(centres + 1, 1, max(1, n - 2))
                x[offset, s] = np.stack([vol[np.clip([j-1,j,j+1],0,n-1)] for j in c])
                mask[offset, s], pos[offset, s] = True, c / max(n-1,1)
        return tuple((torch.from_numpy(x[o]),torch.from_numpy(mask[o]),torch.from_numpy(pos[o])) for o in (0,1))
