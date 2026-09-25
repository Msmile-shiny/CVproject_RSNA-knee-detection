"""Stage 6A: offline smoke/pilot; final epoch, no Gold-selected checkpoint."""
import json, time, random, os
from pathlib import Path
import pandas as pd
import pydicom
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def asset(explicit, name):
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    matches = list(Path('/kaggle/input').rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f'Configure {name} explicitly: {matches}')
    return matches[0]


def read_volume(directory):
    rows = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        ds = pydicom.dcmread(p)
        orient = np.asarray(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(orient[:3], orient[3:])
        position = float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), normal))
        pixels = ds.pixel_array.astype(np.float32)
        if pixels.ndim != 2:
            raise ValueError(f'Non-2D DICOM: {p}')
        pixels = pixels * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
        if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            pixels = -pixels
        rows.append((position, pixels, np.asarray(ds.PixelSpacing, dtype=float)))
    if not rows:
        raise ValueError(f'Empty series: {directory}')
    rows.sort(key=lambda row: row[0])
    spacing = rows[0][2]
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError(f'Invalid spacing: {directory}')
    volume = np.stack([r[1] for r in rows])
    lo, hi = np.percentile(volume, [1, 99])
    volume = np.clip((volume - lo) / max(hi - lo, 1e-6), 0, 1)
    # Preserve physical aspect ratio and full in-plane coverage; no anatomy crop yet.
    physical = np.array(volume.shape[1:]) * spacing
    shape = np.maximum(1, np.round(physical / physical.max() * S6['size']).astype(int))
    result = np.zeros((len(volume), S6['size'], S6['size']), np.uint8)
    y, x = (S6['size'] - shape) // 2
    for i, image in enumerate(volume):
        result[i, y:y+shape[0], x:x+shape[1]] = (cv2.resize(image, (int(shape[1]), int(shape[0]))) * 255).astype(np.uint8)
    return result


class Studies(Dataset):
    def __init__(self, uids, training=False):
        self.uids, self.training = list(uids), training

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        uid = self.uids[index]
        filename=hashlib.sha256(uid.encode()).hexdigest()+'.npz'
        path=cache/filename
        if not path.is_file() and S6['resume']:
            path=Path(S6['resume'])/'pixel_cache'/filename
        with np.load(path) as z:
            images, valid = z['images'].copy(), z['valid'].copy()
        images = torch.from_numpy(images).float() / 255
        if self.training:
            images = (images * random.uniform(.9, 1.1)).clamp(0, 1)
        images = (images - torch.tensor([.485,.456,.406])[None,:,None,None]) / torch.tensor([.229,.224,.225])[None,:,None,None]
        row = labels.loc[uid]
        return images, torch.from_numpy(valid), torch.arange(6).repeat_interleave(S6['windows']), torch.tensor(row[PROB_COLS].to_numpy(dtype=np.float32)), torch.tensor(row[WEIGHT_COLS].to_numpy(dtype=np.float32)), torch.tensor(row[MASK_COLS].to_numpy(dtype=np.float32))


def evaluate(uids, name):
    loader = DataLoader(Studies(uids), batch_size=S6['batch_size'], shuffle=False, num_workers=0)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            x, valid, slot = [v.to(device) for v in batch[:3]]
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                predictions.append(model(x, valid, slot).sigmoid().float().cpu().numpy())
    pred = np.concatenate(predictions)
    pd.DataFrame(pred, index=pd.Index(uids, name='StudyInstanceUID'), columns=TARGET_COLUMNS).to_csv(out / f'{name}_predictions.csv')
    return pred


def run():
    global labels, cache, model, device, out
    started = time.monotonic()
    random.seed(S6['seed']); np.random.seed(S6['seed']); torch.manual_seed(S6['seed'])
    torch.set_num_threads(2)
    out = Path(S6['output']); out.mkdir(parents=True, exist_ok=True)
    cache = out / 'pixel_cache'; cache.mkdir(exist_ok=True)
    root = Path(S6['competition'])
    if not root.is_dir():
        root = Path('/kaggle/input/rsna-knee-abnormality-detection')
    weight = asset(S6['weights'], 'resnet34-b627a593.pth')
    label_path = asset(S6['labels'], 'v5_labels.csv')
    if sha(label_path) != 'c13adffaabf4f8e518abb038282bb1aa09baac7652a9165e030710c457d0be6a':
        raise ValueError('Historical label hash mismatch')
    if not sha(weight).startswith('b627a593'):
        raise ValueError('Official pretrained weight hash mismatch')
    meta = pd.read_csv(root / 'train.csv', dtype={'StudyInstanceUID': str})
    labels = pd.read_csv(label_path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    assert labels.index.is_unique
    values = labels[PROB_COLS + WEIGHT_COLS + MASK_COLS].to_numpy()
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    gold = sorted(meta.loc[meta[TARGET_COLUMNS].notna().all(axis=1), 'StudyInstanceUID'])
    candidates = sorted(set(meta.StudyInstanceUID) - set(gold))
    assert set(candidates).issubset(labels.index), 'Missing labels'
    group_col = 'PatientID' if 'PatientID' in meta and meta.PatientID.notna().all() else 'StudyInstanceUID'
    groups = dict(zip(meta.StudyInstanceUID,meta[group_col].astype(str)))
    gold_groups={groups[u] for u in gold}
    candidates=[u for u in candidates if groups[u] not in gold_groups]
    folds = {u: stable_fold(groups[u]) for u in candidates}
    train = [u for u in candidates if folds[u] != S6['fold']]
    val = [u for u in candidates if folds[u] == S6['fold']]
    if S6['smoke']:
        train, val, gold = train[:8], val[:4], gold[:4]
    assert train and val and not set(train) & set(val)
    assert not {groups[u] for u in train} & {groups[u] for u in val}
    pd.DataFrame([{'StudyInstanceUID': u, 'group': groups[u], 'fold': folds.get(u, -1), 'split': split} for split, ids in [('train', train), ('pseudo_validation', val), ('gold_development', gold)] for u in ids]).to_csv(out / 'split.csv', index=False)
    receipt = dict(config=S6, labels_sha256=sha(label_path), weights_sha256=sha(weight), group_unit=group_col,
                   train_metadata_sha256=sha(root/'train.csv'), series_metadata_sha256=sha(root/'train_series.csv'),
                   gold_independent=False, teacher_oof_provenance_verified=False,
                   status='PREPARING', train_count=len(train), val_count=len(val), gold_count=len(gold))
    resume = Path(S6['resume']) if S6['resume'] else None
    if resume:
        previous=json.loads((resume/'run_receipt.json').read_text())
        for key in ('labels_sha256','weights_sha256','group_unit','train_metadata_sha256','series_metadata_sha256'):
            assert previous[key]==receipt[key], f'Resume mismatch: {key}'
        for key in ('size','windows','batch_size','fold','seed','pooling','backbone_lr','head_lr','smoke','implementation_sha256'):
            assert previous['config'][key]==S6[key], f'Resume mismatch: {key}'
    def report():
        (out / 'run_receipt.json').write_text(json.dumps(receipt, indent=2))
    report()
    series = pd.read_csv(root / 'train_series.csv', dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    series['dir'] = [str(root/'train_series'/u/s) for u,s in zip(series.StudyInstanceUID,series.SeriesInstanceUID)]
    series['n_slices'] = [len(list(Path(d).iterdir())) if Path(d).is_dir() else 0 for d in series.dir]
    series = series[series.n_slices >= 3]
    coverage = []
    for uid in train + val + gold:
        filename=hashlib.sha256(uid.encode()).hexdigest()+'.npz'
        if resume and (resume/'pixel_cache'/filename).is_file():
            # Read prior immutable cache in place; avoid duplicating ~16 GB.
            with np.load(resume/'pixel_cache'/filename) as z:
                assert z['images'].shape==(6*S6['windows'],3,S6['size'],S6['size'])
                assert z['images'].dtype==np.uint8 and z['valid'].any()
                coverage.append({'uid':uid,'valid_windows':int(z['valid'].sum())})
            continue
        windows = np.zeros((6*S6['windows'], 3, S6['size'], S6['size']), np.uint8)
        valid = np.zeros(len(windows), bool)
        matched = match_slots_for_study(series[series.StudyInstanceUID == uid])
        for slot, (name, *_rest) in enumerate(SLOTS):
            selected = matched[name]
            if selected is None:
                continue
            vol = read_volume(Path(selected['dir']))
            for j, center in enumerate(centers(len(vol), S6['windows'])):
                windows[slot*S6['windows']+j] = vol[center-1:center+2]
                valid[slot*S6['windows']+j] = True
        if not valid.any():
            raise RuntimeError(f'No valid slots for {uid}; fix input contract, do not silently drop study')
        np.savez_compressed(cache / (hashlib.sha256(uid.encode()).hexdigest()+'.npz'), images=windows, valid=valid)
        coverage.append({'uid':uid, 'valid_windows':int(valid.sum())})
        if time.monotonic()-started > S6['minutes']*60:
            receipt.update(status='PAUSED_PREPROCESSING'); report(); return
    pd.DataFrame(coverage).to_csv(out/'coverage.csv', index=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KneeResNet(torch.load(weight, map_location='cpu', weights_only=True), pooling=S6['pooling']).to(device)
    optimizer = torch.optim.AdamW([{'params':model.encoder.parameters(), 'lr':S6['backbone_lr']},
        {'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.')], 'lr':S6['head_lr']}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    loader = DataLoader(Studies(train, True), batch_size=S6['batch_size'], shuffle=True, num_workers=0)
    history=[]
    start_epoch=0
    if resume and (resume/'last.pt').is_file():
        saved=torch.load(resume/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch']; history=saved['history']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda':
            torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    def save_checkpoint(epoch):
        tmp=out/'last.tmp'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            epoch=epoch,config=S6,receipt=receipt,history=history,python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),tmp)
        tmp.replace(out/'last.pt')
    save_checkpoint(start_epoch)
    receipt.update(status='TRAINING', device=str(device)); report()
    for epoch in range(start_epoch,S6['epochs']):
        model.train(); total=0.; tick=time.monotonic()
        for batch in loader:
            if time.monotonic()-started>S6['minutes']*60:
                receipt.update(status='PAUSED_TRAINING',completed_epochs=epoch,
                    note='Resume from last complete epoch; partial epoch discarded'); report(); return
            x,valid,slot,y,w,m = [v.to(device) for v in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                loss=loss_fn(model(x,valid,slot), y,w,m)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            if 'first_backbone_gradient_norm' not in receipt:
                grad=model.encoder.conv1.weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
                receipt['first_backbone_gradient_norm']=float(grad.norm())
                report()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer); scaler.update(); total += loss.item()
        history.append(dict(epoch=epoch+1, loss=total/len(loader), seconds=time.monotonic()-tick))
        if epoch==0 or (epoch+1)%4==0:
            validation=evaluate(val,'pseudo_validation')
            history[-1]['pseudo_validation_mse']=float(np.mean((validation-labels.loc[val,PROB_COLS].to_numpy(float))**2))
        print(history[-1], flush=True)
        save_checkpoint(epoch+1)
        (out/'history.json').write_text(json.dumps(history, indent=2))
        if time.monotonic()-started > S6['minutes']*60:
            receipt.update(status='PAUSED_TRAINING', completed_epochs=epoch+1); report(); return
    pred=evaluate(val, 'pseudo_validation')
    truth=labels.loc[val, PROB_COLS].to_numpy(float)
    receipt['pseudo_validation_mse']=float(np.mean((pred-truth)**2))
    pred=evaluate(gold, 'gold_development')
    truth=meta.set_index('StudyInstanceUID').loc[gold,TARGET_COLUMNS].to_numpy(float)
    auc=[float(roc_auc_score(truth[:,i], pred[:,i])) if len(np.unique(truth[:,i]))==2 else None for i in range(12)]
    receipt.update(status='SMOKE_COMPLETE' if S6['smoke'] else 'PILOT_COMPLETE', gold_auc_by_class=dict(zip(TARGET_COLUMNS,auc)),
                   completed_epochs=S6['epochs'], seconds=time.monotonic()-started)
    report()


run()
