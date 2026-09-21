# Stage 5A: streamed DICOM -> frozen features -> mean/MIL/coarse-to-fine controls.
import hashlib, json, copy, shutil, zipfile

session_started = time.monotonic()

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)

def compatible_run(previous, current):
    if previous.get('experiment') != 'stage5a':
        raise ValueError('Resume manifest is not Stage 5A')
    for key in ('checkpoint_sha256', 'labels_sha256'):
        if not previous.get(key) or previous[key] != current[key]:
            raise ValueError(f'Resume mismatch: {key}')
    # Paths, wall-clock budgets and smoke size do not change image features.
    for key in ('coarse_px','fine_px','slices','top_per_class','seed','epochs','head_lr','batch_size'):
        if previous['config'].get(key) != current['config'].get(key):
            raise ValueError(f'Resume configuration mismatch: {key}')
    if previous.get('feature_contract', 'stage5a-v1') != 'stage5a-v1':
        raise ValueError('Unknown feature preprocessing contract')

def validated_cache(path, phase, selected=None):
    try:
        with np.load(path, allow_pickle=False) as z:
            keys=('x','mask','slot','pos','center','scale')
            if not set(keys).issubset(z.files): return False
            b={k:z[k] for k in keys}
        n=len(b['mask'])
        expected=N_SLOT*S5['slices'] if phase=='coarse' else max(1,len(selected))
        if n!=expected or b['x'].shape!=(n,1152) or b['x'].dtype!=np.float16: return False
        if any(b[k].shape!=(n,) for k in keys[1:]): return False
        if b['mask'].dtype!=np.bool_ or not b['mask'].any(): return False
        if any(b[k].dtype!=np.int64 for k in ('slot','center','scale')): return False
        if not all(np.isfinite(b[k]).all() for k in keys): return False
        if not ((b['slot']>=0)&(b['slot']<N_SLOT)).all(): return False
        if not ((b['pos']>=0)&(b['pos']<=1)).all(): return False
        if not (b['scale']==int(phase=='fine')).all(): return False
        valid=b['mask']
        if (b['center'][valid]<1).any(): return False
        if phase=='fine':
            wanted=np.asarray(selected,dtype=np.int64)
            actual=np.stack([b['slot'],b['center']],axis=1)
            if not np.array_equal(actual[valid],wanted[valid]): return False
        return True
    except (OSError,ValueError,KeyError,EOFError,TypeError,IndexError,zipfile.BadZipFile):
        return False

def restore_cache(uid, phase, selected=None):
    dest=cache_path(uid,phase)
    for root in resume_roots:
        src=root/'stage5a_features'/phase/dest.name
        if not src.is_file() or not validated_cache(src,phase,selected): continue
        if src.resolve()!=dest.resolve():
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_suffix('.resume.tmp')
            shutil.copyfile(src,tmp); tmp.replace(dest)
        return True
    return False

def session_budget_reached():
    # Reserve time for reporting; never shorten a test set to meet the budget.
    return time.monotonic()-session_started >= (S5.get('session_minutes',480)-S5.get('reserve_minutes',20))*60

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def find_asset(explicit, filename):
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    hits = list(Path('/kaggle/input').rglob(filename))
    if len(hits) != 1:
        raise FileNotFoundError(f'Set the explicit path for {filename}; found {hits}')
    return hits[0]

output_dir = Path(CFG['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)
assert torch.cuda.is_available(), 'Stage 5A feature extraction requires a GPU.'
asset = find_asset(S5['v5_checkpoint'], 'best_model_s42.pt')
label_asset = find_asset(S5['labels'], 'v5_labels.csv')
assert sha256(label_asset) == 'c13adffaabf4f8e518abb038282bb1aa09baac7652a9165e030710c457d0be6a', 'Use historical v5 labels'
# Only load your own trusted v5 checkpoint. It contains configuration as well as tensors.
checkpoint = torch.load(asset, map_location='cpu', weights_only=False)
assert checkpoint['targets'] == TARGET_COLUMNS
assert checkpoint['config']['seed'] == 42 and checkpoint['config']['image_size'] == 288
state = {k.removeprefix('module.'): v for k, v in checkpoint['model'].items()}
for k, v in (checkpoint.get('ema') or {}).get('shadow', {}).items():
    state[k.removeprefix('module.')] = v
backbone_state = {k[len('dinov2.'):]: v for k, v in state.items() if k.startswith('dinov2.')}
assert backbone_state, 'No v5 DINOv2 backbone found'
encoders = {}
for size in (S5['coarse_px'], S5['fine_px']):
    enc = timm.create_model(CFG['dinov2_variant'], pretrained=False, num_classes=0, img_size=size)
    sd = dict(backbone_state)
    p = sd['pos_embed']; old = math.isqrt(p.shape[1] - 1); new = math.isqrt(enc.pos_embed.shape[1] - 1)
    assert old * old == p.shape[1] - 1
    grid = p[:, 1:].reshape(1, old, old, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, (new, new), mode='bicubic', align_corners=False, antialias=True)
    sd['pos_embed'] = torch.cat([p[:, :1], grid.permute(0, 2, 3, 1).reshape(1, new * new, -1)], 1)
    enc.load_state_dict(sd, strict=True)
    enc.requires_grad_(False).eval().to(DEVICE)
    if N_GPUS > 1:
        class Tokens(nn.Module):
            def __init__(self, model):
                super().__init__(); self.model = model
            def forward(self, x):
                return self.model.forward_features(x)
        enc = nn.DataParallel(Tokens(enc))
    encoders[size] = enc
audit = dict(experiment='stage5a', config=S5, checkpoint_sha256=sha256(asset), labels_sha256=sha256(label_asset),
             gold_is_development_only=True, parent_version_id=347718768, parent_comparison='pending')
resume_roots=[]
for root in [output_dir] + ([Path(S5['resume_input'])] if S5.get('resume_input') else []):
    manifest=root/'stage5a_manifest.json'
    if manifest.is_file():
        compatible_run(json.loads(manifest.read_text(encoding='utf-8')),audit)
        if root.resolve() not in [r.resolve() for r in resume_roots]: resume_roots.append(root)
    elif root!=output_dir:
        raise FileNotFoundError(f'Resume root must contain {manifest.name}: {root}')
    elif (root/'stage5a_features').exists():
        raise RuntimeError('Existing feature cache has no provenance manifest; use a clean output directory')
audit.update(feature_contract='stage5a-v1',completed=False,status='running',resume_sources=[str(r) for r in resume_roots])
atomic_json(output_dir / 'stage5a_manifest.json',audit)
print('Resume roots:',resume_roots,flush=True)
del checkpoint, state, backbone_state
gc.collect()

@torch.inference_mode()
def encode(windows, size):
    outputs = []
    enc = encoders[size]
    for start in range(0, len(windows), S5['encode_batch']):
        x = torch.from_numpy(np.stack(windows[start:start+S5['encode_batch']])).to(DEVICE).float() / 255
        mean = x.new_tensor([.485,.456,.406])[None,:,None,None]
        std = x.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.autocast('cuda', dtype=torch.float16):
            f = enc((x-mean)/std) if isinstance(enc, nn.DataParallel) else enc.forward_features((x-mean)/std)
            patches = f[:,1:]
            feat = torch.cat([f[:,0], patches.mean(1), patches.topk(max(1, patches.shape[1]//8),dim=1).values.mean(1)],1)
        outputs.append(feat.float().cpu().numpy().astype(np.float16))
    return np.concatenate(outputs)

def cache_path(uid, phase):
    return output_dir / 'stage5a_features' / phase / (hashlib.sha256(str(uid).encode()).hexdigest()+'.npz')

def read_slot(info, plane):
    # DICOM decode errors are surfaced rather than converted to fake zero slices.
    lat = None
    files = _list_dcm_files(info['dir'])
    if files:
        ds = pydicom.dcmread(str(Path(info['dir'])/files[0]), stop_before_pixels=True, force=True)
        lat = str(getattr(ds, 'ImageLaterality', '') or getattr(ds, 'Laterality', '')).upper()
    volume, _ = read_series_volume(info['dir'], plane=plane, laterality=lat,
                                    image_size=S5['fine_px'], crop_mm=130.)
    return volume

def extract_study(uid, mapping, phase, selected=None):
    # Fine selections reference actual, physically ordered slice indices.
    n = N_SLOT*S5['slices'] if selected is None else len(selected)
    n = max(1,n)
    x=np.zeros((n,1152),np.float16); mask=np.zeros(n,bool)
    slots=np.zeros(n,np.int64); pos=np.zeros(n,np.float32); centers=np.zeros(n,np.int64)
    size=S5['coarse_px'] if selected is None else S5['fine_px']
    windows=[]; dest=[]; failures=[]
    for s,(name,plane,_,_) in enumerate(SLOTS):
        info=mapping.get(name)
        if info is None: continue
        if selected is not None and not any(int(z[0])==s for z in selected): continue
        try:
            vol=read_slot(info,plane)
            if vol is None or len(vol)<3: raise ValueError('No usable volume')
            candidates=window_centers(len(vol),S5['slices'])
            rows=[(s*S5['slices']+j,int(c)) for j,c in enumerate(candidates)] if selected is None else [(j,int(z[1])) for j,z in enumerate(selected) if int(z[0])==s]
            for j,c in rows:
                if not 1<=c<len(vol)-1: raise ValueError('Fine center outside volume')
                w=vol[c-1:c+2]
                if size!=S5['fine_px']:
                    w=np.stack([cv2.resize(a,(size,size),interpolation=cv2.INTER_AREA) for a in w])
                windows.append((w*255).clip(0,255).round().astype(np.uint8)); dest.append(j)
                slots[j]=s; pos[j]=c/max(1,len(vol)-1); centers[j]=c
        except Exception as e:
            failures.append(dict(uid=uid,slot=name,error=str(e)))
    if windows:
        x[dest]=encode(windows,size); mask[dest]=True
    if not mask.any(): raise RuntimeError(f'No valid images for {uid}: {failures}')
    p=cache_path(uid,phase); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,x=x,mask=mask,slot=slots,pos=pos,center=centers,scale=np.full(n,int(selected is not None),np.int64))
    tmp.replace(p)
    BANKS.pop((uid,phase),None)
    return failures

BANKS = {}

def load_bank(uid, fine=False):
    def get(phase):
        key=(uid,phase)
        if key not in BANKS:
            with np.load(cache_path(uid,phase)) as z: BANKS[key]={k:z[k] for k in z.files}
        return BANKS[key]
    bank=get('coarse')
    if fine:
        extra=get('fine')
        bank={k:np.concatenate([v,extra[k]]) for k,v in bank.items()}
    return bank

def tensor_batch(uids,fine=False):
    banks=[load_bank(u,fine) for u in uids]; n=max(len(b['mask']) for b in banks)
    out=[]
    for key in ('x','mask','slot','pos','scale'):
        arr=np.stack([np.pad(b[key],[(0,n-len(b[key]))]+([(0,0)] if key=='x' else [])) for b in banks])
        t=torch.from_numpy(arr).to(DEVICE)
        out.append(t.float() if key in ('x','pos') else t)
    return out

def fit_head(uids, labels, name, initial=None, pooling='attention', fine=False):
    torch.manual_seed(S5['seed']); rng=np.random.default_rng(S5['seed'])
    head=DenseMIL(pooling=pooling).to(DEVICE)
    if initial is not None: head.load_state_dict(initial.state_dict(),strict=True)
    optimizer=torch.optim.AdamW(head.parameters(),lr=S5['head_lr'],weight_decay=1e-3)
    history=[]
    # Fixed epochs: gold labels never choose epochs or fine selection locations.
    for epoch in range(S5['epochs']):
        head.train(); order=rng.permutation(uids); total=0.; steps=0
        for start in range(0,len(order),S5['batch_size']):
            ids=list(order[start:start+S5['batch_size']]); batch=tensor_batch(ids,fine)
            row=labels.loc[ids]
            y,w,m=[torch.tensor(row[c].to_numpy(np.float32),device=DEVICE) for c in (PROB_COLS,WEIGHT_COLS,MASK_COLS)]
            optimizer.zero_grad(set_to_none=True)
            loss=weighted_bce(head(*batch),y,w,m); loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step(); total+=float(loss.detach()); steps+=1
        history.append(dict(epoch=epoch+1,loss=total/max(1,steps)))
        print(name,history[-1],flush=True)
    head.eval()
    torch.save(dict(state_dict=head.cpu().state_dict(),config=S5,pooling=pooling,fine=fine,audit=audit),output_dir/(name+'.pt'))
    head.to(DEVICE)
    pd.DataFrame(history).to_csv(output_dir/(name+'_history.csv'),index=False)
    return head

def get_head(uids, labels, name, initial=None, pooling='attention', fine=False):
    uid_hash=hashlib.sha256('\n'.join(sorted(uids)).encode()).hexdigest()
    for root in resume_roots:
        path=root/(name+'.pt')
        if not path.is_file(): continue
        # The mounted run is explicitly selected by the user; load its trusted head.
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        compatible_run(ckpt['audit'],audit)
        if ckpt.get('pooling')!=pooling or ckpt.get('fine')!=fine: continue
        old_config=ckpt['config']
        if old_config.get('smoke_studies',0)!=S5.get('smoke_studies',0): continue
        # Legacy full runs use all non-Gold train UIDs, same competition and protocol.
        if ckpt.get('train_uid_sha256',uid_hash)!=uid_hash: continue
        head=DenseMIL(pooling=pooling).to(DEVICE)
        head.load_state_dict(ckpt['state_dict'],strict=True); head.eval()
        dest=output_dir/path.name
        if path.resolve()!=dest.resolve(): shutil.copyfile(path,dest)
        history=root/(name+'_history.csv')
        if history.is_file() and history.resolve()!=(output_dir/history.name).resolve(): shutil.copyfile(history,output_dir/history.name)
        print('Restored completed head:',name,flush=True)
        return head
    head=fit_head(uids,labels,name,initial,pooling,fine)
    path=output_dir/(name+'.pt')
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    ckpt['train_uid_sha256']=uid_hash
    torch.save(ckpt,path)
    return head

@torch.inference_mode()
def predict_head(head,uids,fine=False):
    head.eval(); outputs=[]
    for start in range(0,len(uids),S5['batch_size']):
        outputs.append(head(*tensor_batch(uids[start:start+S5['batch_size']],fine)).sigmoid().cpu().numpy())
    return np.concatenate(outputs)

comp_input=Path(CFG['comp_input'])
if not (comp_input/'train.csv').is_file():
    candidates=[Path('/kaggle/input/rsna-knee-abnormality-detection'),Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
    comp_input=next(p for p in candidates if (p/'train.csv').is_file())
train_meta=pd.read_csv(comp_input/'train.csv',dtype={'StudyInstanceUID':str})
test_meta=pd.read_csv(comp_input/'test.csv',dtype={'StudyInstanceUID':str})
gold=train_meta.loc[train_meta[TARGET_COLUMNS].notna().all(1)].set_index('StudyInstanceUID')
train_uids=sorted(set(train_meta.StudyInstanceUID)-set(gold.index)); gold_uids=sorted(gold.index); test_uids=test_meta.StudyInstanceUID.tolist()
labels=pd.read_csv(label_asset,dtype={'StudyInstanceUID':str}).set_index('StudyInstanceUID')
assert labels.index.is_unique and set(train_uids)<=set(labels.index)
values=labels.loc[train_uids,PROB_COLS+WEIGHT_COLS+MASK_COLS].to_numpy(float)
assert np.isfinite(values).all() and (values>=0).all() and (values<=1).all()
assert not set(train_uids)&set(gold_uids) and not set(train_uids+gold_uids)&set(test_uids)
if S5['smoke_studies']:
    train_uids=train_uids[:S5['smoke_studies']]; gold_uids=gold_uids[:4]
maps={}
for split,ids in [('train',train_uids+gold_uids),('test',test_uids)]:
    metadata=pd.read_csv(comp_input/(split+'_series.csv'),dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    part,_=build_study_slot_map(metadata.loc[metadata.StudyInstanceUID.isin(ids)],comp_input/(split+'_series'))
    maps.update(part)
all_uids=train_uids+gold_uids+test_uids

def export_head(name,head,metrics):
    pred=predict_head(head,gold_uids,fine=name=='fine')
    df=pd.DataFrame(pred,columns=TARGET_COLUMNS); df.insert(0,'StudyInstanceUID',gold_uids)
    df.to_csv(output_dir/f'gold_stage5a_{name}.csv',index=False)
    aucs=[]
    for j,c in enumerate(TARGET_COLUMNS):
        y=gold.loc[gold_uids,c].to_numpy()
        auc=roc_auc_score(y,pred[:,j]) if len(np.unique(y))==2 else float('nan')
        metrics.append(dict(model=name,target=c,auc=auc)); aucs.append(auc)
    print(name,'Gold development macro',np.nanmean(aucs))
    test_pred=predict_head(head,test_uids,fine=name=='fine')
    df=pd.DataFrame(test_pred,columns=TARGET_COLUMNS); df.insert(0,'StudyInstanceUID',test_uids)
    assert np.isfinite(test_pred).all() and df.StudyInstanceUID.tolist()==test_uids
    df.to_csv(output_dir/f'submission_stage5a_{name}.csv',index=False)
    pd.DataFrame(metrics).to_csv(output_dir/'stage5a_auc.csv',index=False)
    gold.loc[gold_uids,TARGET_COLUMNS].to_csv(output_dir/'stage5a_gold_truth.csv')

def run_stage5a():
    failures=[]; selection_log=[]; metrics=[]; heads={}
    reused=dict(coarse=0,fine=0); computed=dict(coarse=0,fine=0)
    for root in resume_roots:
        p=root/'stage5a_decode_failures.json'
        if p.is_file(): failures.extend(json.loads(p.read_text(encoding='utf-8')))

    def persist(stage,done=0,paused=False,completed=False):
        audit.update(status='paused' if paused else ('completed' if completed else 'running'),
                     completed=completed,stage=stage,processed=done,total=len(all_uids),
                     reused=reused,computed=computed,exported_heads=list(heads),
                     train_studies=len(train_uids),gold_studies=len(gold_uids),test_studies=len(test_uids),
                     runtime_minutes=(time.monotonic()-session_started)/60,smoke=bool(S5['smoke_studies']))
        atomic_json(output_dir/'stage5a_manifest.json',audit)
        atomic_json(output_dir/'stage5a_decode_failures.json',failures)
        atomic_json(output_dir/'stage5a_selected_slices.json',selection_log)
        if paused:
            print(f'PAUSED at {stage} {done}/{len(all_uids)}. Existing predictions and caches are saved. '
                  'Mount this output as resume_input in a new run; do not submit a partial run.',flush=True)

    persist('coarse')
    coarse_started=time.monotonic()
    for i,uid in enumerate(all_uids):
        if session_budget_reached() or time.monotonic()-coarse_started>S5['feature_minutes']*60:
            persist('coarse',i,paused=True); return
        if restore_cache(uid,'coarse'): reused['coarse']+=1
        else:
            failures.extend(extract_study(uid,maps.get(uid,{}),'coarse')); computed['coarse']+=1
        if (i+1)%100==0:
            print(f'coarse {i+1}/{len(all_uids)}; reused {reused["coarse"]}',flush=True)
            persist('coarse',i+1)
    # Evaluate every completed control BEFORE starting the costly fine pass.
    for name,pooling in [('mean','mean'),('coarse','attention'),('coarse_continue','attention')]:
        if session_budget_reached(): persist('heads',len(heads),paused=True); return
        head=get_head(train_uids,labels,'stage5a_'+name,
                      initial=heads.get('coarse') if name=='coarse_continue' else None,pooling=pooling)
        export_head(name,head,metrics); heads[name]=head; persist('heads',len(heads))
    persist('fine')
    with torch.inference_mode():
        for i,uid in enumerate(all_uids):
            if session_budget_reached(): persist('fine',i,paused=True); return
            bank=load_bank(uid)
            _,att=heads['coarse'](*tensor_batch([uid]),return_attention=True)
            selected=selected_tokens(att[0].cpu().numpy(),bank['mask'],S5['top_per_class'])
            coordinates=[(int(bank['slot'][j]),int(bank['center'][j])) for j in selected]
            if restore_cache(uid,'fine',coordinates): reused['fine']+=1
            else:
                failures.extend(extract_study(uid,maps.get(uid,{}),'fine',coordinates)); computed['fine']+=1
            selection_log.append(dict(uid=uid,slot_center=coordinates))
            if (i+1)%100==0:
                print(f'fine {i+1}/{len(all_uids)}; reused {reused["fine"]}',flush=True)
                persist('fine',i+1)
    if session_budget_reached(): persist('fine_head',len(all_uids),paused=True); return
    head=get_head(train_uids,labels,'stage5a_fine',initial=heads['coarse'],fine=True)
    export_head('fine',head,metrics); heads['fine']=head
    persist('complete',len(all_uids),completed=True)
    print('Screening complete. Compare with parent936 before creating submission.csv. No automatic leaderboard submission.')

run_stage5a()
