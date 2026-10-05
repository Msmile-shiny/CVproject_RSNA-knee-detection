"""Isolated public reader execution; imported modules belong to this process only."""
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

started=time.monotonic()
src=Path(__file__).resolve().parent
roots=[p.parent for p in Path('/kaggle/input').glob('datasets/*/*/cnxt_v0_fold0.pt')]
roots += [p.parent for p in Path('/kaggle/input').glob('*/cnxt_v0_fold0.pt')]
assert len(set(roots))==1, 'Checkpoint root must be unique'
asset=roots[0]
ckpts=[asset/f'cnxt_v0_fold{i}.pt' for i in range(3)]
assert all(p.is_file() and p.stat().st_size==123035271 for p in ckpts), 'All three exact weights required'
env=Path('/kaggle/working/cnx3_env')
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--target',str(env),'--find-links',str(asset),'pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg'],check=True)
timm_wheels=sorted(Path('/kaggle/input').glob('datasets/*/*/timm-*.whl'))+sorted(Path('/kaggle/input').glob('*/timm-*.whl'))
assert timm_wheels, 'Offline timm wheel required'
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--target',str(env),str(timm_wheels[0])],check=True)
sys.path.insert(0,str(env))
sys.path.insert(0,str(src))
import torch
import numpy as np
import pandas as pd
import infer
from knee import LABELS, build_study_table
from preprocess import MAX_SLICES
torch.set_num_threads(4)
torch.cuda.set_device(0)
torch.cuda.reset_peak_memory_stats(0)
comp=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not comp.is_dir(): comp=Path('/kaggle/input/rsna-knee-abnormality-detection')
test=pd.read_csv(comp/'test.csv',dtype={'StudyInstanceUID':str})
ids=test.StudyInstanceUID.tolist()
assert ids and len(ids)==len(set(ids))
se=pd.read_csv(comp/'test_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
se=se[se.StudyInstanceUID.isin(ids)]
n_files={sr:min(len(os.listdir(comp/'test_series'/st/sr)),MAX_SLICES) for st,sr in zip(se.StudyInstanceUID,se.SeriesInstanceUID)}
table=build_study_table(se,n_files)
ds=infer.DicomStudyDataset(ids,table,str(comp/'test_series'),12)
models=infer.load_models([str(p) for p in ckpts])
assert len(models)==3
config=[]
for p in ckpts:
    ck=torch.load(p,map_location='cpu',weights_only=True)
    assert ck['args']['backbone'].split('.')[0]=='convnext_tiny' and int(ck['args']['res'])==256, 'Unexpected architecture/input'
    config.append(ck['args'])
    del ck
# Full batch and all six slots: memory/latency check only, discarded after use.
x,mask,pos=ds[0]
bx=x.unsqueeze(0).repeat(4,1,1,1,1,1).cuda()
bm=torch.ones((4,6),dtype=torch.bool,device='cuda')
bp=pos.unsqueeze(0).repeat(4,1,1).cuda()
stress_start=time.monotonic()
with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
    for model,res in models:
        probe=model(bx,bm,bp,res=res)
        assert probe.shape==(4,12) and torch.isfinite(probe).all()
torch.cuda.synchronize()
stress_seconds=time.monotonic()-stress_start
stress_peak=torch.cuda.max_memory_allocated(0)
del bx,bm,bp,probe
torch.cuda.empty_cache()
dl=torch.utils.data.DataLoader(ds,batch_size=4,shuffle=False,num_workers=4,pin_memory=True)
out=[];valid=[];inference_start=time.monotonic()
with torch.inference_mode():
    for x,mask,pos in dl:
        assert mask.any(1).all(), 'Empty study'
        valid.extend(mask.sum(1).tolist())
        x,mask,pos=x.cuda(non_blocking=True),mask.cuda(),pos.cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            prob=torch.stack([m(x,mask,pos,res=res).float().sigmoid() for m,res in models]).mean(0)
        assert torch.isfinite(prob).all()
        out.append(prob.cpu().numpy())
values=np.concatenate(out)
assert values.shape==(len(ids),12) and np.isfinite(values).all() and (values>=0).all() and (values<=1).all()
frame=pd.DataFrame(values,columns=LABELS)
frame.insert(0,'StudyInstanceUID',ids)
work=Path('/kaggle/working')
path=work/'cnx3_raw.csv'
frame.to_csv(path,index=False)
receipt=dict(status='COMPLETE',study_count=len(ids),folds=[0,1,2],model_count=3,
    precision='fp16',batch_studies=4,windows_per_slot=12,slots_present=valid,
    checkpoint_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ckpts},
    checkpoint_args=config,stress_pass=True,stress_seconds=stress_seconds,stress_peak_gpu_bytes=stress_peak,
    peak_gpu_bytes=torch.cuda.max_memory_allocated(0),elapsed_seconds=time.monotonic()-started,
    inference_seconds=time.monotonic()-inference_start,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('*.py')},
    raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fallback=0,partial_predictions=0)
(work/'cnx3_worker_receipt.json').write_text(json.dumps(receipt,indent=2,default=str))
print('CNX3 COMPLETE',receipt,flush=True)
