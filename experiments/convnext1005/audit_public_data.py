"""CPU-only public-data replay of the exact ConvNeXt series preprocessing.

Diagnostic only: never makes a competition submission or loads a model.
Run as an embedded Kaggle script with preprocess.py alongside this file.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import pandas as pd
from preprocess import load_series, MAX_SLICES

PLANES=['Sagittal','Coronal','Axial']

def inventory(root, split):
    studies=pd.read_csv(root/f'{split}.csv',dtype={'StudyInstanceUID':str})
    series=pd.read_csv(root/f'{split}_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    rows=[]; slots={}; errors=[]
    for r in series.itertuples(index=False):
        path=root/f'{split}_series'/r.StudyInstanceUID/r.SeriesInstanceUID
        try:
            count=min(len(os.listdir(path)),MAX_SLICES)
            if count<3:
                continue
            slot=PLANES.index(r.Anatomical_Plane)+3*(1-int(r.Fat_Suppression))
            if not 0<=slot<6:
                raise ValueError(f'invalid slot {slot}')
            slots.setdefault((r.StudyInstanceUID,slot),[]).append((count,r.SeriesInstanceUID,path))
        except Exception as exc:
            errors.append(dict(study=r.StudyInstanceUID,series=r.SeriesInstanceUID,error=repr(exc)))
    # Stable descending order exactly matches build_study_table.
    for (study,slot),candidates in slots.items():
        candidates.sort(key=lambda r:-r[0])
        count,sr,path=candidates[0]
        rows.append((split,study,slot,sr,path,count))
    present={r[1] for r in rows}
    return rows,dict(studies=len(studies),series=len(series),selected_series=len(rows),
                    inventory_errors=errors,empty_studies=sorted(set(studies.StudyInstanceUID)-present))

def inspect_one(row):
    split,study,slot,sr,path,count=row
    result=dict(split=split,study=study,slot=slot,series=sr,file_count_capped=count)
    started=time.monotonic()
    try:
        vol,info=load_series(str(path))
        result.update(info=info,shape=list(vol.shape),dtype=str(vol.dtype))
        failures=[]
        if info.get('n_bad',0): failures.append('STRICT_N_BAD_ABORT')
        if len(vol)<3: failures.append('STRICT_SHORT_SERIES_ABORT')
        if vol.ndim!=3 or vol.shape[1:]!=(384,384): failures.append('INVALID_VOLUME_SHAPE')
        result['failures']=failures
        # Preserve representative original decoder errors discarded by load_series.
        if info.get('n_bad',0):
            from preprocess import _read
            result['decode_errors']=[]
            for name in os.listdir(path):
                try: _read(str(path/name))
                except Exception as exc:
                    result['decode_errors'].append(dict(file=name,error=repr(exc)))
                    if len(result['decode_errors'])>=3: break
    except Exception as exc:
        result.update(failures=['PREPROCESS_EXCEPTION'],error=repr(exc),traceback=traceback.format_exc())
    result['seconds']=time.monotonic()-started
    return result

def main():
    cv2.setNumThreads(1)
    root=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
    if not root.exists(): root=Path('/kaggle/input/rsna-knee-abnormality-detection')
    out=Path('/kaggle/working'); started=time.monotonic()
    tasks=[]; inv={}
    for split in ['test','train']:
        rows,inv[split]=inventory(root,split); tasks.extend(rows)
    summary=dict(status='RUNNING',scope='public train and visible test only',gpu=False,
                 preprocessing_sha256=hashlib.sha256(Path(__file__).with_name('preprocess.py').read_bytes()).hexdigest(),
                 inventory=inv,total_selected=len(tasks),processed=0,failed_series=0,failures_by_type={},examples=[],
                 complete=False,hidden_root_cause_confirmed=False)
    def save():
        summary['elapsed_seconds']=time.monotonic()-started
        (out/'public_data_audit.json').write_text(json.dumps(summary,indent=2))
    save()
    # Four bounded tasks at a time; no array retention between series.
    with (out/'public_data_series.jsonl').open('w') as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            iterator=iter(tasks)
            pending={pool.submit(inspect_one,r) for r in [next(iterator,None) for _ in range(4)] if r is not None}
            while pending:
                done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
                for job in done:
                    row=job.result(); handle.write(json.dumps(row)+'\n')
                    summary['processed']+=1
                    if row['failures']:
                        summary['failed_series']+=1
                        for failure in row['failures']:
                            summary['failures_by_type'][failure]=summary['failures_by_type'].get(failure,0)+1
                        if len(summary['examples'])<30: summary['examples'].append(row)
                    next_row=next(iterator,None)
                    if next_row is not None: pending.add(pool.submit(inspect_one,next_row))
                if summary['processed']%100<len(done):
                    handle.flush();save()
                    print('PUBLIC AUDIT',summary['processed'],'/',len(tasks),'failed',summary['failed_series'],flush=True)
    summary.update(status='COMPLETE',complete=True)
    save();print('PUBLIC AUDIT COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__': main()
