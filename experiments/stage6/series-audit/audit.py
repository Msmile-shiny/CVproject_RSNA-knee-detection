SLOTS = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('SAG_FLUID_NOFS', 'Sagittal', True, False), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]

def match_slots_for_study(study_series_df):
    """为单个 study 的每个 slot 匹配最优 series。"""
    slots_found = {}
    for slot_name, plane, fluid, fatsat in SLOTS:
        candidates = study_series_df[(study_series_df['Anatomical_Plane'] == plane) & (study_series_df['Fluid_Sensitive'] == (1 if fluid else 0)) & (study_series_df['Fat_Suppression'] == (1 if fatsat else 0))]
        if len(candidates) == 0 and (not fluid):
            candidates = study_series_df[(study_series_df['Anatomical_Plane'] == plane) & (study_series_df['Fluid_Sensitive'] == 0)]
        if len(candidates) > 0:
            best = candidates.sort_values('n_slices', ascending=False).iloc[0]
            slots_found[slot_name] = {'series_uid': best['SeriesInstanceUID'], 'dir': best['dir'], 'n_slices': int(best['n_slices']), 'plane': plane}
        else:
            slots_found[slot_name] = None
    return slots_found

"""Read-only sampled DICOM-header audit. Does not change training selection.

SLOTS and match_slots_for_study are injected verbatim by build_stage6_audit.py.
Three lexical file positions per series are sampled, NOT a complete volume audit.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom

TAGS=['StudyInstanceUID','SeriesInstanceUID','ImageOrientationPatient','ImagePositionPatient',
      'PixelSpacing','Rows','Columns','SeriesDescription','SequenceName','ScanOptions',
      'Manufacturer','MagneticFieldStrength','SliceThickness','EchoTime','RepetitionTime']


def orientation_plane(iop):
    a=np.asarray(iop,dtype=float)
    if a.shape!=(6,) or not np.isfinite(a).all():
        raise ValueError('Invalid orientation')
    normal=np.cross(a[:3],a[3:]); norm=np.linalg.norm(normal)
    if norm<1e-6:
        raise ValueError('Degenerate orientation')
    normal/=norm
    return ['Sagittal','Coronal','Axial'][int(np.argmax(np.abs(normal)))], normal


def inspect_series(row, root):
    d=root/'train_series'/str(row.StudyInstanceUID)/str(row.SeriesInstanceUID)
    files=sorted(p for p in d.iterdir() if p.is_file()) if d.is_dir() else []
    result=dict(row._asdict(),dir=str(d),n_slices=len(files),header_sample_count=0,
                header_errors=0,uid_mismatch=False,plane_conflict=False,
                shape_varies=False,spacing_varies=False,orientation_varies=False,
                duplicate_sample_position=False,pixel_decode_status='NOT_CHECKED')
    shapes=[]; spacings=[]; normals=[]; positions=[]; planes=[]; errors=[]
    for i in sorted(set([0,len(files)//2,len(files)-1])) if files else []:
        try:
            ds=pydicom.dcmread(files[i],stop_before_pixels=True,specific_tags=TAGS)
            result['header_sample_count']+=1
            result['uid_mismatch'] |= str(ds.StudyInstanceUID)!=str(row.StudyInstanceUID) or str(ds.SeriesInstanceUID)!=str(row.SeriesInstanceUID)
            plane,normal=orientation_plane(ds.ImageOrientationPatient)
            planes.append(plane); normals.append(normal)
            shapes.append((int(ds.Rows),int(ds.Columns)))
            spacing=tuple(float(v) for v in ds.PixelSpacing)
            if len(spacing)!=2 or not np.isfinite(spacing).all() or min(spacing)<=0:
                raise ValueError('Invalid spacing')
            spacings.append(spacing)
            position=np.asarray(ds.ImagePositionPatient,dtype=float)
            if position.shape!=(3,) or not np.isfinite(position).all():
                raise ValueError('Invalid position')
            positions.append(float(np.dot(position,normals[0])))
            if 'Manufacturer' not in result:
                for tag in TAGS[7:]:
                    value=getattr(ds,tag,None)
                    result[tag]=None if value is None else str(value)
                result['sample_shape']=str(shapes[-1]); result['sample_spacing_mm']=str(spacing)
        except Exception as exc:
            result['header_errors']+=1; errors.append(type(exc).__name__+': '+str(exc)[:160])
    result.update(geometric_planes='|'.join(sorted(set(planes))),
                  plane_conflict=any(p!=row.Anatomical_Plane for p in planes),
                  shape_varies=len(set(shapes))>1,spacing_varies=len(set(spacings))>1,
                  orientation_varies=any(not np.allclose(n,normals[0],atol=.01) for n in normals),
                  duplicate_sample_position=len(set(round(p,4) for p in positions))<len(positions),
                  header_error_messages=' | '.join(errors))
    return result


def run():
    started=time.monotonic(); root=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
    if not root.is_dir(): root=Path('/kaggle/input/rsna-knee-abnormality-detection')
    out=Path('/kaggle/working'); out.mkdir(exist_ok=True)
    meta=pd.read_csv(root/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    assert not meta.duplicated(['StudyInstanceUID','SeriesInstanceUID']).any()
    rows=[]
    for row in meta.itertuples(index=False):
        rows.append(inspect_series(row,root))
        if len(rows)%500==0: print('Audited series',len(rows),'/',len(meta),flush=True)
        if time.monotonic()-started>90*60: break
    manifest=pd.DataFrame(rows)
    manifest.to_parquet(out/'series_manifest.parquet',index=False)
    manifest.to_csv(out/'series_manifest.csv',index=False)
    complete=len(rows)==len(meta)
    slots=[]
    if complete:
        for uid,g in manifest.groupby('StudyInstanceUID',sort=False):
            eligible=g[g.n_slices>=3]
            selected=match_slots_for_study(eligible)
            used=[v['series_uid'] for v in selected.values() if v]
            for name,plane,fluid,fat in SLOTS:
                chosen=selected[name]
                slots.append(dict(StudyInstanceUID=uid,slot=name,
                    selected_series=None if chosen is None else chosen['series_uid'],
                    missing=chosen is None,duplicate_selection=chosen is not None and used.count(chosen['series_uid'])>1,
                    candidate_count=int(((eligible.Anatomical_Plane==plane)&(eligible.Fluid_Sensitive==int(fluid))&(eligible.Fat_Suppression==int(fat))).sum())))
        pd.DataFrame(slots).to_csv(out/'slot_manifest.csv',index=False)
    flag_cols=['uid_mismatch','plane_conflict','shape_varies','spacing_varies','orientation_varies','duplicate_sample_position']
    flags=manifest[flag_cols].any(axis=1)|(manifest.header_errors>0)|(manifest.n_slices<3)
    review=manifest[flags].sort_values(['StudyInstanceUID','SeriesInstanceUID'])
    review.head(100).to_csv(out/'review_queue.csv',index=False)
    receipt=dict(status='AUDIT_COMPLETE' if complete else 'AUDIT_PARTIAL',
        series_count=len(rows),expected_series_count=len(meta),study_count=int(manifest.StudyInstanceUID.nunique()),
        train_series_sha256=hashlib.sha256((root/'train_series.csv').read_bytes()).hexdigest(),
        scope='First/middle/last lexical files: header only, no pixel decode or exhaustive sorting check',
        flagged_series=int(flags.sum()),header_error_series=int((manifest.header_errors>0).sum()),
        flags={k:int(manifest[k].sum()) for k in flag_cols},seconds=time.monotonic()-started,
        slot_missing_counts={} if not slots else pd.DataFrame(slots).groupby('slot').missing.sum().astype(int).to_dict(),
        modifies_training=False)
    (out/'audit_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__': run()
