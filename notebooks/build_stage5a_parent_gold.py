"""Wrap the user's downloaded parent V1 to infer on a Gold-only virtual test root."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reference_code/parent936/rsna-knee-abnormality-detectionv1.ipynb'

def build():
    nb=json.loads(SOURCE.read_text(encoding='utf-8'))
    cells=[]
    def add(source,kind='code'):
        if kind=='code': compile(source,'parent-gold','exec')
        c=dict(cell_type=kind,id=f'gold-{len(cells):02d}',metadata={},source=source.splitlines(True))
        if kind=='code': c.update(outputs=[],execution_count=None)
        cells.append(c)
    add('''# Parent V1 Gold diagnostic export — NEVER SUBMIT THIS NOTEBOOK

Source: easoncyy/rsna-knee-abnormality-detectionv1, API currentVersionNumber=1.
User supplied scriptVersionId=347718768; API does not return this ID, so the mapping
is not independently verified. Same assets as the scored notebook are required.
This notebook substitutes a Gold-only test root. Model weights and fusion remain
unchanged. Any Gold training exposure of public model assets is UNKNOWN; this is
a development comparison, not out-of-fold validation. Original attributions follow.
''','markdown')
    add('''import os, json, hashlib
from pathlib import Path
import pandas as pd
_s5_roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
_s5_real=next(p for p in _s5_roots if (p/'train.csv').is_file())
_s5_targets=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
_s5_train=pd.read_csv(_s5_real/'train.csv',dtype={'StudyInstanceUID':str})
_s5_gold=_s5_train.loc[_s5_train[_s5_targets].notna().all(axis=1)].sort_values('StudyInstanceUID')
assert len(_s5_gold)>0 and _s5_gold.StudyInstanceUID.is_unique
_s5_virtual=Path('/kaggle/working/stage5a_gold_input'); _s5_virtual.mkdir(exist_ok=True)
_s5_gold[['StudyInstanceUID']].to_csv(_s5_virtual/'test.csv',index=False)
_s5_sample=_s5_gold[['StudyInstanceUID']].copy()
for _s5_c in _s5_targets: _s5_sample[_s5_c]=.5
_s5_sample.to_csv(_s5_virtual/'sample_submission.csv',index=False)
_s5_series=pd.read_csv(_s5_real/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
_s5_series.loc[_s5_series.StudyInstanceUID.isin(_s5_gold.StudyInstanceUID)].to_csv(_s5_virtual/'test_series.csv',index=False)
for _s5_name in ('train.csv','train_series.csv','train_series'):
    _s5_link=_s5_virtual/_s5_name
    if not _s5_link.exists(): _s5_link.symlink_to(_s5_real/_s5_name,target_is_directory=_s5_name=='train_series')
(_s5_virtual/'test_series').mkdir(exist_ok=True)
for _s5_uid in _s5_gold.StudyInstanceUID:
    _s5_link=_s5_virtual/'test_series'/_s5_uid
    if not _s5_link.exists(): _s5_link.symlink_to(_s5_real/'train_series'/_s5_uid,target_is_directory=True)
_s5_gold[['StudyInstanceUID']+_s5_targets].to_csv('/kaggle/working/parent_gold_truth.csv',index=False)
os.environ['PUBLIC0033_COMPETITION_ROOT']=str(_s5_virtual)
''')
    changed=[]
    for i,original in enumerate(nb['cells']):
        src=''.join(original['source'])
        if original['cell_type']=='code':
            for path in ('/kaggle/input/competitions/rsna-knee-abnormality-detection','/kaggle/input/rsna-knee-abnormality-detection'):
                src=src.replace(path,'/kaggle/working/stage5a_gold_input')
            changed.append(i)
        add(src,original['cell_type'])
        if original['cell_type']=='code':
            add(f'''# Snapshot outputs before later parent cells overwrite or remove them.
import shutil as _s5_shutil
from pathlib import Path as _s5_Path
_s5_snap=_s5_Path('/kaggle/working/parent_gold_stages'); _s5_snap.mkdir(exist_ok=True)
for _s5_file in _s5_Path('/kaggle/working').glob('submission*.csv'):
    _s5_shutil.copyfile(_s5_file,_s5_snap/('cell{i}_'+_s5_file.name))
''')
    add('''import pandas as _s5_pd
from pathlib import Path as _s5_Path
import json as _s5_json
_s5_p=_s5_pd.read_csv('/kaggle/working/submission.csv',dtype={'StudyInstanceUID':str})
_s5_y=_s5_pd.read_csv('/kaggle/working/parent_gold_truth.csv',dtype={'StudyInstanceUID':str})
assert _s5_p.StudyInstanceUID.tolist()==_s5_y.StudyInstanceUID.tolist()
assert _s5_p.columns.tolist()==_s5_y.columns.tolist()
import numpy as _s5_np
assert _s5_np.isfinite(_s5_p.iloc[:,1:].to_numpy()).all()
_s5_p.to_csv('/kaggle/working/parent936_gold.csv',index=False)
_s5_Path('/kaggle/working/submission.csv').rename('/kaggle/working/NOT_FOR_SUBMISSION_gold.csv')
print('Gold export complete: parent936_gold.csv. Do not submit this notebook.')
''')
    nb['cells']=cells
    nb['metadata']['stage5a_parent_source']={'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'api_version':1,'user_script_version_id':347718768,'id_mapping_verified':False,'path_rewritten_cells':changed}
    out=ROOT/'notebooks/kaggle_stage5a_parent936_gold.ipynb'
    out.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8'); print(out)

if __name__=='__main__': build()
