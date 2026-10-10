"""Isolate the verified three-fold reader from the parent stack."""
import ast
import hashlib
import json
from pathlib import Path
import nbformat

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    sources={n:(HERE/n).read_text(encoding='utf-8') for n in ['preprocess.py','knee.py','infer.py']}
    expected=json.loads((HERE/'build_receipt.json').read_text())['sources']
    assert {n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}==expected
    sources['worker.py']=(HERE/'worker_tail.py').read_text(encoding='utf-8')
    for text in sources.values():ast.parse(text)
    code='''from pathlib import Path
import subprocess,sys,json,hashlib
import pandas as pd
import numpy as np
work=Path('/kaggle/working')
src=work/'cnx_solo_source';src.mkdir(exist_ok=True)
for name,text in SOURCES.items(): (src/name).write_text(text,encoding='utf-8')
with (work/'solo_worker.log').open('w') as log:
    try:
        subprocess.run([sys.executable,'-u',str(src/'worker.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2*3600)
    except Exception:
        print((work/'solo_worker.log').read_text()[-8000:])
        raise
receipt=json.loads((work/'cnx3_worker_receipt.json').read_text())
assert receipt['status']=='COMPLETE' and receipt['model_count']==3 and receipt['folds']==[0,1,2]
assert receipt['stress_pass'] and receipt['fallback']==0 and receipt['partial_predictions']==0
frame=pd.read_csv(work/'cnx3_raw.csv',dtype={'StudyInstanceUID':str})
comp=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not comp.exists():comp=Path('/kaggle/input/rsna-knee-abnormality-detection')
ids=pd.read_csv(comp/'test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist()
assert frame.StudyInstanceUID.tolist()==ids and frame.StudyInstanceUID.is_unique
assert frame.columns.tolist()==['StudyInstanceUID','ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
v=frame.iloc[:,1:].to_numpy();assert np.isfinite(v).all() and (v>=0).all() and (v<=1).all()
raw=(work/'cnx3_raw.csv').read_bytes()
assert hashlib.sha256(raw).hexdigest()==receipt['raw_sha256']
target=work/'submission.csv';tmp=work/'solo_submission.tmp';tmp.write_bytes(raw);tmp.replace(target)
receipt.update(candidate='threefold_reader_only',parent_used=False,submission_sha256=hashlib.sha256(raw).hexdigest())
(work/'solo_receipt.json').write_text(json.dumps(receipt,indent=2))
print('SOLO COMPLETE',len(frame),'studies',receipt['elapsed_seconds'],'seconds')
'''.replace('SOURCES',repr(sources))
    ast.parse(code)
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.validate(nb)
    folder=HERE/'solo';folder.mkdir(exist_ok=True)
    raw=nbformat.writes(nb).encode();(folder/'solo.ipynb').write_bytes(raw)
    meta=json.loads((HERE/'cnx3fold30/kernel-metadata.json').read_text())
    meta.update(id='easoncyy/rsna-cnx-threefold-solo',title='RSNA CNX Threefold Solo',code_file='solo.ipynb')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (HERE/'solo_build.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(raw).hexdigest(),
        sources={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},parent_used=False,budget_seconds=7200),indent=2))
    print('Built standalone three-fold reader; worker unchanged from v2')
