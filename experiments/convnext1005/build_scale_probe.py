"""Build a 1024-public-study, three-hour execution diagnostic (no submission)."""
import ast
import hashlib
import json
from pathlib import Path
import nbformat

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def select_studies(rows):
    times={}
    for row in rows:
        if row['split']=='train': times[row['study']]=times.get(row['study'],0)+row['seconds']
    slow=sorted(times,key=lambda uid:(-times[uid],uid))[:64]
    rest=sorted(set(times)-set(slow),key=lambda uid:hashlib.sha256(('scale-20261007:'+uid).encode()).hexdigest())[:960]
    ids=slow+rest
    assert len(ids)==1024 and len(set(ids))==1024
    return ids,slow

def replace_once(source,old,new):
    assert source.count(old)==1, old[:80]
    return source.replace(old,new)

if __name__=='__main__':
    rows=[json.loads(line) for line in (ROOT/'results/cnx_public_audit_v2/public_data_series.jsonl').read_text().splitlines()]
    ids,slow=select_studies(rows)
    sources={name:(HERE/name).read_text(encoding='utf-8') for name in ['preprocess.py','knee.py','infer.py']}
    source_hashes={name:hashlib.sha256(code.encode()).hexdigest() for name,code in sources.items()}
    assert source_hashes==json.loads((HERE/'build_receipt.json').read_text())['sources']
    worker=(HERE/'worker_tail.py').read_text(encoding='utf-8')
    worker=worker.replace("comp/'test.csv'","comp/'train.csv'").replace("comp/'test_series.csv'","comp/'train_series.csv'").replace("comp/'test_series'","comp/'train_series'")
    worker=replace_once(worker,'ids=test.StudyInstanceUID.tolist()',f'ids={ids!r}\nassert set(ids)<=set(test.StudyInstanceUID)')
    worker=replace_once(worker,'out=[];valid=[];inference_start=time.monotonic()', '''import resource
out=[];valid=[];inference_start=time.monotonic()
load_seconds=0.;gpu_seconds=0.;previous_end=inference_start
progress_path=Path('/kaggle/working/scale_progress.json')
def save_progress(batch_index):
    payload=dict(status='RUNNING',requested=len(ids),processed=len(valid),batch_index=batch_index,
        elapsed_seconds=time.monotonic()-started,load_seconds=load_seconds,gpu_seconds=gpu_seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(0),no_accuracy_evaluation=True)
    tmp=progress_path.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(progress_path)
save_progress(-1)''')
    worker=replace_once(worker,"        assert mask.any(1).all(), 'Empty study'",'''        load_seconds+=time.monotonic()-previous_end
        if time.monotonic()-started>3*3600:
            save_progress(batch_index)
            raise TimeoutError('Public scale diagnostic exceeded three-hour budget')
        gpu_start=time.monotonic()
        assert mask.any(1).all(), 'Empty study' ''')
    worker=replace_once(worker,'        out.append(prob.cpu().numpy())','''        out.append(prob.cpu().numpy())
        torch.cuda.synchronize()
        gpu_seconds+=time.monotonic()-gpu_start
        save_progress(batch_index)''')
    worker=replace_once(worker,'values=np.concatenate(out)','''        previous_end=time.monotonic()
values=np.concatenate(out)''')
    worker=worker.replace("path=work/'cnx3_raw.csv'","path=work/'scale_predictions.csv'")
    worker=worker.replace("(work/'cnx3_worker_receipt.json').write_text","receipt.update(scope='public train execution only',no_accuracy_evaluation=True,load_seconds=load_seconds,gpu_seconds=gpu_seconds)\n(work/'scale_receipt.json').write_text")
    ast.parse(worker)
    sources['worker.py']=worker
    code='''from pathlib import Path
import subprocess,sys,json,traceback
src=Path('/kaggle/working/scale_source');src.mkdir(exist_ok=True)
for name,text in SOURCES.items(): (src/name).write_text(text,encoding='utf-8')
with open('/kaggle/working/scale_worker.log','w') as log:
    try:
        result=subprocess.run([sys.executable,'-u',str(src/'worker.py')],stdout=log,stderr=subprocess.STDOUT,timeout=3*3600+120)
        status='COMPLETE' if result.returncode==0 else 'ERROR'
        outcome=dict(status=status,returncode=result.returncode)
    except subprocess.TimeoutExpired:
        outcome=dict(status='BUDGET_LIMIT',seconds=3*3600+120)
Path('/kaggle/working/scale_outcome.json').write_text(json.dumps(outcome,indent=2))
print(outcome)
if outcome['status']!='COMPLETE':
    print(Path('/kaggle/working/scale_worker.log').read_text()[-8000:])
    raise RuntimeError('Public scale diagnostic did not complete; inspect preserved log and progress')
'''.replace('SOURCES',repr(sources))
    ast.parse(code)
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.validate(notebook)
    folder=HERE/'scale-probe';folder.mkdir(exist_ok=True)
    raw=nbformat.writes(notebook).encode()
    (folder/'scale.ipynb').write_bytes(raw)
    meta=json.loads((HERE/'cnx3fold30/kernel-metadata.json').read_text())
    meta.update(id='easoncyy/rsna-cnx-scale-probe',title='RSNA CNX Scale Probe',code_file='scale.ipynb')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (HERE/'scale_build.json').write_text(json.dumps(dict(status='BUILT_NOT_RUN',source_sha256=hashlib.sha256(raw).hexdigest(),
        studies=ids,slow_cases=slow,model_source_hashes=source_hashes,budget_seconds=10800,
        scope='public train execution; not a validation score; never submit'),indent=2))
    print('Built 1024-study diagnostic; original model and pixel source hashes preserved')
