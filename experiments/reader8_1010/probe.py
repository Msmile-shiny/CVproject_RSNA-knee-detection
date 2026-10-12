"""Launch and collect an isolated GPU equivalence/memory probe; never submit."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()

def build():
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager
    nb=json.loads((HERE/'candidate/reader8.ipynb').read_bytes())
    source=''.join(nb['cells'][9]['source']).split('\n',1)[1]
    stream=(HERE/'stream_infer.py').read_text(encoding='utf8')
    built=json.loads((HERE/'stream_build.json').read_text())
    assert sha(source.encode())==built['original_sha256'] and sha(stream.encode())==built['optimized_sha256']
    program='''import sys,importlib.util,gc,time,json,hashlib
from pathlib import Path
import torch,pandas as pd,numpy as np
sys.path.insert(0,'/kaggle/working/own_src')
root=Path('/kaggle/working');data=__DATA__;ck=__CHECKPOINTS__
studies=pd.read_csv(data+'/test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist()
assert len(studies)>0
batch=(studies*4)[:4]
records={};outputs={}
for name in ['original','stream']:
    spec=importlib.util.spec_from_file_location(name,root/'own_src'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    # Four rows exercise the configured full inference batch. Repeated IDs are intentional diagnostic inputs.
    frame=module.run(data,'test',ck,studies=batch,workers=0,bs=4)
    torch.cuda.synchronize()
    outputs[name]=frame.to_numpy()
    assert np.isfinite(outputs[name]).all()
    records[name]=dict(seconds=time.monotonic()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    frame.to_csv(root/f'{name}_probe.csv')
np.testing.assert_allclose(outputs['original'],outputs['stream'],rtol=1e-5,atol=1e-6)
receipt=dict(status='PASS',rows=len(batch),unique_studies=len(set(batch)),workers=0,batch_size=4,records=records,
    max_abs_difference=float(np.max(np.abs(outputs['original']-outputs['stream']))),rtol=1e-5,atol=1e-6,
    source_hashes={name:hashlib.sha256((root/'own_src'/f'{name}.py').read_bytes()).hexdigest() for name in ['original','stream']})
(root/'stream_probe_receipt.json').write_text(json.dumps(receipt,indent=2));print(receipt)
'''
    cells=[nb['cells'][0],nb['cells'][6],nb['cells'][7],nb['cells'][8]]
    cells+=[nbformat.v4.new_code_cell('%%writefile /kaggle/working/own_src/original.py\n'+source),nbformat.v4.new_code_cell('%%writefile /kaggle/working/own_src/stream.py\n'+stream)]
    execute='''import subprocess,sys,os,glob
from pathlib import Path
def _probe_find(name):
    hits=[p for pattern in ['*/'+name,'*/*/'+name,'*/*/*/'+name] for p in Path('/kaggle/input').glob(pattern)]
    assert len(hits)==1, (name,hits)
    return str(hits[0].parent)
_data=_probe_find('test_series.csv');_assets=_probe_find('cnxt_v0_fold0.pt')
_ck=[str(Path(_assets)/f'cnxt_v0_fold{i}.pt') for i in range(3)]
_target='/kaggle/working/probe_env'
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--target',_target,'--find-links',_assets,'pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg'],check=True)
_wheels=[str(p) for pat in ['*/timm-*.whl','*/*/timm-*.whl'] for p in Path('/kaggle/input').glob(pat)]
if _wheels:subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','--target',_target,_wheels[0]],check=True)
_env=dict(os.environ);_env['PYTHONPATH']=_target+os.pathsep+_env.get('PYTHONPATH','')
_program=__PROGRAM__.replace('__DATA__',repr(_data)).replace('__CHECKPOINTS__',repr(_ck))
Path('/kaggle/working/probe_runner.py').write_text(_program,encoding='utf8')
subprocess.run([sys.executable,'/kaggle/working/probe_runner.py'],check=True,timeout=1800,env=_env)
'''.replace('__PROGRAM__',repr(program))
    cells.append(nbformat.v4.new_code_cell(execute))
    result=nbformat.v4.new_notebook(cells=cells,metadata=nb['metadata'])
    for i,c in enumerate(result['cells']):
        c['id']=f'probe-{i}';c.update(outputs=[],execution_count=None)
        ast.parse(TransformerManager().transform_cell(''.join(c['source'])))
    nbformat.validate(result)
    folder=HERE/'probe-candidate';folder.mkdir(exist_ok=True)
    raw=json.dumps(result,ensure_ascii=False,indent=1).encode();(folder/'probe.ipynb').write_bytes(raw)
    meta=json.loads((HERE/'candidate/kernel-metadata.json').read_text());meta.update(id='easoncyy/rsna-reader8-stream-probe',title='RSNA Reader8 Stream Probe',code_file='probe.ipynb')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (HERE/'probe_build.json').write_text(json.dumps(dict(source_sha256=sha(raw),original_sha256=sha(source.encode()),stream_sha256=sha(stream.encode())),indent=2))

def online(action):
    sys.path.insert(0,str(HERE.parent/'convnext1005'))
    from run_scale_probe import prepare_dns
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    prepare_dns();a=KaggleApi();a.authenticate()
    built=json.loads((HERE/'probe_build.json').read_text());raw=(HERE/'probe-candidate/probe.ipynb').read_bytes()
    assert sha(raw)==built['source_sha256'];kernel='easoncyy/rsna-reader8-stream-probe'
    if action in ['launch','repair-launch']:
        if action=='launch':
            assert not (HERE/'probe_launch_attempt.json').exists()
            assert not any(k.ref==kernel for k in a.kernels_list(mine=True,search='rsna-reader8-stream-probe',page_size=100))
        else:
            previous=json.loads((HERE/'probe_launch.json').read_text())
            assert previous['response']['versionNumber']==1 and a.kernels_status(kernel).to_dict()['status']=='ERROR'
            assert not (HERE/'probe_launch_v1.json').exists()
            (HERE/'probe_launch_v1.json').write_text(json.dumps(previous,indent=2))
        (HERE/'probe_launch_attempt.json').write_text(json.dumps(built,indent=2))
        response=a.kernels_push(str(HERE/'probe-candidate'),acc='NvidiaTeslaT4').to_dict()
        (HERE/'probe_launch.json').write_text(json.dumps(dict(build=built,response=response),indent=2));print(response);return
    launch=json.loads((HERE/'probe_launch.json').read_text());version=launch['response']['versionNumber']
    status=a.kernels_status(kernel).to_dict();print(status,flush=True)
    if status['status'] not in ['COMPLETE','ERROR']:return
    out=HERE.parents[1]/'results'/f'reader8_stream_probe_v{version}';out.mkdir(parents=True,exist_ok=True)
    a.kernels_output(kernel,path=str(out),page_size=100)
    assert status['status']=='COMPLETE',f'Inspect failure output {out}'
    req=ApiGetKernelRequest();req.user_name,req.kernel_slug=kernel.split('/')
    with a.build_kaggle_client() as c:remote=c.kernels.kernels_api_client.get_kernel(req)
    assert remote.metadata.to_dict()['currentVersionNumber']==version
    codes=lambda n:[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    assert codes(json.loads(raw))==codes(json.loads(remote.blob.source))
    receipt=json.loads((out/'stream_probe_receipt.json').read_text())
    assert receipt['status']=='PASS' and receipt['rows']==4 and receipt['batch_size']==4
    assert receipt['source_hashes']==dict(original=built['original_sha256'],stream=built['stream_sha256'])
    (HERE/'probe_verified.json').write_text(json.dumps(receipt,indent=2));print(receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['build','launch','repair-launch','collect']);args=p.parse_args()
    if args.action=='build':build()
    else:online(args.action)
