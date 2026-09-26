"""One-variable attention control, fresh initialization, immutable reference cache."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build():
    reference=ROOT/'experiments/stage6/reference_v3/run_receipt.json'
    proof=json.loads(reference.read_text())
    assert proof['status']=='PILOT_COMPLETE' and proof['completed_epochs']==12
    nb=json.loads((ROOT/'experiments/stage6/notebook/stage6a.ipynb').read_text(encoding='utf-8'))
    sources=[''.join(c['source']) for c in nb['cells']]
    assert hashlib.sha256(''.join(sources[2:]).encode()).hexdigest()==proof['config']['implementation_sha256']
    runtime=sources[-1]
    replacements={
        "if not path.is_file() and S6['resume']:":"if not path.is_file() and S6['cache_input']:",
        "path=Path(S6['resume'])/'pixel_cache'/filename":"path=Path(S6['cache_input'])/'pixel_cache'/filename",
        "if resume and (resume/'pixel_cache'/filename).is_file():":"if (Path(S6['cache_input'])/'pixel_cache'/filename).is_file():",
        "with np.load(resume/'pixel_cache'/filename) as z:":"with np.load(Path(S6['cache_input'])/'pixel_cache'/filename) as z:",
    }
    for old,new in replacements.items():
        assert runtime.count(old)==1,old
        runtime=runtime.replace(old,new)
    sources[-1]=runtime
    sources[4]+='''
# CPU preflight on Kaggle: attention gradients and padding invariance.
_probe=KneeResNet(pooling='attention').eval()
_x=torch.randn(1,3,3,32,32)
_valid=torch.tensor([[True,True,False]])
_slots=torch.tensor([[0,1,2]])
_out=_probe(_x,_valid,_slots)
_changed=_x.clone(); _changed[:,2]=1000
assert torch.allclose(_out,_probe(_changed,_valid,_slots),atol=1e-6)
_out.sum().backward()
assert _probe.attention.weight.grad.abs().sum()>0
assert _probe.encoder.conv1.weight.grad.abs().sum()>0
del _probe,_x,_valid,_slots,_out,_changed
print('Attention preflight passed')
'''
    impl=hashlib.sha256(''.join(sources[2:]).encode()).hexdigest()
    config=dict(proof['config'],pooling='attention',epochs=24,minutes=240,resume='',cache_input='',implementation_sha256=impl)
    sources[0]='# Stage 6D attention control\nFresh training, same labels/fold/24 epochs. Cache reuse only; no checkpoint resume or leaderboard submission.'
    sources[1]='S6 = '+repr(config)+f'''
import hashlib, json
from pathlib import Path
matches=[p for p in Path('/kaggle/input').rglob('run_receipt.json')
         if hashlib.sha256(p.read_bytes()).hexdigest()=={hashlib.sha256(reference.read_bytes()).hexdigest()!r}]
assert len(matches)==1, 'Mount the exact reference v3 output'
S6['cache_input']=str(matches[0].parent)
assert (matches[0].parent/'pixel_cache').is_dir()
assert not S6['resume'], 'Attention control must start from official ImageNet weights'
print('Fresh attention training; cache only:', S6['cache_input'])
'''
    for c,s in zip(nb['cells'],sources):
        c['source']=s.splitlines(True)
        if c['cell_type']=='code':ast.parse(s);c['outputs']=[];c['execution_count']=None
    dest=ROOT/'experiments/stage6/attention';dest.mkdir(parents=True,exist_ok=True)
    (dest/'attention.ipynb').write_text(json.dumps(nb,indent=1),encoding='utf-8')
    meta=json.loads((ROOT/'experiments/stage6/notebook/kernel-metadata.json').read_text())
    meta.update(id='easoncyy/rsna-stage6d-attention-control',title='RSNA Stage6D Attention Control',code_file='attention.ipynb')
    meta['kernel_sources'].append('easoncyy/rsna-stage6a-resnet34-reference')
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(dest)


if __name__=='__main__':build()
