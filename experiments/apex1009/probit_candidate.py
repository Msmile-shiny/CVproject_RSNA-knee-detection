"""Single-factor proposal: preserve scored image path; change fusion geometry only."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri, expit

HERE=Path(__file__).resolve().parent

def blend(anchor, reader, weight):
    linear=(1-weight)*anchor+weight*reader
    nonlinear=expit((1-weight)*ndtri(np.clip(anchor,1e-5,1-1e-5))+weight*ndtri(np.clip(reader,1e-5,1-1e-5)))
    return .5*linear+.5*nonlinear+(reader-.5)*1e-4

def build():
    raw=(HERE/'candidate/apex.ipynb').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==json.loads((HERE/'build.json').read_text())['source_sha256']
    nb=copy.deepcopy(json.loads(raw));changes=0
    old='base_fused = (1.0 - w_opt) * rank_sota[col] + w_opt * rank_own[col]'
    new='''from scipy.special import ndtri, expit
            _linear = (1.0 - w_opt) * rank_sota[col] + w_opt * rank_own[col]
            _nonlinear = expit((1.0 - w_opt) * ndtri(_o_np.clip(rank_sota[col],1e-5,1-1e-5)) + w_opt * ndtri(_o_np.clip(rank_own[col],1e-5,1-1e-5)))
            base_fused = 0.5 * _linear + 0.5 * _nonlinear'''
    for c in nb['cells']:
        if c['cell_type']!='code':continue
        text=''.join(c['source'])
        if old in text:
            assert text.count(old)==1
            text=text.replace(old,new);changes+=1
        c.update(source=text,outputs=[],execution_count=None)
    assert changes==1
    import ast
    from IPython.core.inputtransformer2 import TransformerManager
    t=TransformerManager()
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(t.transform_cell(c['source']))
    folder=HERE/'probit-candidate';folder.mkdir(exist_ok=True)
    payload=json.dumps(nb,ensure_ascii=False,indent=1).encode()
    (folder/'apex.ipynb').write_bytes(payload)
    meta=json.loads((HERE/'candidate/kernel-metadata.json').read_text())
    meta.update(id='easoncyy/rsna-apex-probit-only',title='RSNA Apex Probit Only')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (HERE/'probit_build.json').write_text(json.dumps(dict(status='BUILT_NOT_RUN',source_sha256=hashlib.sha256(payload).hexdigest(),parent_submission=56994783,changed_prediction_cells=1,requires_new_formula_verifier=True),indent=2))

if __name__=='__main__':
    # Algebraic checks only: never a performance validation.
    a=np.linspace(.001,1,1000)
    for w in [0,.08,.24,.48,1]:
        p=blend(a,a,w)
        assert np.isfinite(p).all() and (np.diff(p)>0).all()
    assert np.isfinite(blend(np.array([0.,1.]),np.array([1.,0.]),.48)).all()
    build()
    print('Monotonicity/endpoints/syntax PASS; candidate BUILT_NOT_RUN; no score claim')
