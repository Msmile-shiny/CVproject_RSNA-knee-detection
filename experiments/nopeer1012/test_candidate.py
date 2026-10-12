"""Exercise actual candidate fusion code on synthetic inputs; no AUC claim."""
import ast,json,tempfile,textwrap,sys
from pathlib import Path
import numpy as np
import pandas as pd
from run import HERE,ROOT,shared

parent=json.loads((ROOT/'experiments/swa2241010/upstream/parent950.ipynb').read_bytes())
source=''.join(json.loads((HERE/'candidate/reader8.ipynb').read_bytes())['cells'][10]['source'])
tree=ast.parse(''.join(parent['cells'][9]['source']))
weights=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OPTIMAL_WEIGHTS' for t in n.targets))
sys.path.insert(0,str(ROOT/'experiments/convnext1005'))
from run_solo import LABELS
labels=LABELS[1:]
def execute(code,a,r):
    start=code.rindex('tier3_df =');end=code.index('tier3_df.to_csv',start)
    fragment=textwrap.dedent('        '+code[start:end])
    env=dict(sota_df=a,rank_sota=a[labels].rank(pct=True),rank_own=r[labels].rank(pct=True),labels=labels,OPTIMAL_WEIGHTS=weights,_o_np=np)
    exec(fragment,env)
    return env['tier3_df']

rng=np.random.default_rng(1012)
a=pd.DataFrame(rng.random((512,12)),columns=labels);a.insert(0,'StudyInstanceUID',[str(i) for i in range(512)])
r=a.copy();r[labels]=rng.random((512,12))
actual=execute(source,a,r)
modified_a=a.copy();modified_r=r.copy()
modified_a['Effusion']=1-a.Effusion;modified_r['Effusion']=1-r.Effusion
np.testing.assert_array_equal(actual.ACL,execute(source,modified_a,modified_r).ACL)
old=execute(''.join(parent['cells'][9]['source']),a,r)
assert not np.array_equal(old.ACL,actual.ACL),'Synthetic input must distinguish the removed graph'
with tempfile.TemporaryDirectory(dir=ROOT/'results',prefix='nopeer-test-') as directory:
    out=Path(directory)
    a.to_csv(out/'_sota_0949.csv',index=False);r.iloc[::-1].to_csv(out/'_own.csv',index=False);actual.to_csv(out/'submission.csv',index=False)
    receipt=dict(study_count=512,hashes={n:'' for n in ['submission.csv','_sota_0949.csv','_own.csv']})
    shared().verify(out,receipt,use_peers=False)
    old.to_csv(out/'submission.csv',index=False)
    try:shared().verify(out,receipt,use_peers=False)
    except AssertionError:pass
    else:raise AssertionError('Old graph output was incorrectly accepted')
print('Actual candidate code: cross-finding isolation, changed output, ID alignment and old-graph rejection PASS')
