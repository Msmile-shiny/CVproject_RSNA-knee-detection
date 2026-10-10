import json
from pathlib import Path
import tempfile
import time
import unittest
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent

class FusionTests(unittest.TestCase):
    def run_fusion(self, bad_ids=False, ready=True):
        nb=json.loads((HERE/'author-fusion30/fusion.ipynb').read_text(encoding='utf-8'))
        code=''.join(nb['cells'][-1]['source'])
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root=Path(directory)
            labels=[f'label_{i}' for i in range(12)]
            p=pd.DataFrame(np.tile([.1,.8,.4],(12,1)).T,columns=labels)
            p.insert(0,'StudyInstanceUID',['a','b','c'])
            r=pd.DataFrame(np.tile([.9,.2,.6],(12,1)).T,columns=labels)
            r.insert(0,'StudyInstanceUID',['a','b','c'])
            expected=(.7*p[labels].rank(pct=True)+.3*r[labels].rank(pct=True)).rank(pct=True)
            r=r.iloc[::-1].copy()
            if bad_ids:r.loc[r.index[0],'StudyInstanceUID']='missing'
            p.to_csv(root/'submission.csv',index=False)
            r.to_csv(root/'author929.csv',index=False)
            (root/'author_control_receipt.json').write_text('{}')
            code=code.replace("_AFPath('/kaggle/working')",f'_AFPath({str(root)!r})')
            exec(code,dict(_AFPath=Path,_af_time=time,_af_started=time.monotonic(),_sprint_receipt={'ready_for_scoring':ready}))
            result=pd.read_csv(root/'submission.csv')
            np.testing.assert_allclose(result[labels],expected)
            self.assertEqual(result.StudyInstanceUID.tolist(),['a','b','c'])

    def test_permuted_ids_and_formula(self):self.run_fusion()
    def test_missing_ids_fail(self):
        with self.assertRaises(AssertionError):self.run_fusion(bad_ids=True)
    def test_parent_gate_fails(self):
        with self.assertRaises(AssertionError):self.run_fusion(ready=False)

if __name__=='__main__':unittest.main()
