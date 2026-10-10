import json
import unittest
import numpy as np
import pandas as pd
from probit_candidate import blend,HERE

class FormulaTest(unittest.TestCase):
    def test_notebook_matches_verifier(self):
        nb=json.loads((HERE/'probit-candidate/apex.ipynb').read_bytes())
        code=next(''.join(c['source']) for c in nb['cells'] if '            _linear =' in ''.join(c['source']))
        start=code.index('            from scipy.special import ndtri, expit')
        stop=code.index('\n        # Final percentile',start)
        import textwrap
        block=textwrap.dedent(code[start:stop])
        rng=np.random.default_rng(42)
        for w in [.04,.08,.24,.48]:
            a=pd.DataFrame({'test':rng.random(257)}).rank(pct=True)
            r=pd.DataFrame({'test':rng.random(257)}).rank(pct=True)
            dest=a.copy()
            exec(block,dict(_o_np=np,w_opt=w,col='test',rank_sota=a,rank_own=r,tier3_df=dest))
            np.testing.assert_allclose(dest['test'],blend(a['test'].to_numpy(),r['test'].to_numpy(),w),rtol=0,atol=0)
            self.assertTrue(np.isfinite(dest['test']).all())

if __name__=='__main__':unittest.main()
