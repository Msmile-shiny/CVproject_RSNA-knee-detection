import unittest
import numpy as np
import pandas as pd
from rank_blend import rank_blend


class TestRankBlend(unittest.TestCase):
    def setUp(self):
        self.p = pd.DataFrame({'StudyInstanceUID': ['001', '002', '003'], 'ACL': [.1, .9, .5]})
        self.m = pd.DataFrame({'StudyInstanceUID': ['003', '001', '002'], 'ACL': [.9, .5, .1]})

    def test_identity_alignment(self):
        result = rank_blend(self.p, self.m, .1)
        np.testing.assert_allclose(result.ACL, [.9/3 + .2/3, .9 + .1/3, .6 + .1])
        self.assertEqual(result.StudyInstanceUID.tolist(), ['001', '002', '003'])

    def test_bad_inputs(self):
        bad = [self.m.iloc[:2], self.m.iloc[[0,0,1]], self.m.rename(columns={'ACL':'MCL'}), self.m.assign(ACL=[np.nan,.1,.2]), self.m.assign(ACL=[1.1,.1,.2])]
        for member in bad:
            with self.assertRaises(ValueError):
                rank_blend(self.p, member)

    def test_ties_and_endpoints(self):
        tied = self.p.assign(ACL=[.1,.1,.8])
        np.testing.assert_allclose(rank_blend(tied, self.m, 0).ACL, [.5,.5,1])
        np.testing.assert_allclose(rank_blend(tied, self.m, 1).ACL, [2/3,1/3,1])


if __name__ == '__main__':
    unittest.main()
