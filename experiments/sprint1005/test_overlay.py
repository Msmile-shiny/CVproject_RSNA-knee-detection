import csv
import io
import unittest
from overlay import overlay, ranks

LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.parent = 'StudyInstanceUID,' + ','.join(LABELS) + '\n' + '\n'.join(
            uid + ',' + ','.join([value]*12) for uid, value in [('a','0.12345678901234567'),('b','0.5'),('c','0.9')]) + '\n'
        self.bag = 'StudyInstanceUID,Medial Meniscus,Lateral Meniscus\nc,0.1,0.9\na,0.9,0.1\nb,0.5,0.5\n'
    def test_unchanged_tokens_and_uid_reordering(self):
        before = list(csv.reader(io.StringIO(self.parent)))
        after = list(csv.reader(io.StringIO(overlay(self.parent,self.bag))))
        for a,b in zip(before[1:],after[1:]):
            for col in [0,1,2,*range(5,13)]:
                self.assertEqual(a[col],b[col])
        self.assertAlmostEqual(float(after[1][3]),.9/3+.1)
    def test_rank_ties(self):
        self.assertEqual(ranks([3,1,1]),[1,.5,.5])
    def test_missing_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError): overlay(self.parent,self.bag.replace('c,0.1,0.9\n',''))
        with self.assertRaises(ValueError): overlay(self.parent,self.bag.replace('0.9','nan'))

if __name__ == '__main__': unittest.main()
