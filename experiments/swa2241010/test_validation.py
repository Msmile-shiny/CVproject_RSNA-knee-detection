"""Exercise the actual two-stage verifier against copied visible outputs."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import pandas as pd
from run import HERE,verify_frames

class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=HERE)
        self.out=Path(self.tmp.name)
        source=HERE.parents[1]/'results/swa224_fixed15_v1'
        self.receipt=json.loads((source/'swa224_receipt.json').read_text())
        for name in self.receipt['hashes']:shutil.copyfile(source/name,self.out/name)

    def tearDown(self):self.tmp.cleanup()

    def test_actual_result(self):verify_frames(self.out,self.receipt)

    def test_reordered_member(self):
        path=self.out/'independent224.csv'
        df=pd.read_csv(path,dtype={'StudyInstanceUID':str},float_precision='round_trip')
        df.iloc[::-1].to_csv(path,index=False)
        verify_frames(self.out,self.receipt)

    def test_wrong_identity_rejected(self):
        path=self.out/'independent224.csv';df=pd.read_csv(path,dtype={'StudyInstanceUID':str})
        df.loc[0,'StudyInstanceUID']='wrong-study';df.to_csv(path,index=False)
        with self.assertRaises(AssertionError):verify_frames(self.out,self.receipt)

    def test_neutral_parent_rejected(self):
        path=self.out/'parent950.csv';df=pd.read_csv(path,dtype={'StudyInstanceUID':str})
        df.iloc[:,1:]=.5;df.to_csv(path,index=False)
        with self.assertRaises(AssertionError):verify_frames(self.out,self.receipt)

if __name__=='__main__':unittest.main()
