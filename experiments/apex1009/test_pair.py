import ast
import unittest
import numpy as np
import torch
from build import reader_source,HERE

class PairTests(unittest.TestCase):
    def test_equivalence(self):
        old,_=reader_source();tree=ast.parse(old)
        klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DicomStudyDataset')
        for length in [0,2,3,4,9,64]:
            calls=[]
            def loader(path):
                calls.append(path)
                if path.endswith('bad'):raise ValueError('synthetic decode error')
                return np.arange(length*4*4,dtype=np.uint8).reshape(length,4,4),{}
            env=dict(np=np,torch=torch,N_SLOTS=6,SIZE=4,load_series=loader,
                     window_centres=lambda n,k,t:np.linspace(1,n-2,k).round().astype(int))
            exec(compile(ast.Module(body=[klass],type_ignores=[]),'original','exec'),env)
            exec((HERE/'paired_dataset.py').read_text(),env)
            table={'s':[['ok'],['bad'],[],[],[],[]]}
            original=[env['DicomStudyDataset'](['s'],table,'root',4,offset=o)[0] for o in (0,1)]
            self.assertEqual(len(calls),4);calls.clear()
            paired=env['PairedDicomStudyDataset'](['s'],table,'root',4)[0]
            self.assertEqual(len(calls),2)
            for expected,actual in zip(original,paired):
                for a,b in zip(expected,actual):self.assertTrue(torch.equal(a,b))

if __name__=='__main__':unittest.main()
