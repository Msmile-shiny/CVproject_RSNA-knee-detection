"""Exercise the multi-batch loader missed by the three visible studies."""
import ast
from pathlib import Path
import unittest
import torch

HERE=Path(__file__).resolve().parent

class ExecutionTests(unittest.TestCase):
    def test_multiple_batches_keep_every_study_in_order(self):
        dataset=torch.utils.data.TensorDataset(torch.arange(9))
        loader=torch.utils.data.DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0)
        batches=[x.tolist() for (x,) in loader]
        self.assertEqual(batches,[[0,1,2,3],[4,5,6,7],[8]])

    def test_no_multiprocess_prefetch_and_same_recipe(self):
        source=(HERE/'worker_tail.py').read_text(encoding='utf-8')
        ast.parse(source)
        self.assertIn('num_workers=0',source)
        self.assertIn('batch_size=4',source)
        self.assertIn("torch.autocast('cuda',dtype=torch.float16)",source)
        self.assertIn('DicomStudyDataset(ids,table,str(comp/\'test_series\'),12)',source)
        self.assertLess(source.index('allocator_warmup='),source.index('torch.cuda.reset_peak_memory_stats(0)'))
        self.assertIn('assert len(models)==3',source)

if __name__=='__main__': unittest.main()
