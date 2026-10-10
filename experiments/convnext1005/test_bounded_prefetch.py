import threading
import unittest
import torch
from bounded_prefetch import prefetch_one

class PrefetchTests(unittest.TestCase):
    def test_actual_cpu_batches_identical_including_partial_last_batch(self):
        dataset=torch.utils.data.TensorDataset(torch.arange(9*3*8*8,dtype=torch.int64).reshape(9,3,8,8))
        reference=list(torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=0,shuffle=False))
        actual=list(prefetch_one(torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=0,shuffle=False)))
        self.assertEqual(len(actual),3)
        for left,right in zip(reference,actual):self.assertTrue(torch.equal(left[0],right[0]))

    def test_only_one_ahead(self):
        second_ready=threading.Event();produced=[]
        def inputs():
            for i in range(100):
                produced.append(i)
                if i==1:second_ready.set()
                yield i
        values=prefetch_one(inputs())
        self.assertEqual(next(values),0)
        self.assertTrue(second_ready.wait(2))
        self.assertEqual(produced,[0,1])
        values.close()

    def test_source_error_is_not_swallowed(self):
        def inputs():
            yield 'first'
            raise ValueError('bad DICOM')
        values=prefetch_one(inputs())
        self.assertEqual(next(values),'first')
        with self.assertRaisesRegex(ValueError,'bad DICOM'):next(values)

    def test_empty(self):self.assertEqual(list(prefetch_one([])),[])

if __name__=='__main__':unittest.main()
