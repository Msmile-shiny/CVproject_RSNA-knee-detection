"""CPU tensor tests; no weights, data or network required."""
import sys
from pathlib import Path
import unittest
import ast
import tempfile
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks'))
from stage6_resnet_core import KneeResNet, centers, stable_fold, loss_fn


class Stage6Tests(unittest.TestCase):
    def test_centers(self):
        self.assertEqual(centers(2,4).tolist(), [])
        self.assertEqual(centers(3,4).tolist(), [1])
        self.assertEqual(len(set(centers(12,4))),4)

    def test_split(self):
        self.assertEqual(stable_fold('patient-a'),stable_fold('patient-a'))
        self.assertTrue(0 <= stable_fold('patient-a') < 5)

    def test_gradients_and_mask(self):
        torch.set_num_threads(2)
        model=KneeResNet().train()
        images=torch.randn(1,2,3,32,32)
        mask=torch.tensor([[True,False]])
        slot=torch.tensor([[0,1]])
        logits=model(images,mask,slot)
        loss=loss_fn(logits,torch.zeros_like(logits),torch.ones_like(logits),torch.ones_like(logits))
        loss.backward()
        self.assertGreater(model.encoder.conv1.weight.grad.abs().sum().item(),0)
        model.eval()
        with torch.no_grad():
            first=model(images,mask,slot)
            images[:,1]=10000
            torch.testing.assert_close(first,model(images,mask,slot))
        with self.assertRaises(ValueError):
            model(images,torch.zeros_like(mask),slot)

    def test_dicom_geometry_and_real_neighbors(self):
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'notebooks/stage6_resnet_runtime.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read_volume')
        import cv2
        env={'np':np,'pydicom':pydicom,'cv2':cv2,'S6':{'size':32}}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'read_volume','exec'),env)
        with tempfile.TemporaryDirectory() as folder:
            for index,pos in enumerate([2,0,1]):
                meta=FileMetaDataset(); meta.TransferSyntaxUID=ExplicitVRLittleEndian
                ds=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
                ds.Rows=8; ds.Columns=8; ds.SamplesPerPixel=1
                ds.PhotometricInterpretation='MONOCHROME2'
                ds.BitsAllocated=16; ds.BitsStored=16; ds.HighBit=15; ds.PixelRepresentation=0
                ds.ImageOrientationPatient=[1,0,0,0,1,0]; ds.ImagePositionPatient=[0,0,pos]
                ds.PixelSpacing=[2,1]
                ds.PixelData=np.full((8,8),pos*100,dtype=np.uint16).tobytes()
                ds.save_as(Path(folder)/f'{index}.dcm')
            vol=env['read_volume'](Path(folder))
            self.assertEqual(vol.shape,(3,32,32))
            self.assertLess(vol[0].mean(),vol[1].mean())
            self.assertLess(vol[1].mean(),vol[2].mean())
            self.assertTrue((vol[:,:,:8]==0).all())


if __name__=='__main__':
    unittest.main()
