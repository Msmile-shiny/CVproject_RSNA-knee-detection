import tempfile
from pathlib import Path
import unittest
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from audit_public_data import inspect_one

class PublicAuditTests(unittest.TestCase):
    def make_series(self, root):
        for i in range(3):
            meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID=pydicom.uid.MRImageStorage
            meta.MediaStorageSOPInstanceUID=generate_uid()
            ds=FileDataset(str(root/f'{i}.dcm'),{},file_meta=meta,preamble=b'\0'*128)
            ds.Rows=ds.Columns=16;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
            ds.BitsAllocated=ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0
            ds.PixelSpacing=[.4,.4];ds.ImageOrientationPatient=[1,0,0,0,1,0]
            ds.ImagePositionPatient=[0,0,float(i)];ds.InstanceNumber=i
            ds.PixelData=(np.arange(256,dtype=np.uint16).reshape(16,16)+i).tobytes()
            ds.save_as(root/f'{i}.dcm',enforce_file_format=True)

    def test_clean_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_series(root)
            result=inspect_one(('train','synthetic',0,'clean',root,3))
            self.assertEqual(result['failures'],[])
            self.assertEqual(result['shape'],[3,384,384])

    def test_bad_file_captures_original_error_without_hiding_valid_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_series(root)
            (root/'broken.dcm').write_bytes(b'not a dicom')
            result=inspect_one(('train','synthetic',0,'bad',root,4))
            self.assertEqual(result['shape'],[3,384,384])
            self.assertEqual(result['failures'],['STRICT_N_BAD_ABORT'])
            self.assertEqual(result['info']['n_bad'],1)
            self.assertTrue(result['decode_errors'])

if __name__=='__main__': unittest.main()
