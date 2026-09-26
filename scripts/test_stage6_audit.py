"""CPU-only synthetic-header tests; never imports torch."""
import importlib.util
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

PATH=Path(__file__).resolve().parents[1]/'experiments/stage6/series-audit/audit.py'
spec=importlib.util.spec_from_file_location('audit',PATH)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_planes(self):
        self.assertEqual(audit.orientation_plane([1,0,0,0,1,0])[0],'Axial')
        self.assertEqual(audit.orientation_plane([0,1,0,0,0,1])[0],'Sagittal')
        with self.assertRaises(ValueError):audit.orientation_plane([0]*6)

    def test_sampled_headers(self):
        row=namedtuple('Row','StudyInstanceUID SeriesInstanceUID Fluid_Sensitive Fat_Suppression Anatomical_Plane')('1.2.3','1.2.4',1,1,'Coronal')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); d=root/'train_series'/'1.2.3'/'1.2.4';d.mkdir(parents=True)
            for i in range(3):
                meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
                ds=FileDataset(str(d/f'{i}.dcm'),{},file_meta=meta,preamble=b'\0'*128)
                ds.StudyInstanceUID='1.2.3';ds.SeriesInstanceUID='1.2.4'
                ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.ImagePositionPatient=[0,0,i]
                ds.Rows=32;ds.Columns=32;ds.PixelSpacing=[.5,.5]
                ds.save_as(d/f'{i}.dcm')
            r=audit.inspect_series(row,root)
            self.assertEqual(r['header_sample_count'],3);self.assertEqual(r['header_errors'],0)
            self.assertTrue(r['plane_conflict']);self.assertFalse(r['uid_mismatch'])
            self.assertEqual(r['pixel_decode_status'],'NOT_CHECKED')
            (d/'1.dcm').write_bytes(b'not dicom')
            self.assertEqual(audit.inspect_series(row,root)['header_errors'],1)

    def test_matcher_missing_and_largest(self):
        f=pd.DataFrame([dict(Anatomical_Plane='Axial',Fluid_Sensitive=1,Fat_Suppression=1,n_slices=n,SeriesInstanceUID=str(n),dir='unused') for n in [10,20]])
        result=audit.match_slots_for_study(f)
        self.assertEqual(result['AX_FLUID_FS']['series_uid'],'20')
        self.assertIsNone(result['SAG_T1'])


if __name__=='__main__':unittest.main()
