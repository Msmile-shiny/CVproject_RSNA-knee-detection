"""Exercise complete CPU pipeline on synthetic DICOM, NEVER real score evidence.

Usage: python scripts/smoke_stage6_synthetic.py --weights /path/to/official.pth
Only the synthetic labels' hash is substituted in the test namespace.
"""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

parser=argparse.ArgumentParser()
parser.add_argument('--weights',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
notebook=json.loads((root/'experiments/stage6/notebook/stage6a.ipynb').read_text())
env={}
for cell in notebook['cells'][1:-1]:
    exec(''.join(cell['source']),env)
source=''.join(notebook['cells'][-1]['source'])
exec(source.rsplit('\nrun()',1)[0],env)
env['torch'].cuda.is_available=lambda: False  # This integration test is explicitly CPU-only.
with tempfile.TemporaryDirectory() as folder:
    folder=Path(folder)
    truth=[]; series=[]; labels=[]
    for i in range(24):
        uid=f'study{i:02}'
        row={'StudyInstanceUID':uid}
        row.update({c:float(i%2) if i<4 else np.nan for c in env['TARGET_COLUMNS']})
        truth.append(row)
        row={'StudyInstanceUID':uid}
        row.update({c:float(i%2) for c in env['PROB_COLS']})
        row.update({c:1. for c in env['WEIGHT_COLS']+env['MASK_COLS']})
        labels.append(row)
        series.append(dict(StudyInstanceUID=uid,SeriesInstanceUID='sag',Anatomical_Plane='Sagittal',Fluid_Sensitive=1,Fat_Suppression=1))
        directory=folder/'train_series'/uid/'sag'; directory.mkdir(parents=True)
        for j in range(3):
            metadata=FileMetaDataset(); metadata.TransferSyntaxUID=ExplicitVRLittleEndian
            ds=FileDataset(None,{},file_meta=metadata,preamble=b'\0'*128)
            ds.Rows=8;ds.Columns=8;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
            ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0
            ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.ImagePositionPatient=[0,0,j];ds.PixelSpacing=[1,1]
            ds.PixelData=(np.arange(64,dtype=np.uint16).reshape(8,8)+i+j).tobytes()
            ds.save_as(directory/f'{j}.dcm')
    pd.DataFrame(truth).to_csv(folder/'train.csv',index=False)
    pd.DataFrame(series).to_csv(folder/'train_series.csv',index=False)
    pd.DataFrame(labels).to_csv(folder/'v5_labels.csv',index=False)
    real_sha=env['sha']
    env['sha']=lambda p: 'c13adffaabf4f8e518abb038282bb1aa09baac7652a9165e030710c457d0be6a' if Path(p).name=='v5_labels.csv' else real_sha(p)
    env['S6'].update(competition=str(folder),weights=str(args.weights.resolve()),labels=str(folder/'v5_labels.csv'),output=str(folder/'output'),size=32,windows=1)
    env['run']()
    receipt=json.loads((folder/'output/run_receipt.json').read_text())
    assert receipt['status']=='SMOKE_COMPLETE'
    assert receipt['first_backbone_gradient_norm']>0
    # Restore exact optimizer/RNG and finish an additional epoch.
    env['S6'].update(resume=str(folder/'output'),output=str(folder/'resumed'),epochs=2)
    env['run']()
    receipt=json.loads((folder/'resumed/run_receipt.json').read_text())
    assert receipt['completed_epochs']==2
    history=json.loads((folder/'resumed/history.json').read_text())
    assert [h['epoch'] for h in history]==[1,2]
    print('PASS: synthetic DICOM -> train -> export -> resume; not a score estimate')
