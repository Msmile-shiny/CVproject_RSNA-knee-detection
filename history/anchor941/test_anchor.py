"""Exercise the actual notebook's final gate with complete and damaged outputs."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

ROOT = Path(__file__).resolve().parent
NB = json.loads((ROOT/'notebook/anchor941.ipynb').read_text(encoding='utf-8'))
FINAL = ''.join(NB['cells'][-1]['source'])
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

class FinalGate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frame = pd.DataFrame({'StudyInstanceUID': ['a', 'b', 'c'], **{c: [.1, .5, .9] for c in LABELS}})
        self.frame.to_csv(self.root/'submission.csv', index=False)
        self.frame[['StudyInstanceUID']].to_csv(self.root/'test.csv', index=False)
        for name, record in {
            'v50_v2_repro_receipt.json': {}, '_coat_arm_receipt.json': {'models': 3, 'fallback_studies': 0},
            '_coat_raptor_blend_receipt.json': {}, 'probe22_outer_routing_receipt.json': {'status': 'passed'}
        }.items():
            (self.root/name).write_text(json.dumps(record))

    def tearDown(self):
        self.temp.cleanup()

    def run_gate(self):
        real_path = Path
        def mapped(path):
            return self.root if str(path) == '/kaggle/working' else real_path(path)
        with patch('pathlib.Path', side_effect=mapped):
            exec(compile(FINAL, 'final_gate', 'exec'), {'COMP': self.root, 'ANCHOR_STARTED': time.time()})

    def test_valid_output_is_unchanged(self):
        before = (self.root/'submission.csv').read_bytes()
        self.run_gate()
        self.assertEqual(before, (self.root/'submission.csv').read_bytes())
        self.assertTrue(json.loads((self.root/'anchor941_run_receipt.json').read_text())['complete'])

    def test_uid_order_mismatch_rejected(self):
        self.frame.iloc[::-1].to_csv(self.root/'submission.csv', index=False)
        with self.assertRaises(AssertionError): self.run_gate()

    def test_nan_rejected(self):
        self.frame.loc[0, 'ACL'] = float('nan')
        self.frame.to_csv(self.root/'submission.csv', index=False)
        with self.assertRaises(AssertionError): self.run_gate()

    def test_missing_branch_receipt_rejected(self):
        (self.root/'_coat_arm_receipt.json').unlink()
        with self.assertRaises(FileNotFoundError): self.run_gate()

    def test_fallback_branch_rejected(self):
        (self.root/'_coat_arm_receipt.json').write_text(json.dumps({'models': 3, 'fallback_studies': 1}))
        with self.assertRaises(AssertionError): self.run_gate()

if __name__ == '__main__':
    unittest.main()
