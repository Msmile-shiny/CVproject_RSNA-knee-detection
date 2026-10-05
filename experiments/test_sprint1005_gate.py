"""Exercise the result gate with synthetic outputs, including rejected bad recipes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from finish_sprint1005 import check, LABELS, ROOT, PARENT_SHA

class GateTests(unittest.TestCase):
    def fixture(self, out):
        ids=['a','b','c','d']
        parent=pd.DataFrame(np.tile([.1,.4,.6,.9],(12,1)).T,columns=LABELS)
        parent.insert(0,'StudyInstanceUID',ids)
        bag=pd.DataFrame(np.tile([.5,.2,.8,.1],(12,1)).T,columns=LABELS)
        bag.insert(0,'StudyInstanceUID',ids)
        final=parent.copy()
        final[LABELS]=(.7*parent[LABELS].rank(pct=True)+.3*bag[LABELS].rank(pct=True)).rank(pct=True)
        for name,frame in [('parent943_replay.csv',parent),('cnx3_raw.csv',bag),('submission.csv',final)]: frame.to_csv(out/name,index=False)
        build=json.loads((ROOT/'experiments/convnext1005/build_receipt.json').read_text())
        worker=dict(status='COMPLETE',model_count=3,folds=[0,1,2],stress_pass=True,fallback=0,partial_predictions=0,study_count=4,precision='fp16',batch_studies=4,windows_per_slot=12,source_hashes=build['sources'],checkpoint_hashes={f'cnxt_v0_fold{i}.pt':str(i)*64 for i in range(3)})
        (out/'cnx3_worker_receipt.json').write_text(json.dumps(worker))
        digest=lambda name:hashlib.sha256((out/name).read_bytes()).hexdigest()
        receipt=dict(status='COMPLETE',ready_for_scoring=True,parent_sha256=PARENT_SHA,elapsed_seconds=100,study_count=4,reader_weight=.3,upstream_sha256=build['upstream_sha256'],hashes={name:digest(name) for name in ['submission.csv','parent943_replay.csv','cnx3_raw.csv','cnx3_worker_receipt.json']})
        (out/'cnx3_receipt.json').write_text(json.dumps(receipt))
        (out/'sprint_fourway_receipt.json').write_text(json.dumps(dict(ready_for_scoring=True,family_members=['resgated_top3','global96_top3','d4_swa3','repairv1_top3'],submission_sha256=digest('parent943_replay.csv'))))
        (out/'btkd_v559_complete.json').write_text(json.dumps(dict(events=[])))
        return receipt
    def test_valid_recipe_passes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            self.assertEqual(check('cnx3',p)['status'],'VISIBLE_OUTPUT_CHECK_PASS')
    def test_weight_or_output_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=self.fixture(p);r['reader_weight']=.4
            (p/'cnx3_receipt.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):check('cnx3',p)
            self.fixture(p)
            with (p/'submission.csv').open('a') as f:f.write('bogus\n')
            with self.assertRaises(ValueError):check('cnx3',p)
    def test_parent_degradation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            (p/'btkd_v559_complete.json').write_text(json.dumps(dict(events=[dict(kind='partial_predictions')])) )
            with self.assertRaises(ValueError):check('cnx3',p)

if __name__=='__main__':unittest.main()
