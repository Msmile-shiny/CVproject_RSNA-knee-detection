import unittest
import json
import hashlib
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from check_result import LABELS, SPECIAL, FILES, validate_frames, check
from build_outer70 import ROOT, PARENT_SHA, CHANGED


class GateTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(7)
        tr=pd.DataFrame(rng.random((21,12)),columns=LABELS).rank(pct=True)
        cr=pd.DataFrame(rng.random((21,12)),columns=LABELS)
        for f in (tr,cr): f.insert(0,'StudyInstanceUID',[str(i) for i in range(21)])
        frames={FILES[2]:tr,FILES[3]:cr}
        for name,default in [(FILES[0],.7),(FILES[1],.6)]:
            f=tr.copy()
            for l in LABELS:
                w=SPECIAL.get(l,default);f[l]=(1-w)*tr[l]+w*cr[l]
            f[LABELS]=f[LABELS].rank(pct=True);frames[name]=f
        return frames

    def test_valid(self): validate_frames(self.fixture())

    def test_reject_corruption(self):
        for kind in ('ids','nan','range','formula','columns'):
            with self.subTest(kind=kind):
                frames=self.fixture();f=frames[FILES[0]]
                if kind=='ids': f.loc[1,'StudyInstanceUID']='0'
                if kind=='nan': f.loc[0,'ACL']=np.nan
                if kind=='range': f.loc[0,'ACL']=1.1
                if kind=='formula': f.loc[0,'ACL']=.123456
                if kind=='columns': frames[FILES[0]]=f[list(reversed(f.columns))]
                with self.assertRaises(ValueError): validate_frames(frames)

    def test_file_receipts_and_failure_event(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for name,frame in self.fixture().items(): frame.to_csv(out/name,index=False)
            build=json.loads((ROOT/'build_receipt.json').read_text())
            r=dict(status='COMPLETE',ready_for_scoring=True,parent_notebook_sha256=PARENT_SHA,
                   candidate_core_sha256=build['candidate_core_sha256'],changed_labels=CHANGED,
                   default_weight_after=.7,default_weight_before=.6,elapsed_seconds=300,study_count=21,
                   hashes={f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in FILES})
            (out/'outer70_receipt.json').write_text(json.dumps(r))
            (out/'sprint_fourway_receipt.json').write_text(json.dumps(dict(ready_for_scoring=True,
                family_members=['resgated_top3','global96_top3','d4_swa3','repairv1_top3'])))
            p=out/'btkd_v559_complete.json';p.write_text(json.dumps(dict(events=[])))
            self.assertEqual(check(out)['status'],'VISIBLE_OUTPUT_CHECK_PASS')
            p.write_text(json.dumps(dict(events=[dict(kind='a5_decode_failure')])))
            with self.assertRaises(ValueError):check(out)
            p.write_text(json.dumps(dict(events=[])))
            r['elapsed_seconds']=32401
            (out/'outer70_receipt.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):check(out)


if __name__=='__main__': unittest.main()
