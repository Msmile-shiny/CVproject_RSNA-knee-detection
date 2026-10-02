"""CPU checks: exact parent diff and same-pass baseline replay."""
import json
import unittest
import numpy as np
import pandas as pd
from build_outer70 import PARENT, OLD, NEW, CHANGED, make_candidate, code_cells


class Outer70Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = json.loads((PARENT/'sprint-fourway.ipynb').read_text(encoding='utf-8'))
        cls.candidate = make_candidate(cls.parent)

    def test_only_declared_changes(self):
        before = code_cells(self.parent)
        after = code_cells(self.candidate)
        self.assertEqual(len(after), len(before)+1)
        for a,b in zip(before, after):
            expected = a.replace(OLD, NEW).replace("'inference_changes': 'none; post-run audit only'", "'inference_changes': 'outer default weight 0.60 to 0.70; five overrides retained'")
            self.assertEqual(expected, b)

    def test_baseline_replay_and_class_overrides(self):
        labels = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
        special = {'ACL':.75,'Medial Meniscus':.8,'Lateral Meniscus':1.,'Lateral OA':.75,'Fracture':.75}
        rng = np.random.default_rng(42)
        tr = pd.DataFrame(rng.random((61,12)), columns=labels).rank(pct=True)
        cr = pd.DataFrame(rng.random((61,12)), columns=labels)
        frame = tr.copy()
        frame.insert(0, 'StudyInstanceUID', [str(i) for i in range(61)])
        def evaluate(notebook):
            cell = next(c for c in code_cells(notebook) if '_coatnet_weight = {label:' in c)
            source = cell[cell.index('_coatnet_weight = {label:'):cell.index('_blend_output.to_csv(_ke_primary')]
            ns = dict(_blend_labels=labels,_blend_transformer=frame,_blend_tr=tr,_blend_cr=cr,_ke_np=np)
            exec(source, ns)
            return ns['_blend_output']
        baseline, candidate = evaluate(self.parent), evaluate(self.candidate)
        replay = frame.copy()
        for label in labels:
            w = special.get(label, .6)
            replay[label] = (1-w)*tr[label]+w*cr[label]
        replay[labels] = replay[labels].rank(method='average', pct=True)
        pd.testing.assert_frame_equal(baseline, replay)
        pd.testing.assert_frame_equal(baseline[list(special)], candidate[list(special)])
        self.assertEqual(set(CHANGED), set(labels)-set(special))
        self.assertTrue((baseline[CHANGED] != candidate[CHANGED]).any().all())

    def test_reject_missing_assignment(self):
        n = json.loads(json.dumps(self.parent))
        for c in n['cells']:
            c['source'] = ''.join(c['source']).replace(OLD, NEW).splitlines(True)
        with self.assertRaises(ValueError):
            make_candidate(n)


if __name__ == '__main__':
    unittest.main()
