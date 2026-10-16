"""Run only the published S6 member, isolated from the R7 parent artifacts."""
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('s6_author', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.WORK = Path('/kaggle/working/s6_member')
m.OUT = m.WORK / 'audit'
m.TMP = Path('/tmp/s6_member')
m.OUT.mkdir(parents=True, exist_ok=True)
m.TMP.mkdir(parents=True, exist_ok=True)
m.GUARD_S = int(sys.argv[2])
assert m.GUARD_S > 2400
m._LOG = open(m.OUT / 's6.log', 'a')
try:
    m.arm_guard(m.GUARD_S)
    m._RECEIPT['env'] = m.probe_env()
    assert m._RECEIPT['env'].get('cuda_available'), 'CUDA unavailable'
    inputs = m.prepare_inputs()
    m._RECEIPT['B_probe'] = m.probe_b_stack(inputs)
    predictions, logits, info, failed = m.run_b(
        inputs, 'B', inputs.test_ids, 'test', m.guard_left() - m.END_SLACK_S)
    assert not failed, 'S6 incomplete: no baseline fallback allowed'
    assert m.np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1)).all()
    frame = m.pd.DataFrame(predictions, columns=m.LAB)
    frame.insert(0, 'StudyInstanceUID', inputs.test_ids)
    frame.to_csv(m.WORK / 's6_raw.csv', index=False, float_format='%.17g')
    m._RECEIPT.update(status='PASS', B=info, model=m.B_MODEL,
                     rows=len(inputs.test_ids), decode_coverage_verified=False,
                     raw_sha256=m.sha256_file(m.WORK / 's6_raw.csv'))
    m.save_json(m.WORK / 's6_receipt.json', m._RECEIPT)
finally:
    m._GUARD_STOP.set()
    m.faulthandler.cancel_dump_traceback_later()
    m.kill_all_children()
    m._LOG.close()
