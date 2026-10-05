import hashlib as _c3_hash, json as _c3_json
import pandas as _c3_pd, numpy as _c3_np
assert _sprint_receipt['ready_for_scoring'], 'Parent readiness failed'
_c3_root=_C3Path('/kaggle/working')
_c3_parent=_c3_root/'parent943_replay.csv'
assert not _c3_parent.exists()
_c3_parent.write_bytes((_c3_root/'submission.csv').read_bytes())
_c3_frame=_c3_pd.read_csv(_c3_parent,dtype={'StudyInstanceUID':str},float_precision='round_trip')
_c3_own=_c3_pd.read_csv(_c3_root/'cnx3_raw.csv',dtype={'StudyInstanceUID':str},float_precision='round_trip')
assert _c3_frame.columns.tolist()==_c3_own.columns.tolist()
assert _c3_own.StudyInstanceUID.is_unique and set(_c3_own.StudyInstanceUID)==set(_c3_frame.StudyInstanceUID)
_c3_own=_c3_own.set_index('StudyInstanceUID').loc[_c3_frame.StudyInstanceUID].reset_index()
_c3_labels=_c3_frame.columns.tolist()[1:]
assert _c3_np.isfinite(_c3_own[_c3_labels].to_numpy()).all()
if len(_c3_frame)>10:
    assert (_c3_own[_c3_labels].nunique()>1).all(), 'Constant reader outputs'
_c3_final=_c3_frame.copy()
_c3_final[_c3_labels]=(.7*_c3_frame[_c3_labels].rank(method='average',pct=True)+.3*_c3_own[_c3_labels].rank(method='average',pct=True)).rank(method='average',pct=True)
assert _c3_np.isfinite(_c3_final[_c3_labels].to_numpy()).all()
_c3_final.to_csv(_c3_root/'submission.csv',index=False)
_c3_receipt=dict(status='COMPLETE',ready_for_scoring=True,parent_submission=56700487,parent_sha256='__PARENT_SHA__',upstream_sha256='__SOURCE_SHA__',reader_weight=.3,folds=[0,1,2],study_count=len(_c3_final),elapsed_seconds=_c3_time.monotonic()-_c3_started,
    hashes={f:_c3_hash.sha256((_c3_root/f).read_bytes()).hexdigest() for f in ['submission.csv','parent943_replay.csv','cnx3_raw.csv','cnx3_worker_receipt.json']},score=None)
assert _c3_receipt['elapsed_seconds']<8.5*3600
(_c3_root/'cnx3_receipt.json').write_text(_c3_json.dumps(_c3_receipt,indent=2))
_c3_done.set()
print('CNX3 FIXED 30% COMPLETE',_c3_receipt,flush=True)
