# ==== OUR 21ST MEMBER BLEND (fail-closed, mirrors the E11/M2 pattern) ====
# ours = v5 3-seed rank mean (the 0.8959-gold / 0.886-LB base) as one extra member
# final = (1 - OUR_ALPHA) * v47_theirs + OUR_ALPHA * ours_rank_mean
# 产物:
#   submission.csv                    = 融合 (OUR_MEMBER_ACTIVE=True 时)
#   submission_ours_only.csv          = 我们成员单独 (rank mean)
#   submission_v47_theirs_only.csv    = 纯 v47 (回退)
#   ours_apply_audit.json             = 审计 receipt
# scoring 判别: test.csv > 100 行 = 评分环境 (hidden set);
#   ACTIVE=False → 评分时零开销纯复刻; 交互时跑一次冒烟 (不碰 submission.csv)
import shutil as _osh
import hashlib as _ohh
import json as _ojs
from pathlib import Path as _OP
from scipy.stats import rankdata as _ours_rankdata

_ours_primary = _OP('/kaggle/working/submission.csv')
_ours_pres = _OP('/kaggle/working/submission_v47_theirs_only.csv')
_ours_audit_p = _OP('/kaggle/working/ours_apply_audit.json')
_ours_aud = {'status': 'SKIPPED', 'alpha': float(OUR_ALPHA),
             'active': bool(OUR_MEMBER_ACTIVE)}

try:
    _ours_test = pd.read_csv(COMP / 'test.csv', dtype={'StudyInstanceUID': str})
    _ours_scoring = len(_ours_test) > 100
    _ours_run = OUR_MEMBER_ACTIVE or (not _ours_scoring)
    if not _ours_run:
        log('ours member: inactive on scoring replica run; submission.csv untouched')
    else:
        _ours_uids, _ours_raw = _ours_v5_3seed_predict()
        _ours_n = max(len(_ours_uids), 1)
        _ours_ranks = {}
        for _s, _p in _ours_raw.items():
            if _p.shape[1] != len(TARGETS):
                raise RuntimeError(f'ours seed {_s}: {_p.shape[1]} targets != {len(TARGETS)}')
            _ours_ranks[_s] = (_ours_rankdata(_p, axis=0, method='average')
                               / _ours_n).astype(np.float64)
        _ours_base = pd.read_csv(_ours_primary, dtype={'StudyInstanceUID': str})
        if _ours_base.columns.tolist() != ['StudyInstanceUID'] + TARGETS:
            raise RuntimeError('v47 submission schema drift before ours blend')
        _ours_pos = {u: i for i, u in enumerate(_ours_uids)}
        _ours_acc = np.full((len(_ours_base), len(TARGETS)), 0.5, np.float64)
        for _r_i, _u in enumerate(_ours_base['StudyInstanceUID'].astype(str)):
            if _u in _ours_pos:
                _ours_acc[_r_i] = np.mean(
                    [_ours_ranks[_s][_ours_pos[_u]] for _s in _ours_ranks], axis=0)
        _ours_df = _ours_base[['StudyInstanceUID']].copy()
        for _j, _t in enumerate(TARGETS):
            _ours_df[_t] = _ours_acc[:, _j]
        _ours_df.to_csv(_OP('/kaggle/working/submission_ours_only.csv'), index=False)
        _ours_aud.update(status='OURS_READY', studies=int(len(_ours_uids)),
                         scoring=bool(_ours_scoring))
        log(f'ours member ready: {len(_ours_uids)} studies '
            f'(scoring={_ours_scoring}); ours-only saved')
    if OUR_MEMBER_ACTIVE:
        _osh.copy2(_ours_primary, _ours_pres)
        _ours_base = pd.read_csv(_ours_pres, dtype={'StudyInstanceUID': str})
        _ours_th = _ours_base[TARGETS].to_numpy(np.float64)
        _ours_fr = _ours_base.copy()
        for _j, _t in enumerate(TARGETS):
            _ours_fr[_t] = (1.0 - OUR_ALPHA) * _ours_th[:, _j] + OUR_ALPHA * _ours_acc[:, _j]
        if not np.isfinite(_ours_fr[TARGETS].to_numpy()).all():
            raise RuntimeError('ours blend produced non-finite values')
        if _ours_fr[TARGETS].to_numpy().min() < 0 or _ours_fr[TARGETS].to_numpy().max() > 1:
            raise RuntimeError('ours blend out of [0,1]')
        _ours_fr.to_csv(_ours_primary, index=False)
        _ours_aud.update(status='OURS_APPLIED', alpha=float(OUR_ALPHA),
                         selected_sha256=_ohh.sha256(_ours_primary.read_bytes()).hexdigest(),
                         fallback_sha256=_ohh.sha256(_ours_pres.read_bytes()).hexdigest())
        log(f'ours member applied at alpha {OUR_ALPHA} over all 12 targets')
except Exception as _ours_e:
    _ours_aud.update(status='ERROR_THEIRS_PRESERVED',
                     error=f'{type(_ours_e).__name__}: {_ours_e}')
    if OUR_MEMBER_ACTIVE and _ours_pres.is_file():
        _osh.copy2(_ours_pres, _ours_primary)
    log(f'ours blend failed; theirs preserved: {_ours_aud["error"]}')
_ours_audit_p.write_text(_ojs.dumps(_ours_aud, indent=2, sort_keys=True) + '\n')
