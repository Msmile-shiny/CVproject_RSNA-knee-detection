"""Fixed global rank blending with strict identity checks, not weight fitting."""
import numpy as np
import pandas as pd


def rank_blend(parent, member, weight=0.10):
    if not 0 <= weight <= 1:
        raise ValueError('weight must lie in [0, 1]')
    key = 'StudyInstanceUID'
    if parent.columns.tolist() != member.columns.tolist() or parent.columns[0] != key:
        raise ValueError('column order mismatch')
    for frame in (parent, member):
        if frame.empty or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError('empty, missing or duplicate study identifiers')
        if not frame[key].map(lambda x: isinstance(x, str)).all():
            raise ValueError('read identifiers as strings')
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError('invalid probabilities/ranks')
    if set(parent[key]) != set(member[key]):
        raise ValueError('study sets differ')
    aligned = member.set_index(key).loc[parent[key]].reset_index()
    result = parent.copy()
    labels = parent.columns[1:]
    pr = parent[labels].reset_index(drop=True).rank(method='average', pct=True)
    mr = aligned[labels].rank(method='average', pct=True)
    result.loc[:, labels] = ((1-weight)*pr + weight*mr).to_numpy()
    return result
