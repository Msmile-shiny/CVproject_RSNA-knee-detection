"""Measure the physical field of view actually selected by Stage 6 slot matching."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def pair(value):
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(value))
    if len(numbers) != 2:
        raise ValueError(f'Expected a two-dimensional DICOM pair: {value!r}')
    return np.array([float(v) for v in numbers])


def audit(slot_path, series_path, image_size):
    slots = pd.read_csv(slot_path, dtype={'StudyInstanceUID': str, 'selected_series': str})
    series = pd.read_csv(series_path, dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    valid = slots.loc[~slots['missing'].astype(bool)].copy()
    joined = valid.merge(series, left_on=['StudyInstanceUID', 'selected_series'],
                         right_on=['StudyInstanceUID', 'SeriesInstanceUID'], how='left', validate='one_to_one')
    assert len(joined) == len(valid) and joined['sample_shape'].notna().all()
    fov = np.array([pair(shape) * pair(spacing) for shape, spacing in
                    zip(joined['sample_shape'], joined['sample_spacing_mm'])])
    maximum = fov.max(axis=1)
    assert np.isfinite(maximum).all() and (maximum > 0).all()
    joined['fov_mm'] = maximum
    rows = {}
    for name, group in joined.groupby('slot'):
        values = group['fov_mm'].to_numpy()
        rows[name] = {'series': int(len(group)), 'fov_mm_p10_p50_p90': np.quantile(values, [.1, .5, .9]).tolist(),
                      'mm_per_pixel_median': float(np.median(values / image_size)),
                      'fraction_above_0_8_mm_per_pixel': float(np.mean(values / image_size > .8))}
    return {'selected_series': int(len(joined)), 'image_size': image_size,
            'scope': 'DICOM header sample only; no pixel image or pathology visibility audit', 'slots': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('slots', type=Path)
    parser.add_argument('series', type=Path)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.slots, args.series, args.image_size)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
