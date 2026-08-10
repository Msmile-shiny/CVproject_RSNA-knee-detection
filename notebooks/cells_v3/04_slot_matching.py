# ============================================================
# v3: Slot Matching — map DICOM series to 6 clinical slots
# ============================================================
#
# Uses train_series.csv metadata (Anatomical_Plane, Fluid_Sensitive,
# Fat_Suppression) to assign the best series for each clinical slot.
# Falls back to SeriesDescription heuristics when metadata is missing.

def match_slots_for_study(study_series_df):
    """Assign one series per slot for a single study.

    Args:
        study_series_df: DataFrame subset for ONE study, with columns:
            SeriesInstanceUID, Anatomical_Plane, Fluid_Sensitive,
            Fat_Suppression, n_slices (pre-computed), dir (DICOM path)

    Returns:
        dict: {slot_name: dict(series_uid, dir, n_slices, plane) or None}
    """
    slots_found = {}
    for slot_name, plane, fluid, fatsat in SLOTS:
        candidates = study_series_df[
            (study_series_df['Anatomical_Plane'] == plane) &
            (study_series_df['Fluid_Sensitive'] == (1 if fluid else 0)) &
            (study_series_df['Fat_Suppression'] == (1 if fatsat else 0))
        ]

        # Fallback for structural slots (T1): relax fatsat requirement
        if len(candidates) == 0 and not fluid:
            candidates = study_series_df[
                (study_series_df['Anatomical_Plane'] == plane) &
                (study_series_df['Fluid_Sensitive'] == 0)
            ]

        if len(candidates) > 0:
            # Pick series with most slices
            best = candidates.sort_values('n_slices', ascending=False).iloc[0]
            slots_found[slot_name] = {
                'series_uid': best['SeriesInstanceUID'],
                'dir': best['dir'],
                'n_slices': int(best['n_slices']),
                'plane': plane,
            }
        else:
            slots_found[slot_name] = None

    return slots_found


def build_study_slot_map(series_meta, dicom_root):
    """Build slot→series mapping for all studies.

    Args:
        series_meta: DataFrame from train_series.csv
        dicom_root: Path to DICOM directory root

    Returns:
        tuple:
            slot_map: {study_uid: {slot_name: dict(series_uid, dir, n_slices, plane) or None}}
            study_series_map: {study_uid: DataFrame with columns incl. dir, n_slices, SeriesInstanceUID}
    """
    df = series_meta.copy()
    df['StudyInstanceUID'] = df['StudyInstanceUID'].astype(str)
    df['SeriesInstanceUID'] = df['SeriesInstanceUID'].astype(str)

    # Pre-compute DICOM directory and slice count
    dirs = []
    n_slices_list = []
    for _, row in df.iterrows():
        d = str(dicom_root / row['StudyInstanceUID'] / row['SeriesInstanceUID'])
        dirs.append(d)
        if os.path.isdir(d):
            n_slices_list.append(len([f for f in os.listdir(d) if f.endswith('.dcm')]))
        else:
            n_slices_list.append(0)
    df['dir'] = dirs
    df['n_slices'] = n_slices_list

    # Ensure required columns
    for col in ['Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']:
        if col not in df.columns:
            raise KeyError(f'train_series.csv missing column: {col}')

    slot_map = {}
    study_series_map = {}

    for study_uid, grp in df.groupby('StudyInstanceUID'):
        study_series_map[study_uid] = grp
        slot_map[study_uid] = match_slots_for_study(grp)

    # Statistics
    slot_counts = {}
    for slots in slot_map.values():
        for name, sid in slots.items():
            slot_counts[name] = slot_counts.get(name, 0) + (1 if sid is not None else 0)

    if IS_MAIN:
        n_studies = len(slot_map)
        print(f'Slot map: {n_studies} studies')
        for name, count in slot_counts.items():
            print(f'  {name:<18s}: {count:5d}/{n_studies} ({count/n_studies*100:.0f}%)')

    return slot_map, study_series_map
