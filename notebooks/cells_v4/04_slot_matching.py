# ============================================================
# v4: Slot Matching + Laterality Detection + DICOM Header Annotation
# ============================================================

# ---- DICOM Header Annotation (Ref1: annotate_sequences) ----
_SEP = re.compile(r'[_\-.]')
_FATSAT_RX = re.compile(
    r'\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|'
    r'water excit|\btirm\b|\bsting\b|\bfatsup\b'
)
_T1_RX = re.compile(r'\bt1\b|\bt1w\b')
_T2_RX = re.compile(r'\bt2\b|\bt2w\b')
_PD_RX = re.compile(r'\bpd\b|\bpdw\b|proton|\bdp\b|dens')

FATSAT_OPTS = {'FS', 'FATSAT', 'FAT_SAT', 'FSAT'}

_HDR_TAGS = [
    'SeriesDescription', 'SequenceName', 'ScanOptions', 'ScanningSequence',
    'RepetitionTime', 'EchoTime', 'Laterality', 'ImageLaterality',
    'ImagePositionPatient', 'PixelSpacing',
]


def _tag_side(group):
    """从 DICOM Laterality 标签推断侧性。"""
    values = [str(x).strip().upper() for x in group.get('Laterality', pd.Series(dtype=object)).dropna()]
    if 'ImageLaterality' in group.columns:
        values += [str(x).strip().upper() for x in group['ImageLaterality'].dropna()]
    values = [x[0] for x in values if x and x[0] in ('L', 'R')]
    return values[0] if values else None


def _position_side(group, min_offset_mm=5.0):
    """从 ImagePositionPatient[0] 推断侧性：DICOM LPS 中 +x = 患者左侧。"""
    xs = []
    for raw in group.get('ImagePositionPatient', pd.Series(dtype=object)).dropna():
        try:
            xs.append(float(str(raw).split('|')[0]))
        except Exception:
            pass
    if not xs:
        return None
    median_x = float(np.median(xs))
    if abs(median_x) < min_offset_mm:
        return None
    return 'R' if median_x < 0 else 'L'


def detect_laterality(headers_df):
    """为每个 study 确定侧性（左/右），结合标签和几何位置。"""
    tagged, positioned = {}, {}
    for study_uid, group in headers_df.groupby('StudyInstanceUID'):
        tagged[study_uid] = _tag_side(group)
        positioned[study_uid] = _position_side(group)

    comparable = [s for s in tagged if tagged[s] and positioned[s]]
    agreement = float(np.mean([
        tagged[s] == positioned[s] for s in comparable
    ])) if comparable else np.nan

    use_position = bool(comparable) and np.isfinite(agreement) and agreement >= 0.85

    resolved = {
        uid: (tagged[uid] or (positioned[uid] if use_position else None))
        for uid in tagged
    }
    coverage = float(np.mean([v is not None for v in resolved.values()]))

    if IS_MAIN:
        print(f'Laterality: tag_coverage={len([v for v in tagged.values() if v])\max(len(tagged),1):.1%}, '
              f'agreement={agreement:.1%} on {len(comparable)} studies, '
              f'final_coverage={coverage:.1%}')
    return resolved


def annotate_sequences(df):
    """从 DICOM header 推断 Fluid/FatSat/Weight，作为 train_series.csv 的 fallback。"""
    df = df.copy()

    # Fat suppression detection
    desc = (df.get('SeriesDescription', '').fillna('') + ' ' +
            df.get('SequenceName', '').fillna(''))
    desc = desc.str.lower().str.replace(_SEP, ' ', regex=True)

    scan_options = df.get('ScanOptions', '').fillna('').str.upper().str.split('|')
    option_fatsat = scan_options.apply(
        lambda tokens: any(t.strip() in FATSAT_OPTS for t in tokens))
    df['fatsat_detected'] = desc.str.contains(_FATSAT_RX) | option_fatsat

    # Weight detection
    tr = pd.to_numeric(df.get('RepetitionTime', np.nan), errors='coerce')
    te = pd.to_numeric(df.get('EchoTime', np.nan), errors='coerce')
    named_t1 = desc.str.contains(_T1_RX)
    named_t2 = desc.str.contains(_T2_RX)
    named_pd = desc.str.contains(_PD_RX)

    df['weight'] = np.where(
        named_t1 & ~named_t2 & ~named_pd, 'T1',
        np.where(named_t2 & ~named_pd, 'T2',
                 np.where(named_pd, 'PD',
                          np.where(tr < 800, 'T1',
                                   np.where(te > 60, 'T2',
                                            np.where(tr >= 800, 'PD', 'UNK'))))))
    df['fluid_detected'] = df['weight'].isin(['PD', 'T2'])

    return df


# ---- Slot Matching ----
def match_slots_for_study(study_series_df):
    """为单个 study 的每个 slot 匹配最优 series。"""
    slots_found = {}
    for slot_name, plane, fluid, fatsat in SLOTS:
        candidates = study_series_df[
            (study_series_df['Anatomical_Plane'] == plane)
            & (study_series_df['Fluid_Sensitive'] == (1 if fluid else 0))
            & (study_series_df['Fat_Suppression'] == (1 if fatsat else 0))
        ]
        if len(candidates) == 0 and not fluid:
            candidates = study_series_df[
                (study_series_df['Anatomical_Plane'] == plane)
                & (study_series_df['Fluid_Sensitive'] == 0)
            ]
        if len(candidates) > 0:
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
    """为所有 study 构建 slot→series 映射。"""
    df = series_meta.copy()
    df['StudyInstanceUID'] = df['StudyInstanceUID'].astype(str)
    df['SeriesInstanceUID'] = df['SeriesInstanceUID'].astype(str)

    # 计算 DICOM 目录和切片数
    dirs, n_slices_list = [], []
    for _, row in df.iterrows():
        d = str(dicom_root / row['StudyInstanceUID'] / row['SeriesInstanceUID'])
        dirs.append(d)
        if os.path.isdir(d):
            n_slices_list.append(len([f for f in os.listdir(d) if f.endswith('.dcm')]))
        else:
            n_slices_list.append(0)
    df['dir'] = dirs
    df['n_slices'] = n_slices_list

    slot_map, study_series_map = {}, {}
    for study_uid, grp in df.groupby('StudyInstanceUID'):
        study_series_map[study_uid] = grp
        slot_map[study_uid] = match_slots_for_study(grp)

    # 统计
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

print('Slot matching v4 ready.')
