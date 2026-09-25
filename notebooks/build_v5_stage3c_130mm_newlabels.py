"""Complete the 2x2 ablation using the exact Stage 3A code and labels."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_LABEL_SHA = 'efd3d39e7a0b7e8c78ead857c3ef43953eac3eb7a7e1a8a040bb1dac19b49d59'
LABEL = 'pseudo_labels_deepseek_gpt56sol_fused.csv'
OUTPUT = ROOT / 'kaggle_train_v5_stage3c_130mm_newlabels.ipynb'


def main():
    local = ROOT.parent / 'data/processed' / LABEL
    if hashlib.sha256(local.read_bytes()).hexdigest() != EXPECTED_LABEL_SHA:
        raise RuntimeError('Historical Stage 3A labels changed; do not regenerate this experiment silently.')
    source_path = ROOT / 'kaggle_train_v5_stage3a_140mm.ipynb'
    nb = json.loads(source_path.read_text(encoding='utf-8'))
    nb['cells'][0]['source'] = [
        '# Stage 3C: 130 mm + Stage 3A labels\n',
        '\nComplete the 2x2 diagnostic. Same supervision as Stage 3A; crop is 130 mm.\n',
        'Mount rsna-knee-stage3a-labels, competition data and DINOv2 weights.\n',
        'Gold is a development set: it previously influenced label calibration.\n',
    ]
    config_hits = load_hits = 0
    for cell in nb['cells']:
        cell.pop('execution_count', None)
        cell.pop('outputs', None)
        if cell['cell_type'] != 'code':
            continue
        s = ''.join(cell['source'])
        if "'crop_mm': 140.0," in s:
            config_hits += 1
            s = s.replace("'crop_mm': 140.0,", "'crop_mm': 130.0,")
            s = s.replace('stage3a_v5_140mm_gpt56sol_fused', 'stage3c_130mm_newlabels')
            # Log the actual label lineage rather than the inherited v5 header.
            s = s.replace('Fused Soft Labels (text×OOF teacher)', 'Stage 3A report-derived labels')
            s = s.replace("'/kaggle/input/rsna-knee-stage3a-labels'", "'/kaggle/input/datasets/easoncyy/rsna-knee-stage3a-labels'")
            s = s.replace("'/kaggle/input/rsna-dinov2-weights/dinov2_vits14.pth'", "'/kaggle/input/datasets/easoncyy/rsna-dinov2-weights/dinov2_vits14.pth'")
        marker = 'fused_df = pd.read_csv(v5_label_file)'
        if marker in s:
            load_hits += 1
            guard = f'''import hashlib as _audit_hashlib
import json as _audit_json
_label_sha = _audit_hashlib.sha256(v5_label_file.read_bytes()).hexdigest()
if _label_sha != {EXPECTED_LABEL_SHA!r}:
    raise RuntimeError('Stage 3C label hash mismatch: ' + _label_sha)
assert CFG['crop_mm'] == 130.0 and CFG['seed'] == 42
_audit = {{'config': CFG, 'label_sha256': _label_sha,
          'source_notebook_sha256': {hashlib.sha256(source_path.read_bytes()).hexdigest()!r},
          'status': 'preflight_passed'}}
Path(CFG['output_dir']).mkdir(parents=True, exist_ok=True)
(Path(CFG['output_dir']) / 'phase3c_preflight.json').write_text(
    _audit_json.dumps(_audit, indent=2), encoding='utf-8')
print('Stage 3C verified: 130mm / historical Stage 3A labels / seed 42')
'''
            s = s.replace(marker, guard + '\n' + marker)
        s = s.replace('stage3a_manifest.json', 'phase3c_manifest.json')
        cell['source'] = s.splitlines(keepends=True)
        cell['execution_count'] = None
        cell['outputs'] = []
        compile(s, 'stage3c_cell', 'exec')
    if (config_hits, load_hits) != (1, 1):
        raise RuntimeError('Source notebook contract drift')
    OUTPUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(OUTPUT)


if __name__ == '__main__':
    main()
