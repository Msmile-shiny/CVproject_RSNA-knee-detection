"""Generate self-contained offline Stage 5A notebook."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build():
    cells = []
    def add(source, kind='code'):
        if kind == 'code':
            compile(source, f'cell-{len(cells)}', 'exec')
        item = dict(cell_type=kind, id=f's5a-{len(cells):02d}', metadata={}, source=source.splitlines(True))
        if kind == 'code':
            item.update(outputs=[], execution_count=None)
        cells.append(item)
    add('''# Stage 5A — dense DINOv2 MIL and selective high-resolution revisit

Offline screening notebook. GPU T4 x2. First run smoke_studies=8, epochs=1;
then reset the session and use smoke_studies=0, epochs=20. Mount competition,
historical v5_labels.csv and YOUR historical v5s1 best_model_s42.pt (not 3D/4A).
Edit the next configuration cell for explicit file paths. No new pretraining weights.
Outputs mean/coarse/fine predictions; it does not create a ranked competition submission.
Gold was previously involved in label calibration and backbone selection: development only.
Attention locations are hypotheses, not verified lesion localizations.
Recovery: mount the PREVIOUS RUN OUTPUT containing stage5a_manifest.json,
stage5a_features/ and stage5a_*.pt; set resume_input below to that root.
Compatible caches and completed heads are reused. Stage results are exported early.
Budget exhaustion returns PAUSED normally with completed=False in the manifest.
''', 'markdown')
    add((ROOT/'cells_v5/02_imports.py').read_text(encoding='utf-8'))
    cfg=(ROOT/'cells_v5/03_config.py').read_text(encoding='utf-8')
    add(cfg.split('CFG = {')[0]+'''CFG = dict(comp_input='/kaggle/input/competitions/rsna-knee-abnormality-detection',
           output_dir='/kaggle/working', dinov2_variant='vit_small_patch14_dinov2.lvd142m')
# EDIT PATHS HERE. Blank paths search /kaggle/input; ambiguous matches stop.
S5 = dict(v5_checkpoint='/kaggle/input/datasets/easoncyy/v5-bestmodel/best_model_s42.pt',
          labels='/kaggle/input/datasets/easoncyy/rsna-knee-v5-labels/v5_labels.csv',
          resume_input='',  # Directory containing the previous run's stage5a_manifest.json
          coarse_px=168, fine_px=280, slices=32,
          top_per_class=2, encode_batch=48, batch_size=32, epochs=20,
          head_lr=0.0003, seed=42, feature_minutes=300, smoke_studies=0,
          session_minutes=480, reserve_minutes=20)
N_GPUS=torch.cuda.device_count()
DEVICE=torch.device('cuda' if N_GPUS else 'cpu')
import random
random.seed(S5['seed']); np.random.seed(S5['seed']); torch.manual_seed(S5['seed'])
torch.set_num_threads(2)
print('Stage 5A', S5, 'GPUs', N_GPUS)
''')
    add((ROOT/'cells_v5/04_slot_matching.py').read_text(encoding='utf-8'))
    io=(ROOT/'cells_v5/05_dicom_io.py').read_text(encoding='utf-8')
    io=io.replace('slices_info.append(np.zeros((image_size, image_size), dtype=np.float32))', "raise RuntimeError(f'Pixel decode failed: {series_dir / fname}')")
    add(io)
    add((ROOT/'stage5a_core.py').read_text(encoding='utf-8'))
    add((ROOT/'stage5a_runtime.py').read_text(encoding='utf-8'))
    notebook=dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),cells=cells)
    out=ROOT/'kaggle_train_stage5a_dense_mil.ipynb'
    out.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    print(out)


if __name__ == '__main__':
    build()
