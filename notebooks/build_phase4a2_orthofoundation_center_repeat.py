"""Build the input-alignment control after the failed frozen OrthoFoundation probe."""
from pathlib import Path

from build_phase4a_orthofoundation_probe import ROOT, build


if __name__ == '__main__':
    build(
        out=ROOT / 'kaggle_train_phase4a2_orthofoundation_center_repeat.ipynb',
        experiment_name='phase4a2_orthofoundation_center_repeat',
        center_repeat=True,
    )
