"""本地验证 pilkwang 权重包与 enhanced-ensemble notebook 的匹配性.

用 notebook 原代码 (scripts/_verify_fp_src.py) 加载每个 checkpoint,
跑 fingerprint 校验 (tol=2e-3)。全部通过 → Kaggle 上不会因权重/代码
不匹配浪费 GPU 时数。

用法:
    C:\\Users\\eason\\miniconda3\\envs\\d2l\\python.exe scripts/verify_checkpoints_local.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _verify_fp_src as S  # notebook 原代码片段
S.log = lambda msg: print(f"  [log] {msg}")  # notebook 里 log() 的本地存根
# 补齐提取时遗漏的模块级常量（在 notebook cell 3 后半段定义）
S.SLOTS = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]
S.N_SLOT = len(S.SLOTS)
S.GROUP = 3
S.POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}
S.IMG = 336
S.FINGERPRINT_TOL = 2e-3  # check_fingerprint 默认参数在 exec 时求值，须先设
exec(open(Path(__file__).resolve().parent / "_verify_fp_extra.py", encoding="utf-8").read(), S.__dict__)

ARCHIVE = Path(__file__).resolve().parents[1] / "kaggle_dataset" / "archive"
HF_DIR = Path(__file__).resolve().parents[1] / "datasets" / "dinov2_hf_small"

TOL = 2e-3
FINGERPRINT_IMG = 336
FINGERPRINT_SEED = 2026


def main() -> int:
    man = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    members = man["members"]
    print(f"members: {len(members)}")

    dev = torch.device("cpu")
    ok = fail = 0
    errors = []

    for mi, m in enumerate(members):
        f = m["file"]
        ck = torch.load(ARCHIVE / f, map_location="cpu", weights_only=False)

        model = S.build_model(
            int(m["config"]["unfreeze_last"]),
            source=str(HF_DIR),
            variant=m["config"]["variant"],
            pool=m["config"].get("pool", "cls_mean"),
            prior=bool(m["config"].get("prior", False)),
        )
        model.load_state_dict(ck["model"])  # strict=True, 默认
        model.eval()

        d = S.check_fingerprint(
            model, dev, FINGERPRINT_IMG, ck["fingerprint"],
            tol=TOL, tag=f"{f}: ",
        )
        # 同时核对 fingerprint_config
        fc = ck.get("fingerprint_config", {})
        assert int(fc.get("img")) == FINGERPRINT_IMG, f"img {fc.get('img')}"
        assert int(fc.get("group")) == 3, f"group {fc.get('group')}"
        assert int(fc.get("seed")) == FINGERPRINT_SEED, f"seed {fc.get('seed')}"

        ok += 1
        print(f"[{mi+1:2d}/{len(members)}] {f}  PASS  max|diff|={d:.3g}")

        del model, ck

    print(f"\nRESULT: {ok} pass, {fail} fail")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
