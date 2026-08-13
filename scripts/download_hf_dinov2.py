"""下载 HF 格式 DINOv2-small 到本地，供上传 Kaggle Dataset 使用.

kaggle_ensemble_public.ipynb（pilkwang 公开集成）用 transformers.AutoModel
加载 DINOv2，需要 HuggingFace 目录格式（config.json + model.safetensors），
而不是我们训练用的 timm 原始权重 dinov2_vits14.pth。

用法（本地，需联网）:
    python scripts/download_hf_dinov2.py

输出:
    datasets/dinov2_hf_small/
        config.json
        model.safetensors      (~88 MB)
        preprocessor_config.json

之后把整个目录打成 zip 上传为 Kaggle Dataset，推理 notebook 会自动扫描到。
"""

from __future__ import annotations

import os
import ssl
import sys
from pathlib import Path

# 修复公司网络/代理环境下的 SSL 证书问题（与训练环境一致的处理）
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import requests  # noqa: E402

REPO = "facebook/dinov2-small"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
]

OUT_DIR = Path(__file__).resolve().parents[1] / "datasets" / "dinov2_hf_small"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "rsna-knee-dataset-builder/1.0"})

    for name in FILES:
        url = f"{BASE}/{name}"
        dest = OUT_DIR / name
        print(f"downloading {url} ...", flush=True)
        try:
            with session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"  {done / 1e6:.1f}/{total / 1e6:.1f} MB\r",
                                  end="", flush=True)
        except requests.RequestException as exc:
            print(f"\nFAILED: {url}\n  {exc}", flush=True)
            return 1
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)

    # 自检：两个必需文件都在，且 config 是 dinov2
    if not (OUT_DIR / "config.json").is_file() or not (OUT_DIR / "model.safetensors").is_file():
        print("ERROR: missing required files", flush=True)
        return 1

    print(f"\nOK: {OUT_DIR}")
    print("下一步: 打包该目录为 zip，上传为 Kaggle Dataset，")
    print("然后在 kaggle_ensemble_public.ipynb 中挂载（或设 KNEE_DINOV2_DIR）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
