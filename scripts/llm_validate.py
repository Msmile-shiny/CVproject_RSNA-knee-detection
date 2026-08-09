"""NLP 伪标签 — 第一阶段验证脚本.

在 58 个有标签的放射报告上测试 LLM 提取 12 类膝关节异常的能力。
支持任何 OpenAI-compatible API (DeepSeek, GPT-4, Claude via proxy, 本地 vLLM 等).

输出:
    data/pseudo_labels_valid.csv     逐样本对比 (预测 vs 真值)
    data/nlp_validation_report.csv   Per-class 指标
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# 配置 — 在这里改
# ══════════════════════════════════════════════════════════════

API_KEY = "sk-YOUR_DEEPSEEK_API_KEY_HERE"   # ← 填你的 API key
API_BASE = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"
LIMIT = 20           # 跑前 N 个样本 (0 = 全部 58 个)

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ── 标签列名 ──────────────────────────────────────────────────
LABEL_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# ── 已报告中这 12 类的常见表达 (多语言关键词映射) ──────────────
_CLASS_HINTS = {
    "ACL": "前交叉韧带 / LCA / anterior cruciate ligament",
    "MCL": "内侧副韧带 / LCM / medial collateral ligament",
    "Medial Meniscus": "内侧半月板 / menisco medial/interno / medial meniscus",
    "Lateral Meniscus": "外侧半月板 / menisco lateral/externo / lateral meniscus",
    "Medial OA": "内侧骨关节炎 / artrosis medial / medial osteoarthritis / medial chondrosis",
    "Lateral OA": "外侧骨关节炎 / artrosis lateral / lateral osteoarthritis / lateral chondrosis",
    "PF OA": "髌股骨关节炎 / artrosis patelofemoral / patellofemoral osteoarthritis / trochlear cartilage defect",
    "Effusion": "关节积液 / derrame articular / joint effusion / fluid collection",
    "Synovitis": "滑膜炎 / sinovitis / synovitis / synovial thickening",
    "Baker's": "腘窝囊肿 / quiste poplíteo / Baker's cyst / popliteal cyst",
    "Contusion": "骨挫伤 / contusión ósea / bone bruise / bone edema / marrow edema",
    "Fracture": "骨折 / fractura / fracture",
}

# ── System Prompt ────────────────────────────────────────────

SYSTEM_PROMPT = """You are a musculoskeletal radiologist. Your task is to extract 12 knee abnormality findings from an MRI radiology report.

For each of the 12 classes below, output 0 (normal / not mentioned) or 1 (abnormal / positive finding).

Rules:
1. If the report explicitly states an abnormality → 1.
2. If the report explicitly states normal ("intact", "normal", "no tear", "sin signos de", "no se observa") → 0.
3. If the report does NOT mention the structure at all → 0 (assume normal by default).
4. Pay attention to negation words: "no", "not", "sin", "without", "negative for", "unremarkable".
5. Do NOT confuse:
   - PCL (posterior cruciate ligament) with ACL (anterior cruciate ligament)
   - LCL (lateral collateral ligament) with MCL (medial collateral ligament)
   - "osteochondral defect" or "chondrosis" should be classified under the appropriate OA class
   - "bone edema" / "bone bruise" / "marrow edema" = Contusion (if post-traumatic)
   - "effusion" / "derrame" / "fluid" = Effusion

Class definitions:
"""
for col in LABEL_COLS:
    SYSTEM_PROMPT += f"  - {col}: {_CLASS_HINTS[col]}\n"

SYSTEM_PROMPT += """
Output ONLY a valid JSON object with exactly these 12 keys, no extra text:
{"ACL":0,"MCL":0,"Medial Meniscus":0,"Lateral Meniscus":0,"Medial OA":0,"Lateral OA":0,"PF OA":0,"Effusion":0,"Synovitis":0,"Baker's":0,"Contusion":0,"Fracture":0}"""


def parse_llm_response(text: str) -> dict[str, int] | None:
    """从 LLM 回复中提取 JSON 标签.

    容忍 markdown code fences 和其他多余文字.
    """
    # 尝试直接解析
    text = text.strip()

    # 去掉 markdown ```json ... ``` 包裹
    m = re.search(r"\{[^{}]*\"ACL\"[^{}]*\}", text, re.DOTALL)
    if m:
        text = m.group(0)

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # 尝试修复单引号等问题
        try:
            text = text.replace("'", '"')
            result = json.loads(text)
        except json.JSONDecodeError:
            return None

    # 验证所有 12 个 key 都存在且值为 0/1
    parsed = {}
    for col in LABEL_COLS:
        v = result.get(col)
        if v is None:
            return None
        parsed[col] = int(v)
    return parsed


def call_llm(
    report: str,
    api_base: str,
    api_key: str,
    model: str,
    max_retries: int = 2,
) -> dict[str, int] | None:
    """调用 LLM API, 返回解析后的 12 维标签, 失败返回 None."""
    url = f"{api_base.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": report},
        ],
        "temperature": 0.0,          # 确定性输出
        "max_tokens": 500,           # 12 个数字, 200 就够了
        "response_format": {"type": "json_object"},  # OpenAI 兼容的 JSON mode
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = parse_llm_response(content)
            if parsed is not None:
                return parsed
            # JSON 解析失败, 重试
            if attempt < max_retries:
                time.sleep(1)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"  [ERROR] API call failed: {e}")
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description="NLP 伪标签 — 第一阶段验证")
    parser.add_argument(
        "--api-base",
        default=API_BASE,
        help="LLM API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=API_KEY,
        help="LLM API key (或直接在脚本顶部 API_KEY 处填写)",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        help="Model name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=LIMIT,
        help="只跑前 N 个样本 (0=全部)",
    )
    parser.add_argument(
        "--train-csv",
        default="data/metadata/train.csv",
        help="竞赛标签文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="输出目录",
    )
    args = parser.parse_args()

    # ── 项目根目录 ──────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    train_csv = project_root / args.train_csv
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 58 个有标签的报告 ──────────────────────────────
    train = pd.read_csv(train_csv, encoding="utf-8")
    labeled_mask = train[LABEL_COLS].notna().all(axis=1)
    labeled_df = train[labeled_mask].copy()
    for col in LABEL_COLS:
        labeled_df[col] = labeled_df[col].astype(int)

    if args.limit > 0:
        labeled_df = labeled_df.head(args.limit)

    n_samples = len(labeled_df)
    print(f"加载 {n_samples} 个有标签报告")
    print(f"API: {args.api_base}  model: {args.model}")
    print(f"输出: {output_dir}")
    print()

    if not args.api_key or "YOUR_" in args.api_key:
        print("[ERROR] 请先在脚本顶部 API_KEY 处填写你的 DeepSeek API key")
        print("        获取: https://platform.deepseek.com/api_keys")
        sys.exit(1)

    # ── 逐样本调用 LLM ──────────────────────────────────────
    results = []
    n_ok = 0
    n_fail = 0

    for idx, (_, row) in enumerate(labeled_df.iterrows()):
        report = str(row["Report"])
        study_uid = row["StudyInstanceUID"]

        # 截断过长报告 (>3000 chars 通常已足够覆盖所有发现)
        report_truncated = report[:4000]

        print(f"[{idx+1}/{n_samples}] {study_uid[-12:]}... ", end="", flush=True)

        llm_labels = call_llm(
            report_truncated,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
        )

        if llm_labels is None:
            print("❌ FAIL")
            n_fail += 1
            continue

        print("✅")
        n_ok += 1

        true_labels = {col: int(row[col]) for col in LABEL_COLS}
        results.append(
            {
                "StudyInstanceUID": study_uid,
                "Report_snippet": report[:200],
                **{f"true_{col}": true_labels[col] for col in LABEL_COLS},
                **{f"pred_{col}": llm_labels[col] for col in LABEL_COLS},
            }
        )

    if not results:
        print("\n[FATAL] 没有成功解析的样本, 退出.")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "pseudo_labels_valid.csv", index=False, encoding="utf-8")
    print(f"\n{n_ok}/{n_samples} 成功 ({n_fail} 失败)")
    print(f"逐样本对比已保存: {output_dir / 'pseudo_labels_valid.csv'}")

    # ── 计算 per-class 指标 ─────────────────────────────────
    y_true = np.array([[r[f"true_{col}"] for col in LABEL_COLS] for r in results])
    y_pred = np.array([[r[f"pred_{col}"] for col in LABEL_COLS] for r in results])

    print("\n" + "=" * 70)
    print("验证结果 — Per-Class Metrics")
    print("=" * 70)

    rows = []
    for i, col in enumerate(LABEL_COLS):
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        n_pos = int(y_true[:, i].sum())
        fmt = lambda v: f"{v:.3f}" if not np.isnan(v) else "N/A"
        print(
            f"  {col:<20s}  P={fmt(prec)}  R={fmt(rec)}  F1={fmt(f1)}  "
            f"Acc={fmt(acc)}  (n_pos={n_pos})"
        )
        rows.append(
            {
                "class": col,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "accuracy": acc,
                "n_positive": n_pos,
            }
        )

    # ── 总体 ────────────────────────────────────────────────
    overall_acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n  Overall accuracy: {overall_acc:.3f}")
    print(f"  Macro F1:         {macro_f1:.3f}")

    # ── 保存报告 ────────────────────────────────────────────
    metrics_df = pd.DataFrame(rows)
    report_path = output_dir / "nlp_validation_report.csv"
    metrics_df.to_csv(report_path, index=False)
    print(f"\nPer-class 指标已保存: {report_path}")

    # ── 结论 ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if overall_acc >= 0.85 and macro_f1 >= 0.75:
        print("✅ 通过! NLP 伪标签质量足够, 可以进入第二阶段批量标注.")
    elif overall_acc >= 0.75:
        print("⚠️  边缘质量. 建议调优 prompt 后重试, 或用 HIGH/MEDIUM/LOW 分层.")
    else:
        print("❌ 未通过. 需要分析失败样本, 重新设计 prompt 或更换模型.")
    print("=" * 70)


if __name__ == "__main__":
    main()
