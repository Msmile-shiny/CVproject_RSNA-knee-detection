"""NLP 伪标签 — 第二阶段批量标注脚本.

对全部 4349 个无标签放射报告运行 LLM + 规则修正 + 置信度打分,
输出可直接用于模型训练的伪标签 CSV.

输出:
    data/pseudo_labels.csv     StudyInstanceUID + 12 pred + 12 conf
    data/pseudo_labels_stats.csv  置信度分布统计
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ── API 配置 ──────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from api_config import API_KEY, API_BASE, MODEL as DEFAULT_MODEL
except ImportError:
    print("[ERROR] 找不到 scripts/api_config.py")
    print("  1. cp scripts/api_config.example.py scripts/api_config.py")
    print("  2. 编辑 api_config.py, 填入你的 API key")
    sys.exit(1)

# ── 复用验证脚本的所有核心函数 ─────────────────────────────
# 直接 import 避免代码重复
from llm_validate import (
    LABEL_COLS,
    CONFIDENCE_RULES,
    STRONG_NEGATIONS,
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    parse_llm_response,
    _is_negated,
    score_confidence,
    apply_rule_correction,
    _build_messages,
)

import pandas as pd
import requests

# ── 批量参数 ──────────────────────────────────────────────
BATCH_SIZE = 50        # 每 50 条打印一次进度
REQUEST_DELAY = 0.3    # API 限流: 每条之间间隔秒数


def call_llm_single(
    report: str,
    api_base: str,
    api_key: str,
    model: str,
    max_retries: int = 3,
) -> dict[str, int] | None:
    """调用 LLM API, 返回解析后的 12 维标签, 失败返回 None."""
    url = f"{api_base.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": _build_messages(report),
        "temperature": 0.0,
        "max_tokens": 4000,
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)

            if resp.status_code != 200:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
                return None

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if content is None:
                content = ""
            content = content.strip()
            parsed = parse_llm_response(content)
            if parsed is not None:
                return parsed
            if attempt >= max_retries:
                pass  # JSON 解析失败, 返回 None
            else:
                time.sleep(1)
        except Exception:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def main():
    project_root = _SCRIPT_DIR.parent
    train_csv = project_root / "data" / "metadata" / "train.csv"
    output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载所有报告 ──────────────────────────────────────
    train = pd.read_csv(train_csv, encoding="utf-8")
    labeled_mask = train[LABEL_COLS].notna().all(axis=1)
    unlabeled_df = train[~labeled_mask].copy()
    n_total = len(unlabeled_df)

    if n_total == 0:
        print("没有无标签报告, 退出.")
        return

    print(f"加载 {n_total} 个无标签报告", flush=True)
    print(f"API: {API_BASE}  model: {DEFAULT_MODEL}", flush=True)
    print(f"预计 API 调用: {n_total} 次", flush=True)
    print(flush=True)

    if not API_KEY or "YOUR_" in API_KEY:
        print("[ERROR] 请先在 scripts/api_config.py 中填写你的 API key")
        sys.exit(1)

    # ── 逐样本标注 ────────────────────────────────────────
    results: list[dict] = []
    n_ok = 0
    n_fail = 0
    n_corrected_total = 0
    t_start = time.time()

    out_path = output_dir / "pseudo_labels.csv"
    checkpoint_path = output_dir / "pseudo_labels_checkpoint.csv"

    for idx, (_, row) in enumerate(unlabeled_df.iterrows()):
        report = str(row["Report"])
        study_uid = row["StudyInstanceUID"]
        report_truncated = report[:4000]

        t_call_start = time.time()
        llm_labels = call_llm_single(
            report_truncated,
            api_base=API_BASE,
            api_key=API_KEY,
            model=DEFAULT_MODEL,
        )
        t_call = time.time() - t_call_start

        if llm_labels is None:
            n_fail += 1
            results.append({
                "StudyInstanceUID": study_uid,
                **{f"pred_{col}": -1 for col in LABEL_COLS},
                **{f"conf_{col}": "FAIL" for col in LABEL_COLS},
            })
        else:
            corrected = apply_rule_correction(report, llm_labels)
            conf = score_confidence(report, corrected)
            n_corrected = sum(1 for c in LABEL_COLS if corrected[c] != llm_labels[c])
            n_corrected_total += n_corrected
            results.append({
                "StudyInstanceUID": study_uid,
                **{f"pred_{col}": corrected[col] for col in LABEL_COLS},
                **{f"conf_{col}": conf[col] for col in LABEL_COLS},
            })
            n_ok += 1

        # ── 实时进度 (每条都输出) ─────────────────────────
        n_done = idx + 1
        elapsed = time.time() - t_start
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (n_total - n_done) / rate if rate > 0 else 0

        # 状态标记
        status = "OK" if llm_labels else "FAIL"
        corr_str = f" +{n_corrected}corr" if (llm_labels and n_corrected > 0) else ""

        print(
            f"[{n_done:5d}/{n_total}] {study_uid[-12:]} {status} "
            f"({t_call:.1f}s{corr_str}) "
            f"| {elapsed/60:.0f}min elapsed, ~{eta/60:.0f}min remaining "
            f"| {rate:.1f} samples/min",
            flush=True,
        )

        # ── 每 200 条存一次 checkpoint ─────────────────────
        if n_done % 200 == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False, encoding="utf-8")
            print(f"  [checkpoint saved: {checkpoint_path}]", flush=True)

        time.sleep(REQUEST_DELAY)

    # ── 保存最终结果 ──────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n伪标签已保存: {out_path}", flush=True)
    print(f"成功: {n_ok}/{n_total}  ({n_ok/n_total*100:.1f}%)", flush=True)
    print(f"规则修正: {n_corrected_total} 次翻转", flush=True)

    # ── 置信度分布统计 ────────────────────────────────────
    conf_counts = {level: 0 for level in ["HIGH", "MEDIUM", "LOW", "REVIEW", "FAIL"]}
    for _, row in results_df.iterrows():
        for col in LABEL_COLS:
            level = row.get(f"conf_{col}", "FAIL")
            conf_counts[level] = conf_counts.get(level, 0) + 1

    total_preds = n_total * len(LABEL_COLS)
    print(f"\n置信度分布 ({n_total} 报告 x 12 类 = {total_preds} 预测):", flush=True)
    stats_rows = []
    for level in ["HIGH", "MEDIUM", "LOW", "REVIEW", "FAIL"]:
        n = conf_counts.get(level, 0)
        pct = n / total_preds * 100 if total_preds > 0 else 0
        print(f"  {level:<8s}: {n:6d} ({pct:5.1f}%)", flush=True)
        stats_rows.append({"confidence": level, "count": n, "pct": pct})

    stats_path = output_dir / "pseudo_labels_stats.csv"
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    print(f"\n统计已保存: {stats_path}", flush=True)

    # ── 训练用建议 ────────────────────────────────────────
    print("\n训练时使用建议:", flush=True)
    high_mask = results_df[[f"conf_{c}" for c in LABEL_COLS]].apply(
        lambda row: (row == "HIGH").all(), axis=1
    )
    medium_mask = results_df[[f"conf_{c}" for c in LABEL_COLS]].apply(
        lambda row: ((row == "HIGH") | (row == "MEDIUM")).all(), axis=1
    )
    print(f"  全 HIGH 置信度样本: {high_mask.sum()} ({high_mask.sum()/n_total*100:.0f}%) — 直接当 gold 用", flush=True)
    print(f"  HIGH+MEDIUM 样本:   {medium_mask.sum()} ({medium_mask.sum()/n_total*100:.0f}%) — 全量可用", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\n总耗时: {elapsed_total/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
