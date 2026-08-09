"""伪标签加载与合并模块.

从 NLP 管线生成的 pseudo_labels.csv 加载伪标签, 支持置信度过滤和
与 gold label (58 个标注样本) 的合并.

数据格式:
  pseudo_labels.csv:
    StudyInstanceUID, pred_ACL, ..., pred_Fracture, conf_ACL, ..., conf_Fracture
    4350 行, 每行一个 study 的 12 类预测 + 置信度

  pseudo_labels_valid.csv:
    NLP 验证集的详细结果 (163 行), 包含 true_*, pred_*, conf_*, n_votes 等

使用:
    from datasets import PseudoLabelLoader

    loader = PseudoLabelLoader("data/pseudo_labels.csv")
    pseudo_df = loader.load_filtered("HIGH")
    # pseudo_df 可直接传入 Knee25DDataset 作为 labels_df

    # 合并 gold + pseudo:
    train_df, val_df = loader.merge_with_gold(
        gold_csv="data/metadata/train.csv",
        confidence="HIGH",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COLUMNS = [
    "ACL", "MCL",
    "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

CONFIDENCE_LEVELS = ["HIGH", "MEDIUM", "LOW", "REVIEW", "FAIL"]


class PseudoLabelLoader:
    """伪标签加载器.

    加载 NLP 生成的 pseudo_labels.csv, 提供置信度过滤和 gold label 合并.

    Args:
        pseudo_csv: pseudo_labels.csv 路径
        pseudo_valid_csv: pseudo_labels_valid.csv 路径 (可选)
        nlp_report_csv: nlp_validation_report.csv 路径 (可选)
    """

    def __init__(
        self,
        pseudo_csv: str | Path = "data/pseudo_labels.csv",
        pseudo_valid_csv: str | Path | None = "data/pseudo_labels_valid.csv",
        nlp_report_csv: str | Path | None = "data/nlp_validation_report.csv",
        nlp_filtered_csv: str | Path | None = "data/nlp_validation_filtered.csv",
    ):
        self.pseudo_csv = Path(pseudo_csv)
        self.pseudo_valid_csv = Path(pseudo_valid_csv) if pseudo_valid_csv else None
        self.nlp_report_csv = Path(nlp_report_csv) if nlp_report_csv else None
        self.nlp_filtered_csv = Path(nlp_filtered_csv) if nlp_filtered_csv else None

        self._df: pd.DataFrame | None = None
        self._valid_df: pd.DataFrame | None = None
        self._report_df: pd.DataFrame | None = None
        self._filtered_report_df: pd.DataFrame | None = None

    # ── 加载 ─────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """加载原始 pseudo_labels.csv.

        Returns:
            DataFrame: StudyInstanceUID + pred_* (12) + conf_* (12)
        """
        if self._df is None:
            if not self.pseudo_csv.exists():
                raise FileNotFoundError(f"伪标签文件不存在: {self.pseudo_csv}")

            self._df = pd.read_csv(self.pseudo_csv)
            logger.info(
                "加载伪标签: %s 行, %s 列, %s",
                len(self._df),
                len(self._df.columns),
                self.pseudo_csv,
            )
        return self._df

    def load_valid(self) -> pd.DataFrame:
        """加载 NLP 验证集 (pseudo_labels_valid.csv).

        Returns:
            DataFrame: true_* + pred_* + conf_* + n_votes + vote_agreement + n_corrected
        """
        if self._valid_df is None and self.pseudo_valid_csv is not None:
            if not self.pseudo_valid_csv.exists():
                logger.warning("验证集伪标签文件不存在: %s", self.pseudo_valid_csv)
                return pd.DataFrame()
            self._valid_df = pd.read_csv(self.pseudo_valid_csv)
            logger.info("加载伪标签验证集: %s 行", len(self._valid_df))
        return self._valid_df if self._valid_df is not None else pd.DataFrame()

    def load_report(self) -> pd.DataFrame:
        """加载 NLP 验证报告.

        Returns:
            DataFrame: class, precision, recall, f1, accuracy, n_positive
        """
        if self._report_df is None and self.nlp_report_csv is not None:
            if self.nlp_report_csv.exists():
                self._report_df = pd.read_csv(self.nlp_report_csv)
        return self._report_df if self._report_df is not None else pd.DataFrame()

    def load_filtered_report(self) -> pd.DataFrame:
        """加载高置信度过滤后的 NLP 验证报告."""
        if self._filtered_report_df is None and self.nlp_filtered_csv is not None:
            if self.nlp_filtered_csv.exists():
                self._filtered_report_df = pd.read_csv(self.nlp_filtered_csv)
        return self._filtered_report_df if self._filtered_report_df is not None else pd.DataFrame()

    # ── 置信度过滤 ───────────────────────────────────────────────

    def _build_confidence_mask(
        self, df: pd.DataFrame, level: str
    ) -> pd.Series:
        """构建置信度过滤 mask.

        Args:
            df: 包含 conf_* 列的 DataFrame
            level: "HIGH" | "MEDIUM" | "LOW" | "REVIEW" | "FAIL"
                   "HIGH_PLUS_MEDIUM" → HIGH 或 MEDIUM
                   "ALL" → 全部保留

        Returns:
            bool Series, True 表示该行所有 12 类都满足置信度要求.
        """
        if level == "ALL":
            return pd.Series(True, index=df.index)

        if level == "HIGH_PLUS_MEDIUM":
            allowed = {"HIGH", "MEDIUM"}
        else:
            allowed = {level}

        mask = pd.Series(True, index=df.index)
        for col_name in TARGET_COLUMNS:
            conf_col = f"conf_{col_name}"
            if conf_col in df.columns:
                mask &= df[conf_col].isin(allowed)
            # 如果某列不存在, 不参与过滤 (保持宽松)

        return mask

    def filter_by_confidence(
        self,
        df: pd.DataFrame | None = None,
        level: str = "HIGH",
    ) -> pd.DataFrame:
        """按置信度过滤伪标签.

        Args:
            df: 伪标签 DataFrame (None 则加载全部)
            level: 置信度阈值
                "HIGH" — 所有 12 类都是 HIGH
                "HIGH_PLUS_MEDIUM" — 所有 12 类都是 HIGH 或 MEDIUM
                "ALL" — 不做过滤, 全部保留

        Returns:
            过滤后的 DataFrame (仍包含 pred_* 和 conf_* 列).
        """
        if df is None:
            df = self.load()

        mask = self._build_confidence_mask(df, level)
        filtered = df[mask].copy()

        logger.info(
            "置信度过滤 [%s]: %d → %d 行 (%.1f%%)",
            level, len(df), len(filtered),
            100 * len(filtered) / max(len(df), 1),
        )
        return filtered

    # ── 转换为标签 DataFrame ─────────────────────────────────────

    def to_labels_df(
        self,
        confidence: str = "HIGH",
    ) -> pd.DataFrame:
        """将伪标签转换为标准标签 DataFrame.

        Returns:
            DataFrame: index=StudyInstanceUID, columns=12 target classes, values=0/1
            可直接作为 Knee25DDataset 的 labels_df 参数.

        Example:
            loader = PseudoLabelLoader()
            labels_df = loader.to_labels_df("HIGH")
            ds = Knee25DDataset(series_df, labels_df, ...)
        """
        df = self.load()
        filtered = self.filter_by_confidence(df, level=confidence)

        labels_df = filtered[["StudyInstanceUID"]].copy()
        for col_name in TARGET_COLUMNS:
            pred_col = f"pred_{col_name}"
            if pred_col in filtered.columns:
                labels_df[col_name] = (
                    pd.to_numeric(filtered[pred_col], errors="coerce").fillna(0).astype(int)
                )
            else:
                labels_df[col_name] = 0

        labels_df = labels_df.set_index("StudyInstanceUID")
        labels_df = labels_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

        logger.info(
            "伪标签 → labels_df: %d studies, class balance:\n%s",
            len(labels_df),
            labels_df.sum().to_string(),
        )
        return labels_df

    # ── 合并 Gold + Pseudo ───────────────────────────────────────

    def merge_with_gold(
        self,
        gold_csv: str | Path = "data/metadata/train.csv",
        confidence: str = "HIGH",
        val_from_gold: bool = True,
        val_ratio: float = 0.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """合并 gold label 和 pseudo label, 返回 train/val 标签集.

        策略:
        - 伪标签数据作为训练集 (~4000+ studies)
        - Gold label (58) 作为验证集 (确保评估可靠)
        - 可选: 从伪标签中切一部分做验证 (val_ratio > 0)

        Args:
            gold_csv: gold label train.csv 路径 (58 个标注样本)
            confidence: 伪标签置信度过滤级别
            val_from_gold: True → gold 全部做验证, False → gold 也加入训练
            val_ratio: 从伪标签训练集中切出验证集的比例 (0 = 不切)

        Returns:
            (train_labels_df, val_labels_df):
                均为 index=StudyInstanceUID, columns=12 target classes 的 DataFrame
        """
        # ── 1. 加载 gold labels ─────────────────────────────────
        gold_path = Path(gold_csv)
        if not gold_path.exists():
            logger.warning("Gold label CSV 不存在: %s, 仅使用伪标签", gold_path)
            pseudo_df = self.to_labels_df(confidence)
            return pseudo_df, pd.DataFrame()

        gold_df = pd.read_csv(gold_path)

        # 识别 label 列
        label_cols = [c for c in TARGET_COLUMNS if c in gold_df.columns]
        if not label_cols:
            logger.warning("Gold CSV 中未找到标签列, 仅使用伪标签")
            pseudo_df = self.to_labels_df(confidence)
            return pseudo_df, pd.DataFrame()

        # 只保留有标签的行 (排除纯提交样本)
        has_label = gold_df[label_cols].notna().any(axis=1)
        gold_df = gold_df[has_label].copy()

        gold_labels = gold_df[["StudyInstanceUID"] + label_cols].copy()
        for c in TARGET_COLUMNS:
            if c not in gold_labels.columns:
                gold_labels[c] = 0
        gold_labels = gold_labels.set_index("StudyInstanceUID")
        gold_labels = gold_labels.apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0).astype(int)

        logger.info("Gold labels: %d studies", len(gold_labels))

        # ── 2. 加载伪标签 ───────────────────────────────────────
        pseudo_df = self.to_labels_df(confidence)

        # 排除已在 gold 中的 study (避免重复)
        pseudo_df = pseudo_df[~pseudo_df.index.isin(gold_labels.index)]
        logger.info("Pseudo labels (excluding gold): %d studies", len(pseudo_df))

        # ── 3. 构建 train/val split ─────────────────────────────
        if val_from_gold:
            # 标准策略: gold 做验证, pseudo 做训练
            train_df = pseudo_df
            val_df = gold_labels
        else:
            # Gold 也加入训练 (仅当有其他验证集时使用)
            train_df = pd.concat([pseudo_df, gold_labels])
            val_df = pd.DataFrame()

        # 可选: 从伪标签训练集中切一小部分做验证
        if val_ratio > 0 and len(train_df) > 10:
            val_n = max(5, int(len(train_df) * val_ratio))
            val_from_train = train_df.sample(n=val_n, random_state=42)
            train_df = train_df.drop(val_from_train.index)
            val_df = pd.concat([val_df, val_from_train])

        logger.info(
            "Merge完成: train=%d studies, val=%d studies",
            len(train_df), len(val_df),
        )

        return train_df, val_df

    # ── 统计 ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """返回伪标签统计信息.

        Returns:
            {"total": int, "by_confidence": {level: count}, "class_balance": Series}
        """
        df = self.load()

        by_conf = {}
        for col_name in TARGET_COLUMNS:
            conf_col = f"conf_{col_name}"
            if conf_col in df.columns:
                counts = df[conf_col].value_counts().to_dict()
                by_conf[col_name] = counts

        # 整体置信度分布 (取最严格的, 即所有类的 conf 交叉)
        total_by_level = {}
        for level in CONFIDENCE_LEVELS + ["HIGH_PLUS_MEDIUM"]:
            mask = self._build_confidence_mask(df, level)
            total_by_level[level] = int(mask.sum())

        class_balance = {}
        for col_name in TARGET_COLUMNS:
            pred_col = f"pred_{col_name}"
            if pred_col in df.columns:
                class_balance[col_name] = int(df[pred_col].sum())

        return {
            "total": len(df),
            "by_confidence_overall": total_by_level,
            "class_balance": class_balance,
        }


# ── 便捷函数 ─────────────────────────────────────────────────────


def load_train_val_labels(
    pseudo_csv: str | Path = "data/pseudo_labels.csv",
    gold_csv: str | Path = "data/metadata/train.csv",
    confidence: str = "HIGH",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """一行加载训练/验证标签 (便捷函数).

    Returns:
        (train_labels_df, val_labels_df):
            可直接传入 Knee25DDataset 的 labels_df 参数.

    Example:
        from datasets.pseudo_labels import load_train_val_labels

        train_labels, val_labels = load_train_val_labels(
            pseudo_csv="data/pseudo_labels.csv",
            gold_csv="data/metadata/train.csv",
            confidence="HIGH",
        )

        train_ds = Knee25DDataset(series_df, train_labels, ...)
        val_ds = Knee25DDataset(series_df, val_labels, ...)
    """
    loader = PseudoLabelLoader(
        pseudo_csv=pseudo_csv,
        pseudo_valid_csv=None,  # 验证集信息来自 gold labels
    )
    return loader.merge_with_gold(
        gold_csv=gold_csv,
        confidence=confidence,
        val_from_gold=True,
    )
