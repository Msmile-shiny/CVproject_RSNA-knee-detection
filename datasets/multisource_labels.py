"""Fuse official, NLP weak, and model pseudo labels into weighted soft targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]


def _probability(frame: pd.DataFrame, target: str) -> pd.Series:
    for prefix in ("prob_", "pred_", ""):
        column = f"{prefix}{target}"
        if column in frame:
            return pd.to_numeric(frame[column], errors="coerce").clip(0, 1)
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _reliability(frame: pd.DataFrame, target: str, base_weight: float) -> pd.Series:
    weight_col = f"weight_{target}"
    if weight_col in frame:
        reliability = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0).clip(0, 1)
    else:
        conf_col = f"conf_{target}"
        if conf_col in frame:
            reliability = frame[conf_col].astype(str).str.upper().map(
                {"HIGH": 1.0, "MEDIUM": 0.75, "LOW": 0.35, "REVIEW": 0.2, "FAIL": 0.0}
            ).fillna(0.0)
        else:
            reliability = pd.Series(1.0, index=frame.index)
    return reliability.astype(float) * float(base_weight)


class MultiSourceLabelBuilder:
    """Build weighted targets with strict precedence for official labels.

    Official labels always have weight 1.0. NLP and model pseudo labels are
    averaged by their per-target reliability. Their combined effective weight
    is capped below 1.0 because two correlated weak sources are not equivalent
    to a radiologist annotation.
    """

    def __init__(
        self,
        official_weight: float = 1.0,
        nlp_weight: float = 0.5,
        model_weight: float = 0.4,
        weak_weight_cap: float = 0.7,
    ):
        if not 0.3 <= model_weight <= 0.5:
            raise ValueError("model_weight must be between 0.3 and 0.5")
        self.official_weight = float(official_weight)
        self.nlp_weight = float(nlp_weight)
        self.model_weight = float(model_weight)
        self.weak_weight_cap = float(weak_weight_cap)

    @staticmethod
    def _read(value: str | Path | pd.DataFrame | None) -> pd.DataFrame:
        if value is None:
            return pd.DataFrame()
        frame = value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value)
        if "StudyInstanceUID" not in frame.columns:
            raise ValueError("label source is missing StudyInstanceUID")
        frame = frame.copy()
        frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
        if frame["StudyInstanceUID"].duplicated().any():
            raise ValueError("label source contains duplicate StudyInstanceUID")
        return frame.set_index("StudyInstanceUID")

    def build(
        self,
        official: str | Path | pd.DataFrame,
        nlp: str | Path | pd.DataFrame,
        model: str | Path | pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        official_df = self._read(official)
        nlp_df = self._read(nlp)
        model_df = self._read(model)

        official_mask = official_df[TARGET_COLUMNS].notna().all(axis=1)
        gold = official_df.loc[official_mask, TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
        gold_ids = set(gold.index)
        weak_ids = (set(nlp_df.index) | set(model_df.index)) - gold_ids
        weak = pd.DataFrame(index=sorted(weak_ids))

        for target in TARGET_COLUMNS:
            numerator = pd.Series(0.0, index=weak.index)
            denominator = pd.Series(0.0, index=weak.index)
            source_count = pd.Series(0, index=weak.index, dtype=int)
            for frame, base in ((nlp_df, self.nlp_weight), (model_df, self.model_weight)):
                if frame.empty:
                    continue
                probability = _probability(frame, target).reindex(weak.index)
                weight = _reliability(frame, target, base).reindex(weak.index).fillna(0.0)
                valid = probability.notna() & weight.gt(0)
                numerator.loc[valid] += probability.loc[valid] * weight.loc[valid]
                denominator.loc[valid] += weight.loc[valid]
                source_count.loc[valid] += 1

            weak[target] = (numerator / denominator.replace(0, np.nan)).fillna(0.5).astype("float32")
            # Agreement-aware cap: disagreement between two sources lowers confidence.
            effective = denominator.clip(upper=self.weak_weight_cap)
            if not nlp_df.empty and not model_df.empty:
                pn = _probability(nlp_df, target).reindex(weak.index)
                pm = _probability(model_df, target).reindex(weak.index)
                agreement = (1.0 - (pn - pm).abs()).clip(0.25, 1.0).fillna(1.0)
                effective *= agreement
            weak[f"weight_{target}"] = effective.astype("float32")
            weak[f"sources_{target}"] = source_count.astype("uint8")

        weak.insert(0, "StudyInstanceUID", weak.index)
        weak = weak.reset_index(drop=True)

        gold_out = gold.astype("float32").copy()
        for target in TARGET_COLUMNS:
            gold_out[f"weight_{target}"] = self.official_weight
            gold_out[f"sources_{target}"] = 1
        gold_out.insert(0, "StudyInstanceUID", gold_out.index)
        gold_out = gold_out.reset_index(drop=True)
        return weak, gold_out
