#!/usr/bin/env python3
"""Bounded-memory raw-DICOM inference for the optional BTKR M448 member."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import cnx_dicom_geometry as geom
from build_cnx_geometry_manifest import _series_job
from convnext_tokens import ConvNeXtMap
from hierarchical_slot_mil import HierarchicalSlotMIL
from ordered_montage448 import (
    BUDGET_PROFILES,
    MONTAGE_SIZE,
    SLOT_NAMES,
    TILES_PER_MONTAGE,
    MontagePositionConditioner,
    canonical_depth_layout,
    eligible_triplet_centers,
    realized_center_count,
    render_triplet_montages,
    select_ordered_centers,
    split_montage_feature_map,
)
from slot_local2d import SlotLocal2D


LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)
BACKBONE = "convnext_small.dinov3_lvd1689m"
BAND = (0.06, 0.94)
CENTER_BUDGETS = BUDGET_PROFILES["anatomy_axfs"]


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for value in iter(lambda: handle.read(chunk), b""):
            digest.update(value)
    return digest.hexdigest()


def find_competition_root(base: Path | None = None) -> Path:
    base = Path(base) if base is not None else Path(
        os.environ.get("BTKR_CNX_INPUT_ROOT", "/kaggle/input"))
    candidates = (
        base / "competitions" / "rsna-knee-abnormality-detection",
        base / "rsna-knee-abnormality-detection",
    )
    for candidate in candidates:
        if (candidate / "test_series.csv").is_file() \
                and (candidate / "test_series").is_dir() \
                and (candidate / "sample_submission.csv").is_file():
            return candidate
    if base.is_dir():
        for first in sorted(path for path in base.iterdir() if path.is_dir()):
            for candidate in (first, *sorted(
                    path for path in first.iterdir() if path.is_dir())):
                if (candidate / "test_series.csv").is_file() \
                        and (candidate / "test_series").is_dir() \
                        and (candidate / "sample_submission.csv").is_file():
                    return candidate
    raise FileNotFoundError("RSNA Knee competition test mount was not found")


def dread_band(depth: int) -> tuple[int, int]:
    if depth < 1:
        raise ValueError("series depth must be positive")
    low = int(depth * BAND[0])
    high = max(int(depth * BAND[1]) - 1, low)
    return low, min(high, depth - 1)


def _render_training_identity(path: Path, plane: str, side: str | None
                              ) -> tuple[np.ndarray | None, str]:
    try:
        payload = geom.read_pixels(path)
        canonical = geom.canonicalize_payload(payload, plane)
        image, receipt = geom.render_canonical_payload(
            canonical, plane=plane, side=side, crop_mm=130.0,
            output_size=336, interpolation=cv2.INTER_AREA)
        if image is None:
            return None, str(receipt.reason)
        y0, y1, x0, x1 = geom._crop_bounds(canonical, 130.0, 0.0, 0.0)
        rows, columns = canonical.image.shape
        if y0 < 0 or x0 < 0 or y1 > rows or x1 > columns:
            return None, "identity crop outside genuine source support"
        crop = canonical.image[y0:y1, x0:x1]
        low, high = np.percentile(crop[::4, ::4], [1, 99])
        if not np.isfinite(low + high) or high <= low:
            return None, "identity crop has no usable intensity range"
        if image.shape != (336, 336) or image.dtype != np.uint8:
            raise RuntimeError("canonical renderer returned an invalid image")
        return np.ascontiguousarray(image), ""
    except geom.GeometryError as error:
        return None, str(error)


def build_raw_manifest(data: Path, split: str = "test", workers: int = 4,
                       studies: set[str] | None = None) -> pd.DataFrame:
    data = Path(data)
    table_path = data / f"{split}_series.csv"
    series_root = data / f"{split}_series"
    series = pd.read_csv(
        table_path,
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        keep_default_na=False,
    )
    required = {
        "StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane",
        "Fat_Suppression",
    }
    if not required.issubset(series.columns):
        raise RuntimeError(
            f"{table_path} lacks {sorted(required - set(series.columns))}")
    if studies is not None:
        series = series[
            series.StudyInstanceUID.astype(str).isin(studies)].copy()
        if not len(series):
            raise RuntimeError("raw-manifest study filter selected no series")
    series["series_rank"] = series.groupby(
        ["StudyInstanceUID", "Anatomical_Plane", "Fat_Suppression"],
        sort=False,
    ).cumcount()
    jobs = [
        (
            index, str(row.StudyInstanceUID), str(row.SeriesInstanceUID),
            str(row.Anatomical_Plane), int(row.Fat_Suppression),
            int(row.series_rank), str(series_root), False,
        )
        for index, row in enumerate(series.itertuples(index=False))
    ]
    if workers == 1:
        records = [_series_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
                max_workers=workers,
                initializer=geom.initialize_cpu_worker) as pool:
            records = list(pool.map(_series_job, jobs, chunksize=16))
    receipt = pd.DataFrame(records).sort_values("series_index")
    sides = []
    for study, rows in receipt.groupby("StudyInstanceUID", sort=False):
        valid = rows[rows.status.eq("ok")]
        tags: set[str] = set()
        for text in valid.laterality_values.astype(str):
            tags.update(value for value in text.split("|") if value)
        sagittal_x = valid.loc[
            valid.Anatomical_Plane.eq("Sagittal")
            & valid.ipp_x_median.notna(),
            "ipp_x_median",
        ].astype(float).tolist()
        decision = geom.decide_study_side(str(study), tags, sagittal_x)
        sides.append({
            "StudyInstanceUID": str(study),
            "study_side": decision.side or "",
            "side_source": decision.source,
        })
    receipt = receipt.merge(
        pd.DataFrame(sides), on="StudyInstanceUID", how="left",
        validate="many_to_one")
    selected = receipt[receipt.selected_primary].copy().reset_index(drop=True)
    if selected.duplicated(["StudyInstanceUID", "slot"]).any():
        raise RuntimeError("raw manifest selected duplicate study/slot series")
    return selected


@dataclass(frozen=True)
class MontageBatch:
    images: torch.Tensor
    slot: torch.Tensor
    study: torch.Tensor
    tile_z: torch.Tensor
    tile_z_anatomical: torch.Tensor
    montage_z: torch.Tensor
    anatomy_known: torch.Tensor
    tile_log_prior: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.images.shape[0])


def _render_series_component(data: Path, split: str, row: Mapping[str, object]
                             ) -> dict | None:
    if str(row["status"]) != "ok":
        return None
    study = str(row["StudyInstanceUID"])
    series = str(row["SeriesInstanceUID"])
    plane = str(row["Anatomical_Plane"])
    slot = int(row["slot"])
    side = str(row.get("study_side", ""))
    directory = Path(data) / f"{split}_series" / study / series
    ordered = geom.order_series(directory, plane)
    if ordered.header_sha256 != str(row["header_sha256"]):
        raise RuntimeError(
            f"raw header changed after manifest scan: {study}/{series}")
    source_depth = len(ordered.files)
    low, high = dread_band(source_depth)
    band_length = high - low + 1
    usable = np.zeros(band_length, dtype=bool)
    images: dict[int, np.ndarray] = {}
    for source_index in range(low, high + 1):
        image, _ = _render_training_identity(
            ordered.files[source_index], plane, side or None)
        local = source_index - low
        if image is not None and int(image.max()) > 0:
            usable[local] = True
            images[local] = image
    eligible = eligible_triplet_centers(usable)
    realized = realized_center_count(
        len(eligible), CENTER_BUDGETS[slot])
    if realized == 0:
        return None
    centers = select_ordered_centers(
        eligible, realized, training=False,
        rng=np.random.RandomState(0),
        center_target=(source_depth - 1) / 2.0 - low,
    )
    centers, offsets, anatomy_sign, known = canonical_depth_layout(
        centers, slot, side)
    triplets = np.stack([
        np.stack([images[int(center + offset)] for offset in offsets])
        for center in centers
    ])
    montages = render_triplet_montages(triplets)
    montage_count = len(montages)
    raw_centers = low + centers
    z = ((raw_centers.astype(np.float64) - (source_depth - 1) / 2.0)
         / max(source_depth, 1)).astype(np.float32)
    anatomical = (z * anatomy_sign).astype(np.float32)
    return {
        "slot": slot,
        "images": montages,
        "tile_z": z.reshape(montage_count, TILES_PER_MONTAGE),
        "tile_z_anatomical": anatomical.reshape(
            montage_count, TILES_PER_MONTAGE),
        "montage_z": ((np.arange(montage_count, dtype=np.float64)
                       - (montage_count - 1) / 2.0) / montage_count
                      ).astype(np.float32),
        "anatomy_known": np.full(montage_count, known, dtype=bool),
        "tile_log_prior": np.zeros(
            (montage_count, TILES_PER_MONTAGE), dtype=np.float32),
    }


def build_raw_study_batch(data: Path, split: str, study: str,
                          selected: pd.DataFrame, workers: int = 4
                          ) -> MontageBatch:
    rows = selected[
        selected.StudyInstanceUID.astype(str).eq(str(study))
    ].sort_values("slot", kind="stable")
    records = rows.to_dict("records")
    if workers > 1 and len(records) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(records))) as pool:
            components = list(pool.map(
                lambda row: _render_series_component(data, split, row),
                records))
    else:
        components = [
            _render_series_component(data, split, row) for row in records]
    components = [value for value in components if value is not None]
    if not components:
        raise RuntimeError(f"study {study} has no usable montage slot")
    components.sort(key=lambda value: int(value["slot"]))
    images = np.ascontiguousarray(np.concatenate([
        value["images"] for value in components]))
    slot = np.concatenate([
        np.full(len(value["images"]), int(value["slot"]), np.int64)
        for value in components])
    tile_z = np.ascontiguousarray(np.concatenate([
        value["tile_z"] for value in components]))
    anatomical = np.ascontiguousarray(np.concatenate([
        value["tile_z_anatomical"] for value in components]))
    montage_z = np.concatenate([
        value["montage_z"] for value in components])
    known = np.concatenate([
        value["anatomy_known"] for value in components])
    prior = np.ascontiguousarray(np.concatenate([
        value["tile_log_prior"] for value in components]))
    count = len(images)
    if images.shape != (count, 3, MONTAGE_SIZE, MONTAGE_SIZE) \
            or images.dtype != np.uint8:
        raise RuntimeError("raw montage pixel contract failed")
    if slot.shape != (count,) or np.any(slot < 0) or np.any(slot >= 6) \
            or np.any(np.diff(slot) < 0):
        raise RuntimeError("raw montage slot contract failed")
    for value in (tile_z, anatomical, prior):
        if value.shape != (count, TILES_PER_MONTAGE) \
                or value.dtype != np.float32 \
                or not np.isfinite(value).all():
            raise RuntimeError("raw montage tile metadata contract failed")
    if montage_z.shape != (count,) or montage_z.dtype != np.float32 \
            or not np.isfinite(montage_z).all() \
            or known.shape != (count,) or known.dtype != np.bool_:
        raise RuntimeError("raw montage row metadata contract failed")
    return MontageBatch(
        images=torch.from_numpy(images),
        slot=torch.from_numpy(slot).long() + 1,
        study=torch.zeros(count, dtype=torch.long),
        tile_z=torch.from_numpy(tile_z),
        tile_z_anatomical=torch.from_numpy(anatomical),
        montage_z=torch.from_numpy(montage_z),
        anatomy_known=torch.from_numpy(known),
        tile_log_prior=torch.from_numpy(prior),
    )


class MontageSubmissionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = ConvNeXtMap(
            BACKBONE, in_chans=3, pretrained=False, stage=3)
        channels = int(self.enc.num_features)
        self.local = SlotLocal2D(channels)
        self.readout = HierarchicalSlotMIL(
            channels, max_depth=None, continuous_position=True)
        self.montage_tokens = MontagePositionConditioner(channels)


def load_model(checkpoint: Path,
               expected_sha256: str | None = None) -> MontageSubmissionNet:
    checkpoint = Path(checkpoint)
    if expected_sha256 is not None \
            and sha256_file(checkpoint) != str(expected_sha256):
        raise RuntimeError("optional ConvNeXt checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("selection_role") != "primary_validation_selected" \
            or payload.get("submission_eligible") is not True \
            or int(payload.get("fold", -1)) != 0:
        raise RuntimeError("optional ConvNeXt checkpoint is not fold-0 primary")
    cfg = payload.get("cfg", {})
    expected = {
        "backbone": BACKBONE,
        "pool": "hier_slot_mil",
        "cond": "post",
        "norm": "imagenet",
        "triplet": "montage4",
        "depth_remask": True,
        "montage_position_tokens": True,
        "montage_budget_profile": "anatomy_axfs",
        "img": 448,
        "meta": "none",
        "n_meta": 0,
        "site_mode": "none",
        "stem": "native",
    }
    mismatch = {
        key: (cfg.get(key), value) for key, value in expected.items()
        if cfg.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"optional ConvNeXt cfg mismatch: {mismatch}")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("optional ConvNeXt checkpoint has no state_dict")
    model = MontageSubmissionNet()
    model.load_state_dict(state, strict=True)
    return model.eval()


def _normalise(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = images.to(
        device, dtype=torch.float32, non_blocking=True).div_(255.0)
    keep = (value > 0).to(value.dtype)
    mean = value.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = value.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    value = ((value - mean) / std) * keep
    if device.type == "cuda":
        value = value.contiguous(memory_format=torch.channels_last)
    return value


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


@torch.inference_mode()
def _predict_once(model: MontageSubmissionNet, batch: MontageBatch,
                  device: torch.device, micro: int,
                  amp_dtype: torch.dtype | None) -> np.ndarray:
    features = []
    count = int(batch.images.shape[0])
    slot = batch.slot.to(device, non_blocking=True)
    tile_z = batch.tile_z.to(device, non_blocking=True)
    anatomical = batch.tile_z_anatomical.to(device, non_blocking=True)
    montage_z = batch.montage_z.to(device, non_blocking=True)
    known = batch.anatomy_known.to(device, non_blocking=True)
    for start in range(0, count, micro):
        stop = min(start + micro, count)
        image = _normalise(batch.images[start:stop], device)
        with _amp_context(device, amp_dtype):
            feature = model.enc.forward_features(image)
            feature = model.montage_tokens(
                feature, slot[start:stop], tile_z[start:stop],
                anatomical[start:stop], montage_z[start:stop],
                known[start:stop])
            feature = split_montage_feature_map(feature)
        features.append(feature)
        del image
    feature = torch.cat(features)
    expanded_slot = slot.repeat_interleave(TILES_PER_MONTAGE)
    expanded_study = torch.zeros(
        len(expanded_slot), dtype=torch.long, device=device)
    position = torch.stack((tile_z.reshape(-1), anatomical.reshape(-1)), -1)
    prior = batch.tile_log_prior.to(
        device, non_blocking=True).reshape(-1)
    with _amp_context(device, amp_dtype):
        feature = model.local(feature, expanded_slot)
        logits = model.readout(
            feature, expanded_slot, None, expanded_study, 1,
            position=position, log_instance_prior=prior)
    if logits.shape != (1, len(LABELS)) \
            or not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("optional ConvNeXt returned invalid logits")
    return torch.sigmoid(logits.float()).cpu().numpy()[0].astype(np.float32)


def _is_oom(error: RuntimeError) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) \
        or "out of memory" in str(error).lower()


def _is_engine_error(error: RuntimeError) -> bool:
    text = str(error).lower()
    return "unable to find an engine" in text \
        or "cudnn_status_not_supported" in text


def predict_study(model: MontageSubmissionNet, batch: MontageBatch,
                  device: torch.device, micro: int = 4) -> np.ndarray:
    current_micro = max(1, int(micro))
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        amp_dtype: torch.dtype | None = (
            torch.bfloat16 if major >= 8 else torch.float16)
    else:
        amp_dtype = None
    while True:
        try:
            return _predict_once(
                model, batch, device, current_micro, amp_dtype)
        except RuntimeError as error:
            if device.type == "cuda" and _is_oom(error) \
                    and current_micro > 1:
                current_micro = max(1, current_micro // 2)
                torch.cuda.empty_cache()
                continue
            if device.type == "cuda" and amp_dtype is not None \
                    and _is_engine_error(error):
                amp_dtype = None
                current_micro = 1
                torch.cuda.empty_cache()
                continue
            raise


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def run(checkpoint: Path, output: Path, data: Path | None = None,
        expected_checkpoint_sha256: str | None = None,
        workers: int = 4, micro: int = 4) -> pd.DataFrame:
    cv2.setNumThreads(1)
    torch.set_num_threads(max(1, min(int(workers), os.cpu_count() or 1)))
    data = Path(data) if data is not None else find_competition_root()
    sample = pd.read_csv(
        data / "sample_submission.csv", dtype={"StudyInstanceUID": str})
    if sample.columns.tolist() != ["StudyInstanceUID", *LABELS]:
        raise RuntimeError("optional ConvNeXt submission schema drift")
    studies = sample.StudyInstanceUID.astype(str).tolist()
    if len(studies) != len(set(studies)):
        raise RuntimeError("optional ConvNeXt received duplicate study IDs")
    selected = build_raw_manifest(
        data, "test", workers=workers, studies=set(studies))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = load_model(checkpoint, expected_checkpoint_sha256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        model = model.to(device, memory_format=torch.channels_last).eval()
    else:
        model = model.to(device).eval()
    predictions = []
    for study in studies:
        batch = build_raw_study_batch(
            data, "test", study, selected, workers=workers)
        predictions.append(predict_study(
            model, batch, device, micro=micro))
    result = sample.copy()
    result[list(LABELS)] = np.stack(predictions)
    if result.StudyInstanceUID.astype(str).tolist() != studies \
            or not np.isfinite(result[list(LABELS)].to_numpy()).all():
        raise RuntimeError("optional ConvNeXt output validation failed")
    _atomic_csv(result, output)
    return result


__all__ = [
    "BACKBONE", "CENTER_BUDGETS", "LABELS", "MontageBatch",
    "MontageSubmissionNet", "build_raw_manifest", "build_raw_study_batch",
    "find_competition_root", "load_model", "predict_study", "run",
    "sha256_file",
]
