"""Immutable FP32 inference runtime for the ``public0033_meniscus10`` bundle.

This file is copied to the *root* of the private Kaggle Dataset as
``public0033_runtime.py``.  It deliberately contains its own DINOv2 Base and
SlotHead definition instead of importing a repository module: the only model
state accepted is the completed fullfit0033 checkpoint named in the bundle
manifest.

The public route consumes the raw CSV produced here only for the two Meniscus
columns.  It never reads parent predictions, labels, reports, Gold, or Public
LB data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn


BUNDLE_SCHEMA: Final[str] = "public0033_meniscus10_bundle_v1"
BUNDLE_STATUS: Final[str] = "complete_local_not_uploaded"
BUNDLE_ID: Final[str] = "public0033_meniscus10"
RUNTIME_FILENAME: Final[str] = "public0033_runtime.py"
ENTRYPOINT: Final[str] = "run_cached_inference(cache, mask, studies, output_paths)"
CHECKPOINT_FILENAME: Final[str] = "checkpoint_epoch30.safetensors"
SIDECAR_FILENAME: Final[str] = "checkpoint_epoch30.safetensors.json"
RUN_CONTRACT_FILENAME: Final[str] = "run_contract.json"
TASK_COMPLETE_FILENAME: Final[str] = "task_complete.json"
FULLFIT_CONFIG_FILENAME: Final[str] = "fullfit0033_meniscus_bag.yaml"
DINO_CONFIG_FILENAME: Final[str] = "dinov2_config.json"
PUBLIC_CONFIG_FILENAME: Final[str] = "public0033_meniscus10_residual.yaml"
EXPECTED_STATE_TENSORS: Final[int] = 233
N_SLOTS: Final[int] = 6
SLICES_PER_SLOT: Final[int] = 12
IMAGE_SIZE: Final[int] = 336
WINDOW_CHANNELS: Final[int] = 3
BATCH_STUDIES: Final[int] = 32
WINDOW_STARTS: Final[tuple[int, ...]] = tuple(range(10))
HIDDEN_SIZE: Final[int] = 768

TARGETS: Final[tuple[str, ...]] = (
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
)
RAW_TARGETS: Final[tuple[str, str]] = ("Medial Meniscus", "Lateral Meniscus")
RAW_TARGET_INDICES: Final[tuple[int, int]] = tuple(TARGETS.index(name) for name in RAW_TARGETS)  # type: ignore[assignment]
EXPECTED_OUTPUT_KEYS: Final[frozenset[str]] = frozenset(
    {"bag_raw_csv", "receipt_json", "work_dir"}
)


class Public0033InferenceError(RuntimeError):
    """Raised before a score-changing public0033 artifact is emitted."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA256 of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - error detail is the contract
        raise Public0033InferenceError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Public0033InferenceError(f"JSON root must be an object: {path}")
    return value


def _safe_relative(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise Public0033InferenceError(f"unsafe bundle-relative path: {value!r}")
    return path


def _require_equal(value: object, expected: object, name: str) -> None:
    if value != expected:
        raise Public0033InferenceError(
            f"{name} mismatch: observed={value!r}, expected={expected!r}"
        )


def _manifest_file_entry(
    manifest: Mapping[str, Any], relative: str
) -> tuple[str, int]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise Public0033InferenceError("bundle files map missing")
    entry = files.get(relative)
    if not isinstance(entry, Mapping):
        raise Public0033InferenceError(f"bundle file missing from manifest: {relative}")
    digest = entry.get("sha256")
    size = entry.get("bytes")
    if not isinstance(digest, str) or len(digest) != 64:
        raise Public0033InferenceError(f"invalid SHA256 declaration for {relative}")
    if not isinstance(size, int) or size < 0:
        raise Public0033InferenceError(f"invalid byte declaration for {relative}")
    return digest, size


def _validate_manifest_schema(manifest: Mapping[str, Any]) -> None:
    """Validate immutable public0033 route declarations, without file access."""

    _require_equal(manifest.get("schema_version"), BUNDLE_SCHEMA, "bundle schema")
    _require_equal(manifest.get("status"), BUNDLE_STATUS, "bundle status")
    _require_equal(manifest.get("bundle_id"), BUNDLE_ID, "bundle ID")
    _require_equal(manifest.get("target_order"), list(TARGETS), "target order")
    _require_equal(manifest.get("raw_output_targets"), list(RAW_TARGETS), "raw target set")
    _require_equal(manifest.get("precision"), "fp32", "precision")
    _require_equal(manifest.get("required_gpu_count"), 2, "required GPU count")
    _require_equal(manifest.get("required_gpu_family"), "Tesla T4", "required GPU family")
    _require_equal(manifest.get("batch_studies"), BATCH_STUDIES, "batch studies")
    _require_equal(manifest.get("window_starts"), list(WINDOW_STARTS), "window starts")
    _require_equal(manifest.get("state_tensor_count"), EXPECTED_STATE_TENSORS, "state tensor count")

    route = manifest.get("fixed_public_route")
    expected_route = {
        "exact_parent": "exact_0.936",
        "targets": list(RAW_TARGETS),
        "parent_rank_weight": 0.90,
        "bag_rank_weight": 0.10,
        "other_targets_byte_preserved": True,
        "submission_count_max": 1,
    }
    if route != expected_route:
        raise Public0033InferenceError("fixed public route declaration changed")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise Public0033InferenceError("runtime declaration missing")
    _require_equal(runtime.get("bundle_relative_path"), RUNTIME_FILENAME, "runtime path")
    _require_equal(runtime.get("entrypoint"), ENTRYPOINT, "runtime entrypoint")
    runtime_hash = runtime.get("sha256")
    if not isinstance(runtime_hash, str) or len(runtime_hash) != 64:
        raise Public0033InferenceError("runtime SHA256 declaration missing")

    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise Public0033InferenceError("checkpoint declaration missing")
    _require_equal(
        checkpoint.get("bundle_relative_path"), CHECKPOINT_FILENAME, "checkpoint path"
    )
    if not isinstance(checkpoint.get("sha256"), str) or len(str(checkpoint["sha256"])) != 64:
        raise Public0033InferenceError("checkpoint SHA256 declaration missing")
    if not isinstance(checkpoint.get("bytes"), int) or int(checkpoint["bytes"]) <= 0:
        raise Public0033InferenceError("checkpoint byte declaration invalid")
    _require_equal(
        checkpoint.get("state_tensor_count"), EXPECTED_STATE_TENSORS, "checkpoint tensor count"
    )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Public0033InferenceError("artifact declaration missing")
    expected_artifacts = {
        "checkpoint_sidecar": SIDECAR_FILENAME,
        "run_contract": RUN_CONTRACT_FILENAME,
        "task_complete": TASK_COMPLETE_FILENAME,
        "fullfit_config": FULLFIT_CONFIG_FILENAME,
        "dinov2_config": DINO_CONFIG_FILENAME,
        "public0033_config": PUBLIC_CONFIG_FILENAME,
    }
    for key, filename in expected_artifacts.items():
        value = artifacts.get(key)
        if not isinstance(value, Mapping):
            raise Public0033InferenceError(f"artifact declaration missing: {key}")
        _require_equal(value.get("bundle_relative_path"), filename, f"artifact path {key}")
        digest = value.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise Public0033InferenceError(f"artifact SHA256 missing: {key}")

    required_files = {
        RUNTIME_FILENAME,
        CHECKPOINT_FILENAME,
        SIDECAR_FILENAME,
        RUN_CONTRACT_FILENAME,
        TASK_COMPLETE_FILENAME,
        FULLFIT_CONFIG_FILENAME,
        DINO_CONFIG_FILENAME,
        PUBLIC_CONFIG_FILENAME,
        "dataset-metadata.json",
    }
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(map(str, files)) != required_files:
        raise Public0033InferenceError("bundle file set differs from the fixed contract")
    for relative in required_files:
        _manifest_file_entry(manifest, relative)

    forbidden = manifest.get("forbidden_inputs")
    expected_forbidden = {
        "gold_label_values_read": False,
        "gold_metrics_read": False,
        "public_lb_read": False,
        "reports_read": False,
        "portfolio0034_input": False,
    }
    if forbidden != expected_forbidden:
        raise Public0033InferenceError("forbidden-input declaration changed")


def _verify_file(root: Path, manifest: Mapping[str, Any], relative: str) -> None:
    expected_hash, expected_bytes = _manifest_file_entry(manifest, relative)
    path = root / _safe_relative(relative)
    if not path.is_file():
        raise Public0033InferenceError(f"bundle file missing: {path}")
    if path.stat().st_size != expected_bytes:
        raise Public0033InferenceError(f"bundle file size mismatch: {relative}")
    if sha256_file(path) != expected_hash:
        raise Public0033InferenceError(f"bundle file SHA256 mismatch: {relative}")


def verify_bundle(bundle_root: str | Path) -> dict[str, Any]:
    """Hash-check all inference inputs and cross-check fullfit provenance."""

    root = Path(bundle_root).resolve()
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise Public0033InferenceError(f"bundle manifest missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    _validate_manifest_schema(manifest)
    for relative in sorted(map(str, manifest["files"])):
        path = root / _safe_relative(relative)
        # Kaggle consumes dataset-metadata.json as control-plane metadata and
        # does not expose it in the mounted Dataset payload.  It is not a model
        # input; all prediction-bearing files remain mandatory and hash-checked.
        if relative == "dataset-metadata.json" and not path.is_file():
            continue
        _verify_file(root, manifest, relative)

    runtime = manifest["runtime"]
    runtime_hash, _ = _manifest_file_entry(manifest, RUNTIME_FILENAME)
    _require_equal(runtime["sha256"], runtime_hash, "runtime/file hash")

    checkpoint = manifest["checkpoint"]
    checkpoint_hash, checkpoint_bytes = _manifest_file_entry(manifest, CHECKPOINT_FILENAME)
    _require_equal(checkpoint["sha256"], checkpoint_hash, "checkpoint/file hash")
    _require_equal(checkpoint["bytes"], checkpoint_bytes, "checkpoint/file bytes")

    artifacts = manifest["artifacts"]
    sidecar = _read_json(root / SIDECAR_FILENAME)
    _require_equal(sidecar.get("schema_version"), "fullfit0033_checkpoint_v1", "sidecar schema")
    for key in ("checkpoint_sha256", "bytes", "checkpoint", "state_sha256", "architecture_sha256"):
        if key not in sidecar:
            raise Public0033InferenceError(f"checkpoint sidecar field missing: {key}")
    _require_equal(sidecar["checkpoint"], CHECKPOINT_FILENAME, "sidecar checkpoint path")
    _require_equal(sidecar["checkpoint_sha256"], checkpoint_hash, "sidecar checkpoint hash")
    _require_equal(sidecar["bytes"], checkpoint_bytes, "sidecar checkpoint bytes")
    _require_equal(
        artifacts["checkpoint_sidecar"]["sha256"],
        _manifest_file_entry(manifest, SIDECAR_FILENAME)[0],
        "sidecar/file hash",
    )

    run_contract = _read_json(root / RUN_CONTRACT_FILENAME)
    _require_equal(run_contract.get("schema_version"), "fullfit0033_run_contract_v1", "run contract schema")
    _require_equal(run_contract.get("status"), "complete", "run contract status")
    _require_equal(run_contract.get("experiment_id"), "fullfit0033", "run contract experiment")
    _require_equal(run_contract.get("seed"), 2027, "run contract seed")
    run_checkpoint = run_contract.get("checkpoint")
    if not isinstance(run_checkpoint, Mapping):
        raise Public0033InferenceError("run contract checkpoint missing")
    _require_equal(run_checkpoint.get("checkpoint"), CHECKPOINT_FILENAME, "run checkpoint path")
    _require_equal(run_checkpoint.get("checkpoint_sha256"), checkpoint_hash, "run checkpoint hash")
    _require_equal(run_checkpoint.get("bytes"), checkpoint_bytes, "run checkpoint bytes")
    _require_equal(
        artifacts["run_contract"]["sha256"],
        _manifest_file_entry(manifest, RUN_CONTRACT_FILENAME)[0],
        "run contract/file hash",
    )

    task_complete = _read_json(root / TASK_COMPLETE_FILENAME)
    _require_equal(task_complete.get("schema_version"), "fullfit0033_task_complete_v1", "task complete schema")
    _require_equal(task_complete.get("status"), "complete", "task complete status")
    identity = task_complete.get("task_identity")
    if not isinstance(identity, Mapping):
        raise Public0033InferenceError("task identity missing")
    _require_equal(identity.get("experiment_id"), "fullfit0033", "task experiment")
    _require_equal(identity.get("seed"), 2027, "task seed")
    _require_equal(identity.get("task"), "full_data_seed2027", "task name")
    task_hashes = task_complete.get("artifact_sha256")
    if not isinstance(task_hashes, Mapping):
        raise Public0033InferenceError("task artifact hashes missing")
    _require_equal(task_hashes.get(CHECKPOINT_FILENAME), checkpoint_hash, "task checkpoint hash")
    _require_equal(
        task_hashes.get(SIDECAR_FILENAME), _manifest_file_entry(manifest, SIDECAR_FILENAME)[0], "task sidecar hash"
    )
    _require_equal(
        task_hashes.get(RUN_CONTRACT_FILENAME),
        _manifest_file_entry(manifest, RUN_CONTRACT_FILENAME)[0],
        "task run contract hash",
    )
    _require_equal(
        artifacts["task_complete"]["sha256"],
        _manifest_file_entry(manifest, TASK_COMPLETE_FILENAME)[0],
        "task complete/file hash",
    )

    fullfit_config_hash = _manifest_file_entry(manifest, FULLFIT_CONFIG_FILENAME)[0]
    _require_equal(
        artifacts["fullfit_config"]["sha256"], fullfit_config_hash, "fullfit config/file hash"
    )
    _require_equal(run_contract.get("config_sha256"), fullfit_config_hash, "run config hash")
    _require_equal(identity.get("config_sha256"), fullfit_config_hash, "task config hash")

    config_hash = _manifest_file_entry(manifest, DINO_CONFIG_FILENAME)[0]
    _require_equal(
        artifacts["dinov2_config"]["sha256"], config_hash, "DINO config/file hash"
    )
    model = run_contract.get("model")
    if not isinstance(model, Mapping):
        raise Public0033InferenceError("run contract model declaration missing")
    _require_equal(model.get("encoder"), "dinov2_base", "DINO encoder")
    _require_equal(model.get("encoder_config_sha256"), config_hash, "DINO config provenance hash")
    _require_equal(model.get("model_image_size"), IMAGE_SIZE, "model image size")
    _require_equal(model.get("slots"), N_SLOTS, "model slot count")
    _require_equal(model.get("inference"), "overlap10_probability_mean", "inference mode")
    _require_equal(run_contract.get("fold_safe_oof"), False, "fullfit OOF role")
    _require_equal(run_contract.get("promotion"), False, "fullfit promotion role")
    _require_equal(run_contract.get("portfolio0034_input"), False, "portfolio isolation")
    return manifest


def find_unique_bundle_root(input_root: str | Path = "/kaggle/input") -> Path:
    """Find exactly one mounted public0033 bundle, without broad DICOM traversal."""

    explicit = os.environ.get("PUBLIC0033_BUNDLE_ROOT")
    roots = [Path(explicit)] if explicit else [Path(input_root)]
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in (
            "bundle_manifest.json",
            "*/bundle_manifest.json",
            "*/*/bundle_manifest.json",
            "*/*/*/bundle_manifest.json",
        ):
            candidates.extend(root.glob(pattern))
    complete: list[Path] = []
    for manifest_path in sorted({path.resolve() for path in candidates}):
        try:
            manifest = _read_json(manifest_path)
            _validate_manifest_schema(manifest)
        except Exception:
            continue
        complete.append(manifest_path.parent)
    if len(complete) != 1:
        raise Public0033InferenceError(
            f"expected exactly one {BUNDLE_SCHEMA} bundle root, found {len(complete)}"
        )
    return complete[0]


def import_unique_runtime(input_root: str | Path = "/kaggle/input") -> Any:
    """Unique-import the root ``public0033_runtime.py`` after manifest discovery.

    A notebook can use this helper only after obtaining this module by a trusted
    bootstrap.  It is also useful for a local bundle preflight because it avoids
    any accidental same-named module from another mounted Dataset.
    """

    root = find_unique_bundle_root(input_root)
    manifest = verify_bundle(root)
    runtime_path = root / RUNTIME_FILENAME
    expected_hash = str(manifest["runtime"]["sha256"])
    if sha256_file(runtime_path) != expected_hash:
        raise Public0033InferenceError("unique runtime hash mismatch")
    module_name = f"_public0033_runtime_{expected_hash[:16]}"
    existing = __import__("sys").modules.get(module_name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(module_name, runtime_path)
    if specification is None or specification.loader is None:
        raise Public0033InferenceError("cannot create unique runtime import")
    module = importlib.util.module_from_spec(specification)
    __import__("sys").modules[module_name] = module
    specification.loader.exec_module(module)
    function = getattr(module, "run_cached_inference", None)
    if not callable(function):
        raise Public0033InferenceError("unique runtime lacks run_cached_inference")
    return module


class SlotHead(nn.Module):
    """Exact Exp0026/Spec0033 target-specific slot head."""

    def __init__(
        self,
        dim: int,
        n_slot: int = N_SLOTS,
        n_out: int = len(TARGETS),
        hidden: int = 256,
        p: float = 0.2,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU()
        )
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(features) + self.slot_emb
        attention = (
            torch.einsum("bsh,oh->bos", hidden, self.query) / self.hidden**0.5
        )
        attention = attention.masked_fill(
            mask.unsqueeze(1) < 0.5, -1e4
        ).softmax(-1)
        context = self.drop(torch.einsum("bos,bsh->boh", attention, hidden))
        return (
            context * self.out.weight.unsqueeze(0)
        ).sum(-1) + self.out.bias


class Fullfit0033Model(nn.Module):
    """DINOv2 Base + CLS/patch mean + SlotHead, matching the fullfit state."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = SlotHead(HIDDEN_SIZE * 2)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, slots = images.shape[:2]
        if slots != N_SLOTS:
            raise ValueError(f"expected {N_SLOTS} slots, got {slots}")
        values = images.reshape(batch * slots, *images.shape[2:]).float().div_(255.0)
        if values.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            values = F.interpolate(
                values,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        values = (values - self.mean) / self.std
        output = self.backbone(pixel_values=values).last_hidden_state
        if output.shape[1:] != (577, HIDDEN_SIZE):
            raise RuntimeError(f"unexpected DINOv2 output shape: {tuple(output.shape)}")
        features = torch.cat(
            [output[:, 0], output[:, 1:].mean(1)], dim=1
        ).reshape(batch, slots, -1)
        return self.head(features, mask)


def _build_model(config_path: Path) -> Fullfit0033Model:
    from transformers import Dinov2Config, Dinov2Model

    config = Dinov2Config.from_json_file(str(config_path))
    expected = {
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": 12,
        "patch_size": 14,
    }
    for key, value in expected.items():
        if int(getattr(config, key)) != value:
            raise Public0033InferenceError(f"DINO config {key} changed")
    return Fullfit0033Model(Dinov2Model(config))


def _load_fullfit_model(root: Path, device: torch.device) -> Fullfit0033Model:
    """Load exactly the 233-tensor fullfit state with ``strict=True``."""

    model = _build_model(root / DINO_CONFIG_FILENAME)
    state = load_file(str(root / CHECKPOINT_FILENAME), device="cpu")
    if len(state) != EXPECTED_STATE_TENSORS:
        raise Public0033InferenceError(
            f"checkpoint tensor count mismatch: {len(state)} != {EXPECTED_STATE_TENSORS}"
        )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Public0033InferenceError(f"strict state load mismatch: {incompatible}")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise Public0033InferenceError("model parameter precision drifted from FP32")
    model.eval().to(device)
    return model


def _validate_cache_inputs(
    cache: object, mask: object, studies: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Hard-gate the exact dense cache API consumed by this runtime."""

    if not isinstance(cache, np.ndarray):
        raise Public0033InferenceError("cache must be a numpy ndarray or memmap")
    if cache.dtype != np.uint8:
        raise Public0033InferenceError(f"cache dtype must be uint8, got {cache.dtype}")
    if cache.ndim != 5 or cache.shape[1:] != (
        N_SLOTS,
        SLICES_PER_SLOT,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise Public0033InferenceError(
            "cache shape must be (N, 6, 12, 336, 336), "
            f"got {tuple(cache.shape)}"
        )
    if not isinstance(mask, np.ndarray) or mask.dtype != np.float32:
        raise Public0033InferenceError("mask must be a float32 numpy ndarray")
    if mask.shape != (cache.shape[0], N_SLOTS):
        raise Public0033InferenceError(
            f"mask shape mismatch: {tuple(mask.shape)} for cache N={cache.shape[0]}"
        )
    if not np.isfinite(mask).all() or np.any((mask < 0.0) | (mask > 1.0)):
        raise Public0033InferenceError("mask contains non-finite or out-of-range values")
    if np.any(mask.sum(axis=1) <= 0.0):
        raise Public0033InferenceError("whole-study mask failure; neutralization is forbidden")
    ordered = list(studies)
    if len(ordered) != cache.shape[0] or len(ordered) != mask.shape[0]:
        raise Public0033InferenceError("cache/mask/study count mismatch")
    if len(ordered) < 2:
        raise Public0033InferenceError("T4x2 route requires at least two studies")
    if any(not isinstance(uid, str) or not uid for uid in ordered):
        raise Public0033InferenceError("studies must be non-empty string UIDs")
    if len(set(ordered)) != len(ordered):
        raise Public0033InferenceError("studies contain duplicate UIDs")
    if ordered != sorted(ordered):
        raise Public0033InferenceError("studies must be sorted before cached inference")
    return cache, mask, ordered


def _balanced_contiguous_partitions(study_count: int) -> tuple[np.ndarray, np.ndarray]:
    if study_count < 2:
        raise Public0033InferenceError("cannot create two non-empty GPU partitions")
    full = np.arange(study_count, dtype=np.int64)
    left, right = np.array_split(full, 2)
    if not len(left) or not len(right):
        raise Public0033InferenceError("empty GPU partition")
    if len(left) - len(right) not in {0, 1}:
        raise Public0033InferenceError("unbalanced GPU partitions")
    if not np.array_equal(left, np.arange(0, len(left), dtype=np.int64)):
        raise Public0033InferenceError("first GPU partition is not contiguous")
    if not np.array_equal(right, np.arange(len(left), study_count, dtype=np.int64)):
        raise Public0033InferenceError("second GPU partition is not contiguous")
    return left, right


def _require_t4x2() -> list[str]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise Public0033InferenceError(
            f"exactly two CUDA devices are required, got {torch.cuda.device_count()}"
        )
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("T4" not in name.upper() for name in names):
        raise Public0033InferenceError(f"Tesla T4 x2 is required, got {names}")
    return names


def _worker_predict(
    *,
    root: Path,
    cache: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    device_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One isolated FP32 checkpoint replica on one contiguous GPU partition."""

    if indices.ndim != 1 or not len(indices):
        raise Public0033InferenceError("worker received an empty/invalid partition")
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    if torch.is_autocast_enabled():
        raise Public0033InferenceError("autocast is enabled; FP32 route is mandatory")
    started = time.monotonic()
    model: Fullfit0033Model | None = None
    outputs: list[np.ndarray] = []
    try:
        model = _load_fullfit_model(root, device)
        with torch.inference_mode():
            for offset in range(0, len(indices), BATCH_STUDIES):
                selected = indices[offset : offset + BATCH_STUDIES]
                mask_tensor = torch.from_numpy(
                    np.ascontiguousarray(mask[selected])
                ).to(device, non_blocking=False)
                total: torch.Tensor | None = None
                for start in WINDOW_STARTS:
                    batch = np.ascontiguousarray(
                        cache[selected, :, start : start + WINDOW_CHANNELS, :, :]
                    )
                    images = torch.from_numpy(batch).to(device, non_blocking=False)
                    # Intentionally no autocast context: strict FP32 is part of the route.
                    probability = torch.sigmoid(model(images, mask_tensor).float())
                    total = probability if total is None else total + probability
                if total is None:
                    raise Public0033InferenceError("overlap-10 produced no logits")
                values = (total / float(len(WINDOW_STARTS))).cpu().numpy()
                if values.shape != (len(selected), len(TARGETS)):
                    raise Public0033InferenceError(
                        f"worker prediction shape mismatch: {values.shape}"
                    )
                if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
                    raise Public0033InferenceError("non-finite/out-of-range worker prediction")
                outputs.append(values.astype(np.float32, copy=False))
        prediction = np.concatenate(outputs, axis=0)
        if prediction.shape != (len(indices), len(TARGETS)):
            raise Public0033InferenceError("partition concatenation changed prediction shape")
        return indices, prediction, {
            "device_index": device_index,
            "device_name": torch.cuda.get_device_name(device_index),
            "study_count": int(len(indices)),
            "index_start": int(indices[0]),
            "index_stop_exclusive": int(indices[-1]) + 1,
            "strict_load": True,
            "checkpoint_replicated": True,
            "runtime_seconds": time.monotonic() - started,
        }
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


def _validate_output_paths(output_paths: Mapping[str, object]) -> tuple[Path, Path, Path]:
    if set(map(str, output_paths)) != EXPECTED_OUTPUT_KEYS:
        raise Public0033InferenceError(
            f"output_paths keys must be {sorted(EXPECTED_OUTPUT_KEYS)}"
        )
    work_dir = Path(str(output_paths["work_dir"])).resolve()
    bag_path = Path(str(output_paths["bag_raw_csv"])).resolve()
    receipt_path = Path(str(output_paths["receipt_json"])).resolve()
    if bag_path.name != "public0033_bag_raw.csv":
        raise Public0033InferenceError("bag_raw_csv filename drift")
    if receipt_path.name != "public0033_cached_inference_receipt.json":
        raise Public0033InferenceError("receipt_json filename drift")
    if bag_path.parent != work_dir or receipt_path.parent != work_dir:
        raise Public0033InferenceError("outputs must be direct children of work_dir")
    work_dir.mkdir(parents=True, exist_ok=True)
    if bag_path.exists() or receipt_path.exists():
        raise Public0033InferenceError("refusing to overwrite an existing inference artifact")
    return bag_path, receipt_path, work_dir


def _atomic_write_csv(path: Path, studies: Sequence[str], values: np.ndarray) -> None:
    if values.shape != (len(studies), len(RAW_TARGETS)):
        raise Public0033InferenceError("raw CSV value shape mismatch")
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["StudyInstanceUID", *RAW_TARGETS])
            for uid, row in zip(studies, values, strict=True):
                writer.writerow([uid, *[repr(float(value)) for value in row]])
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_cached_inference(
    cache: np.ndarray,
    mask: np.ndarray,
    studies: Sequence[str],
    output_paths: Mapping[str, object],
) -> Mapping[str, Any]:
    """Run fixed fullfit0033 FP32 overlap-10 inference on a dense test cache.

    ``cache`` is exactly ``(N, 6, 12, 336, 336)`` uint8 and ``mask`` is
    ``(N, 6)`` float32.  No decoding, model fallback, prediction neutralisation,
    target substitution, or rank blend occurs here.
    """

    started = time.monotonic()
    bundle_root = Path(__file__).resolve().parent
    manifest = verify_bundle(bundle_root)
    cache_array, mask_array, ordered_studies = _validate_cache_inputs(cache, mask, studies)
    bag_path, receipt_path, _ = _validate_output_paths(output_paths)
    devices = _require_t4x2()
    partitions = _balanced_contiguous_partitions(len(ordered_studies))
    prediction = np.empty((len(ordered_studies), len(TARGETS)), dtype=np.float32)
    worker_receipts: list[dict[str, Any]] = []

    # Two checkpoint copies are intentional: each T4 owns an independent,
    # balanced contiguous partition and both devices run all ten windows.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="public0033-t4") as executor:
        futures = {
            executor.submit(
                _worker_predict,
                root=bundle_root,
                cache=cache_array,
                mask=mask_array,
                indices=partition,
                device_index=device_index,
            ): device_index
            for device_index, partition in enumerate(partitions)
        }
        for future in as_completed(futures):
            device_index = futures[future]
            try:
                indices, values, worker_receipt = future.result()
            except Exception as exc:
                raise Public0033InferenceError(
                    f"GPU worker {device_index} failed; no partial raw CSV is allowed"
                ) from exc
            prediction[indices] = values
            worker_receipts.append(worker_receipt)

    if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
        raise Public0033InferenceError("combined prediction is non-finite/out-of-range")
    raw_values = prediction[:, RAW_TARGET_INDICES]
    _atomic_write_csv(bag_path, ordered_studies, raw_values)
    receipt: dict[str, Any] = {
        "schema_version": "public0033_cached_inference_receipt_v1",
        "status": "passed",
        "bundle_schema_version": BUNDLE_SCHEMA,
        "bundle_manifest_sha256": sha256_file(bundle_root / "bundle_manifest.json"),
        "runtime_sha256": str(manifest["runtime"]["sha256"]),
        "checkpoint": {
            "sha256": str(manifest["checkpoint"]["sha256"]),
            "bytes": int(manifest["checkpoint"]["bytes"]),
            "state_tensor_count": EXPECTED_STATE_TENSORS,
            "strict_load": True,
        },
        "source_hashes": {
            "checkpoint_sidecar": str(manifest["artifacts"]["checkpoint_sidecar"]["sha256"]),
            "run_contract": str(manifest["artifacts"]["run_contract"]["sha256"]),
            "task_complete": str(manifest["artifacts"]["task_complete"]["sha256"]),
            "fullfit_config": str(manifest["artifacts"]["fullfit_config"]["sha256"]),
            "dinov2_config": str(manifest["artifacts"]["dinov2_config"]["sha256"]),
            "public0033_config": str(manifest["artifacts"]["public0033_config"]["sha256"]),
        },
        "precision": "fp32",
        "autocast": False,
        "devices": [
            {"device_index": index, "device_name": name}
            for index, name in enumerate(devices)
        ],
        "partitions": sorted(worker_receipts, key=lambda item: int(item["device_index"])),
        "batch_studies": BATCH_STUDIES,
        "window_starts": list(WINDOW_STARTS),
        "window_pooling": "overlap10_probability_mean",
        "study_count": len(ordered_studies),
        "study_uid_sha256": _sha256_bytes(
            "".join(f"{uid}\n" for uid in ordered_studies).encode("utf-8")
        ),
        "mask_sha256": _sha256_bytes(np.ascontiguousarray(mask_array).tobytes()),
        "cache": {
            "shape": list(cache_array.shape),
            "dtype": str(cache_array.dtype),
        },
        "raw_output": {
            "path": str(bag_path),
            "sha256": sha256_file(bag_path),
            "columns": ["StudyInstanceUID", *RAW_TARGETS],
            "min": {
                target: float(raw_values[:, index].min())
                for index, target in enumerate(RAW_TARGETS)
            },
            "max": {
                target: float(raw_values[:, index].max())
                for index, target in enumerate(RAW_TARGETS)
            },
        },
        "fallback": 0,
        "neutralized_predictions": 0,
        "runtime_seconds": time.monotonic() - started,
    }
    _atomic_write_json(receipt_path, receipt)
    return {
        "status": "passed",
        "bag_raw_csv": str(output_paths["bag_raw_csv"]),
        "receipt_json": str(output_paths["receipt_json"]),
        "study_count": len(ordered_studies),
        "fallback": 0,
    }


def run_cpu_one_study_preflight(bundle_root: str | Path) -> Mapping[str, Any]:
    """Builder-only CPU strict-load/forward smoke test; it is not Kaggle inference."""

    root = Path(bundle_root).resolve()
    manifest = verify_bundle(root)
    model = _load_fullfit_model(root, torch.device("cpu"))
    try:
        with torch.inference_mode():
            images = torch.zeros(
                (1, N_SLOTS, WINDOW_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8
            )
            mask = torch.ones((1, N_SLOTS), dtype=torch.float32)
            logits = model(images, mask)
            probability = torch.sigmoid(logits.float()).cpu().numpy()
        if probability.shape != (1, len(TARGETS)) or not np.isfinite(probability).all():
            raise Public0033InferenceError("CPU strict-load/forward preflight failed")
        return {
            "status": "passed",
            "strict_load": True,
            "state_tensor_count": EXPECTED_STATE_TENSORS,
            "checkpoint_sha256": str(manifest["checkpoint"]["sha256"]),
            "prediction_shape": list(probability.shape),
            "prediction_min": float(probability.min()),
            "prediction_max": float(probability.max()),
        }
    finally:
        del model


def _synthetic_manifest() -> dict[str, Any]:
    digest = "0" * 64
    files = {
        name: {"sha256": digest, "bytes": 1}
        for name in (
            RUNTIME_FILENAME,
            CHECKPOINT_FILENAME,
            SIDECAR_FILENAME,
            RUN_CONTRACT_FILENAME,
            TASK_COMPLETE_FILENAME,
            FULLFIT_CONFIG_FILENAME,
            DINO_CONFIG_FILENAME,
            PUBLIC_CONFIG_FILENAME,
            "dataset-metadata.json",
        )
    }
    return {
        "schema_version": BUNDLE_SCHEMA,
        "status": BUNDLE_STATUS,
        "bundle_id": BUNDLE_ID,
        "target_order": list(TARGETS),
        "raw_output_targets": list(RAW_TARGETS),
        "precision": "fp32",
        "required_gpu_count": 2,
        "required_gpu_family": "Tesla T4",
        "batch_studies": BATCH_STUDIES,
        "window_starts": list(WINDOW_STARTS),
        "state_tensor_count": EXPECTED_STATE_TENSORS,
        "fixed_public_route": {
            "exact_parent": "exact_0.936",
            "targets": list(RAW_TARGETS),
            "parent_rank_weight": 0.90,
            "bag_rank_weight": 0.10,
            "other_targets_byte_preserved": True,
            "submission_count_max": 1,
        },
        "runtime": {
            "bundle_relative_path": RUNTIME_FILENAME,
            "entrypoint": ENTRYPOINT,
            "sha256": digest,
        },
        "checkpoint": {
            "bundle_relative_path": CHECKPOINT_FILENAME,
            "sha256": digest,
            "bytes": 1,
            "state_tensor_count": EXPECTED_STATE_TENSORS,
        },
        "artifacts": {
            "checkpoint_sidecar": {"bundle_relative_path": SIDECAR_FILENAME, "sha256": digest},
            "run_contract": {"bundle_relative_path": RUN_CONTRACT_FILENAME, "sha256": digest},
            "task_complete": {"bundle_relative_path": TASK_COMPLETE_FILENAME, "sha256": digest},
            "fullfit_config": {"bundle_relative_path": FULLFIT_CONFIG_FILENAME, "sha256": digest},
            "dinov2_config": {"bundle_relative_path": DINO_CONFIG_FILENAME, "sha256": digest},
            "public0033_config": {
                "bundle_relative_path": PUBLIC_CONFIG_FILENAME,
                "sha256": digest,
            },
        },
        "files": files,
        "forbidden_inputs": {
            "gold_label_values_read": False,
            "gold_metrics_read": False,
            "public_lb_read": False,
            "reports_read": False,
            "portfolio0034_input": False,
        },
    }


def self_test() -> Mapping[str, Any]:
    """GPU-free synthetic tests for manifest, split, cache input, and CSV schema."""

    manifest = _synthetic_manifest()
    _validate_manifest_schema(manifest)
    bad_manifest = dict(manifest)
    bad_manifest["batch_studies"] = 16
    try:
        _validate_manifest_schema(bad_manifest)
    except Public0033InferenceError:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("manifest batch drift was accepted")
    left, right = _balanced_contiguous_partitions(5)
    if left.tolist() != [0, 1, 2] or right.tolist() != [3, 4]:
        raise AssertionError("balanced contiguous split changed")
    cache = np.zeros((2, N_SLOTS, SLICES_PER_SLOT, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask = np.ones((2, N_SLOTS), dtype=np.float32)
    _validate_cache_inputs(cache, mask, ["a", "b"])
    try:
        _validate_cache_inputs(cache[:, :, :11], mask, ["a", "b"])
    except Public0033InferenceError:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("invalid cache shape was accepted")
    with tempfile.TemporaryDirectory() as temporary_dir:
        target = Path(temporary_dir) / "public0033_bag_raw.csv"
        _atomic_write_csv(
            target,
            ["a", "b"],
            np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        )
        rows = list(csv.reader(target.open("r", encoding="utf-8", newline="")))
        if rows[0] != ["StudyInstanceUID", *RAW_TARGETS] or len(rows) != 3:
            raise AssertionError("raw CSV schema changed")
    return {
        "status": "passed",
        "gpu_used": False,
        "checks": ["manifest", "partition", "cache_shape", "raw_csv_schema"],
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        raise SystemExit("only --self-test is supported as a standalone command")
    print(json.dumps(self_test(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    _main()
