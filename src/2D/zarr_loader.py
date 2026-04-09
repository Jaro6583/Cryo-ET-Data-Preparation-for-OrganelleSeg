"""
Zarr Dataset and DataLoader for OrganelleSeg experiments.

Loads FIB-SEM volumes directly from OME-Zarr stores, with no dependency
on ``cellmap_data`` or any CellMap library.  Full control over resolution
matching, cropping, label encoding, and partial-annotation masking.

Features
--------
- Reads per-class binary Zarr arrays from the CellMap directory layout.
- Handles multi-resolution pyramids: finds the label level that matches
  EM s0 resolution, computes voxel offsets from ``translation`` metadata.
- Converts per-class binary masks → single integer label volume on the fly.
- Extracts random 2D slices or 3D patches from the aligned volumes.
- Supports **partial annotation** via NaN masking: unannotated classes
  get NaN so the loss function can ignore them.
- Lazy loading with LRU caching — only the needed chunks are read.
- Returns batches as ``{'input': Tensor, 'output': Tensor}`` — drop-in
  replacement for the previous NIfTI loader.

Zarr directory layout (CellMap convention)
------------------------------------------
::

    {data_root}/{dataset}/{dataset}.zarr/
        recon-1/
            em/fibsem-uint8/          ← OME-zarr pyramid (s0, s1, …)
            labels/groundtruth/
                {crop}/
                    {class_name}/     ← OME-zarr pyramid per class (binary)

Typical usage
-------------
>>> from organelleseg.data.zarr_loader import get_zarr_dataloader
>>> train_loader, val_loader, vis_loader = get_zarr_dataloader(
...     datalist_path="data/zarr/datalist_35cls.json",
...     classes=["nuc", "mito_mem", "er_mem", "pm", "golgi_mem"],
...     batch_size=24,
...     input_shape=(1, 256, 256),
...     iterations_per_epoch=100,
... )
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, DataLoader


# =====================================================================
# OME-Zarr metadata helpers
# =====================================================================

def _is_valid_zarr_path(path: str) -> bool:
    """Check if a path is a valid zarr array or group.

    A valid zarr store must contain a ``.zarray`` (for arrays) or
    ``.zgroup`` (for groups) metadata file.  Directories that lack
    both are corrupted — e.g. they may contain only data chunks
    without the required metadata.
    """
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, ".zarray")) or os.path.isfile(
        os.path.join(path, ".zgroup")
    )


def _read_multiscale_info(zarr_group_path: str) -> list[dict]:
    """Read OME-zarr multiscale metadata and return per-level info.

    Returns list of dicts with keys:
      ``path``, ``scale``, ``translation``, ``shape``
    """
    if not _is_valid_zarr_path(zarr_group_path):
        return []
    try:
        grp = zarr.open(zarr_group_path, mode="r")
    except (zarr.errors.PathNotFoundError, Exception):
        return []
    ms = grp.attrs.get("multiscales", [])
    if not isinstance(ms, list) or len(ms) == 0:
        return []

    levels = []
    for ds in ms[0].get("datasets", []):
        info = {"path": ds["path"], "scale": None, "translation": None, "shape": None}
        for t in ds.get("coordinateTransformations", []):
            if t["type"] == "scale":
                info["scale"] = t["scale"]
            elif t["type"] == "translation":
                info["translation"] = t["translation"]
        arr_path = os.path.join(zarr_group_path, ds["path"])
        if _is_valid_zarr_path(arr_path):
            try:
                info["shape"] = tuple(zarr.open(arr_path, mode="r").shape)
            except Exception:
                pass
        levels.append(info)
    return levels


def _find_matching_label_level(
    label_levels: list[dict], em_scale: list[float], tol: float = 1e-3
) -> dict | None:
    """Find the label pyramid level whose voxel scale matches the EM."""
    for lvl in label_levels:
        if lvl["scale"] is None:
            continue
        if all(abs(lvl["scale"][i] - em_scale[i]) < tol for i in range(len(em_scale))):
            return lvl
    return None


# =====================================================================
# Per-crop metadata resolver (cached)
# =====================================================================

class CropMetadata:
    """Pre-resolved metadata for one annotated crop.

    Attributes
    ----------
    dataset : str
        Dataset name (e.g. ``jrc_hela-2``).
    crop : str
        Crop name (e.g. ``crop1``).
    em_zarr_path : str
        Path to the EM array at matching resolution (e.g. ``.../fibsem-uint8/s0``).
    em_voxel_offset : list[int] | None
        ``[z0, y0, x0]`` voxel offset into the EM volume.
    output_shape : tuple[int, int, int]
        ``(D, H, W)`` shape of the aligned crop.
    label_paths : dict[str, str]
        Maps class name → path to the zarr array at the matched level.
    annotated_classes : set[str]
        Class names that have annotations in this crop.
    em_scale : list[float]
        Physical voxel size of the EM at s0.
    """

    def __init__(
        self,
        dataset: str,
        crop: str,
        em_zarr_path: str,
        em_voxel_offset: list[int] | None,
        output_shape: tuple[int, int, int],
        label_paths: dict[str, str],
        annotated_classes: set[str],
        em_scale: list[float],
        z_range: tuple[int, int] | None = None,
    ):
        self.dataset = dataset
        self.crop = crop
        self.em_zarr_path = em_zarr_path
        self.em_voxel_offset = em_voxel_offset
        self.output_shape = output_shape
        self.label_paths = label_paths
        self.annotated_classes = annotated_classes
        self.em_scale = em_scale
        self.z_range = z_range  # (z_start, z_end) inclusive, or None for full volume

    @property
    def crop_id(self) -> str:
        suffix = ""
        if self.z_range is not None:
            suffix = f"_z{self.z_range[0]}-{self.z_range[1]}"
        return f"{self.dataset}_{self.crop}{suffix}"


def _has_nonzero_data(zarr_path: str, n_probes: int = 20) -> bool:
    """Check if a zarr array contains any non-zero data.

    Reads a small number of evenly-spaced slices along the first axis
    to detect whether the array is an empty placeholder (all zeros)
    or genuinely annotated.  This avoids reading the entire volume.

    The CellMap zarr format creates directory structures for every class
    in every crop, even when the annotation is entirely empty.  Without
    this check, empty arrays would be treated as annotated (target = 0
    everywhere), which trains the model to suppress those classes.

    Parameters
    ----------
    zarr_path : str
        Path to a zarr array (e.g. ``.../nhchrom/s0``).
    n_probes : int
        Number of slices to sample.  20 probes catches classes occupying
        as little as ~0.01% of the volume while keeping I/O acceptable.
        (Previously 5, which could miss extremely sparse annotations
        like eres_mem and perox_mem.)

    Returns
    -------
    bool
        True if at least one sampled slice contains a non-zero value.
    """
    try:
        arr = zarr.open(zarr_path, mode="r")
        D = arr.shape[0]
        if D == 0:
            return False
        # Evenly spaced probe indices
        step = max(1, D // n_probes)
        indices = list(range(0, D, step))
        if indices[-1] != D - 1:
            indices.append(D - 1)
        for z in indices:
            sl = np.array(arr[z])
            if sl.any():
                return True
        return False
    except Exception:
        return False


def resolve_crop_metadata(
    data_root: str,
    dataset: str,
    crop: str,
    classes: list[str],
    filter_empty_labels: bool = True,
) -> CropMetadata | None:
    """Resolve Zarr paths and resolution matching for one crop.

    Parameters
    ----------
    filter_empty_labels : bool
        If True (default), exclude all-zero annotation arrays via
        ``_has_nonzero_data``.  If False, include them as explicit
        negative supervision (target = 0 instead of NaN).

        v2/R1 (Dice 0.495) ran WITHOUT this filter.  v3 restored it,
        but the resulting lack of negative supervision caused 23/35
        classes to get stuck at Dice 0.  Set to False for new runs.

    Returns ``None`` if essential data is missing.
    """
    zarr_base = os.path.join(data_root, dataset, f"{dataset}.zarr")
    em_group = os.path.join(zarr_base, "recon-1", "em", "fibsem-uint8")
    gt_base = os.path.join(zarr_base, "recon-1", "labels", "groundtruth", crop)

    if not os.path.isdir(em_group) or not os.path.isdir(gt_base):
        return None

    # EM metadata
    em_levels = _read_multiscale_info(em_group)
    if not em_levels or em_levels[0]["scale"] is None:
        return None
    em_scale = em_levels[0]["scale"]

    # Find a reference label class to get spatial metadata
    ref_label_levels = None
    for cls_name in classes:
        cls_dir = os.path.join(gt_base, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        levels = _read_multiscale_info(cls_dir)
        if levels and levels[0]["shape"] is not None:
            ref_label_levels = levels
            break

    if ref_label_levels is None:
        return None

    # Find label level that matches EM s0 resolution
    matched = _find_matching_label_level(ref_label_levels, em_scale)
    if matched is not None:
        use_level = matched
    else:
        # Fallback: use label s0 (may need resampling at load time)
        use_level = ref_label_levels[0]

    output_shape = use_level["shape"]
    level_path = use_level["path"]  # e.g. "s1"

    # Compute EM voxel offset from label translation
    translation = use_level.get("translation")
    em_voxel_offset = None
    if translation is not None:
        em_voxel_offset = [int(round(translation[i] / em_scale[i])) for i in range(3)]

    # Resolve per-class label paths
    #
    # When filter_empty_labels=True (legacy default), we exclude all-zero
    # annotation arrays via _has_nonzero_data.  The CellMap zarr format
    # creates directories for every class in every crop, even when the
    # annotation is entirely zeros.
    #
    # When filter_empty_labels=False (recommended for new runs), all-zero
    # arrays become explicit negative supervision (target=0).  This was
    # the v2/R1 behavior that achieved Dice=0.495.  Without negative
    # supervision, models never learn to suppress false positives.
    #
    # The old comment blamed R3 collapse on removing the filter, but R3
    # also changed EMA/lr_min/epochs simultaneously.  R1 proves that
    # negative supervision works when other hyperparameters are correct.
    label_paths: dict[str, str] = {}
    annotated: set[str] = set()
    for cls_name in classes:
        cls_dir = os.path.join(gt_base, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        matched_path = os.path.join(cls_dir, level_path)
        if _is_valid_zarr_path(matched_path):
            if not filter_empty_labels or _has_nonzero_data(matched_path):
                label_paths[cls_name] = matched_path
                annotated.add(cls_name)
        else:
            s0_path = os.path.join(cls_dir, "s0")
            if _is_valid_zarr_path(s0_path):
                if not filter_empty_labels or _has_nonzero_data(s0_path):
                    label_paths[cls_name] = s0_path
                    annotated.add(cls_name)

    em_arr_path = os.path.join(em_group, "s0")

    return CropMetadata(
        dataset=dataset,
        crop=crop,
        em_zarr_path=em_arr_path,
        em_voxel_offset=em_voxel_offset,
        output_shape=output_shape,
        label_paths=label_paths,
        annotated_classes=annotated,
        em_scale=em_scale,
    )


# =====================================================================
# Zarr array cache  (process-wide, thread-safe reads)
# =====================================================================

_zarr_cache: dict[str, zarr.Array] = {}


def _load_em_context_slices(
    em_arr: zarr.Array,
    z: int,
    safe_D: int,
    safe_H: int,
    safe_W: int,
    oz: int,
    oy: int,
    ox: int,
    context_slices: int = 1,
) -> np.ndarray:
    """Load one or more adjacent z-slices for 2D or 2.5D input.

    For ``context_slices=1`` (standard 2D), returns a single slice.
    For ``context_slices=3`` (2.5D), returns slices ``[z-1, z, z+1]``
    stacked along axis 0.  Boundary slices are clamped (repeated).

    Parameters
    ----------
    em_arr : zarr.Array
        The EM volume array.
    z : int
        Center z-slice index (relative to the crop, before adding ``oz``).
    safe_D : int
        Maximum valid z-index (exclusive) within the crop region.
    safe_H, safe_W : int
        Spatial dimensions to read.
    oz, oy, ox : int
        Voxel offsets into the EM array.
    context_slices : int
        Number of z-slices to load.  Must be odd.  1 = standard 2D,
        3 = 2.5D with adjacent slices as separate channels.

    Returns
    -------
    np.ndarray
        Shape ``(context_slices, H, W)`` float32 image data.
    """
    half = context_slices // 2
    slices = []
    for dz in range(-half, half + 1):
        zz = max(0, min(z + dz, safe_D - 1))
        sl = np.array(
            em_arr[oz + zz, oy : oy + safe_H, ox : ox + safe_W],
            dtype=np.float32,
        )
        slices.append(sl)
    return np.stack(slices, axis=0)  # (context_slices, H, W)


def _open_zarr(path: str) -> zarr.Array:
    """Open a zarr array (cached).

    Raises ``ValueError`` for empty / falsy paths and gives a clear
    diagnostic for corrupted stores (missing ``.zarray`` / ``.zgroup``).
    """
    if not path:
        raise ValueError(f"_open_zarr called with empty/falsy path: {path!r}")
    if path not in _zarr_cache:
        if not _is_valid_zarr_path(path):
            raise FileNotFoundError(
                f"Corrupted zarr store at '{path}': directory exists but "
                f"is missing .zarray / .zgroup metadata"
            )
        _zarr_cache[path] = zarr.open(path, mode="r")
    return _zarr_cache[path]


# =====================================================================
# Global voxel counts per class (from CLASS_REFERENCE.md analysis)
# Used for class-aware sampling weights
# =====================================================================

GLOBAL_VOXEL_COUNTS: dict[str, int] = {
    "ecs":        3_750_644_301,
    "pm":           365_966_884,
    "mito_mem":     740_946_964,
    "mito_lum":     910_826_941,
    "mito_ribo":      1_643_455,
    "golgi_mem":     63_358_310,
    "golgi_lum":     90_271_273,
    "ves_mem":       20_218_267,
    "ves_lum":       11_410_160,
    "endo_mem":      97_744_082,
    "endo_lum":     203_076_684,
    "er_mem":       523_064_850,
    "er_lum":       654_688_548,
    "nuc":        3_533_146_803,
    "lyso_mem":      20_217_077,
    "lyso_lum":      64_241_359,
    "ld_mem":        14_896_191,
    "ld_lum":       120_758_049,
    "eres_mem":       5_456_843,
    "eres_lum":       5_252_100,
    "ne_mem":        56_357_675,
    "ne_lum":        50_414_607,
    "np_out":         4_086_527,
    "np_in":          2_798_532,
    "hchrom":       302_119_910,
    "echrom":         5_585_955,
    "nucpl":        604_553_039,
    "mt_out":        38_248_677,
    "cyto":       7_557_206_558,
    "mt_in":         17_017_019,
    "perox_mem":      5_857_060,
    "perox_lum":     16_430_634,
    "nhchrom":       17_148_855,
    "nechrom":           65_637,
    "nucleo":        13_825_619,
    # Composite classes (sum of their atomics)
    "mito":         1_653_417_360,  # mito_mem + mito_lum + mito_ribo
    "er":           1_177_753_398,  # er_mem + er_lum
    "golgi":          153_629_583,  # golgi_mem + golgi_lum
    "ves":             31_628_427,  # ves_mem + ves_lum
    "endo":           300_820_766,  # endo_mem + endo_lum
    "ne":             106_772_282,  # ne_mem + ne_lum
    "ld":             135_654_240,  # ld_mem + ld_lum
    "mt":              55_265_696,  # mt_out + mt_in
    "perox":           22_287_694,  # perox_mem + perox_lum
    "vacuole":         31_628_427,  # alias for ves
}


def _compute_crop_weights(
    crop_metas: list[CropMetadata],
    classes: list[str],
    class_weight_ratio: float = 0.7,
) -> list[float]:
    """Compute blended sampling weights for each crop.

    Blends class-aware weights with uniform weights to ensure every crop
    gets a minimum sampling probability while still up-weighting rare classes.

    class_aware_weight(crop) = sum over annotated classes c: 1 / sqrt(global_voxel_count(c))
    final_weight = class_weight_ratio × class_aware + (1 - class_weight_ratio) × uniform

    Default ratio 0.7 gives 70% class-aware, 30% uniform — enough to boost
    rare classes while keeping every crop sampled at least ~1× per epoch.
    """
    n = len(crop_metas)
    if n == 0:
        return []

    # Class-aware component
    raw_weights = []
    for meta in crop_metas:
        w = 0.0
        for cls_name in classes:
            if cls_name in meta.label_paths:
                global_count = GLOBAL_VOXEL_COUNTS.get(cls_name, 1e9)
                w += 1.0 / math.sqrt(max(global_count, 1.0))
        raw_weights.append(max(w, 1e-12))
    total_raw = sum(raw_weights)
    class_weights = [w / total_raw for w in raw_weights]

    # Uniform component
    uniform_weight = 1.0 / n

    # Blend
    alpha = class_weight_ratio
    blended = [
        alpha * cw + (1.0 - alpha) * uniform_weight
        for cw in class_weights
    ]
    # Re-normalise
    total_blended = sum(blended)
    return [w / total_blended for w in blended]


# =====================================================================
# Class-aware augmentation helpers
# =====================================================================

def _elastic_deform_2d(
    img: np.ndarray, lbl: np.ndarray, alpha: float = 80.0, sigma: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random elastic deformation to a 2D image + label pair.

    Uses Gaussian-smoothed random displacement fields.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    h, w = img.shape[-2:]
    dx = gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma) * alpha
    dy = gaussian_filter(np.random.randn(h, w).astype(np.float32), sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)]

    # Deform image (bilinear)
    if img.ndim == 2:
        img_out = map_coordinates(img, coords, order=1, mode="reflect").astype(np.float32)
    else:
        img_out = np.stack([
            map_coordinates(img[c], coords, order=1, mode="reflect").astype(np.float32)
            for c in range(img.shape[0])
        ])

    # Deform each label channel (nearest-neighbor to keep binary)
    if lbl.ndim == 2:
        lbl_out = map_coordinates(lbl, coords, order=0, mode="constant", cval=0).astype(np.float32)
    else:
        lbl_out = np.stack([
            map_coordinates(lbl[c], coords, order=0, mode="constant", cval=0).astype(np.float32)
            for c in range(lbl.shape[0])
        ])
    return img_out, lbl_out


def _intensity_augment(img_t: torch.Tensor) -> torch.Tensor:
    """Random brightness, contrast, and Gaussian noise augmentation."""
    # Random brightness shift [-0.1, +0.1]
    if random.random() > 0.5:
        brightness = random.uniform(-0.1, 0.1)
        img_t = img_t + brightness

    # Random contrast scaling [0.8, 1.2]
    if random.random() > 0.5:
        contrast = random.uniform(0.8, 1.2)
        mean_val = img_t.mean()
        img_t = (img_t - mean_val) * contrast + mean_val

    # Additive Gaussian noise (small)
    if random.random() > 0.5:
        noise = torch.randn_like(img_t) * random.uniform(0.01, 0.05)
        img_t = img_t + noise

    return img_t.clamp(0.0, 1.0)


# =====================================================================
# 2D Slice Dataset (with class-aware sampling & augmentation)
# =====================================================================

class ZarrSliceDataset(Dataset):
    """
    PyTorch Dataset that yields random 2D slices from Zarr EM + label volumes.

    Features class-aware crop sampling, Z-slice diversity tracking, and
    class-aware augmentation for rare classes.

    Each sample is ``{'input': Tensor, 'output': Tensor}`` where:

    - ``input``  – ``(C, H, W)`` float32 image normalised to [0, 1].
      ``C=1`` for standard 2D, ``C=context_slices`` for 2.5D.
    - ``output`` – ``(N, H, W)`` float32 multi-channel label.
      Channels for unannotated classes are filled with ``NaN``.

    Parameters
    ----------
    crop_metas : list[CropMetadata]
        Pre-resolved crop metadata objects.
    classes : list[str]
        Ordered class names; determines the channel order.
    crop_size : tuple[int, int] | None
        ``(H, W)`` spatial crop size.  ``None`` = use full slice.
    iterations : int
        Virtual epoch length (total samples per epoch).
    augment : bool
        Enable random flips / 90° rotations.
    class_aware_sampling : bool
        Weight crop selection by inverse class rarity (default True).
    class_aware_augmentation : bool
        Apply stronger augmentation to samples from rare-class crops (default True).
    rare_class_extra_aug_prob : float
        Probability of applying extra augmentation (elastic + intensity) on
        samples containing rare classes (default 0.7).
    context_slices : int
        Number of adjacent z-slices to stack as input channels (default 1).
        Use 3 for 2.5D input (z-1, z, z+1).  Must be odd.
    z_augmentation : bool
        Enable z-axis augmentations for 2.5D mode (default False).
        When True and ``context_slices > 1``, applies two augmentations:

        1. **Z-jitter ±1**: Before loading context slices, shift the center
           z-index by {-1, 0, +1} uniformly at random.  This perturbs which
           physical slice the network sees as "center" and its neighbours,
           providing inter-slice alignment regularisation (Jin et al. 2025).
        2. **Z-flip** (50% probability): Reverse the channel order of the
           loaded context slices, e.g. ``[z-1, z, z+1] → [z+1, z, z-1]``.
           This breaks the assumption that channels follow a fixed
           inferior→superior ordering and has been shown to improve
           generalisation in 2.5D segmentation (Avesta et al. 2023).

        Both augmentations are applied ONLY during training (``augment=True``)
        and ONLY when ``context_slices > 1``.  Labels are not affected since
        they correspond to the center z-slice which remains valid after a
        ±1 jitter (adjacent slices share nearly identical annotations in
        isotropic EM data).
    """

    # Classes with global voxel fraction < 0.1% are considered "rare"
    RARE_THRESHOLD = 0.001

    def __init__(
        self,
        crop_metas: list[CropMetadata],
        classes: list[str],
        crop_size: tuple[int, int] | None = (256, 256),
        iterations: int = 1000,
        augment: bool = False,
        class_aware_sampling: bool = True,
        class_aware_augmentation: bool = True,
        rare_class_extra_aug_prob: float = 0.7,
        context_slices: int = 1,
        z_augmentation: bool = False,
        composite_rules: dict[int, list[int]] | None = None,
    ):
        super().__init__()
        self.crop_metas = crop_metas
        self.classes = classes
        self.crop_size = crop_size
        self.iterations = iterations
        self.augment = augment
        self.class_aware_augmentation = class_aware_augmentation
        self.rare_class_extra_aug_prob = rare_class_extra_aug_prob
        self.context_slices = context_slices
        self.z_augmentation = z_augmentation and context_slices > 1
        self.composite_rules = composite_rules or {}

        # --- Layer 1: Class-aware crop sampling weights ---
        if class_aware_sampling and len(crop_metas) > 0:
            self.crop_weights = _compute_crop_weights(crop_metas, classes)
            # Log the top-5 and bottom-5 weights for visibility
            indexed = sorted(enumerate(self.crop_weights), key=lambda x: -x[1])
            print("  [Sampling] Class-aware crop weights enabled")
            print("    Top-5 weighted crops:")
            for i, (ci, w) in enumerate(indexed[:5]):
                m = crop_metas[ci]
                print(f"      {i+1}. {m.dataset}/{m.crop} "
                      f"(weight={w:.6f}, classes={len(m.annotated_classes)})")
            print("    Bottom-5 weighted crops:")
            for i, (ci, w) in enumerate(indexed[-5:]):
                m = crop_metas[ci]
                print(f"      {len(indexed)-4+i}. {m.dataset}/{m.crop} "
                      f"(weight={w:.6f}, classes={len(m.annotated_classes)})")
        else:
            self.crop_weights = None

        # --- Layer 2: Z-slice diversity tracking ---
        # Track which (crop_index, z) pairs have been seen this epoch
        # so we maximise coverage before repeating slices
        self._crop_z_unseen: dict[int, list[int]] = {}
        self._init_z_tracking()

        # --- Pre-compute which crops contain rare classes ---
        total_voxels = sum(GLOBAL_VOXEL_COUNTS.values())
        self._rare_classes = {
            cls for cls in classes
            if GLOBAL_VOXEL_COUNTS.get(cls, 0) / total_voxels < self.RARE_THRESHOLD
        }
        self._crop_has_rare: dict[int, bool] = {}
        for ci, meta in enumerate(crop_metas):
            self._crop_has_rare[ci] = bool(
                meta.annotated_classes & self._rare_classes
            )
        n_rare_crops = sum(self._crop_has_rare.values())
        if self._rare_classes:
            print(f"  [Sampling] {len(self._rare_classes)} rare classes identified "
                  f"(<{self.RARE_THRESHOLD*100:.1f}% of total voxels)")
            print(f"  [Sampling] {n_rare_crops}/{len(crop_metas)} crops contain rare classes")

    def _init_z_tracking(self):
        """Reset Z-slice tracking — called at start of each virtual epoch."""
        self._crop_z_unseen = {}
        for ci, meta in enumerate(self.crop_metas):
            D = meta.output_shape[0]
            if meta.z_range is not None:
                z_start, z_end = meta.z_range
                zs = list(range(z_start, min(z_end + 1, D)))
            else:
                zs = list(range(D))
            random.shuffle(zs)
            self._crop_z_unseen[ci] = zs

    def _get_diverse_z(self, crop_idx: int, D: int) -> int:
        """Return a Z-slice for this crop, preferring unseen slices.

        Once all slices for a crop are exhausted, it resets the pool
        for that crop (full re-shuffle).  Respects ``z_range`` if set.
        """
        unseen = self._crop_z_unseen.get(crop_idx)
        if not unseen:
            # All slices seen for this crop — reshuffle
            meta = self.crop_metas[crop_idx]
            if meta.z_range is not None:
                z_start, z_end = meta.z_range
                zs = list(range(z_start, min(z_end + 1, D)))
            else:
                zs = list(range(D))
            random.shuffle(zs)
            self._crop_z_unseen[crop_idx] = zs
            unseen = self._crop_z_unseen[crop_idx]
        return unseen.pop()

    def __len__(self) -> int:
        return self.iterations

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # --- Layer 1: Class-aware crop selection ---
        if self.crop_weights is not None:
            crop_idx = random.choices(
                range(len(self.crop_metas)),
                weights=self.crop_weights,
                k=1,
            )[0]
        else:
            crop_idx = random.randint(0, len(self.crop_metas) - 1)

        meta = self.crop_metas[crop_idx]
        D, H, W = meta.output_shape

        # Open EM zarr
        em_arr = _open_zarr(meta.em_zarr_path)

        # --- Layer 2: Z-slice diversity ---
        # Clamp D to valid EM range so we never index out of bounds
        if meta.em_voxel_offset is not None:
            oz, oy, ox = meta.em_voxel_offset
            em_D = em_arr.shape[0]
            safe_D = max(1, min(D, em_D - oz))
        else:
            oz, oy, ox = 0, 0, 0
            safe_D = min(D, em_arr.shape[0])

        z = self._get_diverse_z(crop_idx, safe_D)
        z = min(z, safe_D - 1)  # extra guard against pre-populated Z values

        # ── Z-augmentation (2.5D only, training only) ──
        # Z-jitter: shift center z by ±1 before loading context slices.
        # The label still comes from the original z — in isotropic EM data
        # adjacent slices have near-identical annotations, so the slight
        # mismatch acts as a form of label-smoothing regularisation.
        if self.z_augmentation and self.augment:
            z_shift = random.choice([-1, 0, 1])
            z = max(0, min(z + z_shift, safe_D - 1))

        # ── Determine spatial read region ──
        # When crop_size > annotation extent, expand the EM read region
        # to load real EM context beyond the annotation boundary.
        # Labels stay within annotation extent; border gets NaN (zero gradient).
        # This gives MAGF boundary head real membrane context in the border.
        if self.crop_size is not None:
            ch, cw = self.crop_size
        else:
            ch, cw = H, W

        em_full_H, em_full_W = em_arr.shape[1], em_arr.shape[2]
        expand_h = ch > H or cw > W  # need to read beyond annotation?

        if expand_h and meta.em_voxel_offset is not None:
            # Center the larger crop on the annotation region
            want_oy = oy + H // 2 - ch // 2
            want_ox = ox + W // 2 - cw // 2
            # Clamp to volume boundaries
            read_oy = max(0, min(want_oy, em_full_H - ch))
            read_ox = max(0, min(want_ox, em_full_W - cw))
            read_h = min(ch, em_full_H - read_oy)
            read_w = min(cw, em_full_W - read_ox)
            # Where the annotation sits within the expanded crop
            label_y0 = oy - read_oy
            label_x0 = ox - read_ox
        else:
            # Normal: read at annotation size
            read_oy, read_ox = oy, ox
            read_h = min(H, em_full_H - oy)
            read_w = min(W, em_full_W - oy if oy == read_oy else em_full_W - read_oy)
            read_h = min(H, em_full_H - read_oy)
            read_w = min(W, em_full_W - read_ox)
            label_y0, label_x0 = 0, 0

        # Read EM slice(s) — single slice for 2D, multiple for 2.5D
        img_slices = _load_em_context_slices(
            em_arr, z, safe_D, read_h, read_w, oz, read_oy, read_ox, self.context_slices
        )  # (C, read_h, read_w)

        # Z-flip: reverse the channel order with 50% probability.
        # Breaks the fixed inferior→superior z-ordering assumption.
        if self.z_augmentation and self.augment and random.random() < 0.5:
            img_slices = img_slices[::-1].copy()  # reverse along axis 0

        # Random spatial crop (only when EM read is larger than crop_size)
        if self.crop_size is not None:
            cur_h, cur_w = img_slices.shape[1], img_slices.shape[2]
            if cur_h > ch:
                y0 = random.randint(0, cur_h - ch)
                label_y0 = max(0, label_y0 - y0)
            else:
                y0 = 0
            if cur_w > cw:
                x0 = random.randint(0, cur_w - cw)
                label_x0 = max(0, label_x0 - x0)
            else:
                x0 = 0
            img_slices = img_slices[:, y0 : y0 + min(ch, cur_h), x0 : x0 + min(cw, cur_w)]

            # Pad if EM read is still smaller than crop (volume edge case)
            ph, pw = img_slices.shape[1], img_slices.shape[2]
            if ph < ch or pw < cw:
                img_slices = np.pad(
                    img_slices,
                    ((0, 0), (0, ch - ph), (0, cw - pw)),
                    mode="constant",
                    constant_values=0,
                )
        else:
            y0, x0 = 0, 0

        # Normalise to [0, 1]
        img_t = torch.from_numpy(img_slices).float() / 255.0
        img_t = img_t.clamp(0.0, 1.0)

        # Build multi-channel label
        n_classes = len(self.classes)
        out_h, out_w = img_slices.shape[1], img_slices.shape[2]
        label_t = torch.full((n_classes, out_h, out_w), float("nan"), dtype=torch.float32)

        for c_idx, cls_name in enumerate(self.classes):
            if cls_name not in meta.label_paths:
                continue  # leave as NaN (unannotated)
            lbl_arr = _open_zarr(meta.label_paths[cls_name])
            lbl_shape = lbl_arr.shape

            # Read the label slice — handle potential shape mismatch
            if lbl_shape == meta.output_shape:
                lbl_slice = np.array(lbl_arr[z, :H, :W], dtype=np.float32)
            else:
                # Need to read from a different resolution and nearest-neighbor resample
                ratio = [lbl_shape[i] / meta.output_shape[i] for i in range(3)]
                src_z = min(int(z * ratio[0]), lbl_shape[0] - 1)
                src_h = lbl_shape[1]
                src_w = lbl_shape[2]
                raw = np.array(lbl_arr[src_z], dtype=np.float32)
                if raw.shape != (H, W):
                    from scipy.ndimage import zoom as scipy_zoom
                    raw = scipy_zoom(raw, (H / raw.shape[0], W / raw.shape[1]), order=0)
                lbl_slice = raw

            # Place label within the (possibly expanded) crop.
            # When crop_size > annotation extent, the label occupies a sub-region
            # of the output; the border stays NaN (zero gradient, no false negatives).
            lbl_binary = (lbl_slice > 0).astype(np.float32)
            if expand_h and self.crop_size is not None:
                # Compute where this label slice sits in the output tensor
                # label_y0/label_x0 = offset of annotation within expanded crop
                # But we may have random-cropped (y0, x0) from the expanded EM
                # The label's position in the annotation is (0,0) to (H,W)
                # After random crop of the expanded region, label starts at label_y0
                ly0 = label_y0
                lx0 = label_x0
                # Clamp label placement to output bounds
                src_y0 = max(0, -ly0)
                src_x0 = max(0, -lx0)
                dst_y0 = max(0, ly0)
                dst_x0 = max(0, lx0)
                copy_h = min(H - src_y0, ch - dst_y0)
                copy_w = min(W - src_x0, cw - dst_x0)
                if copy_h > 0 and copy_w > 0:
                    label_t[c_idx, dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = \
                        torch.from_numpy(lbl_binary[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w])
            else:
                # Normal path: annotation >= crop, apply same spatial crop
                if self.crop_size is not None:
                    lbl_binary = lbl_binary[y0 : y0 + min(ch, H), x0 : x0 + min(cw, W)]
                    lh, lw = lbl_binary.shape
                    if lh < ch or lw < cw:
                        lbl_binary = np.pad(
                            lbl_binary,
                            ((0, ch - lh), (0, cw - lw)),
                            mode="constant",
                            constant_values=0,
                        )
                label_t[c_idx] = torch.from_numpy(lbl_binary)

        # --- Compute composite channels from atomics (for multi-modal training) ---
        if self.composite_rules:
            for comp_idx, atomic_indices in self.composite_rules.items():
                if comp_idx >= n_classes:
                    continue
                atomics = label_t[atomic_indices]
                # Only compute if at least one atomic is annotated (not all NaN)
                if not atomics.isnan().all():
                    label_t[comp_idx] = atomics.nan_to_num(0).amax(dim=0)

        # --- Standard augmentation (geometric) ---
        if self.augment:
            if random.random() > 0.5:
                img_t = img_t.flip(-1)
                label_t = label_t.flip(-1)
            if random.random() > 0.5:
                img_t = img_t.flip(-2)
                label_t = label_t.flip(-2)
            k = random.randint(0, 3)
            if k > 0:
                img_t = torch.rot90(img_t, k, [-2, -1])
                label_t = torch.rot90(label_t, k, [-2, -1])

        # --- Layer 3: Class-aware augmentation for rare classes ---
        if (self.augment and self.class_aware_augmentation
                and self._crop_has_rare.get(crop_idx, False)):
            if random.random() < self.rare_class_extra_aug_prob:
                # Elastic deformation — creates plausible shape variations
                # of rare structures (e.g. nuclear pore, peroxisome)
                img_np = img_t.numpy()
                lbl_np = label_t.numpy()
                img_np, lbl_np = _elastic_deform_2d(
                    img_np, lbl_np,
                    alpha=random.uniform(50.0, 120.0),
                    sigma=random.uniform(8.0, 15.0),
                )
                img_t = torch.from_numpy(img_np).float()
                label_t = torch.from_numpy(lbl_np).float()

                # Intensity augmentation — brightness, contrast, noise
                img_t = _intensity_augment(img_t)

        voxel_spacing = torch.tensor(float(np.mean(meta.em_scale[1:])))
        return {"input": img_t, "output": label_t, "voxel_spacing": voxel_spacing}


# =====================================================================
# 3D Volume Patch Dataset
# =====================================================================

class ZarrVolumeDataset(Dataset):
    """
    Dataset that yields random 3D patches from Zarr EM + label volumes.

    Features class-aware crop sampling and augmentation, matching
    the 2D ZarrSliceDataset strategy.

    Returns ``{'input': (1, D, H, W), 'output': (C, D, H, W)}``.
    """

    def __init__(
        self,
        crop_metas: list[CropMetadata],
        classes: list[str],
        patch_size: tuple[int, int, int] = (32, 256, 256),
        iterations: int = 1000,
        augment: bool = False,
        class_aware_sampling: bool = True,
    ):
        super().__init__()
        self.crop_metas = crop_metas
        self.classes = classes
        self.patch_size = patch_size
        self.iterations = iterations
        self.augment = augment

        # Class-aware sampling weights (same as 2D)
        if class_aware_sampling and len(crop_metas) > 0:
            self.crop_weights = _compute_crop_weights(crop_metas, classes)
        else:
            self.crop_weights = None

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        # Class-aware crop selection
        if self.crop_weights is not None:
            crop_idx = random.choices(
                range(len(self.crop_metas)),
                weights=self.crop_weights,
                k=1,
            )[0]
        else:
            crop_idx = random.randint(0, len(self.crop_metas) - 1)

        meta = self.crop_metas[crop_idx]
        D, H, W = meta.output_shape
        pd, ph, pw = self.patch_size

        em_arr = _open_zarr(meta.em_zarr_path)

        # Random patch origin — respect z_range if set
        if meta.z_range is not None:
            z_lo, z_hi = meta.z_range
            z_hi_clamped = min(z_hi, D - 1)
            z_range_len = z_hi_clamped - z_lo + 1
            z0 = z_lo + random.randint(0, max(z_range_len - pd, 0))
        else:
            z0 = random.randint(0, max(D - pd, 0))
        y0 = random.randint(0, max(H - ph, 0))
        x0 = random.randint(0, max(W - pw, 0))

        actual_d = min(pd, D)
        actual_h = min(ph, H)
        actual_w = min(pw, W)

        # Read EM patch
        if meta.em_voxel_offset is not None:
            oz, oy, ox = meta.em_voxel_offset
            img_patch = np.array(
                em_arr[
                    oz + z0 : oz + z0 + actual_d,
                    oy + y0 : oy + y0 + actual_h,
                    ox + x0 : ox + x0 + actual_w,
                ],
                dtype=np.float32,
            )
        else:
            img_patch = np.array(
                em_arr[z0 : z0 + actual_d, y0 : y0 + actual_h, x0 : x0 + actual_w],
                dtype=np.float32,
            )

        # Pad to exact patch size
        cd, ch, cw = img_patch.shape
        if cd < pd or ch < ph or cw < pw:
            img_patch = np.pad(
                img_patch,
                ((0, pd - cd), (0, ph - ch), (0, pw - cw)),
                mode="constant",
                constant_values=0,
            )

        img_t = torch.from_numpy(img_patch).unsqueeze(0).float() / 255.0
        img_t = img_t.clamp(0.0, 1.0)

        # Build multi-channel label patch
        n_cls = len(self.classes)
        label_t = torch.full((n_cls, pd, ph, pw), float("nan"), dtype=torch.float32)

        for c_idx, cls_name in enumerate(self.classes):
            if cls_name not in meta.label_paths:
                continue
            lbl_arr = _open_zarr(meta.label_paths[cls_name])
            lbl_shape = lbl_arr.shape

            if lbl_shape == meta.output_shape:
                lbl_patch = np.array(
                    lbl_arr[z0 : z0 + actual_d, y0 : y0 + actual_h, x0 : x0 + actual_w],
                    dtype=np.float32,
                )
            else:
                ratio = [lbl_shape[i] / meta.output_shape[i] for i in range(3)]
                src_z0 = int(z0 * ratio[0])
                src_z1 = min(int(np.ceil((z0 + actual_d) * ratio[0])), lbl_shape[0])
                src_y0 = int(y0 * ratio[1])
                src_y1 = min(int(np.ceil((y0 + actual_h) * ratio[1])), lbl_shape[1])
                src_x0 = int(x0 * ratio[2])
                src_x1 = min(int(np.ceil((x0 + actual_w) * ratio[2])), lbl_shape[2])
                raw = np.array(lbl_arr[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1], dtype=np.float32)
                if raw.shape != (actual_d, actual_h, actual_w):
                    from scipy.ndimage import zoom as scipy_zoom
                    zf = (actual_d / max(raw.shape[0], 1),
                           actual_h / max(raw.shape[1], 1),
                           actual_w / max(raw.shape[2], 1))
                    raw = scipy_zoom(raw, zf, order=0)
                lbl_patch = raw

            ld, lh, lw = lbl_patch.shape
            if ld < pd or lh < ph or lw < pw:
                lbl_patch = np.pad(
                    lbl_patch,
                    ((0, pd - ld), (0, ph - lh), (0, pw - lw)),
                    mode="constant",
                    constant_values=0,
                )

            label_t[c_idx] = torch.from_numpy((lbl_patch > 0).astype(np.float32))

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img_t = img_t.flip(-1)
                label_t = label_t.flip(-1)
            if random.random() > 0.5:
                img_t = img_t.flip(-2)
                label_t = label_t.flip(-2)

        voxel_spacing = torch.tensor(float(np.mean(meta.em_scale[1:])))
        return {"input": img_t, "output": label_t, "voxel_spacing": voxel_spacing}


# =====================================================================
# Deterministic Validation Dataset (2D slices)
# =====================================================================

class ZarrDeterministicValDataset(Dataset):
    """Deterministic 2D validation dataset that systematically covers all crops.

    Unlike the training ``ZarrSliceDataset`` which randomly samples crops and
    Z-slices, this dataset:

    1. **Covers every crop** — iterates through all validation crops in order.
    2. **Fixed Z-stride** — samples slices at regular intervals (default every
       ``z_stride`` slices) for reproducible, stable metrics.
    3. **Center crop** — uses deterministic center-crop instead of random spatial
       crop, eliminating run-to-run variance.
    4. **Multi-crop tiling** (optional) — when ``spatial_tiles > 1``, tiles the
       spatial dimensions with non-overlapping crops for better spatial coverage.

    This gives much more reliable per-class metrics, especially for rare classes
    that may only appear in a few crops or specific Z-ranges.

    Parameters
    ----------
    crop_metas : list[CropMetadata]
        Pre-resolved crop metadata objects.
    classes : list[str]
        Ordered class names.
    crop_size : tuple[int, int] | None
        ``(H, W)`` spatial crop size. ``None`` = use full slice.
    z_stride : int
        Sample every ``z_stride``-th slice from each crop (default 5).
    spatial_tiles : int
        Number of non-overlapping spatial tiles per slice (default 1 = center
        crop only). Use 4 for a 2×2 grid covering ~4× more area.
    context_slices : int
        Number of adjacent z-slices to stack as input channels (default 1).
        Use 3 for 2.5D input.
    """

    def __init__(
        self,
        crop_metas: list[CropMetadata],
        classes: list[str],
        crop_size: tuple[int, int] | None = (256, 256),
        z_stride: int = 5,
        spatial_tiles: int = 1,
        context_slices: int = 1,
        composite_rules: dict[int, list[int]] | None = None,
    ):
        super().__init__()
        self.crop_metas = crop_metas
        self.classes = classes
        self.crop_size = crop_size
        self.composite_rules = composite_rules or {}
        self.spatial_tiles = spatial_tiles
        self.context_slices = context_slices

        # Pre-compute the full sample list: (crop_idx, z, tile_idx)
        self._samples: list[tuple[int, int, int]] = []
        for ci, meta in enumerate(crop_metas):
            D = meta.output_shape[0]
            if meta.z_range is not None:
                z_start, z_end = meta.z_range
                z_end = min(z_end, D - 1)
            else:
                z_start, z_end = 0, D - 1

            for z in range(z_start, z_end + 1, z_stride):
                for tile_idx in range(spatial_tiles):
                    self._samples.append((ci, z, tile_idx))

        n_crops = len(crop_metas)
        n_total = len(self._samples)
        print(f"  [DeterministicVal] {n_crops} crops, z_stride={z_stride}, "
              f"tiles={spatial_tiles} → {n_total} samples")

    def __len__(self) -> int:
        return len(self._samples)

    def _get_tile_origin(
        self, H: int, W: int, ch: int, cw: int, tile_idx: int
    ) -> tuple[int, int]:
        """Return (y0, x0) for the given tile index.

        For ``spatial_tiles=1``: center crop.
        For ``spatial_tiles=4``: 2×2 grid (TL, TR, BL, BR).
        For ``spatial_tiles=9``: 3×3 grid.
        """
        if self.spatial_tiles == 1:
            y0 = max(0, (H - ch) // 2)
            x0 = max(0, (W - cw) // 2)
            return y0, x0

        # General n×n grid
        n = int(self.spatial_tiles ** 0.5)
        n = max(n, 1)
        row = tile_idx // n
        col = tile_idx % n
        # Evenly space tiles across the spatial dimensions
        y_step = max(1, (H - ch) // max(n - 1, 1)) if H > ch else 0
        x_step = max(1, (W - cw) // max(n - 1, 1)) if W > cw else 0
        y0 = min(row * y_step, max(0, H - ch))
        x0 = min(col * x_step, max(0, W - cw))
        return y0, x0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        crop_idx, z, tile_idx = self._samples[idx]
        meta = self.crop_metas[crop_idx]
        D, H, W = meta.output_shape

        em_arr = _open_zarr(meta.em_zarr_path)

        # Clamp z to safe range
        if meta.em_voxel_offset is not None:
            oz, oy, ox = meta.em_voxel_offset
            em_D = em_arr.shape[0]
            safe_D = max(1, min(D, em_D - oz))
        else:
            oz, oy, ox = 0, 0, 0
            safe_D = min(D, em_arr.shape[0])

        z = min(z, safe_D - 1)

        # ── Determine spatial read region (same logic as ZarrSliceDataset) ──
        if self.crop_size is not None:
            ch, cw = self.crop_size
        else:
            ch, cw = H, W

        em_full_H, em_full_W = em_arr.shape[1], em_arr.shape[2]
        expand_h = ch > H or cw > W

        if expand_h and meta.em_voxel_offset is not None:
            want_oy = oy + H // 2 - ch // 2
            want_ox = ox + W // 2 - cw // 2
            read_oy = max(0, min(want_oy, em_full_H - ch))
            read_ox = max(0, min(want_ox, em_full_W - cw))
            read_h = min(ch, em_full_H - read_oy)
            read_w = min(cw, em_full_W - read_ox)
            label_y0 = oy - read_oy
            label_x0 = ox - read_ox
        else:
            read_oy, read_ox = oy, ox
            read_h = min(H, em_full_H - read_oy)
            read_w = min(W, em_full_W - read_ox)
            label_y0, label_x0 = 0, 0

        # Read EM slice(s) — single slice for 2D, multiple for 2.5D
        img_slices = _load_em_context_slices(
            em_arr, z, safe_D, read_h, read_w, oz, read_oy, read_ox, self.context_slices
        )  # (C, read_h, read_w)

        # Deterministic spatial crop (center or tile)
        if self.crop_size is not None:
            cur_h, cur_w = img_slices.shape[1], img_slices.shape[2]
            y0, x0 = self._get_tile_origin(cur_h, cur_w, ch, cw, tile_idx)
            if y0 > 0:
                label_y0 = max(0, label_y0 - y0)
            if x0 > 0:
                label_x0 = max(0, label_x0 - x0)
            img_slices = img_slices[:, y0 : y0 + min(ch, cur_h), x0 : x0 + min(cw, cur_w)]

            # Pad if EM read is still smaller than crop (volume edge case)
            ph, pw = img_slices.shape[1], img_slices.shape[2]
            if ph < ch or pw < cw:
                img_slices = np.pad(
                    img_slices,
                    ((0, 0), (0, ch - ph), (0, cw - pw)),
                    mode="constant",
                    constant_values=0,
                )
        else:
            y0, x0 = 0, 0

        img_t = torch.from_numpy(img_slices).float() / 255.0
        img_t = img_t.clamp(0.0, 1.0)

        # Multi-channel labels
        n_classes = len(self.classes)
        out_h, out_w = img_slices.shape[1], img_slices.shape[2]
        label_t = torch.full(
            (n_classes, out_h, out_w), float("nan"), dtype=torch.float32
        )

        for c_idx, cls_name in enumerate(self.classes):
            if cls_name not in meta.label_paths:
                continue
            lbl_arr = _open_zarr(meta.label_paths[cls_name])
            lbl_shape = lbl_arr.shape

            if lbl_shape == meta.output_shape:
                lbl_slice = np.array(lbl_arr[z, :H, :W], dtype=np.float32)
            else:
                ratio = [lbl_shape[i] / meta.output_shape[i] for i in range(3)]
                src_z = min(int(z * ratio[0]), lbl_shape[0] - 1)
                raw = np.array(lbl_arr[src_z], dtype=np.float32)
                if raw.shape != (H, W):
                    from scipy.ndimage import zoom as scipy_zoom
                    raw = scipy_zoom(
                        raw, (H / raw.shape[0], W / raw.shape[1]), order=0
                    )
                lbl_slice = raw

            # Place label within the (possibly expanded) crop.
            lbl_binary = (lbl_slice > 0).astype(np.float32)
            if expand_h and self.crop_size is not None:
                ly0 = label_y0
                lx0 = label_x0
                src_y0 = max(0, -ly0)
                src_x0 = max(0, -lx0)
                dst_y0 = max(0, ly0)
                dst_x0 = max(0, lx0)
                copy_h = min(H - src_y0, ch - dst_y0)
                copy_w = min(W - src_x0, cw - dst_x0)
                if copy_h > 0 and copy_w > 0:
                    label_t[c_idx, dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = \
                        torch.from_numpy(lbl_binary[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w])
            else:
                if self.crop_size is not None:
                    lbl_binary = lbl_binary[y0 : y0 + min(ch, H), x0 : x0 + min(cw, W)]
                    lh, lw = lbl_binary.shape
                    if lh < ch or lw < cw:
                        lbl_binary = np.pad(
                            lbl_binary,
                            ((0, ch - lh), (0, cw - lw)),
                            mode="constant",
                            constant_values=0,
                        )
                label_t[c_idx] = torch.from_numpy(lbl_binary)

        # --- Compute composite channels from atomics ---
        if self.composite_rules:
            for comp_idx, atomic_indices in self.composite_rules.items():
                if comp_idx >= n_classes:
                    continue
                atomics = label_t[atomic_indices]
                if not atomics.isnan().all():
                    label_t[comp_idx] = atomics.nan_to_num(0).amax(dim=0)

        voxel_spacing = torch.tensor(float(np.mean(meta.em_scale[1:])))
        return {"input": img_t, "output": label_t, "voxel_spacing": voxel_spacing}


# =====================================================================
# Factory function  (main public API)
# =====================================================================

def get_zarr_dataloader(
    datalist_path: str | Path | None = None,
    classes: Sequence[str] | None = None,
    batch_size: int = 1,
    input_shape: tuple[int, ...] = (1, 256, 256),
    iterations_per_epoch: int = 1000,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    augment_train: bool = True,
    random_validation: bool = True,
    max_volumes: int | None = None,
    context_slices: int = 1,
    z_augmentation: bool = False,
    filter_empty_labels: bool = True,
    composite_rules: dict[int, list[int]] | None = None,
    **kwargs,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """
    Create train and validation DataLoaders from a Zarr datalist JSON.

    Drop-in replacement for the old ``get_nifti_dataloader()``.
    Returns the same ``{'input': Tensor, 'output': Tensor}`` batch format.

    Parameters
    ----------
    datalist_path : str or Path
        Path to a datalist JSON (``datalist_35cls.json``).
        If None, defaults to ``data/zarr/datalist_35cls.json`` relative to repo root.
    classes : list of str
        Ordered class names.  If None, read from the datalist.
    batch_size : int
        Batch size.
    input_shape : tuple
        ``(1, H, W)`` for 2D or ``(D, H, W)`` for 3D.
    iterations_per_epoch : int
        Number of samples per epoch.
    num_workers : int
        DataLoader workers.
    augment_train : bool
        Apply random augmentation to training data.
    random_validation : bool
        If True, validation samples are random slices.
    max_volumes : int, optional
        Limit the validation set to N crops.
    filter_empty_labels : bool
        If True (default), exclude all-zero annotation arrays.
        If False, include them as negative supervision (target=0).

    Returns
    -------
    (train_loader, val_loader, vis_loader)
        ``val_loader`` = random validation for metrics (may be None).
        ``vis_loader`` = deterministic validation for visualization (may be None).
    """
    # ── Resolve datalist path ──────────────────────────────────────
    if datalist_path is None:
        from organelleseg.config import DATALIST_35CLS
        datalist_path = DATALIST_35CLS
    datalist_path = Path(datalist_path)

    with open(datalist_path) as f:
        datalist = json.load(f)

    # ── Determine classes ──────────────────────────────────────────
    class_defs = datalist.get("class_names", [])
    if classes is None:
        classes = [c["name"] for c in class_defs]

    # ── Resolve data_root ──────────────────────────────────────────
    data_root = datalist.get("data_root", "")
    if not data_root:
        # Default: data/zarr relative to repo root
        from organelleseg.config import ZARR_DATA_PATH
        data_root = str(ZARR_DATA_PATH)

    # ── Build CropMetadata for train/val ───────────────────────────
    def _resolve_entries(entries: list[dict]) -> list[CropMetadata]:
        metas = []
        for entry in entries:
            ds = entry["dataset"]
            crop = entry["crop"]
            meta = resolve_crop_metadata(
                data_root, ds, crop, list(classes),
                filter_empty_labels=filter_empty_labels,
            )
            if meta is not None:
                # Apply Z-range restriction if specified in the datalist entry.
                # This enables splitting a single crop into disjoint train/val
                # sub-volumes along the Z-axis (e.g. for rare classes that exist
                # in only one crop — see FINDINGS.md "val_exclude_from_mean").
                if "z_range" in entry:
                    z_start, z_end = entry["z_range"]
                    meta.z_range = (int(z_start), int(z_end))
                    # Clamp to actual volume depth
                    D = meta.output_shape[0]
                    meta.z_range = (max(0, meta.z_range[0]), min(D - 1, meta.z_range[1]))
                metas.append(meta)
            else:
                print(f"  [WARN] Could not resolve crop {ds}/{crop}, skipping")
        return metas

    train_entries = datalist.get("training", [])
    val_entries = datalist.get("validation", [])

    print(f"  [Zarr] Resolving {len(train_entries)} training crops...")
    t0 = time.monotonic()
    train_metas = _resolve_entries(train_entries)
    dt = time.monotonic() - t0
    print(f"  [Zarr] Resolved {len(train_metas)}/{len(train_entries)} training crops ({dt:.1f}s)")

    if val_entries:
        if max_volumes is not None:
            val_entries = val_entries[:max_volumes]
            print(f"  [max_volumes={max_volumes}] Using {len(val_entries)} validation crops")
        print(f"  [Zarr] Resolving {len(val_entries)} validation crops...")
        val_metas = _resolve_entries(val_entries)
        print(f"  [Zarr] Resolved {len(val_metas)}/{len(val_entries)} validation crops")
    else:
        val_metas = []

    # ── Determine 2D vs 3D ─────────────────────────────────────────
    # 2.5D mode (context_slices > 1) still uses 2D slice datasets;
    # input_shape[0] equals context_slices, NOT a spatial depth dim.
    is_2d = len(input_shape) == 3 and (input_shape[0] == 1 or context_slices > 1)
    spatial_shape = input_shape[-2:]  # (H, W) for 2D

    # ── Worker init for RNG diversity ──────────────────────────────
    def _worker_init_fn(worker_id: int) -> None:
        """Re-seed each DataLoader worker so they produce independent samples."""
        worker_seed = torch.initial_seed() + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2**32))

    # ── DataLoader kwargs ──────────────────────────────────────────
    dl_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "worker_init_fn": _worker_init_fn,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor

    # ── Build datasets ─────────────────────────────────────────────
    # iterations_per_epoch = number of *batches*, so total samples = iterations * batch_size
    train_samples = iterations_per_epoch * batch_size
    val_samples = kwargs.get("val_iterations", 500)  # random val samples per epoch

    if is_2d:
        train_ds = ZarrSliceDataset(
            crop_metas=train_metas,
            classes=list(classes),
            crop_size=spatial_shape,
            iterations=train_samples,
            augment=augment_train,
            context_slices=context_slices,
            z_augmentation=z_augmentation,
            composite_rules=composite_rules,
        )
        train_loader = DataLoader(train_ds, **dl_kwargs)

        val_loader = None
        vis_loader = None
        if val_metas:
            val_z_stride = kwargs.get("val_z_stride", 5)
            val_spatial_tiles = kwargs.get("val_spatial_tiles", 1)

            # Random validation loader — covers all crops uniformly
            # for reliable per-class metrics (like v2/R1).
            val_ds = ZarrSliceDataset(
                crop_metas=val_metas,
                classes=list(classes),
                crop_size=spatial_shape,
                iterations=val_samples,
                augment=False,
                context_slices=context_slices,
                z_augmentation=False,
                composite_rules=composite_rules,
            )
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=pin_memory)

            # Deterministic visualization loader — fixed crops for
            # epoch-over-epoch visual comparison (same slices every time).
            # If vis_crops is specified, only use those datasets for vis.
            vis_crops_filter = kwargs.get("vis_crops", None)
            if vis_crops_filter:
                vis_metas = [m for m in val_metas if m.dataset in vis_crops_filter]
                if not vis_metas:
                    vis_metas = val_metas  # fallback if none match
                else:
                    print(f"  [Vis] Filtered to {len(vis_metas)} crops: {[m.dataset for m in vis_metas]}")
            else:
                vis_metas = val_metas
            vis_ds = ZarrDeterministicValDataset(
                crop_metas=vis_metas,
                classes=list(classes),
                crop_size=spatial_shape,
                composite_rules=composite_rules,
                z_stride=val_z_stride,
                spatial_tiles=val_spatial_tiles,
                context_slices=context_slices,
            )
            vis_loader = DataLoader(vis_ds, batch_size=1, num_workers=0, pin_memory=pin_memory)
    else:
        patch_size = input_shape  # (D, H, W)
        train_ds = ZarrVolumeDataset(
            crop_metas=train_metas,
            classes=list(classes),
            patch_size=patch_size,
            iterations=train_samples,
            augment=augment_train,
        )
        train_loader = DataLoader(train_ds, **dl_kwargs)

        val_loader = None
        vis_loader = None
        if val_metas:
            val_ds = ZarrVolumeDataset(
                crop_metas=val_metas,
                classes=list(classes),
                patch_size=patch_size,
                iterations=max(len(val_metas) * 5, 100),
                augment=False,
            )
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=pin_memory)
            vis_loader = None  # 3D doesn't need separate vis loader

    return train_loader, val_loader, vis_loader
