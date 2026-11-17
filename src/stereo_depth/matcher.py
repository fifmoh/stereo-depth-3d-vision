"""Block-matching stage: StereoBM / StereoSGBM wrapped behind one call.

No learned model is involved: the disparity is a deterministic function of the pixels.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Calib, MatcherCfg


def create_matcher(cfg: MatcherCfg):
    """Instantiate the OpenCV matcher described by ``cfg``."""
    method = cfg.method.lower()
    if method == "bm":
        return cv2.StereoBM_create(
            numDisparities=cfg.num_disparities, blockSize=cfg.block_size
        )
    if method == "sgbm":
        return cv2.StereoSGBM_create(**cfg.to_sgbm_kwargs())
    raise ValueError(f"unknown method {cfg.method!r} (expected 'bm' or 'sgbm')")


def num_disparities_for(calib: Calib, step: int = 16) -> int:
    """Smallest valid ``numDisparities`` (multiple of ``step``) covering the scene.

    ``Calib.scaled()`` already scales ``ndisp``, so the working resolution must NOT be
    applied again here (that bug silently truncated the search range to 32 px).
    """
    value = int(np.ceil(float(calib.ndisp) / step) * step)
    return max(step, value)


def suggested_cfg(calib: Calib, method: str = "sgbm") -> MatcherCfg:
    """A sensible starting configuration derived from the (already scaled) geometry."""
    return MatcherCfg(
        method=method,
        num_disparities=num_disparities_for(calib),
        block_size=5 if method == "sgbm" else 15,
    )


def compute_disparity(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    cfg: MatcherCfg | None = None,
) -> np.ndarray:
    """Return a float32 disparity map in pixels at the resolution of the inputs.

    OpenCV's SGBM/BM natively return signed 16-bit values scaled by 16, with
    ``-16`` (i.e. ``-1.0``) marking invalid pixels.
    """
    if left_gray.shape != right_gray.shape:
        raise ValueError(
            f"stereo pair must have identical shapes, got {left_gray.shape} and {right_gray.shape}"
        )
    cfg = cfg or MatcherCfg()
    left = left_gray if left_gray.dtype == np.uint8 else left_gray.astype(np.uint8)
    right = right_gray if right_gray.dtype == np.uint8 else right_gray.astype(np.uint8)

    raw = create_matcher(cfg).compute(left, right)
    return raw.astype(np.float32) / 16.0


def disparity_valid_mask(disparity: np.ndarray, min_valid: float = 0.0) -> np.ndarray:
    """Pixels with a usable (finite, positive) disparity."""
    mask = np.isfinite(disparity) & (disparity > min_valid)
    return mask


def remove_speckles(disparity: np.ndarray, valid: np.ndarray, max_size: int = 200) -> np.ndarray:
    """Drop connected components of ``valid`` smaller than ``max_size`` pixels."""
    mask = valid.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = valid.copy()
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] < max_size:
            out[labels == label] = False
    return out


def left_right_consistent(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    cfg: MatcherCfg,
    tolerance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both directions and return (left disparity, LRC validity mask)."""
    disp_left = compute_disparity(left_gray, right_gray, cfg)
    disp_right = compute_disparity(right_gray, left_gray, cfg)

    height, width = disp_left.shape
    xs = np.arange(width)[None, :].repeat(height, axis=0)
    targets = np.clip((xs - disp_left).astype(np.int32), 0, width - 1)
    sampled = np.take_along_axis(disp_right, targets, axis=1)
    valid = (
        disparity_valid_mask(disp_left)
        & disparity_valid_mask(sampled)
        & (np.abs(disp_left - sampled) <= tolerance)
    )
    return disp_left, valid
