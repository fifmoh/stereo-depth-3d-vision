"""Preprocessing: decoding, greyscale, rescaling and *verification* of rectification.

Phase 1 of the assignment asks for "geometric rectification and perfect horizontal
alignment of the stereo pair". Middlebury ``-perfect`` scenes are already rectified
(``dyavg = 0``), so the pipeline must be able to *measure* residual vertical
misalignment instead of assuming it is zero: see :func:`estimate_vertical_offset`.
The corrective rectification (ECC/homography estimation) lives in M1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image, raising a clear error instead of returning ``None``."""
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"could not decode image: {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize(img: np.ndarray, scale: float, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize by ``scale``; prefer :func:`cv2.resize` with INTER_AREA for downscaling."""
    if scale == 1.0:
        return img
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    height, width = img.shape[:2]
    size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(img, size, interpolation=interpolation)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Local contrast equalisation (helps SGBM on dark scenes such as ``Shelves``)."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(gray)


@dataclass
class DyEstimate:
    """Residual vertical misalignment of a stereo pair.

    Two complementary views are reported:

    * ``dy_median`` / ``dy_mad`` / ``dy_p90`` — statistics of the original
      horizontal-alignment convention on *subpixel-refined* feature matches;
    * ``dy_avg_model`` / ``dy_max_model`` — mean and max ``|dy|`` predicted by a robust
      affine model over the whole image, directly comparable to the ``dyavg`` / ``dymax``
      published in Middlebury ``calib.txt``.
    """

    dy_median: float
    dy_mad: float
    dy_p90: float
    dx_median: float
    dy_avg_model: float
    dy_max_model: float
    rotation_deg: float
    scale: float
    n_matches: int
    n_inliers: int
    affine: np.ndarray | None = None

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        return (
            f"dy_median={self.dy_median:+.3f}px mad={self.dy_mad:.3f}px "
            f"p90={self.dy_p90:.3f}px | model dyavg={self.dy_avg_model:.3f}px "
            f"dymax={self.dy_max_model:.3f}px | rot={self.rotation_deg:+.3f}deg "
            f"(n={self.n_matches}, inliers={self.n_inliers})"
        )


def _match_features(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_features: int,
    ratio: float,
    max_dx: float,
    max_dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched point pairs (left, right) after ratio test and plausibility filters."""
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=7)
    kp0, des0 = orb.detectAndCompute(left_gray, None)
    kp1, des1 = orb.detectAndCompute(right_gray, None)
    if des0 is None or des1 is None or len(kp0) < 8 or len(kp1) < 8:
        raise RuntimeError("not enough ORB features to estimate the vertical offset")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    good = [
        m
        for pair in matcher.knnMatch(des0, des1, k=2)
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < ratio * n.distance
    ]
    if len(good) < 8:
        raise RuntimeError(f"not enough reliable matches ({len(good)}) to estimate dy")

    pts0 = np.array([kp0[m.queryIdx].pt for m in good], dtype=np.float32)
    pts1 = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float32)

    # A left pixel is matched to a right pixel shifted left by its disparity.
    dx = pts0[:, 0] - pts1[:, 0]
    dy = pts0[:, 1] - pts1[:, 1]
    keep = (dx > 0) & (dx < max_dx) & (np.abs(dy) < max_dy)
    if keep.sum() < 8:
        raise RuntimeError(f"not enough consistent matches ({int(keep.sum())}) to estimate dy")
    return pts0[keep], pts1[keep]


def _refine_subpixel(img: np.ndarray, pts: np.ndarray, win: int = 5) -> np.ndarray:
    """Sub-pixel corner refinement: ORB keypoints are quantised to whole pixels."""
    refined = np.ascontiguousarray(pts.reshape(-1, 1, 2).astype(np.float32))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.01)
    cv2.cornerSubPix(img, refined, (win, win), (-1, -1), criteria)
    return refined.reshape(-1, 2)


def estimate_vertical_offset(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_features: int = 6000,
    ratio: float = 0.75,
    max_dx: float = 800.0,
    max_dy: float = 15.0,
    refine: bool = True,
) -> DyEstimate:
    """Measure the residual vertical misalignment between two rectified views.

    A correctly rectified pair yields ``dy ≈ 0``; Middlebury ``-imperfect`` scenes yield
    ``dy_avg_model`` matching their published ``dyavg`` (≈ 0.45 … 0.92 px).

    The robust affine model (partial affine + RANSAC) is what makes the measurement
    comparable to ``calib.txt``: a residual rectification error is a smooth function of
    the position (rotation + shear + shift), not a constant offset, so a single median
    would be undetectable once quantised to whole pixels.
    """
    if left_gray.shape != right_gray.shape:
        raise ValueError(
            f"stereo pair must have identical shapes, got {left_gray.shape} and {right_gray.shape}"
        )

    pts0, pts1 = _match_features(
        left_gray, right_gray, max_features, ratio, max_dx, max_dy
    )
    if refine:
        pts0 = _refine_subpixel(left_gray, pts0)
        pts1 = _refine_subpixel(right_gray, pts1)

    dy = pts0[:, 1] - pts1[:, 1]
    dx = pts0[:, 0] - pts1[:, 0]

    # Fit the *vertical* displacement field dy(x, y) = a*xn + b*yn + c robustly, with
    # coordinates centred and normalised so the least-squares problem is well conditioned
    # (raw pixel coordinates make the design matrix rank deficient for narrow bands of
    # matches, which produced inf coefficients).
    # A residual rectification error is a smooth function of position (rotation, shear,
    # scale mismatch), so a single global median is not enough: once quantised to whole
    # pixels it collapses to 0 and hides a real misalignment.
    height, width = left_gray.shape
    cx, cy = width / 2.0, height / 2.0
    xn = (pts1[:, 0] - cx) / width
    yn = (pts1[:, 1] - cy) / height
    design = np.stack([xn, yn, np.ones_like(xn)], axis=1)
    coef, inliers = _robust_affine_fit(design, dy)

    xs = np.linspace(0, width - 1, 32)
    ys = np.linspace(0, height - 1, 32)
    grid_x, grid_y = np.meshgrid((xs - cx) / width, (ys - cy) / height)
    dy_grid = coef[0] * grid_x + coef[1] * grid_y + coef[2]

    dy_in = dy[inliers]
    med = float(np.median(dy_in))
    mad = float(np.median(np.abs(dy_in - med)))
    model_at_inliers = np.einsum("ij,j->i", design[inliers], coef)
    residual = np.abs(dy_in - model_at_inliers)

    # dy per pixel in x: rotation angle; dy per pixel in y: vertical scale/shear mismatch.
    slope_x = coef[0] / width
    slope_y = coef[1] / height

    return DyEstimate(
        dy_median=med,
        dy_mad=mad,
        dy_p90=float(np.percentile(residual, 90)) if residual.size else 0.0,
        dx_median=float(np.median(dx)),
        dy_avg_model=float(np.mean(np.abs(dy_grid))),
        dy_max_model=float(np.max(np.abs(dy_grid))),
        rotation_deg=float(np.degrees(np.arctan(slope_x))),
        scale=float(np.hypot(slope_x, slope_y - 1.0)),
        n_matches=int(dy.size),
        n_inliers=int(inliers.sum()),
        affine=np.array([[slope_x, slope_y, float(coef[2])]], dtype=np.float64),
    )


def _robust_affine_fit(design: np.ndarray, values: np.ndarray, iterations: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively reweighted least squares with MAD-based outlier rejection."""
    keep = np.ones(values.size, dtype=bool)
    coef = np.array([0.0, 0.0, float(np.median(values))])
    for _ in range(iterations):
        coef, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        if not np.all(np.isfinite(coef)):
            break
        # einsum, not ``@``: BLAS matmul on these strided arrays raises spurious
        # divide-by-zero / overflow FloatingPointErrors on macOS Accelerate.
        residual = values - np.einsum("ij,j->i", design, coef)
        med = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - med))
        threshold = max(3.0 * 1.4826 * mad, 0.25)
        new_keep = np.abs(residual - med) <= threshold
        if new_keep.sum() < 8 or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= 8 else keep
            break
        keep = new_keep
    coef, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
    return coef, keep
