"""Triangulation: disparity → metric depth → 3D points.

Both formulations are implemented on purpose and must agree:

    Z = f * B / (d + doffs)                    (official Middlebury convention)
    Z = Q-derived projection                   (cv2.reprojectImageTo3D)

``doffs = cx1 - cx0`` is the disparity offset of the rectified pair as published in
``calib.txt`` and it is **added** to the disparity (Middlebury dataset description:
"Z = baseline * f / (d + doffs)"). Dropping it biases every depth by a large constant
factor — see ``tests/test_depth_math.py``.
"""

from __future__ import annotations

import numpy as np

from .config import Calib


def disparity_to_depth(
    disparity: np.ndarray,
    calib: Calib,
    min_depth: float = 1e-6,
    max_depth: float | None = None,
) -> np.ndarray:
    """Convert disparity (pixels) to depth in the same unit as ``calib.baseline``.

    Pixels whose ``d + doffs`` is not strictly positive (e.g. invalid or clipped
    disparities) are ``nan``.
    """
    focal = calib.focal
    baseline = calib.baseline
    denom = disparity.astype(np.float64) + calib.doffs

    with np.errstate(divide="ignore", invalid="ignore"):
        depth = focal * baseline / denom

    depth = np.where(denom > 1e-3, depth, np.nan)
    depth = np.where(depth > min_depth, depth, np.nan)
    if max_depth is not None:
        depth = np.where(depth < max_depth, depth, np.nan)
    return depth.astype(np.float32)


def depth_to_disparity(depth: np.ndarray, calib: Calib) -> np.ndarray:
    """Inverse mapping, mostly useful for tests and sanity checks."""
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = calib.focal * calib.baseline / depth.astype(np.float64) - calib.doffs
    return np.where(depth > 0, disp, np.nan).astype(np.float32)


def relative_depth(depth: np.ndarray) -> np.ndarray:
    """Scale-free depth: ``Z / median(Z)`` (``nan``-aware)."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.full_like(depth, np.nan)
    return (depth / float(np.median(finite))).astype(np.float32)


def projection_matrix(calib: Calib) -> np.ndarray:
    """``Q`` matrix for :func:`cv2.reprojectImageTo3D` from rectified geometry.

    For a standard rectified pair (``Tx = baseline``, principal points offset by
    ``doffs``), Q is::

        1  0  0       -cx
        0  1  0       -cy
        0  0  0         f
        0  0 1/B   doffs/B

    which yields ``qw = (d + doffs)/B`` and therefore ``Z = qz/qw = f*B/(d + doffs)``.
    """
    baseline = calib.baseline
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    q = np.zeros((4, 4), dtype=np.float64)
    q[0, 0] = 1.0
    q[1, 1] = 1.0
    q[0, 3] = -calib.cx
    q[1, 3] = -calib.cy
    q[2, 3] = calib.focal
    q[3, 2] = 1.0 / baseline
    q[3, 3] = calib.doffs / baseline
    return q


def disparity_to_points(disparity: np.ndarray, calib: Calib) -> np.ndarray:
    """Reproject a disparity map to 3D points (H, W, 3) in ``calib`` units."""
    import cv2

    disp = np.nan_to_num(disparity.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # reprojectImageTo3D expects the *shifted* disparity used by Q; our Q absorbs doffs.
    points = cv2.reprojectImageTo3D(disp, projection_matrix(calib))
    points[~np.isfinite(points)] = np.nan
    return points
