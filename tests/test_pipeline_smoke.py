"""End-to-end smoke test on the offline ``skimage.data.stereo_motorcycle()`` pair.

This test needs no network access and no dataset download: if it passes, the whole
chain (decode → grey → block matching → depth) runs on the machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from stereo_depth.config import Calib
from stereo_depth.depth import disparity_to_depth
from stereo_depth.matcher import compute_disparity, disparity_valid_mask, suggested_cfg
from stereo_depth.preprocess import estimate_vertical_offset, resize, to_gray

# Motorcycle-perfect published geometry. scikit-image ships this scene resized by
# exactly 0.25 (2964x2000 -> 741x500), including the GT disparity.
MOTO_FULL = Calib(
    fx=3979.911,
    fy=3979.911,
    cx=1244.772,
    cy=1019.507,
    baseline=193.001,
    doffs=124.343,
    width=2964,
    height=2000,
    ndisp=270,
    vmin=23,
    vmax=245,
)
SKIMAGE_SCALE = 0.25


@pytest.fixture(scope="module")
def motorcycle():
    skimage_data = pytest.importorskip("skimage.data")
    left, right, disp = skimage_data.stereo_motorcycle()
    return to_gray(left), to_gray(right), disp


def test_gt_disparity_is_plausible(motorcycle):
    _, _, disp = motorcycle
    finite = disp[np.isfinite(disp)]
    assert finite.size > 0.5 * disp.size
    # Resized by 0.25: expect median ≈ 155/4 ≈ 39 px, inside the calibrated range.
    calib = MOTO_FULL.scaled(SKIMAGE_SCALE)
    median = float(np.median(finite))
    assert 0 < median < calib.vmax
    assert median == pytest.approx(38.7, abs=2.0)


def test_disparity_not_degenerate(motorcycle):
    left, right, gt = motorcycle
    # Work at the resolution scikit-image ships (already 0.25x the published scene).
    calib = MOTO_FULL.scaled(SKIMAGE_SCALE)
    cfg = suggested_cfg(calib, method="sgbm")

    disp = compute_disparity(left, right, cfg)
    valid = disparity_valid_mask(disp)
    assert valid.mean() > 0.3, "less than 30% of pixels got a disparity"
    assert disp[valid].std() > 5.0, "disparity map is flat: matching failed"
    # The recovered disparity must correlate with the ground truth.
    both = valid & np.isfinite(gt) & (gt > calib.vmin)
    corr = np.corrcoef(disp[both], gt[both])[0, 1]
    assert corr > 0.8, f"disparity barely correlates with GT (r={corr:.2f})"


def test_depth_range_is_physical(motorcycle):
    left, right, gt = motorcycle
    calib = MOTO_FULL.scaled(SKIMAGE_SCALE)
    disp = compute_disparity(left, right, suggested_cfg(calib))
    depth = disparity_to_depth(disp, calib)
    finite = depth[np.isfinite(depth)]
    assert finite.size > 1000
    # Middlebury scenes are room-sized and depths are in millimetres.
    assert 250 < np.median(finite) < 20000

    # Cross-check against the GT disparity rather than trusting our own matching.
    gt_depth = disparity_to_depth(gt, calib)
    gt_finite = gt_depth[np.isfinite(gt_depth)]
    assert abs(np.log10(np.median(finite) / np.median(gt_finite))) < 0.5


def test_vertical_offset_measures_zero_on_rectified_pair(motorcycle):
    left, right, _ = motorcycle
    estimate = estimate_vertical_offset(left, right)
    assert abs(estimate.dy_median) < 1.0
    assert estimate.n_matches > 50


def test_vertical_offset_detects_a_shift(motorcycle):
    """Injecting a known vertical shift must be recovered within a fraction of a pixel."""
    left, right, _ = motorcycle
    shift = 3  # pixels
    matrix = np.float32([[1, 0, 0], [0, 1, shift]])
    shifted = np.roll(right, shift, axis=0)  # pure vertical translation of the right view
    _ = matrix  # documented intent: dy = y_left - y_right becomes -shift
    estimate = estimate_vertical_offset(left, shifted)
    assert estimate.dy_median == pytest.approx(-shift, abs=0.5)
