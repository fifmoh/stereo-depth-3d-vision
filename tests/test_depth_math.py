"""Depth/triangulation maths tests: scale invariance, doffs handling, Q consistency."""

from __future__ import annotations

import numpy as np
import pytest

from stereo_depth.config import Calib
from stereo_depth.depth import (
    depth_to_disparity,
    disparity_to_depth,
    disparity_to_points,
    projection_matrix,
)

CALIB = Calib(
    fx=4161.221,
    fy=4161.221,
    cx=1445.577,
    cy=984.686,
    baseline=176.252,
    doffs=209.059,
    width=2880,
    height=1988,
    ndisp=280,
)


def test_closed_form_depth():
    disp = np.array([[300.0, 220.0]], dtype=np.float32)
    depth = disparity_to_depth(disp, CALIB)
    expected = CALIB.focal * CALIB.baseline / (disp.astype(np.float64) + CALIB.doffs)
    np.testing.assert_allclose(depth, expected, rtol=1e-6)


def test_reference_scene_depth_is_in_metres():
    """Motorcycle-perfect: median GT disparity 154.9 px -> ~2.75 m (doffs added)."""
    calib = Calib(
        fx=3979.911, fy=3979.911, cx=1244.772, cy=1019.507,
        baseline=193.001, doffs=124.343, width=2964, height=2000, ndisp=270,
    )
    depth = disparity_to_depth(np.array([[154.9]], dtype=np.float32), calib)[0, 0]
    assert depth == pytest.approx(2751.0, rel=0.01)  # millimetres


def test_doffs_cannot_be_ignored():
    """Ignoring doffs produces a large systematic bias, not a rounding error."""
    disp = np.array([[150.0]], dtype=np.float32)
    with_doffs = float(disparity_to_depth(disp, CALIB)[0, 0])
    without = float(CALIB.focal * CALIB.baseline / 150.0)
    assert (without - with_doffs) / with_doffs > 1.0  # > 100 % relative error


def test_depth_invariant_to_working_scale():
    """Z must not depend on the resolution the images were matched at."""
    disp_full = 300.0
    z_full = disparity_to_depth(np.array([[disp_full]], np.float32), CALIB)[0, 0]
    scale = 0.25
    calib_s = CALIB.scaled(scale)
    z_scaled = disparity_to_depth(np.array([[disp_full * scale]], np.float32), calib_s)[0, 0]
    assert z_scaled == pytest.approx(z_full, rel=1e-5)


def test_no_non_positive_denominator():
    # d + doffs must be strictly positive; d = -doffs gives an exact zero denominator.
    disp = np.array([[0.0, -CALIB.doffs, -500.0]], dtype=np.float32)
    depth = disparity_to_depth(disp, CALIB)
    assert np.isfinite(depth[0, 0])  # d = 0 is a legitimate pixel (Z = f*B/doffs)
    assert np.isnan(depth[0, 1:]).all()
    # ... while a denominator just above the floor yields an enormous finite depth.
    depth_edge = disparity_to_depth(np.array([[-CALIB.doffs + 0.5]], dtype=np.float32), CALIB)[0, 0]
    assert np.isfinite(depth_edge) and depth_edge > 1e6


def test_roundtrip_disparity_depth():
    disp = np.array([[260.0, 300.0, 400.0]], dtype=np.float32)
    depth = disparity_to_depth(disp, CALIB)
    np.testing.assert_allclose(depth_to_disparity(depth, CALIB), disp, rtol=1e-5)


def test_reproject_matches_closed_form():
    """cv2.reprojectImageTo3D with our Q must agree with Z = f*B/(d+doffs)."""
    h, w = 8, 12
    disp = np.full((h, w), 300.0, dtype=np.float32)
    disp[0, 0] = -CALIB.doffs  # zero denominator -> invalid

    closed = disparity_to_depth(disp, CALIB)
    points = disparity_to_points(disp, CALIB)
    np.testing.assert_allclose(points[..., 2][1:, 1:], closed[1:, 1:], rtol=1e-4)
    z00 = float(points[0, 0, 2])
    assert not np.isfinite(z00) or abs(z00) > 1e6  # degenerate: zero denominator

    # X/Y must follow the pinhole formula around the principal point.
    v, u = 4, 6
    z = closed[v, u]
    assert points[v, u, 0] == pytest.approx((u - CALIB.cx) * z / CALIB.focal, rel=1e-4)
    assert points[v, u, 1] == pytest.approx((v - CALIB.cy) * z / CALIB.focal, rel=1e-4)


def test_projection_matrix_shape():
    q = projection_matrix(CALIB)
    assert q.shape == (4, 4)
    assert q[3, 2] == pytest.approx(1.0 / CALIB.baseline)
    assert q[3, 3] == pytest.approx(CALIB.doffs / CALIB.baseline)
