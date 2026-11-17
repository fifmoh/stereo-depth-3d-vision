"""Phase-1 rectification checks.

The image-content estimator is validated on **known-answer** synthetic corruptions
(inject a rotation / vertical shift, check it is recovered), then cross-checked on the
real Middlebury scenes, where ``-perfect`` must be near zero and ``-imperfect`` must land
in the range published in ``calib.txt``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from stereo_depth.preprocess import estimate_vertical_offset, to_gray

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def motorcycle_left():
    skimage_data = pytest.importorskip("skimage.data")
    left, _right, _disp = skimage_data.stereo_motorcycle()
    return to_gray(left)


def _rotate(img: np.ndarray, angle_deg: float, center: tuple[float, float] | None = None):
    h, w = img.shape[:2]
    center = center or (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def test_recovers_injected_rotation(motorcycle_left):
    """A 0.3 deg rotation must be recovered with the expected dy statistics."""
    left = motorcycle_left
    right = _rotate(left, 0.3)
    estimate = estimate_vertical_offset(left, right)

    height, width = left.shape
    # dy(x) = tan(theta) * (x - cx) for a rotation about the image centre.
    expected_avg = np.tan(np.radians(0.3)) * np.mean(np.abs(np.arange(width) - width / 2.0))
    expected_max = np.tan(np.radians(0.3)) * (width / 2.0)

    assert abs(estimate.rotation_deg) == pytest.approx(0.3, abs=0.05)
    assert estimate.dy_avg_model == pytest.approx(expected_avg, rel=0.2)
    assert estimate.dy_max_model == pytest.approx(expected_max, rel=0.2)


def test_recovers_injected_vertical_shift(motorcycle_left):
    left = motorcycle_left
    shift = 4
    right = np.roll(left, shift, axis=0)
    estimate = estimate_vertical_offset(left, right)
    # A downward-shifted right view yields dy = y_left - y_right = -shift.
    assert estimate.dy_avg_model == pytest.approx(shift, abs=0.25)
    assert estimate.rotation_deg == pytest.approx(0.0, abs=0.02)


def test_rectified_pair_is_clean(motorcycle_left):
    """The shipped motorcycle pair is already rectified: dy must be small."""
    skimage_data = pytest.importorskip("skimage.data")
    _left, right, _disp = skimage_data.stereo_motorcycle()
    estimate = estimate_vertical_offset(motorcycle_left, to_gray(right))
    assert estimate.dy_avg_model < 0.5
    assert estimate.n_inliers > 100


# --------------------------------------------------------------------------------------
# Real scene checks (skipped when the dataset has not been downloaded)
# --------------------------------------------------------------------------------------
needs_data = pytest.mark.skipif(
    not (DATA / "Adirondack-perfect").is_dir(), reason="run `make data` first"
)


@needs_data
def test_real_perfect_scene_is_aligned():
    import stereo_depth.io_middlebury as io
    from stereo_depth.preprocess import imread

    pair = io.load_pair(DATA / "Adirondack-perfect")
    left = to_gray(imread(pair.left_path))
    right = to_gray(imread(pair.right_path))
    estimate = estimate_vertical_offset(left, right)
    # Published dyavg = 0.0, dymax = 0.0; the measured noise floor is well below 0.2 px.
    assert estimate.dy_avg_model < 0.2


@needs_data
def test_real_imperfect_scene_matches_published_value():
    import stereo_depth.io_middlebury as io
    from stereo_depth.preprocess import imread

    pair = io.load_pair(DATA / "Adirondack-imperfect")
    left = to_gray(imread(pair.left_path))
    right = to_gray(imread(pair.right_path))
    estimate = estimate_vertical_offset(left, right)
    published = pair.calib.dyavg
    assert published == pytest.approx(0.735, abs=1e-3)
    # Recovered within ~0.15 px of the published value.
    assert estimate.dy_avg_model == pytest.approx(published, abs=0.15)
