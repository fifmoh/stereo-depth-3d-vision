"""PFM and calib.txt parsing tests (the phase-0 bugs that silently poison every metric)."""

from __future__ import annotations

import numpy as np
import pytest

from stereo_depth.io_middlebury import parse_calib, read_pfm, write_pfm


def test_pfm_roundtrip_single_channel(tmp_path):
    data = np.array([[1.0, 2.5], [np.inf, -3.0]], dtype=np.float32)
    path = tmp_path / "disp.pfm"
    write_pfm(path, data)
    back = read_pfm(path)
    assert back.shape == data.shape
    # np.inf == np.inf is True; use array comparison to be explicit.
    np.testing.assert_array_equal(back, data)


def test_pfm_roundtrip_rgb(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.random((4, 6, 3)).astype(np.float32)
    path = tmp_path / "img.pfm"
    write_pfm(path, data)
    np.testing.assert_array_equal(read_pfm(path), data)


def test_pfm_preserves_row_order(tmp_path):
    """PFM stores rows bottom-to-top: the reader must flip them back."""
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "rows.pfm"
    write_pfm(path, data)
    np.testing.assert_array_equal(read_pfm(path), data)


def test_pfm_rejects_non_pfm(tmp_path):
    path = tmp_path / "nope.pfm"
    path.write_bytes(b"P6\n2 2\n255\n")
    with pytest.raises(ValueError):
        read_pfm(path)


CALIB = """cam0=[4161.221 0 1445.577; 0 4161.221 984.686; 0 0 1]
cam1=[4161.221 0 1654.636; 0 4161.221 984.686; 0 0 1]
doffs=209.059
baseline=176.252
width=2880
height=1988
ndisp=280
isint=0
vmin=25
vmax=248
dyavg=0
dymax=0
"""

IMPERFECT = CALIB.replace("dyavg=0", "dyavg=0.735").replace("dymax=0", "dymax=1.694")


def _write(tmp_path, text):
    path = tmp_path / "calib.txt"
    path.write_text(text)
    return path


def test_parse_calib_fields(tmp_path):
    calib = parse_calib(_write(tmp_path, CALIB))
    assert calib.focal == pytest.approx(4161.221)
    assert calib.cx == pytest.approx(1445.577)
    assert calib.cy == pytest.approx(984.686)
    assert calib.baseline == pytest.approx(176.252)
    assert calib.doffs == pytest.approx(209.059)
    assert (calib.width, calib.height) == (2880, 1988)
    assert calib.ndisp == 280
    assert (calib.vmin, calib.vmax) == (25, 248)
    assert calib.dyavg == 0.0


def test_parse_calib_imperfect_reports_misalignment(tmp_path):
    calib = parse_calib(_write(tmp_path, IMPERFECT))
    assert calib.dyavg == pytest.approx(0.735)
    assert calib.dymax == pytest.approx(1.694)


def test_scaled_calib_matches_real_resize(tmp_path):
    calib = parse_calib(_write(tmp_path, CALIB)).scaled(0.5)
    assert calib.focal == pytest.approx(2080.6105)
    assert calib.doffs == pytest.approx(104.5295)
    assert calib.ndisp == 140
    assert calib.width == 1440
    # baseline is metric: it must NOT be scaled
    assert calib.baseline == pytest.approx(176.252)


def test_registered_matrices_are_consistent(tmp_path):
    """cam1 differs from cam0 only by the disparity offset (pure horizontal shift)."""
    calib = parse_calib(_write(tmp_path, CALIB))
    # Middlebury encodes the rig offset as cx1 - cx0, which equals doffs.
    cx1_delta = 1654.636 - 1445.577
    assert cx1_delta == pytest.approx(calib.doffs, abs=1e-2)
