"""Readers for the Middlebury Stereo 2014 dataset (no login, direct HTTP).

Layout of one scene directory ``datasets/<Scene>-<perfect|imperfect>/``::

    im0.png      left  image (rectified)
    im1.png      right image (rectified)
    disp0.pfm    ground-truth disparity, left  view (float32 PFM)
    disp1.pfm    ground-truth disparity, right view (float32 PFM)
    calib.txt    rectified intrinsics + baseline + disparity offset

PFM notes: the header stores a *negative* scale for little-endian data, the raster is
written bottom-to-top, and unknown disparities are ``inf`` (not ``0``). All three
details are handled here and unit-tested in ``tests/test_pfm.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Calib

REQUIRED_FILES = ("im0.png", "im1.png", "disp0.pfm", "disp1.pfm", "calib.txt")


# ----------------------------------------------------------------------------------
# PFM
# ----------------------------------------------------------------------------------
def read_pfm(path: str | Path) -> np.ndarray:
    """Read a PFM file into a float32 array (H, W) or (H, W, 3), top-to-bottom."""
    path = Path(path)
    with open(path, "rb") as fh:
        magic = fh.readline().rstrip()
        if magic == b"PF":
            channels = 3
        elif magic == b"Pf":
            channels = 1
        else:
            raise ValueError(f"{path}: not a PFM file (magic={magic!r})")

        # Skip any comment / blank lines between header entries.
        def _next_token() -> bytes:
            line = fh.readline()
            while line.startswith(b"#") or line.strip() == b"":
                line = fh.readline()
            return line

        dims = _next_token().split()
        if len(dims) != 2:
            raise ValueError(f"{path}: bad dimensions line {dims!r}")
        width, height = int(dims[0]), int(dims[1])

        scale_line = _next_token().strip()
        scale = float(scale_line)
        endian = "<" if scale < 0 else ">"  # negative scale == little endian

        count = width * height * channels
        payload = fh.read(count * 4)
        if len(payload) != count * 4:
            raise ValueError(f"{path}: truncated payload ({len(payload)} of {count * 4} bytes)")

    dtype = np.dtype(f"{endian}f4")
    data = np.frombuffer(payload, dtype=dtype)
    shape = (height, width, channels) if channels == 3 else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float32, copy=False)


def write_pfm(path: str | Path, data: np.ndarray, scale: float = -1.0) -> None:
    """Write a float32 array as a PFM file (used by tests and by our own exports)."""
    path = Path(path)
    data = np.asarray(data, dtype=np.float32)
    height, width = data.shape[:2]
    channels = 3 if data.ndim == 3 else 1
    if channels == 3 and data.shape[2] != 3:
        raise ValueError("only 1 or 3 channel PFM is supported")
    endian = "<" if scale < 0 else ">"
    header = (
        f"{'PF' if channels == 3 else 'Pf'}\n"
        f"{width} {height}\n"
        f"{scale}\n"
    ).encode("ascii")
    payload = np.flipud(data).astype(np.dtype(f"{endian}f4")).tobytes()
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(payload)


# ----------------------------------------------------------------------------------
# calib.txt
# ----------------------------------------------------------------------------------
_CAM_RE = re.compile(r"cam(\d)\s*=\s*\[([^\]]+)\]")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_calib(path: str | Path) -> Calib:
    """Parse a Middlebury ``calib.txt`` into a :class:`Calib`."""
    path = Path(path)
    text = path.read_text()

    matrices: dict[int, np.ndarray] = {}
    scalars: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:  # pragma: no cover - defensive
            continue
        cam = _CAM_RE.match(line)
        if cam:
            values = [float(v) for v in _NUM_RE.findall(cam.group(2))]
            if len(values) != 9:
                raise ValueError(f"{path}: cam{cam.group(1)} must be 3x3, got {values}")
            matrices[int(cam.group(1))] = np.array(values).reshape(3, 3)
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            try:
                scalars[key.strip()] = float(value.strip())
            except ValueError:  # pragma: no cover - unknown non-numeric field
                continue

    if 0 not in matrices or 1 not in matrices:
        raise ValueError(f"{path}: both cam0 and cam1 are required")

    def _opt(name: str):
        return scalars.get(name)

    return Calib(
        fx=float(matrices[0][0, 0]),
        fy=float(matrices[0][1, 1]),
        cx=float(matrices[0][0, 2]),
        cy=float(matrices[0][1, 2]),
        baseline=float(scalars["baseline"]),
        doffs=float(scalars["doffs"]),
        width=int(scalars["width"]),
        height=int(scalars["height"]),
        ndisp=int(scalars["ndisp"]),
        vmin=_opt("vmin"),
        vmax=_opt("vmax"),
        isint=None if _opt("isint") is None else int(scalars["isint"]),
        dyavg=_opt("dyavg"),
        dymax=_opt("dymax"),
    )


# ----------------------------------------------------------------------------------
# Scene loading
# ----------------------------------------------------------------------------------
@dataclass
class StereoPair:
    """One middlebury scene pair with its geometry (images are *not* loaded yet)."""

    scene: str
    grade: str          # "perfect" | "imperfect"
    root: Path
    calib: Calib

    @property
    def left_path(self) -> Path:
        return self.root / "im0.png"

    @property
    def right_path(self) -> Path:
        return self.root / "im1.png"

    @property
    def disp_left_path(self) -> Path:
        return self.root / "disp0.pfm"

    @property
    def disp_right_path(self) -> Path:
        return self.root / "disp1.pfm"

    def read_gt_disparity(self) -> np.ndarray:
        """Ground-truth left disparity, with ``inf`` (unknown) mapped to ``nan``."""
        disp = read_pfm(self.disp_left_path)
        disp = np.where(np.isfinite(disp), disp, np.nan)
        return disp.astype(np.float32)

    def valid_mask(self, tol: float = 1e-6) -> np.ndarray:
        """Pixels carrying a usable ground-truth disparity."""
        disp = self.read_gt_disparity()
        mask = np.isfinite(disp)
        if self.calib.vmin is not None:
            mask &= disp > (self.calib.vmin - tol)
        return mask


def load_pair(scene_dir: str | Path) -> StereoPair:
    """Load one ``<Scene>-<grade>`` directory into a :class:`StereoPair`."""
    root = Path(scene_dir)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{root}: missing {missing} (run scripts/fetch_data.py)")
    name = root.name
    scene, _, grade = name.rpartition("-")
    if not scene:
        scene, grade = name, "unknown"
    return StereoPair(scene=scene, grade=grade, root=root, calib=parse_calib(root / "calib.txt"))
