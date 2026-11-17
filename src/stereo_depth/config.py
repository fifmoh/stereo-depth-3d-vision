"""Single source of truth for all geometry and matcher parameters.

The most common bug in stereo pipelines is applying a resize factor to the images
but not to every derived quantity (focal length, principal point, disparity range,
``doffs``). Everything scale-dependent therefore goes through :func:`Calib.scaled`.

Invariant that must hold (and is unit-tested):
    Z = f * B / (d + doffs)  is invariant to the working scale.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Calib:
    """Rectified stereo geometry (Middlebury ``calib.txt`` layout).

    Attributes
    ----------
    fx, fy : focal length in pixels
    cx, cy : principal point in pixels
    baseline : stereo baseline in millimetres (metric, never rescaled)
    doffs : x-difference of principal points (``doffs = cx1 - cx0``); it is **added**
        to the disparity when converting to depth: ``Z = f*B/(d + doffs)``
    width, height : image size of the reference (already rectified) pair
    ndisp : nominal disparity range (used to size ``numDisparities``)
    vmin, vmax : min/max GT disparity present in the scene
    isint : 1 if GT disparity is integer-valued
    dyavg, dymax : residual vertical misalignment of the rectification
    """

    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float
    doffs: float
    width: int
    height: int
    ndisp: int
    vmin: float | None = None
    vmax: float | None = None
    isint: int | None = None
    dyavg: float | None = None
    dymax: float | None = None

    @property
    def focal(self) -> float:
        """Single focal length (Middlebury rectified pairs are square-pixel)."""
        return self.fx

    def scaled(self, scale: float) -> "Calib":
        """Return the same geometry expressed for images resized by ``scale``.

        ``baseline`` is metric and therefore NOT scaled; ``focal``, the principal
        point, the disparity offset and the disparity range all are.
        """
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        return replace(
            self,
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
            doffs=self.doffs * scale,
            width=int(round(self.width * scale)),
            height=int(round(self.height * scale)),
            ndisp=int(round(self.ndisp * scale)),
            vmin=None if self.vmin is None else self.vmin * scale,
            vmax=None if self.vmax is None else self.vmax * scale,
        )


@dataclass(frozen=True)
class MatcherCfg:
    """Block-matching parameters (SGBM by default).

    ``p1``/``p2`` default to the canonical SGBM heuristic ``8*c*bs^2`` and
    ``32*c*bs^2``; expose ``lambda_p2`` to tune their ratio instead of each value.
    """

    method: str = "sgbm"            # "sgbm" | "bm"
    num_disparities: int = 288      # multiple of 16, >= ndisp of the scene
    block_size: int = 5             # odd
    min_disparity: int = 0
    uniqueness_ratio: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1        # built-in left-right consistency check
    pre_filter_cap: int = 63
    texture_threshold: int = 10     # BM only
    lambda_p2: float = 4.0          # P2 = lambda_p2 * P1
    channels: int = 1
    mode: str = "sgbm_3way"         # sgbm_3way | sgbm_hh | sgbm
    use_wls: bool = False           # needs opencv-contrib (cv2.ximgproc)

    def to_sgbm_kwargs(self) -> dict:
        p1 = 8 * self.channels * self.block_size ** 2
        p2 = int(round(self.lambda_p2 * p1))
        return dict(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=self.disp12_max_diff,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window,
            speckleRange=self.speckle_range,
            preFilterCap=self.pre_filter_cap,
            mode=_sgbm_mode(self.mode),
        )

    def to_bm_kwargs(self) -> dict:
        return dict(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            preFilterCap=self.pre_filter_cap,
            textureThreshold=self.texture_threshold,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window,
            speckleRange=self.speckle_range,
            disp12MaxDiff=self.disp12_max_diff,
        )


def _sgbm_mode(name: str) -> int:
    import cv2

    mapping = {
        "sgbm": cv2.STEREO_SGBM_MODE_SGBM,
        "sgbm_hh": cv2.STEREO_SGBM_MODE_HH,
        "sgbm_3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    }
    try:
        return mapping[name]
    except KeyError:
        raise ValueError(f"unknown sgbm mode {name!r}, expected one of {sorted(mapping)}")


# --------------------------------------------------------------------------------------
# Dataset splits (Middlebury Stereo 2014). Strict separation so that BP-2.0 numbers
# reported in the report are not contaminated by the scenes used to tune parameters.
# --------------------------------------------------------------------------------------

SPLIT_TUNE = ("Adirondack", "Backpack", "Bicycle1", "Cable", "Classroom1", "Couch")
SPLIT_VAL = ("Flowers", "Jadeplant", "Mask", "Motorcycle")
SPLIT_TEST = ("Pipes", "Playroom", "Playtable", "Recycle")
SPLIT_RECT = ("Adirondack", "Backpack", "Motorcycle", "Shelves")  # -imperfect variants

DEFAULT_SCENES_PERFECT = SPLIT_TUNE + SPLIT_VAL + SPLIT_TEST

MIDDLEBURY_BASE = "https://vision.middlebury.edu/stereo/data"
