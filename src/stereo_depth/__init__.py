"""Classical stereo → depth → 3D pipeline (CPU only, no learned models).

Stages
------
1. preprocess : rectification / verification of horizontal alignment (dy ≈ 0)
2. matcher    : StereoBM / StereoSGBM block matching (+ deterministic post-filters)
3. depth      : disparity → metric depth by triangulation, and 3D reprojection
4. objects    : per-object distance estimation
"""

__version__ = "0.1.0"

from .config import Calib, MatcherCfg  # noqa: F401
