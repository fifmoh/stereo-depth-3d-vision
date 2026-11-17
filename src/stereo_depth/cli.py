"""Command line entry points for the stereo → depth pipeline.

Examples
--------
    python -m stereo_depth.cli info      --scene data/raw/Adirondack-perfect
    python -m stereo_depth.cli dy        --scene data/raw/Adirondack-imperfect --scale 0.5
    python -m stereo_depth.cli disparity --scene data/raw/Adirondack-perfect --scale 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import MatcherCfg  # noqa: F401  (used by suggested_cfg overrides)
from .depth import disparity_to_depth
from .io_middlebury import load_pair
from .matcher import compute_disparity, disparity_valid_mask, suggested_cfg
from .preprocess import estimate_vertical_offset, imread, resize, to_gray


def _load_gray_pair(scene: Path, scale: float):
    pair = load_pair(scene)
    left = to_gray(imread(pair.left_path))
    right = to_gray(imread(pair.right_path))
    if scale != 1.0:
        left = resize(left, scale)
        right = resize(right, scale)
    return pair, left, right


def cmd_info(args: argparse.Namespace) -> int:
    pair = load_pair(args.scene)
    calib = pair.calib
    payload = {
        "scene": pair.scene,
        "grade": pair.grade,
        "root": str(pair.root),
        "focal_px": calib.focal,
        "principal_point_px": [calib.cx, calib.cy],
        "baseline_mm": calib.baseline,
        "doffs_px": calib.doffs,
        "image_size": [calib.width, calib.height],
        "ndisp": calib.ndisp,
        "disparity_range": [calib.vmin, calib.vmax],
        "rectification": {"dyavg": calib.dyavg, "dymax": calib.dymax},
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_dy(args: argparse.Namespace) -> int:
    pair, left, right = _load_gray_pair(args.scene, args.scale)
    estimate = estimate_vertical_offset(left, right)

    # Express every dy statistic in FULL-RESOLUTION pixels so that it is directly
    # comparable to the dyavg / dymax published in calib.txt, whatever --scale was used.
    s = 1.0 / args.scale
    scaled = {
        key: getattr(estimate, key) * s
        for key in ("dy_median", "dy_mad", "dy_p90", "dy_avg_model", "dy_max_model")
    }
    payload = {
        "scene": pair.root.name,
        "scale": args.scale,
        "units": "full-resolution pixels",
        "calib_dyavg": pair.calib.dyavg,
        "calib_dymax": pair.calib.dymax,
        "measured": {
            **scaled,
            "dx_median": estimate.dx_median * s,
            "rotation_deg": estimate.rotation_deg,
            "scale_xy": estimate.scale,
            "n_matches": estimate.n_matches,
            "n_inliers": estimate.n_inliers,
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_disparity(args: argparse.Namespace) -> int:
    pair, left, right = _load_gray_pair(args.scene, args.scale)
    calib = pair.calib.scaled(args.scale)
    cfg = suggested_cfg(calib, method=args.method)
    overrides = {}
    if args.num_disparities:
        overrides["num_disparities"] = args.num_disparities
    if args.block_size:
        overrides["block_size"] = args.block_size
    if overrides:
        from dataclasses import replace

        cfg = replace(cfg, **overrides)

    disparity = compute_disparity(left, right, cfg)
    depth = disparity_to_depth(disparity, calib)
    valid = disparity_valid_mask(disparity) & np.isfinite(depth)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{pair.root.name}_disp.npy", disparity)
    np.save(out_dir / f"{pair.root.name}_depth.npy", depth)

    finite = depth[valid]
    summary = {
        "scene": pair.root.name,
        "method": cfg.method,
        "scale": args.scale,
        "num_disparities": cfg.num_disparities,
        "block_size": cfg.block_size,
        "valid_ratio": float(valid.mean()),
        "disparity_px": {
            "min": float(np.nanmin(disparity[valid])) if valid.any() else None,
            "max": float(np.nanmax(disparity[valid])) if valid.any() else None,
        },
        "depth_mm": {
            "median": float(np.median(finite)) if finite.size else None,
            "p10": float(np.percentile(finite, 10)) if finite.size else None,
            "p90": float(np.percentile(finite, 90)) if finite.size else None,
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stereo_depth.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, helptext in (
        ("info", cmd_info, "print the scene calibration as JSON"),
        ("dy", cmd_dy, "measure residual vertical misalignment (phase 1 check)"),
        ("disparity", cmd_disparity, "compute disparity + depth and save them as .npy"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--scene", type=Path, required=True, help="path to <Scene>-<grade> dir")
        p.add_argument("--scale", type=float, default=0.5, help="working scale (default 0.5)")
        if name == "disparity":
            p.add_argument("--method", default="sgbm", choices=["sgbm", "bm"])
            p.add_argument("--block-size", type=int, default=0, help="0 = auto (5 sgbm / 15 bm)")
            p.add_argument("--num-disparities", type=int, default=0, help="0 = derive from ndisp")
            p.add_argument("--out", type=Path, default="outputs")
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
