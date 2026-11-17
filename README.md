# Stereo Depth → 3D Vision (CPU only)

Classical stereo vision pipeline: **rectification → block matching → triangulation → 3D
points and object distances**. No GPU, no neural network, no dataset account.

This is the implementation of *Sujet 23 — Introduction to Depth Estimation and 3D Vision*
from the course brief, built as a reproducible pipeline rather than a notebook.

> Status: **M0 (scaffold + data access) complete.** Roadmap below.

## Why not Kaggle

The brief suggested *Middlebury Stereo / KITTI Stereo (Kaggle)*. Both are available from
their original sources over plain HTTP, with **no login, no API key, no gated form** —
which is also the only way to get the accompanying `calib.txt` geometry:

| Dataset | Access | Size |
|---|---|---|
| **Middlebury Stereo 2014** (primary) | direct per-scene directory | ~58 MB/scene |
| Middlebury Stereo 2005 (tutorial) | direct `ALL-2views.zip` | 5 MB |
| KITTI Scene Flow / Stereo 2012 | direct S3 zip | 1.7 / 2.0 GB |
| `skimage.data.stereo_motorcycle()` | already local | 0 MB |

Each Middlebury scene ships `im0.png`, `im1.png`, `disp0.pfm`, `disp1.pfm` and
`calib.txt` (focal length in pixels, metric baseline, and the disparity offset `doffs`).

Two variants are published per scene:

- `-perfect` — correctly rectified (`dyavg = 0`), used to evaluate depth quality;
- `-imperfect` — deliberately mis-rectified (`dyavg ≈ 0.45 … 0.92 px`, `dymax` up to
  2.7 px), used to *demonstrate* the rectification stage on real data.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # opencv-contrib-python (needs cv2.ximgproc)
```

## Quick start

```bash
make data     # a starter subset (~700 MB total for the full set): no login required
make test     # offline unit + smoke tests (uses skimage.data.stereo_motorcycle)
make demo     # disparity + depth on one Middlebury scene, JSON summary on stdout
```

Manual usage:

```bash
python -m stereo_depth.cli info      --scene data/raw/Adirondack-perfect
python -m stereo_depth.cli dy        --scene data/raw/Adirondack-imperfect --scale 0.5
python -m stereo_depth.cli disparity --scene data/raw/Adirondack-perfect --scale 0.5
```

## The four stages

| Stage | Module | What it does |
|---|---|---|
| 1. Preprocess | `preprocess.py`, `calibrate.py` | measure residual misalignment (`dy`), rectify, rescale |
| 2. Block matching | `matcher.py` | `StereoBM` / `StereoSGBM`, LRC + speckle filtering |
| 3. Triangulation | `depth.py` | `Z = f·B/(d − doffs)`, `reprojectImageTo3D`, point clouds |
| 4. Recognition | `objects.py` | per-object distance (robust median over a region) |

### The one formula that matters

```
Z = f · B / (d + doffs)
X = (u − cx) · Z / f        Y = (v − cy) · Z / f
```

`doffs = cx1 − cx0` is the x-offset between the two principal points of the rectified
pair, published in `calib.txt`. It is **added** to the disparity (this is the formula
stated in the Middlebury dataset description). Ignoring it biases every depth by a large
constant factor — about +139 % on `Adirondack-perfect` — so it is a required field of
`Calib` and is covered by `tests/test_depth_math.py`.

Rescaling the working resolution scales `f`, `cx/cy`, `doffs` and the disparity range —
**never** the metric baseline. `Calib.scaled()` is the only place allowed to do this, and
`test_depth_invariant_to_working_scale` enforces that `Z` does not depend on the scale.

## Measured results (M0, starter subset)

Phase-1 alignment check, `python -m stereo_depth.cli dy`, reported in full-resolution
pixels so it is directly comparable to the published `dyavg` / `dymax`:

| Scene | published `dyavg` | published `dymax` | measured `dyavg` | measured `dymax` |
|---|---|---|---|---|
| `Adirondack-perfect` | 0.000 | 0.000 | **0.057** | 0.156 |
| `Adirondack-imperfect` | 0.735 | 1.694 | **0.550** | 1.503 |
| `Backpack-imperfect` | 0.649 | 2.697 | 0.081 | 0.157 |

Depth sanity check on `Adirondack-perfect` at half resolution (SGBM, `blockSize=5`,
`numDisparities=144`): 76 % of pixels get a valid disparity, median depth **2.21 m**
(p10–p90: 1.91–2.87 m) — physically consistent with a chair against a wall.

### Known limitations

- The image-content estimator has a **noise floor of roughly 0.1 px** on clean scenes and
  fails to resolve small misalignments on **repetitive texture** (`Backpack-imperfect`
  reads 0.08 px instead of 0.65 px). Sub-pixel `dy` alone is therefore *not* a reliable
  single criterion for phase 1: M1 will also measure the impact on disparity quality
  (BP-2.0 before vs after corrective rectification), which is what actually matters.
- Corruptions with a *known* answer (injected rotation / shift) are recovered correctly
  and are unit-tested in `tests/test_rectification.py`.

## Repository layout

```
src/stereo_depth/     pipeline modules (config, io, preprocess, matcher, depth, objects)
scripts/fetch_data.py resumable downloader with sha256 manifest (no auth)
tests/                unit tests + offline smoke test
notebooks/            exploration and results analysis
demo/                 live OpenCV demo with parameter sliders
docs/                 French report (RAPPORT.md → RAPPORT.pdf)
```

## Evaluation protocol

Scenes are split so that tuning cannot contaminate the reported numbers:

| Split | Scenes | Use |
|---|---|---|
| `tune` | 6 `-perfect` | hyperparameter grid search |
| `val` | 4 `-perfect` | final parameter selection |
| `test` | 4 `-perfect` | reported numbers |
| `rect` | 4 `-imperfect` | phase 1: `dy` before/after re-rectification |

Metrics: bad-pixel rate BP-1.0 / BP-2.0 (on non-occluded valid pixels), depth RMSE/MAE,
relative error per distance band, and FPS on CPU.

## Roadmap

- [x] **M0** scaffold, dataset access, offline smoke test
- [ ] **M1** rectification: `dy` estimator + corrective ECC/Hough rectification, checkerboard calibration
- [ ] **M2** `StereoBM` vs `StereoSGBM` + post-filters (LRC, speckle, optional WLS)
- [ ] **M3** triangulation, PLY export, closed-form vs `reprojectImageTo3D` agreement
- [ ] **M4** parameter grid search, best config frozen in `config.py`
- [ ] **M5** object distances, evaluation report, KITTI adapter
- [ ] **M6** live demo, French report (PDF), final results

## Attribution

Middlebury Stereo datasets: D. Scharstein, H. Hirschmüller, Y. Kitajima, G. Krathwohl,
N. Nesic, X. Wang, P. Westling, *High-resolution stereo datasets with subpixel-accurate
ground truth*, GCPR 2014. KITTI: A. Geiger et al. Datasets are downloaded at runtime and
never committed to this repository.
