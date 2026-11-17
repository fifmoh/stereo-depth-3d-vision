"""Download the Middlebury Stereo datasets without any account, API key or login.

Everything here is plain HTTP/1.1 with a browser User-Agent: the Middlebury server
returns 403 to clients that do not send one. Files are written atomically, resumable,
and recorded in ``data/manifest.json`` with a sha256 so later runs can verify integrity.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

BASE = "https://vision.middlebury.edu/stereo/data"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) stereo-depth-3d-vision/0.1"
SCENE_FILES = ("im0.png", "im1.png", "disp0.pfm", "disp1.pfm", "calib.txt")
CHUNK = 1 << 20


@dataclass(frozen=True)
class Remote:
    """A single downloadable file."""

    url: str
    dest: Path


def scene_files(scene: str, grade: str, root: Path) -> list[Remote]:
    folder = f"{scene}-{grade}"
    base = f"{BASE}/scenes2014/datasets/{folder}"
    return [Remote(f"{base}/{name}", root / folder / name) for name in SCENE_FILES]


def tutorial_2005(root: Path) -> list[Remote]:
    return [
        Remote(
            f"{BASE}/scenes2005/ThirdSize/zip-2views/ALL-2views.zip",
            root / "scenes2005" / "ALL-2views.zip",
        )
    ]


def _download(remote: Remote, verify_sha: bool, manifest: dict) -> dict:
    dest = remote.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    key = str(dest)

    if dest.exists():
        entry = manifest.get(key, {})
        if verify_sha and entry.get("sha256") and _sha256(dest) == entry["sha256"]:
            print(f"  ok (cached)   {dest}")
            return entry
        if not verify_sha:
            print(f"  ok (cached)   {dest}")
            return entry or {"size": dest.stat().st_size}
        print(f"  re-download   {dest} (sha256 mismatch)")

    request = urllib.request.Request(remote.url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=f"  dl {dest.name}",
                leave=False,
            ) as bar:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    bar.update(len(chunk))
            tmp.replace(dest)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code} for {remote.url}\n  (a browser User-Agent is required)")
    except urllib.error.URLError as exc:
        raise SystemExit(f"network error for {remote.url}: {exc.reason}")

    entry = {"url": remote.url, "size": dest.stat().st_size, "sha256": _sha256(dest)}
    print(f"  saved         {dest}")
    return entry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _free_space(path: Path) -> str:
    usage = shutil.disk_usage(path)
    return f"{usage.free / 1e9:.1f} GB free"


def build_plan(args) -> list[Remote]:
    root: Path = args.dest
    remotes: list[Remote] = []

    perfect = args.scene or []
    imperfect = args.imperfect or []

    if args.split:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from stereo_depth.config import SPLIT_RECT, SPLIT_TEST, SPLIT_TUNE, SPLIT_VAL

        splits = {"tune": SPLIT_TUNE, "val": SPLIT_VAL, "test": SPLIT_TEST, "rect": SPLIT_RECT}
        for name in args.split:
            if name not in splits:
                raise SystemExit(f"unknown split {name!r}, expected {sorted(splits)}")
            if name == "rect":
                imperfect.extend(splits[name])
            else:
                perfect.extend(splits[name])

    for scene in dict.fromkeys(perfect):
        remotes.extend(scene_files(scene, "perfect", root))
    for scene in dict.fromkeys(imperfect):
        remotes.extend(scene_files(scene, "imperfect", root))
    if args.tutorial:
        remotes.extend(tutorial_2005(root))
    return remotes


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Middlebury Stereo 2014 scenes (no login, no API key)."
    )
    parser.add_argument("--dest", type=Path, default=Path("data/raw"), help="download root")
    parser.add_argument("--scene", action="append", default=[], help="'-perfect' scene name")
    parser.add_argument(
        "--imperfect", action="append", default=[], help="'-imperfect' scene name (for phase 1)"
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        choices=["tune", "val", "test", "rect"],
        help="download a predefined split instead of listing scenes",
    )
    parser.add_argument("--tutorial", action="store_true", help="also fetch the 2005 sample archive")
    parser.add_argument("--no-verify", action="store_true", help="skip sha256 verification")
    args = parser.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    remotes = build_plan(args)
    if not remotes:
        parser.error("nothing to do: pass --scene, --imperfect or --split")

    manifest_path = args.dest / "manifest.json"
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    print(f"destination: {args.dest}  ({_free_space(args.dest)})")
    print(f"files: {len(remotes)}")
    for remote in remotes:
        manifest[str(remote.dest)] = _download(remote, not args.no_verify, manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"done. manifest: {manifest_path} ({len(manifest)} entries)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
