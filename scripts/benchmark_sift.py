#!/usr/bin/env python
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark SIFT feature extraction throughput across backends.

Measures the end-to-end extraction cost (decode + detect + describe, as each
backend reads the image files itself) that the ``sfm sift --extract`` pipeline
pays, excluding only the ``.sift`` write. These fixed scenarios track sfmtool's
throughput against COLMAP across both image scales.

Datasets (checked-in, selected with --dataset; default runs both):
    - ``small``: seoul_bull_sculpture (17 @ 270x480) -- scenarios 1 / 17 / 100
      images (100 = the 17 copied cyclically into a temp dir). Tiny images where
      per-image rayon can't saturate the cores, so cross-image concurrency wins.
    - ``large``: dino_dog_toy (85 @ 2040x1536) -- scenarios 1 / 17 images. High
      resolution where the extract dominates and decode/save overhead is small.

Backends: ``sfmtool``, ``colmap-cpu`` (COLMAP SIFT, ``use_gpu=False``),
``colmap-gpu`` (COLMAP SiftGPU, ``use_gpu=True`` -- on a host without a real GPU
this runs on *software* OpenGL and is not a CPU measurement), and ``opencv``.
The default compares all three CPU-relevant paths so the GPU-path confound is
visible rather than hidden behind ``get_colmap_feature_options()``'s
``use_gpu=True`` default.

Usage:
    pixi run -e test python scripts/benchmark_sift.py
    pixi run -e test python scripts/benchmark_sift.py --backends sfmtool,colmap-cpu
    pixi run -e test python scripts/benchmark_sift.py --dataset large
    pixi run -e test python scripts/benchmark_sift.py --sizes 1,17 --runs 7
"""

import argparse
import contextlib
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_ROOT = REPO_ROOT / "test-data" / "images"

# Benchmark datasets. ``small`` (seoul_bull) is the original tiny-image case
# where per-image rayon can't saturate the cores; ``large`` (dino) is a
# high-resolution case where the extract dominates and decode/save overhead is a
# smaller slice. Each carries its own default scenario sizes (overridable with
# --sizes); large-image batches are kept modest because every backend -- COLMAP
# especially -- is far slower per image there.
DATASETS = {
    "small": {
        "dir": "seoul_bull_sculpture",
        "glob": "seoul_bull_sculpture_*.jpg",
        "sizes": [1, 17, 100],
    },
    "large": {
        "dir": "dino_dog_toy",
        "glob": "dino_dog_toy_*.jpg",
        "sizes": [1, 17],
    },
}


def check_build_profile(
    profile: str, timing_sfmtool: bool, *, allow_debug: bool
) -> None:
    """Refuse a debug-built sfmtool extension unless explicitly allowed.

    A debug build of the Rust SIFT kernels runs ~10-15x slower than ``--release``
    (rebuild with ``pixi run maturin develop --release``). Comparing those
    timings against COLMAP -- a prebuilt, always-optimized C++ library -- makes
    sfmtool look catastrophically slow for reasons that have nothing to do with
    the algorithm, which is exactly the trap this guard exists to prevent.
    """
    if not timing_sfmtool or profile == "release":
        return
    msg = (
        f"sfmtool extension is a {profile.upper()} build -- SIFT timings will be "
        "~10-15x too slow and are not comparable to release or to COLMAP.\n"
        "Rebuild with:  pixi run maturin develop --release"
    )
    if not allow_debug:
        sys.exit(f"ERROR: {msg}\n(pass --allow-debug to benchmark it anyway)")
    print(
        f"WARNING: {msg}\n(continuing because --allow-debug was given)\n",
        file=sys.stderr,
    )


def dataset_images(spec: dict) -> list[Path]:
    """The checked-in jpgs for one dataset spec, sorted by name."""
    directory = IMAGES_ROOT / spec["dir"]
    images = sorted(directory.glob(spec["glob"]))
    if not images:
        sys.exit(f"No images found in {directory}")
    return images


def make_image_set(images: list[Path], count: int, workdir: Path) -> list[Path]:
    """Return ``count`` image paths, copying cyclically into ``workdir`` if needed.

    For ``count <= len(images)`` this is just the first ``count`` originals; for
    larger counts the originals are copied under fresh names so every backend
    (including COLMAP, which keys images by name) sees ``count`` distinct files.
    """
    if count <= len(images):
        return images[:count]

    workdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        src = images[i % len(images)]
        dst = workdir / f"img_{i:04d}{src.suffix}"
        if not dst.exists():
            shutil.copyfile(src, dst)
        paths.append(dst)
    return paths


def get_backend(name: str):
    """Resolve a backend name to ``(extract_fn, feature_options, label)``.

    ``colmap-gpu`` selects COLMAP's SiftGPU path (``use_gpu=True``); on a host
    without a real GPU this runs on software OpenGL, so it is *not* a CPU SIFT
    measurement. ``colmap`` / ``colmap-cpu`` select COLMAP's CPU SIFT
    (``use_gpu=False``), the like-for-like comparison against sfmtool and OpenCV.
    """
    if name == "sfmtool":
        from sfmtool.sift.extract_sfmtool import (
            extract_sift_with_sfmtool,
            get_default_sfmtool_feature_options,
        )

        return (
            extract_sift_with_sfmtool,
            get_default_sfmtool_feature_options(),
            "sfmtool",
        )
    if name in ("colmap", "colmap-cpu", "colmap-gpu"):
        from sfmtool.sift.extract_colmap import (
            extract_sift_with_colmap,
            get_colmap_feature_options,
        )

        use_gpu = name == "colmap-gpu"
        return (
            extract_sift_with_colmap,
            get_colmap_feature_options(use_gpu=use_gpu),
            f"colmap({'gpu' if use_gpu else 'cpu'})",
        )
    if name == "opencv":
        from sfmtool.sift.extract_opencv import (
            extract_sift_with_opencv,
            get_default_opencv_feature_options,
        )

        return (
            extract_sift_with_opencv,
            get_default_opencv_feature_options(),
            "opencv",
        )
    sys.exit(
        f"Unknown backend: {name!r} "
        f"(choose from sfmtool, colmap-cpu, colmap-gpu, opencv)"
    )


@contextlib.contextmanager
def silenced(active: bool):
    """Redirect fds 1 and 2 to /dev/null so backend chatter stays out of the table.

    Uses os-level ``dup2`` (not ``contextlib.redirect_stdout``) so it also
    swallows COLMAP's C++ glog output, which is written straight to fd 2.
    """
    if not active:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def run_once(extract_fn, paths, feature_options, quiet: bool) -> tuple[float, int]:
    """Run one full extraction over ``paths``; return (seconds, total_features).

    The result is fully materialized (the sfmtool backend yields a generator) so
    all decode + extract work is forced before the timer stops.
    """
    with silenced(quiet):
        t0 = time.perf_counter()
        results = list(extract_fn(paths, feature_options))
        elapsed = time.perf_counter() - t0
    # results[i] = (feature_tool_metadata, metadata, positions, affine, desc, thumb)
    total_features = sum(len(r[2]) for r in results)
    return elapsed, total_features


def benchmark(extract_fn, paths, feature_options, runs: int, quiet: bool) -> dict:
    """Warm up once, then time ``runs`` extractions; return summary stats."""
    _, total_features = run_once(extract_fn, paths, feature_options, quiet)  # warmup
    times = [
        run_once(extract_fn, paths, feature_options, quiet)[0] for _ in range(runs)
    ]
    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "features": total_features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backends",
        default="sfmtool,colmap-cpu,colmap-gpu",
        help="Comma-separated backends to compare "
        "(sfmtool, colmap-cpu, colmap-gpu, opencv).",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *DATASETS],
        help="Which dataset(s) to benchmark: small (270x480), large (2040x1536), "
        "or all (default).",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="Comma-separated image counts, overriding each dataset's defaults "
        "(small: 1,17,100; large: 1,17).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Timed runs per scenario after a warmup (default: 5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Don't suppress backend stdout/stderr during timed runs.",
    )
    parser.add_argument(
        "--allow-debug",
        action="store_true",
        help="Proceed even if the sfmtool extension is a debug build (timings "
        "will be ~10-15x too slow and must not be compared against release).",
    )
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    size_override = (
        [int(s) for s in args.sizes.split(",") if s.strip()] if args.sizes else None
    )
    dataset_names = list(DATASETS) if args.dataset == "all" else [args.dataset]

    from sfmtool._sfmtool import build_profile

    profile = build_profile()
    check_build_profile(profile, "sfmtool" in backends, allow_debug=args.allow_debug)

    print(f"SIFT extraction benchmark  (cpus={os.cpu_count()}, sfmtool={profile})")
    print(f"Backends: {', '.join(backends)}   runs={args.runs}/scenario (+1 warmup)")

    try:
        import cv2
    except Exception:
        cv2 = None

    with tempfile.TemporaryDirectory(prefix="sift_bench_") as tmp:
        tmpdir = Path(tmp)
        # Resolve every backend up front so a missing dependency fails fast.
        resolved = [get_backend(b) for b in backends]

        for name in dataset_names:
            spec = DATASETS[name]
            images = dataset_images(spec)
            sizes = size_override or spec["sizes"]
            if cv2 is not None:
                h, w = cv2.imread(str(images[0])).shape[:2]
                res = f"{w}x{h}"
            else:
                res = "?"

            print()
            print(f"Dataset: {name} ({spec['dir']})  {len(images)} images @ {res}")

            for count in sizes:
                paths = make_image_set(images, count, tmpdir / f"{name}_{count}")
                label = f"{count} image" + ("" if count == 1 else "s")
                print(f"Scenario: {label}")
                print(
                    f"  {'backend':<14}{'median_s':>10}{'per_img_ms':>12}"
                    f"{'img/s':>9}{'features':>10}"
                )
                for extract_fn, feature_options, blabel in resolved:
                    stats = benchmark(
                        extract_fn, paths, feature_options, args.runs, not args.verbose
                    )
                    med = stats["median_s"]
                    per_img_ms = med / count * 1000.0
                    img_per_s = count / med if med > 0 else float("inf")
                    print(
                        f"  {blabel:<14}{med:>10.4f}{per_img_ms:>12.2f}"
                        f"{img_per_s:>9.2f}{stats['features']:>10}"
                    )
                print()


if __name__ == "__main__":
    main()
