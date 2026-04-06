#!/usr/bin/env python
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark flow matching pipeline throughput.

Measures the end-to-end flow matching over a short image sequence,
including optical flow computation, flow composition, and feature matching.

Requires a workspace with extracted SIFT features. Set one up with:
    bash scripts/init_dataset_dino_dog_toy.sh
    pixi run sfm init --dsp dino_dog_toy_ws
    pixi run sfm sift --extract dino_dog_toy_ws/images/*.jpg

Usage:
    pixi run python scripts/benchmark_flow_matching.py [image_dir] [options]

Examples:
    pixi run python scripts/benchmark_flow_matching.py
    pixi run python scripts/benchmark_flow_matching.py dino_dog_toy_ws/images --n-images 20
    pixi run python scripts/benchmark_flow_matching.py dino_dog_toy_ws/images --preset fast
"""

import argparse
import glob
import sys
import time
from pathlib import Path


def find_sift_dir(image_dir: Path) -> Path | None:
    """Find the sift feature directory for an image directory."""
    features_dir = image_dir / "features"
    if not features_dir.is_dir():
        return None
    for d in sorted(features_dir.iterdir()):
        if d.is_dir() and d.name.startswith("sift-"):
            return d
    return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark flow matching pipeline")
    parser.add_argument(
        "image_dir",
        nargs="?",
        default="dino_dog_toy_ws/images",
        help="Directory containing images (default: dino_dog_toy_ws/images)",
    )
    parser.add_argument(
        "--preset",
        default="high_quality",
        choices=["fast", "default", "high_quality"],
    )
    parser.add_argument(
        "--n-images", type=int, default=20, help="Number of images to use (default: 20)"
    )
    parser.add_argument(
        "--window-size", type=int, default=5, help="Flow window size (default: 5)"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start index in image list (default: 0)"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of benchmark runs (default: 3)"
    )
    parser.add_argument(
        "--trace", type=str, default=None,
        help="Write Perfetto trace JSON to this path (only for the first run)",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"Error: {image_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find images
    images = sorted(
        glob.glob(str(image_dir / "*.jpg"))
        + glob.glob(str(image_dir / "*.png"))
    )
    if len(images) < 2:
        print(f"Error: Need at least 2 images in {image_dir}", file=sys.stderr)
        sys.exit(1)

    # Find sift features
    sift_dir = find_sift_dir(image_dir)
    if sift_dir is None:
        print(f"Error: No sift feature directory found in {image_dir}/features/", file=sys.stderr)
        print("Run: pixi run sfm sift --extract <image_dir>", file=sys.stderr)
        sys.exit(1)

    # Select subset
    end = min(args.start + args.n_images, len(images))
    images = images[args.start:end]
    n = len(images)
    if n < 2:
        print(f"Error: Need at least 2 images in range [{args.start}, {end})", file=sys.stderr)
        sys.exit(1)

    image_paths = [Path(p) for p in images]
    sift_paths = [sift_dir / (Path(p).name + ".sift") for p in images]

    # Verify sift files exist
    for sp in sift_paths:
        if not sp.exists():
            print(f"Error: Missing sift file: {sp}", file=sys.stderr)
            sys.exit(1)

    # Import after arg parsing for fast --help
    import cv2
    from sfmtool._sfmtool import compute_optical_flow, gpu_available
    from sfmtool.feature_match._flow_matching import flow_match_sequential

    # Print setup info
    img = cv2.imread(str(images[0]))
    h, w = img.shape[:2]
    print(f"Images: {n} (frames {args.start} to {end - 1})")
    print(f"Resolution: {w}x{h} ({w*h:,} pixels)")
    print(f"Preset: {args.preset}")
    print(f"Window size: {args.window_size}")
    print(f"GPU available: {gpu_available()}")
    print()

    # Warmup: run one flow computation to initialize GPU
    if gpu_available():
        print("Warming up GPU...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(str(images[1]))
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        compute_optical_flow(gray, gray2, preset=args.preset)
        print()

    # Benchmark
    n_pairs = n - 1
    run_times = []
    run_match_counts = []

    for run_idx in range(args.runs):
        print(f"--- Run {run_idx + 1}/{args.runs} ---")
        t0 = time.perf_counter()
        trace_path = Path(args.trace) if args.trace and run_idx == 0 else None
        matches = flow_match_sequential(
            image_paths=image_paths,
            sift_paths=sift_paths,
            preset=args.preset,
            window_size=args.window_size,
            trace_path=trace_path,
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0
        run_times.append(elapsed)
        total_matches = sum(len(m) for m in matches.values())
        run_match_counts.append(total_matches)
        print(f"  Time: {elapsed:.2f}s  |  {len(matches)} pairs  |  {total_matches} matches")
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Images: {n}, Resolution: {w}x{h}, Preset: {args.preset}")
    print()
    median_time = sorted(run_times)[len(run_times) // 2]
    for i, (t, mc) in enumerate(zip(run_times, run_match_counts)):
        print(f"  Run {i+1}: {t:.2f}s total, {t / n_pairs:.2f}s/pair, {mc} matches")
    print(f"  Median: {median_time:.2f}s total, {median_time / n_pairs:.2f}s/pair")


if __name__ == "__main__":
    main()
