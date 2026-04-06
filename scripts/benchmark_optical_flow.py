#!/usr/bin/env python
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark GPU vs CPU optical flow performance.

Usage:
    pixi run python scripts/benchmark_optical_flow.py [image_dir_or_glob] [--preset PRESET] [--runs N]

Examples:
    pixi run python scripts/benchmark_optical_flow.py
    pixi run python scripts/benchmark_optical_flow.py test-data/images/dino_dog_toy --preset fast
    pixi run python scripts/benchmark_optical_flow.py "test-data/images/seoul_bull_sculpture/seoul_bull_*.jpg" --runs 10
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

from sfmtool._sfmtool import compute_optical_flow, gpu_available


def load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def benchmark(img0, img1, preset, use_gpu, n_runs):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        compute_optical_flow(img0, img1, preset=preset, use_gpu=use_gpu)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU vs CPU optical flow")
    parser.add_argument(
        "image_dir",
        nargs="?",
        default="test-data/images/dino_dog_toy",
        help="Directory containing images or glob pattern (default: test-data/images/dino_dog_toy)",
    )
    parser.add_argument(
        "--preset",
        default="high_quality",
        choices=["fast", "default", "high_quality"],
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument(
        "--frame-a", type=int, default=0, help="Index of first image (default: 0)"
    )
    parser.add_argument(
        "--frame-b", type=int, default=1, help="Index of second image (default: 1)"
    )
    args = parser.parse_args()

    # Support both a directory of images and a glob pattern
    if os.path.isdir(args.image_dir):
        images = sorted(
            glob.glob(os.path.join(args.image_dir, "*.jpg"))
            + glob.glob(os.path.join(args.image_dir, "*.png"))
        )
    else:
        images = sorted(glob.glob(args.image_dir))
    if len(images) < 2:
        print(f"Error: Need at least 2 images in {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    if args.frame_a >= len(images) or args.frame_b >= len(images):
        print(
            f"Error: Frame indices out of range (have {len(images)} images)",
            file=sys.stderr,
        )
        sys.exit(1)

    img0 = load_grayscale(images[args.frame_a])
    img1 = load_grayscale(images[args.frame_b])
    h, w = img0.shape

    print(f"Images: {os.path.basename(images[args.frame_a])} -> "
          f"{os.path.basename(images[args.frame_b])}")
    print(f"Resolution: {w}x{h} ({w*h:,} pixels)")
    print(f"Preset: {args.preset}")
    print(f"GPU available: {gpu_available()}")
    print()

    # Warmup GPU
    if gpu_available():
        print("Warming up GPU...")
        compute_optical_flow(img0, img1, preset=args.preset, use_gpu=True)
        print()

    # GPU benchmark
    if gpu_available():
        print(f"GPU benchmark ({args.runs} runs):")
        gpu_times = benchmark(img0, img1, args.preset, True, args.runs)
        for i, ms in enumerate(gpu_times):
            print(f"  Run {i+1}: {ms:.1f}ms")
        gpu_median = sorted(gpu_times)[len(gpu_times) // 2]
        print(f"  Median: {gpu_median:.1f}ms")
        print(f"  Min:    {min(gpu_times):.1f}ms")
        print()

    # CPU benchmark
    print(f"CPU benchmark ({args.runs} runs):")
    cpu_times = benchmark(img0, img1, args.preset, False, args.runs)
    for i, ms in enumerate(cpu_times):
        print(f"  Run {i+1}: {ms:.1f}ms")
    cpu_median = sorted(cpu_times)[len(cpu_times) // 2]
    print(f"  Median: {cpu_median:.1f}ms")
    print(f"  Min:    {min(cpu_times):.1f}ms")
    print()

    # Summary
    if gpu_available():
        print(f"Speedup: {cpu_median/gpu_median:.1f}x (GPU {gpu_median:.0f}ms vs CPU {cpu_median:.0f}ms)")


if __name__ == "__main__":
    main()
