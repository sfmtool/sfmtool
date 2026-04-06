#!/usr/bin/env python
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark advect_points performance at various keypoint counts.

Usage:
    pixi run python scripts/benchmark_advect.py
"""

import time

import numpy as np

from sfmtool._sfmtool import advect_points


def main():
    # Create a synthetic flow field (1920x1080)
    h, w = 1080, 1920
    rng = np.random.default_rng(42)
    flow_u = rng.standard_normal((h, w)).astype(np.float32) * 5.0
    flow_v = rng.standard_normal((h, w)).astype(np.float32) * 5.0

    counts = [1_000, 10_000, 100_000, 1_000_000]
    runs = 20

    print(f"Flow field: {w}x{h}")
    print(f"Runs per count: {runs}")
    print()
    print(f"{'N points':>12}  {'median':>10}  {'min':>10}  {'max':>10}")
    print("-" * 50)

    for n in counts:
        # Random points within the flow field
        points = np.column_stack([
            rng.uniform(0, w, size=n).astype(np.float32),
            rng.uniform(0, h, size=n).astype(np.float32),
        ])

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            advect_points(points, flow_u, flow_v)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        times.sort()
        median = times[len(times) // 2]
        lo = times[0]
        hi = times[-1]

        def fmt(t):
            if t < 0.001:
                return f"{t * 1_000_000:.0f} us"
            return f"{t * 1000:.2f} ms"

        print(f"{n:>12,}  {fmt(median):>10}  {fmt(lo):>10}  {fmt(hi):>10}")


if __name__ == "__main__":
    main()
