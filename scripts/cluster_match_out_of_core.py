# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster matching in memory versus out of core, across corpus scales.

The background-floor matcher is a self-join: every descriptor is both an index
entry and a query. So a file-backed index is not on its own enough to take it out
of core — holding all N queries would defeat the point. The out-of-core path
therefore reads each query descriptor back out of the `.kdf` it is querying, uses
it, and drops it, visiting rows in stored order so a bounded cache serves most of
the reads.

What that leaves resident is the cache budget plus the `N x (d + 1)` neighbour
table the clustering stage consumes — not the corpus, and not the forest. The
clustering stage itself never looks at a descriptor, only at neighbour indexes and
distances, which is what makes the split possible at all.

This runs both paths at increasing corpus sizes and compares peak memory, wall
time, and — the check that makes the timings meaningful — that the two produce
identical clusters.

Run:
    pixi run python scripts/cluster_match_out_of_core.py --workspace WS
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
import json
import threading
import time
from pathlib import Path

import numpy as np

from sfmtool._sfmtool import build_profile
from sfmtool._sfmtool.matching import (
    background_floor_clusters,
    background_floor_clusters_kdf,
)
from sfmtool._sfmtool.spatial import KdForest, write_kdf
from sfmtool.sift.file import SiftReader

KIB = 1 << 10
MIB = 1 << 20
GIB = 1 << 30


class _Counters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_GET_MEMORY_INFO = ctypes.WinDLL("psapi").GetProcessMemoryInfo
_GET_MEMORY_INFO.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
_GET_MEMORY_INFO.restype = wintypes.BOOL


def current_rss() -> int:
    """Resident working set of this process right now, in bytes.

    `argtypes` matters: without it ctypes truncates the struct pointer on 64-bit
    and the call quietly reports zero.
    """
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    if not _GET_MEMORY_INFO(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return 0
    return int(counters.WorkingSetSize)


class PeakSampler:
    """Highest resident set seen while a block of work runs.

    The process-wide peak counter is monotonic, so it cannot separate one stage
    from an earlier, larger one — sampling the current value does. 5 ms is short
    against stages that run for seconds and cheap enough not to perturb them.
    """

    def __init__(self, interval: float = 0.005) -> None:
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakSampler":
        self.peak = current_rss()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, current_rss())

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak = max(self.peak, current_rss())


def load_corpus(features: Path, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Descriptors plus CSR image offsets, image by image."""
    files = sorted(features.glob("*.sift"))
    blocks, starts = [], [0]
    for path in files:
        reader = SiftReader(path)
        desc = reader.read_descriptors()
        reader.close()
        blocks.append(desc)
        starts.append(starts[-1] + len(desc))
        if limit is not None and starts[-1] >= limit:
            break
    return np.vstack(blocks), np.asarray(starts, dtype=np.uint32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--features")
    p.add_argument("--scratch", required=True)
    p.add_argument(
        "--scales",
        default="50000,200000,800000",
        help="comma-separated corpus sizes (descriptors)",
    )
    p.add_argument("--trees", type=int, default=4)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--d", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--block-bytes", type=int, default=4 * KIB)
    p.add_argument("--chunk-bytes", type=int, default=1 * MIB)
    p.add_argument("--cache-bytes", type=int, default=256 * MIB)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--out")
    args = p.parse_args()

    if build_profile() != "release":
        raise SystemExit("refusing to benchmark a debug build of _sfmtool")

    workspace = Path(args.workspace)
    features = Path(args.features) if args.features else None
    if features is None:
        roots = sorted((workspace / "images/features").glob("sift-*")) or sorted(
            (workspace / "frames/features").glob("sift-*")
        )
        if not roots:
            raise SystemExit(f"no sift-* directory under {workspace}")
        features = roots[0]

    scales = [int(s) for s in args.scales.split(",")]
    out = Path(args.scratch)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    print(
        f"{'N':>10} {'images':>7} {'kdf MB':>8} | {'build s':>12} {'mem s':>9}"
        f" {'peak MB':>9} | {'kdf s':>12} {'peak MB':>9} | {'us/q mem':>6} {'us/q kdf':>7}"
        f" | {'clusters':>9} {'same':>5}"
    )
    for target in scales:
        descriptors, starts = load_corpus(features, target)
        n = len(descriptors)
        n_images = len(starts) - 1

        # ── In-memory path: corpus resident, forest built each run.
        with PeakSampler() as sampler:
            t = time.perf_counter()
            eager = background_floor_clusters(
                descriptors,
                starts,
                d=args.d,
                alpha=args.alpha,
                num_trees=args.trees,
                leaf_size=args.leaf_size,
                max_leaf_checks=args.budget,
                seed=args.seed,
            )
            mem_match = time.perf_counter() - t
        mem_peak = sampler.peak

        # ── Export the same forest, so both paths match against one index.
        forest = KdForest(
            descriptors,
            num_trees=args.trees,
            leaf_size=args.leaf_size,
            max_leaf_checks=args.budget,
            seed=args.seed,
        )
        t = time.perf_counter()
        path = out / f"corpus{n}.kdf"
        if path.exists():
            path.unlink()
        write_kdf(
            forest,
            str(path),
            layout="shared",
            chunk_bytes=args.chunk_bytes,
            descriptor_block_bytes=args.block_bytes,
        )
        build = time.perf_counter() - t
        kdf_mb = path.stat().st_size / 1e6
        del forest

        # ── Out-of-core path: drop the corpus first, so its memory cannot be
        # credited to the file-backed run. Only image_starts survives.
        descriptors_bytes = descriptors.nbytes
        del descriptors
        with PeakSampler() as sampler:
            t = time.perf_counter()
            lazy = background_floor_clusters_kdf(
                str(path),
                starts,
                d=args.d,
                alpha=args.alpha,
                max_leaf_checks=args.budget,
                cache_bytes=args.cache_bytes,
                query_workers=args.workers,
            )
            kdf_match = time.perf_counter() - t
        kdf_peak = sampler.peak

        same = all(np.array_equal(a, b) for a, b in zip(eager, lazy))
        for expected, actual in zip(eager, lazy, strict=True):
            np.testing.assert_array_equal(actual, expected)
        clusters = len(eager[0]) - 1
        rows.append(
            {
                "n": n,
                "images": n_images,
                "corpus_bytes": int(descriptors_bytes),
                "kdf_bytes": int(path.stat().st_size),
                "memory_match_seconds": mem_match,
                "memory_peak_bytes": mem_peak,
                "kdf_build_seconds": build,
                "kdf_match_seconds": kdf_match,
                "kdf_peak_bytes": kdf_peak,
                "clusters": clusters,
                "identical": bool(same),
            }
        )
        print(
            f"{n:>10,} {n_images:>7} {kdf_mb:>8.1f} | {build:>12.1f} {mem_match:>9.1f}"
            f" {mem_peak / MIB:>9.0f} | {kdf_match:>12.1f} {kdf_peak / MIB:>9.0f}"
            f" | {mem_match / n * 1e6:>6.1f} {kdf_match / n * 1e6:>7.1f}"
            f" | {clusters:>9,} {'yes' if same else 'NO':>5}"
        )
        if not args.keep:
            path.unlink()

    if args.out:
        Path(args.out).write_text(json.dumps({"results": rows}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
