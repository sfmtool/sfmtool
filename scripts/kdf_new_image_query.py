# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Fitting new images into an existing capture, from a `.kdf` or from memory.

The realistic question: a capture has been matched, more images arrive, and each
one's descriptors need their nearest neighbours in the existing corpus. This is the
access pattern the file-backed path was built for — a few thousand queries against
a corpus of millions — as opposed to the whole-corpus self-join, where every
descriptor is a query.

What makes it interesting is what each path has to do *before* answering. A
persistent index is opened; an in-memory one has to be rebuilt, which means reading
every `.sift` file and running the build. That cost is unavoidable for a fresh
process and irrelevant for a long-running one, so both are reported.

A selection of images is withheld from the index rather than just the last one, and
they are spread through the sequence. Spread matters for a video capture like
DinoLedge: a withheld frame's immediate neighbours stay in the index, which is the
situation a real arrival is in. Withholding a contiguous run would instead measure
the hardest case, where nothing nearby has been seen.

Run:
    pixi run python scripts/kdf_new_image_query.py --workspace WS --scratch DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from sfmtool._sfmtool import build_profile
from sfmtool._sfmtool.spatial import KdForest, LazyKdForest, read_kdf, write_kdf
from sfmtool.sift.file import SiftReader

KIB = 1 << 10
MIB = 1 << 20


def load_images(
    features: Path, limit: int | None
) -> tuple[list[np.ndarray], list[str]]:
    """Descriptors per image, in filename order."""
    blocks, names, total = [], [], 0
    for path in sorted(features.glob("*.sift")):
        reader = SiftReader(path)
        desc = reader.read_descriptors()
        reader.close()
        blocks.append(desc)
        names.append(path.name.removesuffix(".sift"))
        total += len(desc)
        if limit is not None and total >= limit:
            break
    return blocks, names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--features")
    p.add_argument("--scratch", required=True)
    p.add_argument("--limit", type=int, help="cap the corpus size (descriptors)")
    p.add_argument(
        "--holdout", type=int, default=10, help="images withheld from the index"
    )
    p.add_argument(
        "--contiguous",
        action="store_true",
        help="withhold a consecutive run instead of a spread selection",
    )
    p.add_argument("--trees", type=int, default=4)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--k", type=int, default=11)
    p.add_argument("--budget", type=int, default=128)
    p.add_argument("--block-bytes", type=int, default=2 * KIB)
    p.add_argument("--chunk-bytes", type=int, default=1 * MIB)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--caches", default="64,256,1024", help="cache budgets in MiB to try"
    )
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

    t = time.perf_counter()
    images, names = load_images(features, args.limit)
    read_seconds = time.perf_counter() - t
    if len(images) <= args.holdout:
        raise SystemExit(f"need more than {args.holdout} images; found {len(images)}")

    if args.contiguous:
        start = (len(images) - args.holdout) // 2
        held = list(range(start, start + args.holdout))
    else:
        step = len(images) / args.holdout
        held = sorted(
            {
                min(len(images) - 1, int(i * step + step / 2))
                for i in range(args.holdout)
            }
        )
    held_set = set(held)
    index = np.vstack([d for i, d in enumerate(images) if i not in held_set])
    arrivals = [images[i] for i in held]

    print(
        f"capture {len(index):,} descriptors from {len(images) - len(held)} images"
        f" ({read_seconds:.1f}s to read)"
    )
    print(
        f"withheld {len(held)} {'consecutive' if args.contiguous else 'spread'} images:"
        f" {sum(len(a) for a in arrivals):,} descriptors,"
        f" {statistics.median(len(a) for a in arrivals):.0f} median each\n"
    )

    t = time.perf_counter()
    forest = KdForest(
        index,
        num_trees=args.trees,
        leaf_size=args.leaf_size,
        max_leaf_checks=args.budget,
        seed=args.seed,
    )
    build_seconds = time.perf_counter() - t

    out = Path(args.scratch)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "capture.kdf"
    if path.exists():
        path.unlink()
    t = time.perf_counter()
    write_kdf(
        forest,
        str(path),
        chunk_bytes=args.chunk_bytes,
        descriptor_block_bytes=args.block_bytes,
    )
    export_seconds = time.perf_counter() - t
    file_mb = path.stat().st_size / 1e6
    print(
        f"index: build {build_seconds:.1f}s, export {export_seconds:.1f}s, file {file_mb:.0f} MB"
    )
    print(
        f"a fresh in-memory process pays {read_seconds + build_seconds:.1f}s"
        f" before its first answer; opening the file pays the open below\n"
    )

    reference = [
        forest.query(a, k=args.k, max_leaf_checks=args.budget) for a in arrivals
    ]
    eager = []
    for arrival in arrivals:
        t = time.perf_counter()
        forest.query(arrival, k=args.k, max_leaf_checks=args.budget)
        eager.append(time.perf_counter() - t)

    rows = [
        {
            "path": "memory",
            "read_seconds": read_seconds,
            "build_seconds": build_seconds,
            "per_image_seconds": eager,
        }
    ]
    print(
        f"{'path':>20} {'open ms':>8} {'1st img s':>10} {'later s':>8}"
        f" {'us/desc':>8} {'reads 1st':>10} {'evict':>8} {'match':>6}"
    )
    print(
        f"{'in memory':>20} {'-':>8} {eager[0]:>10.2f} {statistics.median(eager[1:]):>8.2f}"
        f" {statistics.median(eager) / statistics.median(len(a) for a in arrivals) * 1e6:>8.0f}"
        f" {'-':>10} {'-':>8} {'ref':>6}"
    )

    # Loading the file into the in-memory structure: no build, no .sift reads,
    # then full in-memory query speed.
    t = time.perf_counter()
    # The load walks the corpus in storage order, so a modest cache suffices
    # however large the file — it only ever holds the blocks in flight.
    loaded = read_kdf(str(path), cache_bytes=256 * MIB, max_compressed_bytes=64 * MIB)
    load_seconds = time.perf_counter() - t
    loaded_per_image, agree = [], True
    for i, arrival in enumerate(arrivals):
        t = time.perf_counter()
        got = loaded.query(arrival, k=args.k, max_leaf_checks=args.budget)
        loaded_per_image.append(time.perf_counter() - t)
        for actual, expected in zip(got, reference[i], strict=True):
            np.testing.assert_array_equal(actual, expected)
        agree = agree and np.array_equal(got[0], reference[i][0])
    print(
        f"{'kdf -> memory':>20} {load_seconds * 1e3:>8.0f} {loaded_per_image[0]:>10.2f}"
        f" {statistics.median(loaded_per_image[1:]):>8.2f}"
        f" {statistics.median(loaded_per_image) / statistics.median(len(a) for a in arrivals) * 1e6:>8.0f}"
        f" {'-':>10} {'-':>8} {'yes' if agree else 'NO':>6}"
    )
    rows.append(
        {
            "path": "kdf-loaded",
            "load_seconds": load_seconds,
            "per_image_seconds": loaded_per_image,
            "identical": bool(agree),
        }
    )
    del loaded

    for mib in (int(v) for v in args.caches.split(",")):
        budget = mib * MIB
        t = time.perf_counter()
        lazy = LazyKdForest(
            str(path),
            cache_bytes=budget,
            max_in_flight_bytes=budget,
            max_chunk_bytes=min(budget, 4 * MIB),
            # Not tied to the cache budget: this caps a per-entry decode buffer,
            # and the hash directory read at open grows with the block count.
            max_compressed_bytes=64 * MIB,
            query_workers=args.workers,
        )
        open_ms = (time.perf_counter() - t) * 1e3

        per_image, reads_first, same = [], None, True
        for i, arrival in enumerate(arrivals):
            lazy.reset_io_stats()
            t = time.perf_counter()
            got = lazy.query(arrival, k=args.k, max_leaf_checks=args.budget)
            per_image.append(time.perf_counter() - t)
            if i == 0:
                reads_first = lazy.io_stats()["read_calls"]
            for actual, expected in zip(got, reference[i], strict=True):
                np.testing.assert_array_equal(actual, expected)
            same = same and np.array_equal(got[0], reference[i][0])
        io = lazy.io_stats()

        print(
            f"{f'kdf {mib} MiB':>20} {open_ms:>8.0f} {per_image[0]:>10.2f}"
            f" {statistics.median(per_image[1:]):>8.2f}"
            f" {statistics.median(per_image) / statistics.median(len(a) for a in arrivals) * 1e6:>8.0f}"
            f" {reads_first:>10,} {io['evictions']:>8,} {'yes' if same else 'NO':>6}"
        )
        rows.append(
            {
                "path": f"kdf-{mib}MiB",
                "cache_bytes": budget,
                "open_seconds": open_ms / 1e3,
                "per_image_seconds": per_image,
                "reads_first_image": reads_first,
                "evictions_last_image": io["evictions"],
                "io_last_image": io,
                "identical": bool(same),
            }
        )
        del lazy

    if not args.keep:
        path.unlink()
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "capture_descriptors": int(len(index)),
                    "capture_images": len(images) - len(held),
                    "withheld": [names[i] for i in held],
                    "contiguous": args.contiguous,
                    "k": args.k,
                    "budget": args.budget,
                    "file_bytes": int(file_mb * 1e6),
                    "results": rows,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
