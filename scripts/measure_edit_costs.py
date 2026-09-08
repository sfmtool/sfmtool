# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Measure what an in-viewer point edit would cost on a real reconstruction.

Step 1 of the editing plan in ``specs/drafts/sfm-explorer-editing.md`` asks for
three numbers before the design is settled: how big a reconstruction is and how
much of that is the heavy columns (patch bitmaps and thumbnails), what a full
clone of one costs, and what it costs to materialise a base plus a handful of
point edits into a plain CSR value again. Those set the history memory budget,
the materialisation threshold, and whether two bases should share their heavy
columns.

Usage::

    pixi run -e dev python scripts/measure_edit_costs.py <file.sfmr> \
        [--edits N] [--repeat K] [--new M] [--seed S]

The materialisation modelled here is the one
``specs/drafts/sfm-explorer-editing-overlay.md`` describes: ``--edits N`` points
are deleted from the base and re-added as additions with one extra observation
each, a further ``N // 2`` points are deleted outright, and ``--new M`` points
are genuinely new. Materialising puts each modified point back at the base index
it replaced, closes the slots the outright deletions left (a monotone shift by a
prefix count of deletions), appends the new points, then re-sorts the
observations by ``(point, image)`` and rebuilds the CSR offsets. The re-sort is
timed on its own, since it is the dominant term.

The result is verified before it is reported: the CSR offsets agree with the
counts and with the sorted point column, every modified point carries its extra
observation, and the row map is a bijection from the surviving base indexes onto
the materialised ones.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from sfmtool._sfmtool.io import read_sfmr

# The two columns that dominate memory by an order of magnitude over everything
# else, and the reason the umbrella draft asks whether bases should share them.
HEAVY_COLUMNS = ("patch_bitmaps_y_x_rgba", "thumbnails_y_x_rgb")

# Point-indexed columns, one row per point. Every one of these is carried
# through a materialisation under the row map.
POINT_COLUMNS = (
    "positions_xyzw",
    "colors_rgb",
    "reprojection_errors",
    "normals_xyz",
    "normal_confidence",
    "point_constraints",
    "constraint_distances",
    "constraint_reference_images",
    "patch_u_halfvec_xyz",
    "patch_v_halfvec_xyz",
    "patch_bitmaps_y_x_rgba",
)

# Observation-indexed columns, one row per track row. `point_indexes` is
# handled separately: it is what the re-sort is keyed on.
OBSERVATION_COLUMNS = (
    "image_indexes",
    "point_indexes",
    "feature_indexes",
    "keypoints_xy",
    "observation_confidence",
)


def arrays_of(data: dict) -> dict[str, np.ndarray]:
    """Every numpy array the dict actually carries, by key.

    `read_sfmr` returns each optional column's key with a `None` value rather
    than omitting it, so presence has to be tested on the value.
    """
    return {
        key: value
        for key, value in data.items()
        if isinstance(value, np.ndarray) and value.size >= 0
    }


def megabytes(n_bytes: int) -> float:
    return n_bytes / (1024.0 * 1024.0)


def median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def report_sizes(arrays: dict[str, np.ndarray]) -> None:
    """Section 1: bytes per array, the total, and the heavy columns' share."""
    print("== Column sizes ==")
    print(f"{'column':<32} {'MB':>10} {'dtype':>10}  shape")
    total = 0
    heavy = 0
    for key in sorted(arrays, key=lambda k: -arrays[k].nbytes):
        array = arrays[key]
        total += array.nbytes
        if key in HEAVY_COLUMNS:
            heavy += array.nbytes
        print(
            f"{key:<32} {megabytes(array.nbytes):>10.2f} "
            f"{str(array.dtype):>10}  {array.shape}"
        )
    print(f"{'TOTAL':<32} {megabytes(total):>10.2f}")
    share = 100.0 * heavy / total if total else 0.0
    print(
        f"{'heavy (bitmaps + thumbnails)':<32} {megabytes(heavy):>10.2f}  "
        f"({share:.1f}% of total)"
    )
    print(f"{'light (everything else)':<32} {megabytes(total - heavy):>10.2f}")
    print()


def time_clone(arrays: dict[str, np.ndarray], repeat: int) -> None:
    """Section 2: what a whole-value copy costs, with and without the heavy columns."""
    light = {k: v for k, v in arrays.items() if k not in HEAVY_COLUMNS}

    def clone(source: dict[str, np.ndarray]) -> float:
        start = time.perf_counter()
        copies = {key: value.copy() for key, value in source.items()}
        elapsed = time.perf_counter() - start
        # Keep the copies alive across the measurement so the allocator cannot
        # hand the same pages back on the next repeat.
        del copies
        return elapsed

    full = [clone(arrays) for _ in range(repeat)]
    without = [clone(light) for _ in range(repeat)]
    print("== Full clone (deep copy of every array) ==")
    print(f"{'all columns':<32} {median_ms(full):>10.2f} ms  (median of {repeat})")
    print(f"{'excluding heavy columns':<32} {median_ms(without):>10.2f} ms")
    print()


def build_edits(
    data: dict,
    n_edits: int,
    n_new: int,
    rng: np.random.Generator,
) -> dict:
    """Choose the point edits and build the addition set they imply.

    Returns the plan the materialisation consumes: which base points were
    modified (deleted and re-added with one extra observation), which were
    deleted outright, and the observation rows the additions carry.
    """
    counts = data["observation_counts"].astype(np.int64)
    n_points = counts.shape[0]
    offsets = np.zeros(n_points + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    n_edits = min(n_edits, n_points)
    n_deleted = min(n_edits // 2, n_points - n_edits)
    picked = rng.choice(n_points, size=n_edits + n_deleted, replace=False)
    modified = np.sort(picked[:n_edits])
    deleted = np.sort(picked[n_edits:])

    n_images = data["camera_indexes"].shape[0]
    return {
        "modified": modified,
        "deleted": deleted,
        "n_new": n_new,
        "offsets": offsets,
        "counts": counts,
        "n_images": n_images,
        "rng": rng,
    }


def addition_rows(data: dict, plan: dict) -> dict[str, np.ndarray]:
    """The observation rows the additions hold, per modified point and per new point.

    A modified point's track is its base track plus one row: the added
    observation the first track edit produces, in an image chosen at random.
    A new point carries two rows, the fewest a triangulated point can have.
    """
    modified = plan["modified"]
    offsets = plan["offsets"]
    rng = plan["rng"]
    n_images = plan["n_images"]

    # Base rows of every modified track, gathered once.
    base_rows = (
        np.concatenate(
            [np.arange(offsets[p], offsets[p + 1], dtype=np.int64) for p in modified]
        )
        if modified.size
        else np.zeros(0, dtype=np.int64)
    )
    # One extra row per modified point (the observation the edit adds), and two
    # rows per new point, all appended after the gathered base rows and sorted
    # into place by the re-sort below.
    n_new = plan["n_new"]
    new_counts = np.full(n_new, 2, dtype=np.int64)
    total_extra = int(modified.shape[0]) + int(new_counts.sum())
    extra_images = rng.integers(0, max(n_images, 1), size=total_extra, dtype=np.uint32)

    return {
        "base_rows": base_rows,
        "new_counts": new_counts,
        "extra_images": extra_images,
    }


def materialise(data: dict, plan: dict, rows: dict) -> tuple[dict, float]:
    """Produce the plain CSR value the base plus its edits stands for.

    Returns the materialised arrays and the seconds the track re-sort alone
    took, so the dominant term can be reported on its own.
    """
    counts = plan["counts"]
    modified = plan["modified"]
    deleted = plan["deleted"]
    n_new = plan["n_new"]
    n_points = counts.shape[0]

    # --- The row map: monotone, a prefix count of the outright deletions. ---
    # A modified point keeps its place (it is re-added at the index it
    # replaced); only an outright deletion closes a slot and shifts what is
    # after it.
    is_deleted = np.zeros(n_points, dtype=bool)
    is_deleted[deleted] = True
    shift = np.cumsum(is_deleted, dtype=np.int64)
    row_map = np.arange(n_points, dtype=np.int64) - shift
    row_map[is_deleted] = -1
    n_survivors = n_points - deleted.shape[0]
    n_out_points = n_survivors + n_new

    # --- Point columns: survivors in place, new points appended. ---
    survivors = ~is_deleted
    out: dict[str, np.ndarray] = {}
    for key in POINT_COLUMNS:
        column = data.get(key)
        if column is None:
            continue
        kept = column[survivors]
        if n_new:
            tail = np.zeros((n_new, *column.shape[1:]), dtype=column.dtype)
            kept = np.concatenate([kept, tail], axis=0)
        out[key] = kept

    # --- Observation rows: unmodified tracks from the base, modified tracks
    # from the additions, new tracks appended. ---
    is_modified = np.zeros(n_points, dtype=bool)
    is_modified[modified] = True
    keep_base_row = np.repeat(survivors & ~is_modified, counts)
    base_kept = np.flatnonzero(keep_base_row)

    added_rows = rows["base_rows"]
    new_counts = rows["new_counts"]
    n_extra = int(modified.shape[0]) + int(new_counts.sum())

    # The point index each observation belongs to, in materialised numbering.
    point_of_base_kept = row_map[np.repeat(np.arange(n_points), counts)[base_kept]]
    point_of_added = row_map[np.repeat(modified, counts[modified])]
    point_of_extra = np.concatenate(
        [
            row_map[modified],
            np.repeat(np.arange(n_survivors, n_out_points, dtype=np.int64), new_counts),
        ]
    )

    def gather_observations(key: str) -> np.ndarray | None:
        column = data.get(key)
        if column is None:
            return None
        parts = [column[base_kept], column[added_rows]]
        # The extra rows (one per modified point, two per new point) are
        # synthesized: an added observation is a pixel and an image, and this
        # measures the cost of carrying it, not its photometric fit.
        tail = np.zeros((n_extra, *column.shape[1:]), dtype=column.dtype)
        if key == "image_indexes":
            tail = rows["extra_images"].astype(column.dtype)
        parts.append(tail)
        return np.concatenate(parts, axis=0)

    observations: dict[str, np.ndarray] = {}
    for key in OBSERVATION_COLUMNS:
        if key == "point_indexes":
            continue
        gathered = gather_observations(key)
        if gathered is not None:
            observations[key] = gathered
    point_indexes = np.concatenate([point_of_base_kept, point_of_added, point_of_extra])

    # --- The re-sort into CSR order, timed on its own. ---
    sort_start = time.perf_counter()
    order = np.lexsort((observations["image_indexes"], point_indexes))
    point_indexes = point_indexes[order]
    for key, column in observations.items():
        observations[key] = column[order]
    sort_seconds = time.perf_counter() - sort_start

    out_counts = np.bincount(point_indexes, minlength=n_out_points).astype(np.uint32)
    out_offsets = np.zeros(n_out_points + 1, dtype=np.int64)
    np.cumsum(out_counts, out=out_offsets[1:])

    out.update(observations)
    out["point_indexes"] = point_indexes.astype(np.uint32)
    out["observation_counts"] = out_counts
    out["observation_offsets"] = out_offsets
    out["row_map"] = row_map
    return out, sort_seconds


def verify(data: dict, plan: dict, out: dict) -> None:
    """Check the materialisation before its timing is believed."""
    counts = out["observation_counts"].astype(np.int64)
    offsets = out["observation_offsets"]
    point_indexes = out["point_indexes"].astype(np.int64)

    assert offsets[0] == 0, "CSR offsets must start at zero"
    assert offsets[-1] == point_indexes.shape[0], (
        f"CSR offsets end at {offsets[-1]} but there are {point_indexes.shape[0]} rows"
    )
    assert np.array_equal(np.diff(offsets), counts), "offsets disagree with counts"
    assert np.all(np.diff(point_indexes) >= 0), "tracks are not sorted by point"
    for key, column in out.items():
        if key in OBSERVATION_COLUMNS:
            assert column.shape[0] == point_indexes.shape[0], (
                f"observation column {key} has {column.shape[0]} rows, "
                f"expected {point_indexes.shape[0]}"
            )

    row_map = out["row_map"]
    survivors = row_map >= 0
    mapped = np.sort(row_map[survivors])
    assert np.array_equal(mapped, np.arange(mapped.shape[0])), (
        "the row map is not a bijection onto the surviving indexes"
    )

    base_counts = plan["counts"]
    modified = plan["modified"]
    for point in modified:
        new_index = row_map[point]
        assert counts[new_index] == base_counts[point] + 1, (
            f"modified point {point} carries {counts[new_index]} observations, "
            f"expected {base_counts[point] + 1}"
        )
    n_new = plan["n_new"]
    if n_new:
        tail = counts[-n_new:]
        assert np.all(tail == 2), f"new points carry {tail}, expected 2 each"


def time_materialisation(data: dict, args: argparse.Namespace) -> None:
    """Section 3: what a materialisation of a base plus a handful of edits costs."""
    samples: list[float] = []
    sorts: list[float] = []
    out = plan = None
    for i in range(args.repeat):
        rng = np.random.default_rng(args.seed + i)
        plan = build_edits(data, args.edits, args.new, rng)
        rows = addition_rows(data, plan)
        start = time.perf_counter()
        out, sort_seconds = materialise(data, plan, rows)
        samples.append(time.perf_counter() - start)
        sorts.append(sort_seconds)
    verify(data, plan, out)

    n_points = data["observation_counts"].shape[0]
    n_obs = data["point_indexes"].shape[0]
    print("== Materialisation ==")
    print(
        f"base: {n_points} points, {n_obs} observations; "
        f"edits: {plan['modified'].shape[0]} modified, "
        f"{plan['deleted'].shape[0]} deleted, {plan['n_new']} new"
    )
    print(f"{'whole materialisation':<32} {median_ms(samples):>10.2f} ms")
    print(f"{'of which the track re-sort':<32} {median_ms(sorts):>10.2f} ms")
    print(f"(median of {args.repeat}; verified)")
    print()


def time_hash(arrays: dict[str, np.ndarray], repeat: int) -> None:
    """Section 4: XXH128 over every array's bytes, the base-hash cost estimate."""
    print("== Base hash (XXH128 over every array) ==")
    try:
        import xxhash
    except ImportError:
        print("xxhash is not importable in this environment; skipped")
        print()
        return

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        digest = xxhash.xxh128()
        for key in sorted(arrays):
            digest.update(np.ascontiguousarray(arrays[key]).data)
        samples.append(time.perf_counter() - start)
    total = sum(a.nbytes for a in arrays.values())
    seconds = statistics.median(samples)
    rate = megabytes(total) / seconds / 1024.0 if seconds else float("inf")
    print(
        f"{'hash of all columns':<32} {median_ms(samples):>10.2f} ms  ({rate:.2f} GB/s)"
    )
    print(f"(median of {repeat}; digest {digest.hexdigest()})")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure clone, materialisation and hash costs on a .sfmr file.",
    )
    parser.add_argument("path", type=Path, help="the .sfmr file to measure")
    parser.add_argument(
        "--edits",
        type=int,
        default=8,
        help="how many points are modified (deleted and re-added with one extra "
        "observation); half as many again are deleted outright (default: 8)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="repeats to take the median of (default: 5)",
    )
    parser.add_argument(
        "--new",
        type=int,
        default=None,
        help="how many genuinely new points the additions carry (default: --edits // 4)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    args = parser.parse_args(argv)
    if args.new is None:
        args.new = max(1, args.edits // 4)

    print(f"file: {args.path}")
    start = time.perf_counter()
    data = read_sfmr(str(args.path))
    print(f"read_sfmr: {(time.perf_counter() - start) * 1000.0:.2f} ms")
    metadata = data.get("metadata") or {}
    print(
        f"feature source: {metadata.get('feature_source', '?')}, "
        f"format version {metadata.get('version', '?')}"
    )
    print()

    arrays = arrays_of(data)
    report_sizes(arrays)
    time_clone(arrays, args.repeat)
    time_materialisation(data, args)
    time_hash(arrays, args.repeat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
