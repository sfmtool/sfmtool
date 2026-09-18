# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure a *progressive* patch constellation query, offline, from one table.

The shipped query takes a constellation of a fixed size and answers once. A
progressive query would instead start from the few features nearest the patch
centre, answer, and widen only if the answer is thin: query the nearest `S`,
then `S + D`, then `S + 2D`, up to a cap, reusing each feature's forest lookups
across stages. This script measures whether that is worth building.

**Why one measurement covers every schedule.** A constellation feature's forest
hits do not depend on which other features are in the constellation;
`one_hit_per_image` collapses a (feature, candidate image) cell on its own; and
RANSAC seeds per candidate image from `seed + image_index`. So the stage that
holds the nearest `n` features returns exactly what a plain
`constellation_query` on those `n` features returns, and a progressive run is a
*subsequence* of the prefix queries. Running every prefix once, storing the full
per-candidate result, and replaying schedules and stopping rules offline is
therefore exact rather than approximate. `min_inliers` is likewise a reporting
filter applied after the per-image fit (`constellation.rs`, after
`fit_affine_ransac`), so querying at a low floor and thresholding afterwards is
equivalent to querying at the higher floor -- which is what makes the acceptance
bar sweepable offline.

Three modes:

* `measure` samples patches the way
  [`kdf_constellation_eval.py`](kdf_constellation_eval.py) does -- same PCG64
  seed, same "random registered image, then random keypoint" rule, so the
  centres match that harness's -- sorts each image's keypoints by distance from
  the centre, and queries every prefix in `--prefixes`. For every candidate
  image at every prefix it stores the inlier and correspondence counts, the
  warp's scale, and the ground truth's own residuals under that warp on three
  reference sets: the stage's own correspondences, those of the nearest-50 disc
  (so warps fitted at different stages are compared on one fixed set), and those
  of the nearest ten (the patch centre's own neighbourhood, which is what a
  caller warping the centre pixel actually depends on). Timing is separated into
  a cold first-touch pass, a warm pass, a warm pass at the production
  `min_inliers`, and a sweep of bare `LazyKdForest.query` batch sizes, so the
  cost of an increment can be modelled apart from the cost of re-fitting.
* `timing` re-times one finished table's queries and batch searches in random
  order, each measurement warm for its own prefix, and folds the result back
  into the file. The ascending sweep `measure` takes is enough on a capture
  whose blocks all stay cached, and is not on a large one, where it reports the
  cost of a stage in a progressive run rather than of a query on its own.
* `analyze` replays schedules, acceptance bars and stopping rules over the
  stored tables and prints the comparison.

Run:
    pixi run -e test python scripts/kdf_constellation_progressive_eval.py measure \\
        --workspace WS --kdf capture.kdf --sfmr WS/sfmr/solve.sfmr \\
        --label NAME --out target/constellation-progressive/NAME.json
    pixi run -e test python scripts/kdf_constellation_progressive_eval.py analyze \\
        target/constellation-progressive/*.json --out analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

# The script's own directory leads sys.path when it is run as a file, which is
# how every harness in scripts/ is run, so the sibling module imports by name.
from kdf_constellation_eval import (
    NO_NEIGHBOR,
    GroundTruth,
    ImageDescriptors,
    align_images,
    corpus_offsets,
    grow_radius,
    image_of_feature,
    load_keypoints,
    resolve_sift_paths,
)

from sfmtool._sfmtool import build_profile
from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.spatial import LazyKdForest
from sfmtool.sift.file import SiftReader

KIB = 1 << 10
MIB = 1 << 20
# Prefix sizes queried. Every schedule replayed offline has to land on these, so
# they are dense at the small end, where a progressive scheme spends its stages.
DEFAULT_PREFIXES = (5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50)
# Round two narrows to the stages its arms need: a lock start, the cap, and the
# sizes the refits select on.
ROUND2_PREFIXES = (15, 20, 25, 30, 35, 50)
# Batch sizes timed through the bare forest search, to separate an increment's
# lookup cost from its fixed per-call overhead.
DEFAULT_BATCHES = (1, 2, 5, 8, 10, 15, 20, 25, 30, 40, 50)
WARP_TOLERANCE_PX = 3.0
# The nearest-ten disc is the "centre" reference set: a caller that warps the
# patch centre pixel depends on the affine being right there and nowhere else.
CENTRE_FEATURES = 10


# ── Measurement ──────────────────────────────────────────────────────────────


def nearest_order(xy: np.ndarray, centre: np.ndarray, cap: int) -> np.ndarray:
    """The `cap` keypoint rows nearest `centre`, nearest first.

    Prefixes of this array are the progressive scheme's stages, so they have to
    nest exactly; `grow_radius` returns a radius's membership instead, which
    differs from the nearest `n` only when keypoints tie at the radius.
    """
    d = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1])
    return np.argsort(d, kind="stable")[:cap]


def sample_patches(rng, positions, registered, count: int, centres: str, sizes):
    """`(image, centre)` pairs, on keypoints or at uniformly random pixels.

    The keypoint rule is `kdf_constellation_eval.py`'s, drawn from the same
    seeded generator in the same order, so the two harnesses see the same
    patches. The pixel rule is what a hand-placed query looks like: the centre
    is nowhere in particular and the nearest keypoint may be far away.
    """
    patches = []
    while len(patches) < count:
        image = int(registered[rng.integers(len(registered))])
        xy = positions[image]
        if len(xy) < 4:
            continue
        if centres == "keypoint":
            centre = np.asarray(xy[rng.integers(len(xy))], dtype=np.float32)
        else:
            width, height = sizes[image]
            centre = np.asarray(
                [rng.uniform(0, width), rng.uniform(0, height)], dtype=np.float32
            )
        patches.append((image, centre))
    return patches


def score_matches(
    matches, positions, xy_cap, gt_slot, gt_image, gt_feature, offsets, covisible, n
) -> list[dict]:
    """One record per candidate image, scored against the ground truth.

    The scoring is `kdf_constellation_eval.measure`'s, per candidate rather than
    pooled per patch, and evaluated on three reference sets at once so a warp
    fitted at one stage can be judged on another stage's points. `res_n` is the
    stage's own set and reproduces that harness's `warp ok`; `res_50` is the
    whole cap disc; `res_10` is the centre's neighbourhood.
    """
    records = []
    for match in matches:
        other = int(match["image_index"])
        take = gt_image == other
        slots = gt_slot[take]
        feats = gt_feature[take]
        affine = np.asarray(match["affine"], dtype=np.float64)
        if len(slots):
            predicted = xy_cap[slots] @ affine[:, :2].T + affine[:, 2]
            actual = positions[other][feats].astype(np.float64)
            residual = np.hypot(*(predicted - actual).T)
        else:
            residual = np.zeros(0)
        in_stage = slots < n
        in_centre = slots < CENTRE_FEATURES

        columns = match["inlier_correspondences"]
        ids = np.asarray(columns["feature_id"], dtype=np.int64)
        images = image_of_feature(offsets, ids)
        recovered = set(
            zip(
                np.asarray(columns["query_index"], dtype=np.int64).tolist(),
                (ids - offsets[images]).tolist(),
            )
        )
        hit = np.array(
            [(int(s), int(f)) in recovered for s, f in zip(slots, feats)], dtype=bool
        )
        if not len(slots):
            hit = np.zeros(0, dtype=bool)

        def med(mask) -> float | None:
            return float(np.median(residual[mask])) if mask.any() else None

        linear = affine[:, :2]
        records.append(
            {
                "i": other,
                "inl": int(match["inliers"]),
                "corr": int(match["correspondences"]),
                "scale": float(math.sqrt(abs(float(np.linalg.det(linear))))),
                "gt_n": int(in_stage.sum()),
                "gt_50": int(len(slots)),
                "gt_10": int(in_centre.sum()),
                "res_n": med(in_stage),
                "res_50": med(np.ones(len(slots), dtype=bool)) if len(slots) else None,
                "res_10": med(in_centre),
                "rec_n": int((hit & in_stage).sum()),
                "rec_50": int(hit.sum()),
                "ncov": other not in covisible,
            }
        )
    return records


def measure_patch(lazy, gt, positions, offsets, image, order, prefixes, knobs) -> dict:
    """Every prefix query for one patch, scored against the ground truth."""
    xy_cap = positions[image][order].astype(np.float64)
    query_xy = np.ascontiguousarray(positions[image][order], dtype=np.float32)
    ids = (offsets[image] + order).astype(np.uint32).tolist()
    gt_slot, gt_image, gt_feature = gt.correspondences(image, order)
    covisible = gt.covisible(image)

    stages = {}
    for n in prefixes:
        start = time.perf_counter()
        matches = lazy.constellation_query(
            query_xy[:n], feature_ids=ids[:n], image_index=image, **knobs
        )
        elapsed = (time.perf_counter() - start) * 1e3
        in_stage = gt_slot < n
        per_image: dict[int, int] = {}
        for other in gt_image[in_stage]:
            per_image[int(other)] = per_image.get(int(other), 0) + 1
        stages[str(n)] = {
            "gt_images_1": sum(1 for c in per_image.values() if c >= 1),
            "gt_images_3": sum(1 for c in per_image.values() if c >= 3),
            "gt_image_list_3": [i for i, c in per_image.items() if c >= 3],
            "gt_corr": int(in_stage.sum()),
            "ms": elapsed,
            "cands": score_matches(
                matches,
                positions,
                xy_cap,
                gt_slot,
                gt_image,
                gt_feature,
                offsets,
                covisible,
                n,
            ),
        }
    return stages


def time_queries(lazy, positions, offsets, patches, prefixes, knobs) -> list[dict]:
    """Wall time of every prefix query, results discarded."""
    out = []
    for image, centre in patches:
        order = nearest_order(positions[image], centre, max(prefixes))
        query_xy = np.ascontiguousarray(positions[image][order], dtype=np.float32)
        ids = (offsets[image] + order).astype(np.uint32).tolist()
        row = {}
        for n in prefixes:
            start = time.perf_counter()
            lazy.constellation_query(
                query_xy[:n], feature_ids=ids[:n], image_index=image, **knobs
            )
            row[str(n)] = (time.perf_counter() - start) * 1e3
        out.append(row)
    return out


def time_search_batches(lazy, descriptors, positions, patches, batches, knobs):
    """Bare `LazyKdForest.query` time against batch size, warm.

    An increment of `D` features pays one forest search of `D` queries. Whether
    that is `D` times the cost of one, or one fixed cost plus `D` times a
    smaller slope, is what decides whether a small increment is wasteful.
    """
    out = []
    for image, centre in patches:
        order = nearest_order(positions[image], centre, max(batches))
        vectors = descriptors.rows(image, order)
        row = {}
        for size in batches:
            block = np.ascontiguousarray(vectors[:size], dtype=np.uint8)
            start = time.perf_counter()
            lazy.query(block, k=knobs["k"], max_leaf_checks=knobs["max_leaf_checks"])
            row[str(size)] = (time.perf_counter() - start) * 1e3
        out.append(row)
    return out


def repeat_timed(rng, jobs, run) -> None:
    """Time each job warm and in random order, second of two consecutive runs.

    Timing a prefix sweep in ascending order measures the wrong thing on a
    file-backed forest: the small prefixes pay the patch's first-touch reads and
    the large ones inherit a cache the small ones filled, so the curve reports a
    progressive run's stage costs rather than what each query costs on its own.
    Shuffling separates the jobs, and running each one twice makes the recorded
    time that of a query whose own blocks are resident.
    """
    order = list(range(len(jobs)))
    rng.shuffle(order)
    for index in order:
        run(jobs[index], False)
        run(jobs[index], True)


def run_timing(args) -> None:
    """Re-time one dataset's queries properly and fold it into its table."""
    if build_profile() != "release":
        raise SystemExit("refusing to measure a debug build of _sfmtool")
    raw = json.loads(Path(args.out).read_text())
    lazy, positions, offsets, sift_paths, _, _ = open_corpus(
        Path(raw["workspace"]),
        raw["kdf"],
        Path(args.features) if args.features else None,
        raw["cache_mib"],
    )
    prefixes, batches = raw["prefixes"], raw["batches"]
    patches = [
        (r["image"], np.asarray(r["centre"], dtype=np.float32)) for r in raw["records"]
    ]
    knobs = dict(raw["params"], min_inliers=raw["production_inliers"])

    query_ms = [{} for _ in patches]
    orders = [nearest_order(positions[i], c, max(prefixes)) for i, c in patches]
    ids = [
        (offsets[i] + o).astype(np.uint32).tolist()
        for (i, _), o in zip(patches, orders)
    ]
    xy = [
        np.ascontiguousarray(positions[i][o], dtype=np.float32)
        for (i, _), o in zip(patches, orders)
    ]

    def one_query(job, keep: bool) -> None:
        patch, n = job
        start = time.perf_counter()
        lazy.constellation_query(
            xy[patch][:n],
            feature_ids=ids[patch][:n],
            image_index=patches[patch][0],
            **knobs,
        )
        if keep:
            query_ms[patch][str(n)] = (time.perf_counter() - start) * 1e3

    rng = np.random.default_rng(raw["seed"])
    repeat_timed(
        rng,
        [(p, n) for p in range(len(patches)) for n in prefixes],
        one_query,
    )
    print("query timing done", flush=True)

    descriptors = ImageDescriptors(sift_paths)
    blocks = [
        np.ascontiguousarray(descriptors.rows(i, o), dtype=np.uint8)
        for (i, _), o in zip(patches, orders)
    ]
    search_ms = [{} for _ in patches]

    def one_search(job, keep: bool) -> None:
        patch, size = job
        start = time.perf_counter()
        lazy.query(
            blocks[patch][:size],
            k=knobs["k"],
            max_leaf_checks=knobs["max_leaf_checks"],
        )
        if keep:
            search_ms[patch][str(size)] = (time.perf_counter() - start) * 1e3

    repeat_timed(
        rng,
        [(p, b) for p in range(len(patches)) for b in batches],
        one_search,
    )
    raw["repeat_production_ms"] = query_ms
    raw["repeat_search_ms"] = search_ms
    Path(args.out).write_text(json.dumps(raw))
    print(f"updated {args.out}")


def open_corpus(workspace: Path, kdf: str, features: Path | None, cache_mib: int):
    """The forest, per-image keypoints, corpus offsets, `.sift` paths and names."""
    budget = cache_mib * MIB
    lazy = LazyKdForest(
        kdf,
        cache_bytes=budget,
        max_in_flight_bytes=budget,
        max_chunk_bytes=min(budget, 4 * MIB),
        max_compressed_bytes=64 * MIB,
    )
    table = lazy.image_table()
    if table is None:
        raise SystemExit(f"{kdf} carries no SIFT sources")
    names = list(table["names"])
    sift_paths = resolve_sift_paths(workspace, names, features)
    positions = load_keypoints(sift_paths)
    sizes = []
    for path in sift_paths:
        meta = SiftReader(path).metadata
        sizes.append((meta["image_width"], meta["image_height"]))
    return lazy, positions, corpus_offsets(lazy, positions), sift_paths, names, sizes


def run_measure(args) -> None:
    if build_profile() != "release":
        raise SystemExit("refusing to measure a debug build of _sfmtool")

    workspace = Path(args.workspace)
    lazy, positions, offsets, sift_paths, kdf_names, sizes = open_corpus(
        workspace,
        args.kdf,
        Path(args.features) if args.features else None,
        args.cache_mib,
    )

    recon = SfmrReconstruction.load(args.sfmr)
    kdf_to_sfmr = align_images(kdf_names, list(recon.image_names))
    gt = GroundTruth(recon, kdf_to_sfmr)
    registered = np.flatnonzero(kdf_to_sfmr >= 0)
    print(
        f"{len(kdf_names)} indexed images, {len(registered)} in the ground truth,"
        f" {lazy.len:,} descriptors",
        flush=True,
    )

    prefixes = [int(v) for v in args.prefixes.split(",")]
    batches = [int(v) for v in args.batches.split(",")]
    rng = np.random.default_rng(args.seed)
    patches = sample_patches(
        rng, positions, registered, args.patches, args.centres, sizes
    )

    knobs = {
        "k": args.k,
        "max_leaf_checks": args.budget,
        "threshold_px": args.threshold,
        "iterations": args.iterations,
        "min_correspondences": args.min_correspondences,
        "one_hit_per_image": True,
        "same_image_ratio": 1.0,
        "min_inliers": args.floor_inliers,
        "max_scale": args.max_scale,
        "seed": args.seed,
    }
    production = dict(knobs, min_inliers=args.production_inliers)

    cold = []
    if args.cold_pass:
        print("cold pass…", flush=True)
        cold = time_queries(lazy, positions, offsets, patches, prefixes, knobs)

    print("warm pass…", flush=True)
    records = []
    for index, (image, centre) in enumerate(patches):
        order = nearest_order(positions[image], centre, max(prefixes))
        stages = measure_patch(
            lazy, gt, positions, offsets, image, order, prefixes, knobs
        )
        radii = {}
        for n in prefixes:
            radius, _ = grow_radius(positions[image], centre, n)
            radii[str(n)] = radius
        records.append(
            {
                "image": int(image),
                "centre": [float(centre[0]), float(centre[1])],
                "radius": radii,
                "keypoints": int(len(positions[image])),
                "stages": stages,
            }
        )
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(patches)} patches", flush=True)

    production_times = []
    if args.cold_pass:
        print("production-floor timing pass…", flush=True)
        production_times = time_queries(
            lazy, positions, offsets, patches, prefixes, production
        )

    print("search batch timing…", flush=True)
    search = time_search_batches(
        lazy, ImageDescriptors(sift_paths), positions, patches, batches, knobs
    )

    out = {
        "label": args.label or workspace.name,
        "workspace": str(workspace),
        "kdf": args.kdf,
        "sfmr": args.sfmr,
        "centres": args.centres,
        "images": len(kdf_names),
        "registered_images": int(len(registered)),
        "descriptors": lazy.len,
        "observations": int(len(gt.image)),
        "cache_mib": args.cache_mib,
        "params": knobs,
        "production_inliers": args.production_inliers,
        "prefixes": prefixes,
        "batches": batches,
        "patches": args.patches,
        "seed": args.seed,
        "records": records,
        "cold_ms": cold,
        "production_ms": production_times,
        "search_ms": search,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out}")


# ── Acceptance bars ──────────────────────────────────────────────────────────


def make_bar(spec: str):
    """A predicate `(inliers, correspondences, n) -> bool` from a name.

    `c8` is the shipped constant floor. `f6_0.3` is `max(6, ceil(0.3 n))`, which
    rises with the stage. `s4_0.5_8` is `min(8, max(4, ceil(0.5 n)))`, which
    rises and then stops: that is the shape a progressive query wants if eight
    inliers out of ten features is too much to ask early but eight is still the
    right floor once the constellation is large. `r8_0.3` adds an inlier-ratio
    floor to a constant, which is the other way to say "the consensus has to be
    a real share of what was offered".
    """
    kind, _, rest = spec.partition("_")
    if kind.startswith("c"):
        floor = int(kind[1:])
        return lambda inl, corr, n: inl >= floor
    if kind.startswith("f"):
        base, fraction = int(kind[1:]), float(rest)
        return lambda inl, corr, n: inl >= max(base, math.ceil(fraction * n))
    if kind.startswith("s"):
        fraction, cap = rest.split("_")
        base, fraction, cap = int(kind[1:]), float(fraction), int(cap)
        return lambda inl, corr, n: inl >= min(cap, max(base, math.ceil(fraction * n)))
    if kind.startswith("r"):
        base, ratio = int(kind[1:]), float(rest)
        return lambda inl, corr, n: inl >= base and (corr and inl / corr >= ratio)
    raise SystemExit(f"unknown bar {spec!r}")


DEFAULT_BARS = (
    "c4",
    "c5",
    "c6",
    "c7",
    "c8",
    "c10",
    "c12",
    "f5_0.3",
    "f6_0.3",
    "f5_0.4",
    "f6_0.25",
    "s4_0.5_8",
    "s5_0.4_8",
    "s4_0.3_8",
    "r6_0.3",
    "r8_0.3",
)


# ── Replay ───────────────────────────────────────────────────────────────────


def accepted(stage: dict, bar, n: int) -> list[dict]:
    return [c for c in stage["cands"] if bar(c["inl"], c["corr"], n)]


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def score_selection(record: dict, chosen: list[dict], n_for_gt: int) -> dict:
    """Score one patch's accepted candidates against its ground truth.

    `n_for_gt` names the constellation whose ground truth is the denominator.
    Recall at a fixed denominator is what makes two stages comparable, so the
    caller passes the cap for cross-stage tables and the stage's own size when
    reproducing a single-query measurement.
    """
    stage = record["stages"][str(n_for_gt)]
    truth = set(stage["gt_image_list_3"])
    found = {c["i"] for c in chosen}
    corr_den = stage["gt_corr"]
    # A candidate found at stage `n` can only have inliers among the first `n`
    # features, so `rec_n` and `rec_50` agree for it and either counts the
    # ground-truth correspondences it recovered.
    recovered = sum(c["rec_n"] for c in chosen)
    return {
        "found": len(found),
        "hits_3": len(found & truth),
        "truth_3": len(truth),
        "recall_3": len(found & truth) / len(truth) if truth else None,
        "precision_3": len(found & truth) / len(found) if found else None,
        "corr_recall": recovered / corr_den if corr_den else None,
        "false": sum(1 for c in chosen if c["gt_50"] == 0),
        "never_covis": sum(1 for c in chosen if c["ncov"]),
        "warp_checked": sum(1 for c in chosen if c["gt_n"] >= 3),
        "warp_ok": sum(
            1 for c in chosen if c["gt_n"] >= 3 and c["res_n"] <= WARP_TOLERANCE_PX
        ),
        "warp50_checked": sum(1 for c in chosen if c["gt_50"] >= 3),
        "warp50_ok": sum(
            1 for c in chosen if c["gt_50"] >= 3 and c["res_50"] <= WARP_TOLERANCE_PX
        ),
        "warp10_checked": sum(1 for c in chosen if c["gt_10"] >= 3),
        "warp10_ok": sum(
            1 for c in chosen if c["gt_10"] >= 3 and c["res_10"] <= WARP_TOLERANCE_PX
        ),
        "res_n": median([c["res_n"] for c in chosen]),
        "res_50": median([c["res_50"] for c in chosen]),
        "res_10": median([c["res_10"] for c in chosen]),
    }


def pool(scores: list[dict]) -> dict:
    """Per-patch means for the rates, pooled sums for the warp shares."""

    def share(ok: str, checked: str) -> float | None:
        total = sum(s[checked] for s in scores)
        return sum(s[ok] for s in scores) / total if total else None

    return {
        "patches": len(scores),
        "found": mean([s["found"] for s in scores]),
        "recall_3": mean([s["recall_3"] for s in scores]),
        "precision_3": mean([s["precision_3"] for s in scores]),
        "corr_recall": mean([s["corr_recall"] for s in scores]),
        "false": mean([s["false"] for s in scores]),
        "never_covis": mean([s["never_covis"] for s in scores]),
        "warp_ok": share("warp_ok", "warp_checked"),
        "warp_checked": sum(s["warp_checked"] for s in scores),
        "warp50_ok": share("warp50_ok", "warp50_checked"),
        "warp50_checked": sum(s["warp50_checked"] for s in scores),
        "warp10_ok": share("warp10_ok", "warp10_checked"),
        "warp10_checked": sum(s["warp10_checked"] for s in scores),
        "res_n": median([s["res_n"] for s in scores]),
        "res_50": median([s["res_50"] for s in scores]),
        "res_10": median([s["res_10"] for s in scores]),
    }


def bootstrap_ci(values, statistic=np.mean, draws: int = 2000, seed: int = 0):
    """A percentile bootstrap interval over patches, for the 60-patch noise."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    sample = statistic(data[rng.integers(0, len(data), (draws, len(data)))], axis=1)
    return [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))]


def single_stage(raw: dict, bar, n: int, fixed_denominator: bool) -> dict:
    """Metrics of one plain query at prefix `n`, at one acceptance bar."""
    cap = max(raw["prefixes"])
    scores = []
    for record in raw["records"]:
        stage = record["stages"][str(n)]
        scores.append(
            score_selection(
                record, accepted(stage, bar, n), cap if fixed_denominator else n
            )
        )
    out = pool(scores)
    out["recall_3_ci"] = bootstrap_ci([s["recall_3"] for s in scores])
    out["ms"] = median([record["stages"][str(n)]["ms"] for record in raw["records"]])
    return out


def first_hit(raw: dict, bar, schedule: list[int]) -> dict:
    """Stop at the first stage of `schedule` where any image passes."""
    cap = max(raw["prefixes"])
    scores, stops = [], []
    for record in raw["records"]:
        chosen, stop = [], schedule[-1]
        for n in schedule:
            here = accepted(record["stages"][str(n)], bar, n)
            if here:
                chosen, stop = here, n
                break
        stops.append(stop)
        scores.append(score_selection(record, chosen, cap))
    out = pool(scores)
    out["stop_median"] = median(stops)
    out["stop_histogram"] = {str(n): stops.count(n) for n in schedule}
    out["never_stopped"] = sum(
        1
        for record, stop in zip(raw["records"], stops)
        if stop == schedule[-1] and not accepted(record["stages"][str(stop)], bar, stop)
    )
    out["stops"] = stops
    return out


def locked(raw: dict, bar, schedule: list[int], rule: str) -> dict:
    """Run every stage; each image keeps the warp of one of its passing stages.

    `first` freezes at the earliest stage that accepts the image, which is what
    a progressive query would naturally return; `last` and `ratio` are the two
    alternatives worth asking about -- the widest evidence, and the stage whose
    consensus was the largest share of what it was offered.
    """
    cap = max(raw["prefixes"])
    scores = []
    for record in raw["records"]:
        best: dict[int, tuple[float, dict]] = {}
        for n in schedule:
            for cand in accepted(record["stages"][str(n)], bar, n):
                key = cand["i"]
                if rule == "first":
                    rank = -n
                elif rule == "last":
                    rank = n
                else:
                    rank = cand["inl"] / cand["corr"] if cand["corr"] else 0.0
                if key not in best or rank > best[key][0]:
                    best[key] = (rank, cand)
        scores.append(score_selection(record, [c for _, c in best.values()], cap))
    return pool(scores)


def plateau(raw: dict, bar, schedule: list[int]) -> dict:
    """Stop when the best accepted image's inlier count stops growing."""
    cap = max(raw["prefixes"])
    scores, stops = [], []
    for record in raw["records"]:
        chosen, stop, previous = [], schedule[-1], -1
        for n in schedule:
            here = accepted(record["stages"][str(n)], bar, n)
            top = max((c["inl"] for c in here), default=0)
            if here and top <= previous:
                chosen, stop = here, n
                break
            previous = top
            chosen, stop = here, n
        stops.append(stop)
        scores.append(score_selection(record, chosen, cap))
    out = pool(scores)
    out["stop_median"] = median(stops)
    out["stops"] = stops
    return out


def small_versus_wide(raw: dict, bar, small: int) -> dict:
    """Q3: when a small stage and the cap both find an image, which warp wins?

    Compared at the patch centre (`res_10`) and over the cap disc (`res_50`),
    both on the same fixed points, so the only difference is which affine was
    fitted. Also counts the images one stage finds and the other does not, split
    by whether the ground truth corroborates them.
    """
    cap = max(raw["prefixes"])
    centre_win = centre_loss = centre_tie = 0
    disc_win = disc_loss = disc_tie = 0
    only_small = only_cap = 0
    only_small_true = only_cap_true = 0
    only_small_covis = 0
    centre_gain, disc_gain = [], []
    for record in raw["records"]:
        at_small = {
            c["i"]: c for c in accepted(record["stages"][str(small)], bar, small)
        }
        at_cap = {c["i"]: c for c in accepted(record["stages"][str(cap)], bar, cap)}
        for key in set(at_small) - set(at_cap):
            only_small += 1
            only_small_true += int(at_small[key]["gt_50"] > 0)
            only_small_covis += int(not at_small[key]["ncov"])
        for key in set(at_cap) - set(at_small):
            only_cap += 1
            only_cap_true += int(at_cap[key]["gt_50"] > 0)
        for key in set(at_small) & set(at_cap):
            a, b = at_small[key], at_cap[key]
            for field, gate, counters, gains in (
                ("res_10", "gt_10", "centre", centre_gain),
                ("res_50", "gt_50", "disc", disc_gain),
            ):
                if a[gate] < 3 or a[field] is None or b[field] is None:
                    continue
                gains.append(b[field] - a[field])
                # A tenth of a pixel, or a tenth of the larger residual: below
                # that the two warps are the same answer.
                slack = max(0.1, 0.1 * max(a[field], b[field]))
                if a[field] < b[field] - slack:
                    won = 1
                elif b[field] < a[field] - slack:
                    won = -1
                else:
                    won = 0
                if counters == "centre":
                    centre_win += won == 1
                    centre_loss += won == -1
                    centre_tie += won == 0
                else:
                    disc_win += won == 1
                    disc_loss += won == -1
                    disc_tie += won == 0
    return {
        "small": small,
        "cap": cap,
        "centre_small_better": centre_win,
        "centre_cap_better": centre_loss,
        "centre_tie": centre_tie,
        "centre_gain_median": median(centre_gain),
        "disc_small_better": disc_win,
        "disc_cap_better": disc_loss,
        "disc_tie": disc_tie,
        "disc_gain_median": median(disc_gain),
        "only_small": only_small,
        "only_small_true": only_small_true,
        "only_small_covisible": only_small_covis,
        "only_cap": only_cap,
        "only_cap_true": only_cap_true,
    }


def false_floor(raw: dict) -> dict:
    """The inlier distribution of wrong candidates, per prefix size.

    `false` is a candidate the ground truth gives no correspondence with the cap
    constellation; `never covisible` is the stricter one a sparse solve cannot
    manufacture. Where these sit against the constellation size is what an
    acceptance bar has to clear.
    """
    out = {}
    for n in raw["prefixes"]:
        false_inliers, strict_inliers, true_inliers = [], [], []
        for record in raw["records"]:
            for cand in record["stages"][str(n)]["cands"]:
                if cand["ncov"]:
                    strict_inliers.append(cand["inl"])
                if cand["gt_50"] == 0:
                    false_inliers.append(cand["inl"])
                else:
                    true_inliers.append(cand["inl"])
        out[str(n)] = {
            "false_n": len(false_inliers),
            "false_median": median(false_inliers),
            "false_p90": (
                float(np.percentile(false_inliers, 90)) if false_inliers else None
            ),
            "false_max": max(false_inliers, default=None),
            "strict_n": len(strict_inliers),
            "strict_median": median(strict_inliers),
            "strict_p90": (
                float(np.percentile(strict_inliers, 90)) if strict_inliers else None
            ),
            "strict_max": max(strict_inliers, default=None),
            "true_n": len(true_inliers),
            "true_median": median(true_inliers),
        }
    return out


def cost_model(raw: dict) -> dict:
    """Per-stage search, fit and total time, and the fixed cost of a call.

    A `timing` run's shuffled, individually-warmed measurements are preferred
    where they exist: the sweep `measure` takes in ascending order reports each
    prefix with its predecessors' blocks already cached, which is a progressive
    run's stage cost rather than a standalone query's.
    """
    rows = raw.get("repeat_search_ms") or raw["search_ms"]
    search = {size: median([row[str(size)] for row in rows]) for size in raw["batches"]}
    warm = {
        n: median([record["stages"][str(n)]["ms"] for record in raw["records"]])
        for n in raw["prefixes"]
    }
    cold = {
        n: median([row[str(n)] for row in raw["cold_ms"]]) if raw["cold_ms"] else None
        for n in raw["prefixes"]
    }
    timed = raw.get("repeat_production_ms") or raw["production_ms"]
    production = {
        n: (median([row[str(n)] for row in timed]) if timed else None)
        for n in raw["prefixes"]
    }

    def search_at(size: float) -> float:
        keys = sorted(search)
        return float(np.interp(size, keys, [search[k] for k in keys]))

    base = production if production[raw["prefixes"][0]] is not None else warm
    fit = {n: max(0.0, base[n] - search_at(n)) for n in raw["prefixes"]}
    # A straight line through the two ends of the batch sweep: its intercept is
    # what a call costs before it has looked anything up, which is the number
    # that decides whether a five-feature increment is worth issuing.
    lo, hi = min(search), max(search)
    slope = (search[hi] - search[lo]) / (hi - lo)
    return {
        "search_ms": {str(k): v for k, v in search.items()},
        "search_slope_ms_per_feature": slope,
        "search_intercept_ms": search[lo] - slope * lo,
        "warm_ms": {str(k): v for k, v in warm.items()},
        "cold_ms": {str(k): v for k, v in cold.items()},
        "production_ms": {str(k): v for k, v in production.items()},
        "fit_ms": {str(k): v for k, v in fit.items()},
    }


def schedule_stages(start: int, increment, cap: int, prefixes) -> list[int]:
    """The stage sizes of one schedule, snapped to the measured prefixes."""
    stages, n = [], start
    while n < cap:
        if n in prefixes:
            stages.append(n)
        n = n * 2 if increment == "x2" else n + int(increment)
    stages.append(cap)
    return sorted(set(stages))


def schedule_cost(costs: dict, stages: list[int], stop: int) -> float:
    """Modelled wall time of a progressive run that stops at `stop`.

    Each stage pays the forest search for its own increment -- the earlier
    features' hits are already held -- and a full re-fit at its size, because
    RANSAC is per candidate image over that image's whole correspondence list.
    """
    search = {int(k): v for k, v in costs["search_ms"].items()}
    keys = sorted(search)

    def search_at(size: float) -> float:
        return float(np.interp(size, keys, [search[k] for k in keys]))

    fit = {int(k): v for k, v in costs["fit_ms"].items()}
    total, previous = 0.0, 0
    for n in stages:
        total += search_at(n - previous) + fit[n]
        previous = n
        if n >= stop:
            break
    return total


def run_analyze(args) -> None:
    bars = args.bars.split(",")
    results = {}
    for path in args.inputs:
        raw = json.loads(Path(path).read_text())
        label = raw["label"] + ("" if raw["centres"] == "keypoint" else "-pixel")
        cap = max(raw["prefixes"])
        costs = cost_model(raw)
        entry = {
            "images": raw["images"],
            "descriptors": raw["descriptors"],
            "observations": raw["observations"],
            "centres": raw["centres"],
            "patches": len(raw["records"]),
            "cost": costs,
            "false_floor": false_floor(raw),
            "single": {},
            "first_hit": {},
            "locked": {},
            "plateau": {},
            "small_versus_wide": {},
            "schedules": {},
        }
        for spec in bars:
            bar = make_bar(spec)
            entry["single"][spec] = {
                str(n): single_stage(raw, bar, n, fixed_denominator=True)
                for n in raw["prefixes"]
            }
        # The old harness's numbers, for the sanity check: the stage's own
        # ground truth as the denominator, at the shipped floor.
        entry["single_local_c8"] = {
            str(n): single_stage(raw, make_bar("c8"), n, fixed_denominator=False)
            for n in raw["prefixes"]
        }
        for start in args.starts:
            for increment in args.increments:
                stages = schedule_stages(start, increment, cap, raw["prefixes"])
                name = f"S{start}D{increment}"
                entry["schedules"][name] = {
                    "stages": stages,
                    # What a rule that always reaches the cap costs: every
                    # stage's increment searched and every stage re-fitted.
                    "full_ms": schedule_cost(costs, stages, cap),
                }
                for spec in args.schedule_bars.split(","):
                    bar = make_bar(spec)
                    hit = first_hit(raw, bar, stages)
                    hit["modelled_ms"] = median(
                        [
                            schedule_cost(costs, stages, stop)
                            for stop in hit.pop("stops")
                        ]
                    )
                    entry["first_hit"][f"{name}/{spec}"] = hit
                    for rule in ("first", "last", "ratio"):
                        entry["locked"][f"{name}/{spec}/{rule}"] = locked(
                            raw, bar, stages, rule
                        )
                    flat = plateau(raw, bar, stages)
                    flat["modelled_ms"] = median(
                        [
                            schedule_cost(costs, stages, stop)
                            for stop in flat.pop("stops")
                        ]
                    )
                    entry["plateau"][f"{name}/{spec}"] = flat
        for spec in bars:
            bar = make_bar(spec)
            for small in args.smalls:
                entry["small_versus_wide"][f"{spec}/{small}"] = small_versus_wide(
                    raw, bar, small
                )
        entry["baseline_ms"] = costs["production_ms"].get(str(cap)) or costs[
            "warm_ms"
        ].get(str(cap))
        results[label] = entry
        print(f"{label}: analysed {len(raw['records'])} patches", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


# ── Round two: the two-stage locked warp, measured where the caller uses it ──
#
# Round one scored a warp by the ground truth's residuals over the patch disc,
# which is a proxy. What the caller does -- `bench/search.rs` -- is apply the
# affine to the patch centre pixel and seed an observation there. So round two
# centres each patch on a keypoint that carries a ground-truth track, **holds
# that keypoint out of the constellation**, and measures the distance between
# where the affine puts the centre and where the track says the point is. No
# three-point model can pass through a feature that is not in the constellation,
# so the measurement is held out rather than fitted.


def build_corpus_sources(workspace: Path, paths, names):
    """The `write_kdf` sources mapping for a list of `.sift` files."""
    descriptors, positions, shapes = [], [], []
    for path in paths:
        reader = SiftReader(path)
        descriptors.append(np.asarray(reader.read_descriptors(), dtype=np.uint8))
        xy, shape = reader.read_positions_and_shapes()
        positions.append(np.asarray(xy, dtype=np.float32))
        shapes.append(np.asarray(shape, dtype=np.float32))
        reader.close()
    counts = [len(d) for d in descriptors]
    image_of = np.concatenate(
        [np.full(c, i, dtype=np.uint32) for i, c in enumerate(counts)]
    )
    feature_of = np.concatenate([np.arange(c, dtype=np.uint32) for c in counts])
    sources = {
        "workspace": {
            "absolute_path": str(workspace),
            "relative_path": ".",
            "contents": {
                "feature_tool": "sfmtool",
                "feature_type": "sift",
                "feature_options": json.dumps({}),
                "feature_prefix_dir": "features",
            },
        },
        "image_names": names,
        # Placeholders: nothing here verifies provenance, and the table's size
        # does not depend on which bytes the hashes hold.
        "feature_tool_hashes": [bytes(16)] * len(names),
        "sift_content_hashes": [bytes(16)] * len(names),
        "image_indexes": image_of.tolist(),
        "image_feature_indexes": feature_of.tolist(),
        "positions": np.vstack(positions),
        "affine_shapes": np.vstack(shapes),
    }
    return np.vstack(descriptors), sources


def run_build_kdf(args) -> None:
    """Build one `.kdf`, optionally over a subset of a workspace's images.

    `--sfmr` restricts the corpus to the images a reconstruction registered and
    `--stride` keeps every Nth of those. Together they manufacture a
    wider-baseline capture out of a video walk without re-solving anything: the
    solve's own tracks still describe the subset, so the ground truth is the
    same trusted one seen through a sparser set of viewpoints.
    """
    from sfmtool._sfmtool.spatial import KdForest, write_kdf

    workspace = Path(args.workspace).resolve()
    paths = []
    for directory in args.features:
        paths.extend(sorted(Path(directory).resolve().glob("*.sift")))

    def name_of(path: Path) -> str:
        image = path.parent.parent.parent / path.name.removesuffix(".sift")
        return image.relative_to(workspace).as_posix()

    if args.sfmr:
        wanted = set(SfmrReconstruction.load(args.sfmr).image_names)
        paths = [p for p in paths if name_of(p) in wanted]
    paths = paths[:: max(1, args.stride)]
    names = [name_of(p) for p in paths]

    corpus, sources = build_corpus_sources(workspace, paths, names)
    print(f"{len(names)} images, {len(corpus):,} descriptors", flush=True)
    forest = KdForest(
        corpus, num_trees=args.trees, leaf_size=args.leaf_size, seed=args.seed
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    write_kdf(
        forest,
        str(out),
        chunk_bytes=args.chunk_bytes,
        descriptor_block_bytes=args.block_bytes,
        sources=sources,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)


def offered_correspondences(lazy, offsets, image: int, descriptors, knobs):
    """The correspondence list the query itself fits on, rebuilt from the hits.

    The query returns only the inliers it kept, and the refit arms need the
    whole list a candidate image was offered: to re-select inliers under a
    refitted model, and to ask which correspondences near the centre agree with
    an early warp. This repeats the same batch search at the same `k` and leaf
    budget, drops the query image's own hits, and applies the same per-cell
    collapse -- keep the nearest hit of each (constellation feature, candidate
    image) cell, which is the first occurrence in the neighbour list because
    that list is ascending in distance.
    """
    ids, _ = lazy.query(
        np.ascontiguousarray(descriptors, dtype=np.uint8),
        k=knobs["k"],
        max_leaf_checks=knobs["max_leaf_checks"],
    )
    ids = np.asarray(ids)
    n, k = ids.shape
    live = (ids != NO_NEIGHBOR).ravel()
    rows = np.repeat(np.arange(n, dtype=np.int64), k)[live]
    feature_ids = ids.ravel()[live].astype(np.int64)
    images = image_of_feature(offsets, feature_ids)
    keep = images != image
    rows, feature_ids, images = rows[keep], feature_ids[keep], images[keep]
    features = feature_ids - offsets[images]

    per_image: dict[int, list[tuple[int, int]]] = {}
    seen: set[tuple[int, int]] = set()
    for row, other, feature in zip(rows, images, features):
        cell = (int(row), int(other))
        if cell in seen:
            continue
        seen.add(cell)
        per_image.setdefault(int(other), []).append((int(row), int(feature)))
    return per_image


def centre_track(gt: GroundTruth, image: int, row: int):
    """The point the centre keypoint observes and its other observations."""
    point = int(gt.point_of(image, np.asarray([row]))[0])
    if point < 0:
        return None
    lo, hi = gt.point_start[point], gt.point_start[point + 1]
    obs = [
        (int(gt.obs_image[o]), int(gt.obs_feature[o]))
        for o in range(lo, hi)
        if int(gt.obs_image[o]) != image
    ]
    return point, obs


def sample_patches2(rng, positions, registered, gt, count, centres, sizes, min_obs=2):
    """Patch centres for round two, with the centre's own ground truth attached.

    A keypoint centre is only usable if the solve tracked it into at least
    `min_obs` other images of **this corpus**, because the measurement is "where
    does the affine put this point, and where is it really". Two is the default,
    a track of length three. A corpus that is a subsample of a solved capture
    has to drop to one: a video walk's tracks span consecutive frames, so
    keeping every twentieth frame leaves almost every track with a single
    surviving observation even though the solve that made it saw dozens. A
    pixel centre has no track of its own, so the proxy is its five nearest
    keypoints, held out of the constellation in exactly the same way.
    """
    patches, tries = [], 0
    # Centres are drawn without replacement: a capture with few tracked
    # keypoints would otherwise hand back the same patch many times and the
    # bootstrap would read the repeats as independent evidence.
    seen: set[tuple[int, int]] = set()
    while len(patches) < count and tries < count * 400:
        tries += 1
        image = int(registered[rng.integers(len(registered))])
        xy = positions[image]
        if len(xy) < 60:
            continue
        if centres == "keypoint":
            row = int(rng.integers(len(xy)))
            if (image, row) in seen:
                continue
            found = centre_track(gt, image, row)
            if found is None or len(found[1]) < min_obs:
                continue
            seen.add((image, row))
            patches.append((image, np.asarray(xy[row], dtype=np.float32), row))
        else:
            width, height = sizes[image]
            centre = np.asarray(
                [rng.uniform(0, width), rng.uniform(0, height)], dtype=np.float32
            )
            d = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1])
            # A hand-placed pixel in empty sky is not a patch. Fifty keypoints
            # inside a quarter of the frame diagonal is the floor.
            if np.partition(d, 54)[54] > 0.25 * math.hypot(width, height):
                continue
            patches.append((image, centre, None))
    if len(patches) < count:
        print(f"only {len(patches)} of {count} patches met the criteria", flush=True)
    return patches


def run_measure2(args) -> None:
    if build_profile() != "release":
        raise SystemExit("refusing to measure a debug build of _sfmtool")

    workspace = Path(args.workspace)
    lazy, positions, offsets, sift_paths, kdf_names, sizes = open_corpus(
        workspace,
        args.kdf,
        Path(args.features) if args.features else None,
        args.cache_mib,
    )
    recon = SfmrReconstruction.load(args.sfmr)
    kdf_to_sfmr = align_images(kdf_names, list(recon.image_names))
    gt = GroundTruth(recon, kdf_to_sfmr)
    registered = np.flatnonzero(kdf_to_sfmr >= 0)
    print(
        f"{len(kdf_names)} indexed images, {len(registered)} in the ground truth,"
        f" {lazy.len:,} descriptors",
        flush=True,
    )

    prefixes = [int(v) for v in args.prefixes.split(",")]
    cap = max(prefixes)
    knobs = {
        "k": args.k,
        "max_leaf_checks": args.budget,
        "threshold_px": args.threshold,
        "iterations": args.iterations,
        "min_correspondences": args.min_correspondences,
        "one_hit_per_image": True,
        "same_image_ratio": 1.0,
        "min_inliers": args.floor_inliers,
        "max_scale": args.max_scale,
        "seed": args.seed % (1 << 32),
    }
    rng = np.random.default_rng(args.seed)
    patches = sample_patches2(
        rng,
        positions,
        registered,
        gt,
        args.patches,
        args.centres,
        sizes,
        args.min_centre_obs,
    )
    descriptors = ImageDescriptors(sift_paths)

    records, offered_checked, offered_agree = [], 0, 0
    for index, (image, centre, centre_row) in enumerate(patches):
        xy = positions[image]
        d = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1])
        order = np.argsort(d, kind="stable")
        # Hold the centre out. For a keypoint centre that is the centre feature
        # itself; for a pixel centre it is the five nearest, which carry the
        # proxy ground truth. Either way nothing the error is measured on can
        # reach the three-point solve.
        held = np.asarray([centre_row]) if centre_row is not None else order[:5]
        keep = order[~np.isin(order, held)][:cap]
        xy_cap = xy[keep].astype(np.float64)
        query_xy = np.ascontiguousarray(xy[keep], dtype=np.float32)
        ids = (offsets[image] + keep).astype(np.uint32).tolist()
        radius = float(d[keep[-1]])

        record = {
            "image": int(image),
            "centre": [float(centre[0]), float(centre[1])],
            "radius": radius,
            "xy": xy_cap.tolist(),
            "stages": {},
        }
        if centre_row is not None:
            point, obs = centre_track(gt, image, int(centre_row))
            record["centre_gt"] = [
                [o, f, float(positions[o][f][0]), float(positions[o][f][1])]
                for o, f in obs
            ]
        else:
            proxy = []
            for slot, row in enumerate(held):
                found = centre_track(gt, image, int(row))
                if found is None:
                    continue
                for o, f in found[1]:
                    proxy.append(
                        [
                            slot,
                            o,
                            f,
                            float(xy[row][0]),
                            float(xy[row][1]),
                            float(positions[o][f][0]),
                            float(positions[o][f][1]),
                        ]
                    )
            record["proxy_gt"] = proxy

        gt_slot, gt_image, gt_feature = gt.correspondences(image, keep)
        record["gt"] = [
            [int(s), int(o), float(positions[o][f][0]), float(positions[o][f][1])]
            for s, o, f in zip(gt_slot, gt_image, gt_feature)
        ]

        for n in prefixes:
            matches = lazy.constellation_query(
                query_xy[:n], feature_ids=ids[:n], image_index=image, **knobs
            )
            record["stages"][str(n)] = [
                {
                    "i": int(m["image_index"]),
                    "inl": int(m["inliers"]),
                    "corr": int(m["correspondences"]),
                    "A": np.asarray(m["affine"], dtype=np.float64).ravel().tolist(),
                    "in": [
                        [int(q), float(p[0]), float(p[1])]
                        for q, p in zip(
                            m["inlier_correspondences"]["query_index"],
                            m["inlier_correspondences"]["position"],
                        )
                    ],
                }
                for m in matches
            ]

        candidates = {c["i"] for c in record["stages"][str(cap)]}
        per_image = offered_correspondences(
            lazy, offsets, image, descriptors.rows(image, keep), knobs
        )
        record["offered"] = {
            str(other): [
                [q, float(positions[other][f][0]), float(positions[other][f][1])]
                for q, f in rows
            ]
            for other, rows in per_image.items()
            if other in candidates
        }
        # The rebuilt list must be the one the query fitted: same count per
        # candidate image. A mismatch would invalidate every refit arm.
        for candidate in record["stages"][str(cap)]:
            offered_checked += 1
            offered_agree += int(
                len(record["offered"].get(str(candidate["i"]), [])) == candidate["corr"]
            )
        records.append(record)
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(patches)} patches", flush=True)

    stage_ms = {}
    if args.time_stages:
        stage_ms = time_stages2(lazy, positions, offsets, patches, prefixes, knobs)

    out = {
        "label": args.label or workspace.name,
        "workspace": str(workspace),
        "kdf": args.kdf,
        "sfmr": args.sfmr,
        "centres": args.centres,
        "images": len(kdf_names),
        "registered_images": int(len(registered)),
        "descriptors": lazy.len,
        "observations": int(len(gt.image)),
        "cache_mib": args.cache_mib,
        "params": knobs,
        "prefixes": prefixes,
        "patches": len(records),
        "seed": args.seed,
        "offered_checked": offered_checked,
        "offered_agree": offered_agree,
        "stage_ms": stage_ms,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(
        f"wrote {args.out}; rebuilt correspondence lists matched the query's"
        f" count in {offered_agree} of {offered_checked} candidates"
    )


def time_stages2(lazy, positions, offsets, patches, prefixes, knobs):
    """Per-prefix wall time at the production floor, shuffled and warm.

    Same discipline as `timing`: each (patch, prefix) job is run twice in a
    shuffled order and the second run is the one recorded, so no prefix
    inherits the cache another one filled.
    """
    production = dict(knobs, min_inliers=8)
    prepared = []
    for image, centre, centre_row in patches:
        xy = positions[image]
        d = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1])
        order = np.argsort(d, kind="stable")
        held = np.asarray([centre_row]) if centre_row is not None else order[:5]
        keep = order[~np.isin(order, held)][: max(prefixes)]
        prepared.append(
            (
                image,
                np.ascontiguousarray(xy[keep], dtype=np.float32),
                (offsets[image] + keep).astype(np.uint32).tolist(),
            )
        )
    out = [{} for _ in prepared]

    def one(job, keep_it: bool) -> None:
        patch, n = job
        image, query_xy, ids = prepared[patch]
        start = time.perf_counter()
        lazy.constellation_query(
            query_xy[:n], feature_ids=ids[:n], image_index=image, **production
        )
        if keep_it:
            out[patch][str(n)] = (time.perf_counter() - start) * 1e3

    repeat_timed(
        np.random.default_rng(0),
        [(p, n) for p in range(len(prepared)) for n in prefixes],
        one,
    )
    return {
        str(n): median([row[str(n)] for row in out if str(n) in row]) for n in prefixes
    }


# ── Round two: the arms ──────────────────────────────────────────────────────


def fit_affine_ls(src: np.ndarray, dst: np.ndarray, weights=None):
    """Least squares 2x3 affine through correspondences, or `None`.

    The shipped affine is the best *three-point* model with no refinement, so
    part of any gain a staged query shows could be noise reduction rather than
    locality. These refits are the control for that, and they are the cheap
    option: a single query already holds every correspondence they need.
    """
    if len(src) < 3:
        return None
    design = np.column_stack([src, np.ones(len(src))])
    if weights is not None:
        root = np.sqrt(weights)[:, None]
        design, dst = design * root, dst * root
    try:
        solution, *_ = np.linalg.lstsq(design, dst, rcond=None)
    except np.linalg.LinAlgError:
        return None
    affine = solution.T
    if not np.all(np.isfinite(affine)):
        return None
    determinant = affine[0, 0] * affine[1, 1] - affine[0, 1] * affine[1, 0]
    # The same guards the three-point solve applies: a reflected or wildly
    # scaled model is impossible rather than merely unlikely.
    if determinant <= 0 or not (0.25 <= math.sqrt(determinant) <= 4.0):
        return None
    return affine


def apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ affine[:, :2].T + affine[:, 2]


def reselect_refit(affine, src, dst, threshold: float, rounds: int = 3):
    """Re-select inliers under `affine` and refit, up to `rounds` times."""
    current = affine
    for _ in range(rounds):
        residual = np.hypot(*(apply_affine(current, src) - dst).T)
        keep = residual <= threshold
        if keep.sum() < 4:
            return current
        refit = fit_affine_ls(src[keep], dst[keep])
        if refit is None or np.allclose(refit, current):
            return current
        current = refit
    return current


def bar_at(n: int) -> int:
    """The acceptance bar round one recommended: 8, easing below n=20."""
    return min(8, max(5, math.ceil(0.4 * n)))


def stage_index(record: dict, n: int) -> dict:
    return {c["i"]: c for c in record["stages"][str(n)]}


def arm_affines(record: dict, cap: int, stages, threshold: float) -> dict:
    """Every arm's affine for every image the cap query accepted.

    One pass per patch. The image set never changes -- it is always the cap
    query's, at the shipped floor of eight -- so the arms differ in exactly one
    thing, which affine they report, and every comparison below is paired over
    the same (patch, image) cases.
    """
    centre = np.asarray(record["centre"], dtype=np.float64)
    xy = np.asarray(record["xy"], dtype=np.float64)
    radius = record["radius"]
    at = {n: stage_index(record, n) for n in stages}
    accepted = {
        n: {i for i, c in at[n].items() if c["inl"] >= bar_at(n)} for n in stages
    }
    out: dict[int, dict[str, np.ndarray]] = {}

    for image in sorted(accepted[cap]):
        cap_candidate = at[cap][image]
        cap_affine = np.asarray(cap_candidate["A"], dtype=np.float64).reshape(2, 3)
        arms: dict[str, np.ndarray] = {"A_cap": cap_affine}

        inliers = np.asarray(cap_candidate["in"], dtype=np.float64)
        if len(inliers):
            rows = inliers[:, 0].astype(np.int64)
            src, dst = xy[rows], inliers[:, 1:]
            distance = np.hypot(*(src - centre).T)
        else:
            rows = np.zeros(0, dtype=np.int64)
            src = dst = np.zeros((0, 2))
            distance = np.zeros(0)

        offered = np.asarray(
            record["offered"].get(str(image), []), dtype=np.float64
        ).reshape(-1, 3)
        offered_rows = offered[:, 0].astype(np.int64)
        offered_src = xy[offered_rows] if len(offered) else np.zeros((0, 2))
        offered_dst = offered[:, 1:] if len(offered) else np.zeros((0, 2))

        def refit(mask, weights=None) -> np.ndarray:
            model = fit_affine_ls(src[mask], dst[mask], weights)
            return cap_affine if model is None else model

        # C1: least squares on every inlier the cap stage kept.
        arms["C1_ls"] = (
            refit(np.ones(len(src), dtype=bool)) if len(src) >= 3 else cap_affine
        )
        arms["C1_ls_iter"] = (
            reselect_refit(arms["C1_ls"], offered_src, offered_dst, threshold)
            if len(offered) >= 4
            else arms["C1_ls"]
        )
        # C2: least squares on the cap inliers that lie near the centre.
        for near in (15, 25):
            mask = rows < near
            arms[f"C2_ls{near}"] = refit(mask) if mask.sum() >= 4 else arms["C1_ls"]
        # C3: every cap inlier, weighted by distance from the centre.
        for sigma in (0.25, 0.5, 1.0):
            if len(src) >= 3:
                weights = np.exp(-0.5 * (distance / (sigma * radius)) ** 2)
                arms[f"C3_w{sigma}"] = (
                    refit(np.ones(len(src), dtype=bool), weights)
                    if weights.sum() > 1e-9
                    else arms["C1_ls"]
                )
            else:
                arms[f"C3_w{sigma}"] = cap_affine

        # B: the affine of the first stage that accepted this image.
        for start in stages:
            if start == cap:
                continue
            early = at[start].get(image)
            arms[f"B_lock{start}"] = (
                np.asarray(early["A"], dtype=np.float64).reshape(2, 3)
                if image in accepted[start]
                else cap_affine
            )
        for first, second in ((15, 30), (20, 35)):
            if first in at and second in at:
                if image in accepted[first]:
                    pick = at[first][image]
                elif image in accepted[second]:
                    pick = at[second][image]
                else:
                    pick = cap_candidate
                arms[f"B_lock{first}_{second}"] = np.asarray(
                    pick["A"], dtype=np.float64
                ).reshape(2, 3)

        # C4: the stage-25 lock, refitted by least squares on its own inliers.
        early = at.get(25, {}).get(image)
        if early is not None and image in accepted.get(25, set()):
            early_inliers = np.asarray(early["in"], dtype=np.float64)
            model = None
            if len(early_inliers) >= 4:
                model = fit_affine_ls(
                    xy[early_inliers[:, 0].astype(np.int64)], early_inliers[:, 1:]
                )
            arms["C4_lock25_ls"] = (
                model
                if model is not None
                else np.asarray(early["A"], dtype=np.float64).reshape(2, 3)
            )
            # The hybrid: the early warp only *selects*, and least squares does
            # the rest. Selecting among the correspondences below row 25 would
            # be the stage-25 inlier set exactly -- a feature's hits do not
            # depend on the constellation, so "offered rows < 25 that agree
            # with the stage-25 model" *is* that model's inlier set, and the
            # arm would be C4 under another name. So the early warp selects
            # over the whole cap disc instead, which is a claim it has not
            # already made.
            early_affine = np.asarray(early["A"], dtype=np.float64).reshape(2, 3)
            if len(offered) >= 4:
                predicted = apply_affine(early_affine, offered_src)
                agree = np.hypot(*(predicted - offered_dst).T) <= threshold
                chosen = (
                    fit_affine_ls(offered_src[agree], offered_dst[agree])
                    if agree.sum() >= 4
                    else None
                )
                arms["C5_hybrid25"] = cap_affine if chosen is None else chosen
            else:
                arms["C5_hybrid25"] = cap_affine
        else:
            arms["C4_lock25_ls"] = cap_affine
            arms["C5_hybrid25"] = cap_affine

        # D: a guard on the lock. Whichever of the two affines explains more of
        # the cap's own correspondences near the centre wins, ties to the early
        # one, which is the defence against locking onto a wrong early model.
        def votes(model, points, places) -> int:
            if not len(points):
                return 0
            return int(
                (np.hypot(*(apply_affine(model, points) - places).T) <= threshold).sum()
            )

        for start in (20, 25, 30):
            early = at.get(start, {}).get(image)
            if early is None or image not in accepted.get(start, set()):
                arms[f"D_guard{start}"] = cap_affine
                arms[f"D_strict{start}"] = cap_affine
                continue
            early_affine = np.asarray(early["A"], dtype=np.float64).reshape(2, 3)
            near = offered_rows < start
            if near.sum() == 0:
                arms[f"D_guard{start}"] = early_affine
                arms[f"D_strict{start}"] = early_affine
                continue
            source, target = offered_src[near], offered_dst[near]
            early_votes = votes(early_affine, source, target)
            cap_votes = votes(cap_affine, source, target)
            arms[f"D_guard{start}"] = (
                early_affine if early_votes >= cap_votes else cap_affine
            )
            # The plain guard nearly always keeps the early model, because the
            # early model was fitted to maximise agreement on exactly the
            # correspondences the guard counts. The strict form asks something
            # the early model has not already answered: does it also explain
            # the correspondences it never saw, out to the rim of the disc?
            early_all = votes(early_affine, offered_src, offered_dst)
            cap_all = votes(cap_affine, offered_src, offered_dst)
            arms[f"D_strict{start}"] = (
                early_affine
                if early_votes >= cap_votes and early_all >= 0.8 * cap_all
                else cap_affine
            )

        out[image] = arms
    return out


def score_arms(record: dict, arms_by_image: dict, cap: int, stages) -> dict:
    """Centre error and the warp-quality columns, per arm, for one patch."""
    centre = np.asarray(record["centre"], dtype=np.float64)
    xy = np.asarray(record["xy"], dtype=np.float64)
    radius = record["radius"]
    gt_rows = np.asarray(record["gt"], dtype=np.float64).reshape(-1, 4)
    centre_obs = {
        int(o): np.asarray([x, y])
        for o, _f, x, y in np.asarray(
            record.get("centre_gt", []), dtype=np.float64
        ).reshape(-1, 4)
    }
    proxy = np.asarray(record.get("proxy_gt", []), dtype=np.float64).reshape(-1, 7)

    out: dict[str, list[dict]] = {}
    for image, arms in arms_by_image.items():
        take = gt_rows[:, 1].astype(np.int64) == image
        disc_src = xy[gt_rows[take, 0].astype(np.int64)] if take.any() else None
        disc_dst = gt_rows[take, 2:] if take.any() else None
        near15 = (
            gt_rows[take, 0].astype(np.int64) < 15 if take.any() else np.zeros(0, bool)
        )
        truth_local = None
        if take.any() and near15.sum() >= 5:
            truth_local = fit_affine_ls(disc_src[near15], disc_dst[near15])

        target = centre_obs.get(image)
        proxy_here = (
            proxy[proxy[:, 1].astype(np.int64) == image] if len(proxy) else None
        )

        for name, affine in arms.items():
            row = {"image": image}
            scale = math.sqrt(
                abs(affine[0, 0] * affine[1, 1] - affine[0, 1] * affine[1, 0])
            )
            if target is not None:
                error = float(
                    np.hypot(*(apply_affine(affine, centre[None, :])[0] - target))
                )
                row["centre_err"] = error
                row["centre_norm"] = error / max(radius * scale, 1e-9)
            elif proxy_here is not None and len(proxy_here) >= 2:
                predicted = apply_affine(affine, proxy_here[:, 3:5])
                error = float(np.mean(np.hypot(*(predicted - proxy_here[:, 5:7]).T)))
                row["centre_err"] = error
                row["centre_norm"] = error / max(radius * scale, 1e-9)
            if take.any() and take.sum() >= 3:
                residual = np.hypot(*(apply_affine(affine, disc_src) - disc_dst).T)
                row["disc_med"] = float(np.median(residual))
            if truth_local is not None:
                a, g = affine[:, :2], truth_local[:, :2]
                row["lin_rel"] = float(
                    np.linalg.norm(a - g, "fro") / max(np.linalg.norm(g, "fro"), 1e-9)
                )
                row["scale_ratio"] = float(
                    scale
                    / max(math.sqrt(abs(g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0])), 1e-9)
                )
                row["rot_deg"] = float(
                    abs(
                        math.degrees(
                            math.atan2(a[1, 0], a[0, 0]) - math.atan2(g[1, 0], g[0, 0])
                        )
                    )
                )
            out.setdefault(name, []).append(row)
    return out


def arm_summary(cases: list[dict]) -> dict:
    errors = np.asarray([c["centre_err"] for c in cases if "centre_err" in c])
    disc = np.asarray([c["disc_med"] for c in cases if "disc_med" in c])
    norm = np.asarray([c["centre_norm"] for c in cases if "centre_norm" in c])
    linear = np.asarray([c["lin_rel"] for c in cases if "lin_rel" in c])
    rotation = np.asarray([c["rot_deg"] for c in cases if "rot_deg" in c])

    def q(values, p):
        return float(np.percentile(values, p)) if len(values) else None

    return {
        "cases": int(len(errors)),
        "median": q(errors, 50),
        "p75": q(errors, 75),
        "p90": q(errors, 90),
        "le2": float((errors <= 2).mean()) if len(errors) else None,
        "le3": float((errors <= 3).mean()) if len(errors) else None,
        "le5": float((errors <= 5).mean()) if len(errors) else None,
        "gt10": float((errors > 10).mean()) if len(errors) else None,
        "norm_median": q(norm, 50),
        "norm_p90": q(norm, 90),
        "disc_cases": int(len(disc)),
        "disc_ok3": float((disc <= 3).mean()) if len(disc) else None,
        "disc_median": q(disc, 50),
        "lin_cases": int(len(linear)),
        "lin_rel_median": q(linear, 50),
        "rot_median": q(rotation, 50),
    }


def paired_compare(per_patch, left: str, right: str, draws: int, seed: int = 0) -> dict:
    """Paired bootstrap over patches for two arms on the same cases.

    Cases inside a patch share a constellation and a centre, so they are
    correlated and resampling them would understate the interval. The patch is
    the independent unit, so the patch is what is resampled.
    """
    patches = []
    wins = losses = ties = 0
    for cases in per_patch:
        a = {c["image"]: c for c in cases.get(left, []) if "centre_err" in c}
        b = {c["image"]: c for c in cases.get(right, []) if "centre_err" in c}
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        left_errors = np.asarray([a[i]["centre_err"] for i in shared])
        right_errors = np.asarray([b[i]["centre_err"] for i in shared])
        # A tenth of a pixel, or a tenth of the larger error: below that the two
        # affines put the centre in the same place.
        slack = np.maximum(0.1, 0.1 * np.maximum(left_errors, right_errors))
        wins += int((left_errors < right_errors - slack).sum())
        losses += int((right_errors < left_errors - slack).sum())
        ties += int(
            len(shared)
            - (left_errors < right_errors - slack).sum()
            - (right_errors < left_errors - slack).sum()
        )
        patches.append((left_errors, right_errors))
    if not patches:
        return {"cases": 0}

    rng = np.random.default_rng(seed)
    share, median_gap = [], []
    index = rng.integers(0, len(patches), (draws, len(patches)))
    for draw in index:
        left_all = np.concatenate([patches[i][0] for i in draw])
        right_all = np.concatenate([patches[i][1] for i in draw])
        share.append((left_all <= 3).mean() - (right_all <= 3).mean())
        median_gap.append(np.median(left_all) - np.median(right_all))
    left_all = np.concatenate([p[0] for p in patches])
    right_all = np.concatenate([p[1] for p in patches])
    return {
        "cases": int(len(left_all)),
        "patches": len(patches),
        "le3_left": float((left_all <= 3).mean()),
        "le3_right": float((right_all <= 3).mean()),
        "d_le3": float((left_all <= 3).mean() - (right_all <= 3).mean()),
        "d_le3_ci": [
            float(np.percentile(share, 2.5)),
            float(np.percentile(share, 97.5)),
        ],
        "d_median": float(np.median(left_all) - np.median(right_all)),
        "d_median_ci": [
            float(np.percentile(median_gap, 2.5)),
            float(np.percentile(median_gap, 97.5)),
        ],
        "win": wins,
        "loss": losses,
        "tie": ties,
    }


def harm_cases(per_patch, records, left: str, right: str, limit: int = 40) -> dict:
    """Where the left arm is much worse than the right, and what those look like."""
    worse5, gross, listed = 0, 0, []
    for cases, record in zip(per_patch, records):
        a = {c["image"]: c for c in cases.get(left, []) if "centre_err" in c}
        b = {c["image"]: c for c in cases.get(right, []) if "centre_err" in c}
        for image in sorted(set(a) & set(b)):
            gap = a[image]["centre_err"] - b[image]["centre_err"]
            if gap > 5:
                worse5 += 1
            if a[image]["centre_err"] > 10 and b[image]["centre_err"] <= 3:
                gross += 1
                if len(listed) < limit:
                    stages = {
                        n: next(
                            (
                                (c["inl"], c["corr"])
                                for c in record["stages"][n]
                                if c["i"] == image
                            ),
                            None,
                        )
                        for n in record["stages"]
                    }
                    listed.append(
                        {
                            "patch_image": record["image"],
                            "candidate": image,
                            "left_err": a[image]["centre_err"],
                            "right_err": b[image]["centre_err"],
                            "stages": stages,
                        }
                    )
    return {"worse_by_5px": worse5, "gross": gross, "examples": listed}


def run_cost2(args) -> None:
    """Re-time one round-2 table's stages, searches and refits, in place.

    What a staged implementation pays over a single query is one extra fit per
    extra stage, because the forest search of the first `S` features is reused.
    So the three numbers to measure are the whole query at each prefix, the bare
    search at each prefix, and how long a least-squares refit takes. All three
    are timed with the shuffled, second-of-two-runs discipline, except the
    refits, which are timed in numpy and are therefore an **upper bound** on
    what the same arithmetic costs in Rust.
    """
    if build_profile() != "release":
        raise SystemExit("refusing to measure a debug build of _sfmtool")
    raw = json.loads(Path(args.out).read_text())
    lazy, positions, offsets, sift_paths, _, _ = open_corpus(
        Path(raw["workspace"]),
        raw["kdf"],
        Path(args.features) if args.features else None,
        raw["cache_mib"],
    )
    prefixes = raw["prefixes"]
    knobs = dict(raw["params"], min_inliers=8)
    subset = raw["records"][: args.patches]
    descriptors = ImageDescriptors(sift_paths)

    prepared = []
    for record in subset:
        image = record["image"]
        xy = np.asarray(record["xy"], dtype=np.float32)
        rows = np.asarray(
            [
                int(np.flatnonzero(np.all(positions[image] == point, axis=1))[0])
                for point in xy
            ]
        )
        prepared.append(
            (
                image,
                np.ascontiguousarray(xy),
                (offsets[image] + rows).astype(np.uint32).tolist(),
                np.ascontiguousarray(descriptors.rows(image, rows), dtype=np.uint8),
            )
        )

    query_ms = [{} for _ in prepared]
    search_ms = [{} for _ in prepared]

    def one_query(job, keep: bool) -> None:
        patch, n = job
        image, xy, ids, _ = prepared[patch]
        start = time.perf_counter()
        lazy.constellation_query(
            xy[:n], feature_ids=ids[:n], image_index=image, **knobs
        )
        if keep:
            query_ms[patch][str(n)] = (time.perf_counter() - start) * 1e3

    def one_search(job, keep: bool) -> None:
        patch, n = job
        block = prepared[patch][3][:n]
        start = time.perf_counter()
        lazy.query(block, k=knobs["k"], max_leaf_checks=knobs["max_leaf_checks"])
        if keep:
            search_ms[patch][str(n)] = (time.perf_counter() - start) * 1e3

    jobs = [(p, n) for p in range(len(prepared)) for n in prefixes]
    repeat_timed(np.random.default_rng(0), jobs, one_query)
    repeat_timed(np.random.default_rng(1), jobs, one_search)

    # The refit, timed over the table's own inlier sets.
    cap = max(prefixes)
    refit_us = []
    for record in subset:
        xy = np.asarray(record["xy"], dtype=np.float64)
        for candidate in record["stages"][str(cap)]:
            if candidate["inl"] < 8 or len(candidate["in"]) < 3:
                continue
            block = np.asarray(candidate["in"], dtype=np.float64)
            src, dst = xy[block[:, 0].astype(np.int64)], block[:, 1:]
            start = time.perf_counter()
            fit_affine_ls(src, dst)
            refit_us.append((time.perf_counter() - start) * 1e6)

    raw["cost2"] = {
        "patches": len(prepared),
        "query_ms": {
            str(n): median([row[str(n)] for row in query_ms if str(n) in row])
            for n in prefixes
        },
        "search_ms": {
            str(n): median([row[str(n)] for row in search_ms if str(n) in row])
            for n in prefixes
        },
        "refit_us_median": median(refit_us),
        "refit_us_p90": float(np.percentile(refit_us, 90)) if refit_us else None,
        "refits": len(refit_us),
    }
    Path(args.out).write_text(json.dumps(raw))
    print(f"updated {args.out} with cost2")


def run_analyze2(args) -> None:
    results = {}
    for path in args.inputs:
        raw = json.loads(Path(path).read_text())
        label = raw["label"] + ("" if raw["centres"] == "keypoint" else "-pixel")
        stages = raw["prefixes"]
        cap = max(stages)
        threshold = raw["params"]["threshold_px"]

        per_patch, grew, total = [], 0, 0
        for record in raw["records"]:
            arms = arm_affines(record, cap, stages, threshold)
            per_patch.append(score_arms(record, arms, cap, stages))
            # How many candidates gained a correspondence between stage 25 and
            # the cap: the ones that did not need no refit at all.
            at25 = {c["i"]: c["corr"] for c in record["stages"].get("25", [])}
            for candidate in record["stages"][str(cap)]:
                if candidate["inl"] >= 8 and candidate["i"] in at25:
                    total += 1
                    grew += int(candidate["corr"] > at25[candidate["i"]])

        names = sorted({name for cases in per_patch for name in cases})
        pooled = {
            name: arm_summary([c for cases in per_patch for c in cases.get(name, [])])
            for name in names
        }
        best = max(
            (n for n in names if n.startswith(("B_", "D_"))),
            key=lambda n: (pooled[n]["le3"] or 0, -(pooled[n]["median"] or 1e9)),
            default="B_lock25",
        )
        best_refit = max(
            (n for n in names if n.startswith("C")),
            key=lambda n: (pooled[n]["le3"] or 0, -(pooled[n]["median"] or 1e9)),
            default="C1_ls",
        )
        compare = {
            f"{name}|A_cap": paired_compare(per_patch, name, "A_cap", args.draws)
            for name in names
            if name != "A_cap"
        }
        compare[f"{best}|{best_refit}"] = paired_compare(
            per_patch, best, best_refit, args.draws
        )

        # Images an early stage accepts that the cap rejects: round one said
        # these are vanishingly rare, and this is the check at higher power.
        early_only = {}
        for start in stages:
            if start == cap:
                continue
            count = true_count = 0
            for record in raw["records"]:
                at_cap = {
                    c["i"]
                    for c in record["stages"][str(cap)]
                    if c["inl"] >= bar_at(cap)
                }
                gt_images = {int(r[1]) for r in record["gt"]}
                for candidate in record["stages"][str(start)]:
                    if (
                        candidate["inl"] >= bar_at(start)
                        and candidate["i"] not in at_cap
                    ):
                        count += 1
                        true_count += int(candidate["i"] in gt_images)
            early_only[str(start)] = {
                "per_patch": count / len(raw["records"]),
                "gt_true": true_count,
                "count": count,
            }

        results[label] = {
            "images": raw["images"],
            "descriptors": raw["descriptors"],
            "observations": raw["observations"],
            "patches": raw["patches"],
            "centres": raw["centres"],
            "cache_mib": raw["cache_mib"],
            "offered_agree": [raw["offered_agree"], raw["offered_checked"]],
            "stage_ms": raw.get("stage_ms", {}),
            "arms": pooled,
            "best_staged": best,
            "best_refit": best_refit,
            "compare": compare,
            "harm_best_vs_cap": harm_cases(per_patch, raw["records"], best, "A_cap"),
            "harm_best_vs_refit": harm_cases(
                per_patch, raw["records"], best, best_refit
            ),
            "grew_between_25_and_cap": [grew, total],
            "early_only": early_only,
        }
        print(f"{label}: {raw['patches']} patches, best staged {best}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    m = sub.add_parser("measure", help="run every prefix query and store the table")
    m.add_argument("--workspace", required=True)
    m.add_argument("--kdf", required=True)
    m.add_argument("--sfmr", required=True)
    m.add_argument("--features", help="directory of .sift files, when not derivable")
    m.add_argument("--out", required=True)
    m.add_argument("--label", default="")
    m.add_argument("--patches", type=int, default=60)
    m.add_argument("--centres", choices=("keypoint", "pixel"), default="keypoint")
    m.add_argument("--prefixes", default=",".join(str(n) for n in DEFAULT_PREFIXES))
    m.add_argument("--batches", default=",".join(str(n) for n in DEFAULT_BATCHES))
    m.add_argument("--cache-mib", type=int, default=64)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--k", type=int, default=32)
    m.add_argument("--budget", type=int, default=512)
    m.add_argument("--threshold", type=float, default=8.0)
    m.add_argument("--iterations", type=int, default=200)
    m.add_argument("--min-correspondences", type=int, default=3)
    m.add_argument(
        "--floor-inliers",
        type=int,
        default=4,
        help="the floor every query runs at, so the bar can be swept offline",
    )
    m.add_argument("--production-inliers", type=int, default=8)
    m.add_argument("--max-scale", type=float, default=4.0)
    m.add_argument(
        "--no-cold-pass",
        dest="cold_pass",
        action="store_false",
        help="skip the cold and production timing passes",
    )

    t = sub.add_parser(
        "timing",
        help="re-time one table's queries warm, shuffled, and fold the result in",
    )
    t.add_argument(
        "--out", required=True, help="the table to re-time, updated in place"
    )
    t.add_argument("--features", help="directory of .sift files, when not derivable")

    b = sub.add_parser("build-kdf", help="build a .kdf over a workspace's .sift files")
    b.add_argument("--workspace", required=True)
    b.add_argument("--features", action="append", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--sfmr", help="keep only images this reconstruction names")
    b.add_argument("--stride", type=int, default=1, help="keep every Nth image")
    b.add_argument("--trees", type=int, default=4)
    b.add_argument("--leaf-size", type=int, default=16)
    b.add_argument("--block-bytes", type=int, default=2 * KIB)
    b.add_argument("--chunk-bytes", type=int, default=1 * MIB)
    b.add_argument("--seed", type=int, default=0)

    m2 = sub.add_parser(
        "measure2", help="round-2 table: held-out centre, affines and correspondences"
    )
    m2.add_argument("--workspace", required=True)
    m2.add_argument("--kdf", required=True)
    m2.add_argument("--sfmr", required=True)
    m2.add_argument("--features")
    m2.add_argument("--out", required=True)
    m2.add_argument("--label", default="")
    m2.add_argument("--patches", type=int, default=400)
    m2.add_argument("--centres", choices=("keypoint", "pixel"), default="keypoint")
    m2.add_argument("--prefixes", default=",".join(str(n) for n in ROUND2_PREFIXES))
    m2.add_argument("--cache-mib", type=int, default=1024)
    m2.add_argument("--seed", type=int, default=20260918)
    m2.add_argument("--k", type=int, default=32)
    m2.add_argument("--budget", type=int, default=512)
    m2.add_argument("--threshold", type=float, default=8.0)
    m2.add_argument("--iterations", type=int, default=200)
    m2.add_argument("--min-correspondences", type=int, default=3)
    m2.add_argument("--floor-inliers", type=int, default=4)
    m2.add_argument("--max-scale", type=float, default=4.0)
    m2.add_argument("--time-stages", action="store_true", help="also time each stage")
    m2.add_argument(
        "--min-centre-obs",
        type=int,
        default=2,
        help="other images of this corpus the centre's track must reach; 1 for a"
        " corpus that subsamples a solved capture, whose tracks are cut short",
    )

    c2 = sub.add_parser(
        "cost2", help="re-time one round-2 table's stages, searches and refits"
    )
    c2.add_argument(
        "--out", required=True, help="the table to re-time, updated in place"
    )
    c2.add_argument("--features")
    c2.add_argument("--patches", type=int, default=60)

    a2 = sub.add_parser("analyze2", help="score the round-2 arms and compare them")
    a2.add_argument("inputs", nargs="+")
    a2.add_argument("--out", required=True)
    a2.add_argument("--draws", type=int, default=2000)

    a = sub.add_parser("analyze", help="replay schedules and rules over the tables")
    a.add_argument("inputs", nargs="+")
    a.add_argument("--out", required=True)
    a.add_argument("--bars", default=",".join(DEFAULT_BARS))
    a.add_argument(
        "--schedule-bars",
        default="c6,c8,s4_0.5_8,s5_0.4_8,r6_0.3",
        help="bars the schedule replay runs at; the full --bars list is swept"
        " per stage, which is where the bar itself is chosen",
    )
    a.add_argument("--starts", type=int, nargs="+", default=[5, 10, 15, 20, 25])
    a.add_argument("--increments", nargs="+", default=["5", "10", "15", "20", "x2"])
    a.add_argument("--smalls", type=int, nargs="+", default=[10, 15, 20, 25])

    args = p.parse_args()
    if args.mode == "measure":
        run_measure(args)
    elif args.mode == "timing":
        run_timing(args)
    elif args.mode == "build-kdf":
        run_build_kdf(args)
    elif args.mode == "measure2":
        run_measure2(args)
    elif args.mode == "cost2":
        run_cost2(args)
    elif args.mode == "analyze2":
        run_analyze2(args)
    else:
        run_analyze(args)


if __name__ == "__main__":
    main()
