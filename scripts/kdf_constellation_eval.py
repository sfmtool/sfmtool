# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure how large a patch constellation has to be to find the right images.

The constellation query answers "which other photographs contain this patch" by
looking every feature inside a radius up in a `.kdf` index and asking which
candidate images hold them under one consistent affine. How many features that
takes is an empirical question, and this script answers it against a
reconstruction's own tracks.

For a workspace with a `.kdf` over its `.sift` files and a `sift_files` ground
truth `.sfmr`, it samples patches centred on random keypoints, grows the radius
around each until the constellation holds roughly N features for a sweep of N,
and for every size records:

* **image recall and precision** against the images the reconstruction says are
  covisible with the constellation, at a floor of one and of three shared
  observations;
* **correspondence recall**, the fraction of the ground truth's own
  correspondences among the constellation's features that come back as inliers;
* **warp accuracy**, the pixel residual of those ground-truth correspondences
  under the affine the query fitted, median and 90th percentile, and the share
  of found images whose warp puts them within a few pixels;
* **false candidates**, images the query reports that share no ground-truth
  correspondence with the constellation, and their inlier counts, because those
  are what a wrong warp would seed; and, as a floor that a sparse solve cannot
  inflate, the ones that share no point with the query image anywhere in it;
* **wall time** per query.

Parameter sweeps over `k`, `max_leaf_checks`, `min_inliers`, `threshold_px` and
`iterations` run at one chosen size, one knob at a time with the rest at their
defaults.

The ground truth is the reconstruction's tracks, which came from the same SIFT
descriptors the index holds, so "recall" here is agreement with that solve and
not with the world: a correspondence the solve never made counts against the
query as a false candidate. `--dump-disagreements` writes image crops for a few
of them so they can be judged by eye.

Run:
    pixi run -e test python scripts/kdf_constellation_eval.py \
        --workspace WS --kdf capture.kdf --sfmr WS/sfmr/solve.sfmr --out out.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sfmtool._sfmtool import build_profile
from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.spatial import LazyKdForest
from sfmtool.sift.file import SiftReader

MIB = 1 << 20
DEFAULT_SIZES = (10, 25, 50, 100, 200, 400, 800)


# ── Corpus and ground-truth tables ───────────────────────────────────────────


def resolve_sift_paths(workspace: Path, names: list[str], features: Path | None):
    """A `.sift` path for every image the `.kdf` names, in index order."""
    paths = []
    for name in names:
        if features is not None:
            candidate = features / (Path(name).name + ".sift")
            if candidate.exists():
                paths.append(candidate)
                continue
        image = workspace / name
        from sfmtool.sift.file import get_sift_path_for_image

        candidate = get_sift_path_for_image(image)
        if not candidate.exists():
            raise SystemExit(f"no .sift for {name}: looked at {candidate}")
        paths.append(candidate)
    return paths


def load_keypoints(sift_paths):
    """Per-image `(K, 2)` float32 keypoint positions."""
    positions = []
    for path in sift_paths:
        reader = SiftReader(path)
        xy, _ = reader.read_positions_and_shapes()
        reader.close()
        positions.append(np.asarray(xy, dtype=np.float32))
    return positions


def corpus_offsets(lazy, positions):
    """Row of each image's first feature, checked against the file's origins.

    The corpus is the per-image descriptor blocks concatenated in image order,
    so the offsets are a cumulative sum; the check is there because every
    feature ID this script maps back to an image depends on that holding.
    """
    counts = np.array([len(p) for p in positions], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    probe = np.concatenate([offsets[:-1], offsets[1:] - 1]).astype(np.uint32)
    origin_images, origin_features = lazy.resolve_origins(probe.tolist())
    images = np.asarray(origin_images, dtype=np.int64)
    features = np.asarray(origin_features, dtype=np.int64)
    n = len(counts)
    want_images = np.concatenate([np.arange(n), np.arange(n)])
    want_features = np.concatenate([np.zeros(n, dtype=np.int64), counts - 1])
    if not (
        np.array_equal(images, want_images) and np.array_equal(features, want_features)
    ):
        raise SystemExit("the .kdf corpus is not the images concatenated in order")
    return offsets


def align_images(kdf_names: list[str], sfmr_names: list[str]):
    """`.kdf` image index -> `.sfmr` image index, or -1 when the solve dropped it."""
    exact = {name: i for i, name in enumerate(sfmr_names)}
    by_base: dict[str, list[int]] = {}
    for i, name in enumerate(sfmr_names):
        by_base.setdefault(Path(name).name, []).append(i)
    out = np.full(len(kdf_names), -1, dtype=np.int64)
    for i, name in enumerate(kdf_names):
        if name in exact:
            out[i] = exact[name]
            continue
        hits = by_base.get(Path(name).name, [])
        if len(hits) == 1:
            out[i] = hits[0]
        elif len(hits) > 1:
            raise SystemExit(f"{name} matches several .sfmr images by basename")
    return out


class GroundTruth:
    """The reconstruction's observations, indexed both ways.

    `point_of` answers "which point does feature f of image i belong to"; the
    CSR arrays answer "which observations does point p have". Together they turn
    a constellation into the set of correspondences the solve believes in.
    """

    def __init__(self, recon: SfmrReconstruction, kdf_to_sfmr: np.ndarray):
        if recon.feature_source != "sift_files":
            raise SystemExit(
                f"ground truth is {recon.feature_source!r}; only sift_files tracks"
                " carry the .sift feature indexes this comparison needs"
            )
        sfmr_to_kdf = np.full(len(recon.image_names), -1, dtype=np.int64)
        for kdf_index, sfmr_index in enumerate(kdf_to_sfmr):
            if sfmr_index >= 0:
                sfmr_to_kdf[sfmr_index] = kdf_index
        image = sfmr_to_kdf[np.asarray(recon.track_image_indexes, dtype=np.int64)]
        feature = np.asarray(recon.track_feature_indexes, dtype=np.int64)
        point = np.asarray(recon.track_point_indexes, dtype=np.int64)
        keep = image >= 0
        self.image, self.feature, self.point = image[keep], feature[keep], point[keep]

        # (image, feature) -> observation row, as a sorted key array.
        self.keys = (self.image << np.int64(32)) | self.feature
        order = np.argsort(self.keys, kind="stable")
        self.keys = self.keys[order]
        self.key_point = self.point[order]

        # point -> its observation rows, CSR over point index.
        by_point = np.argsort(self.point, kind="stable")
        self.obs_image = self.image[by_point]
        self.obs_feature = self.feature[by_point]
        counts = np.bincount(self.point, minlength=int(self.point.max()) + 1)
        self.point_start = np.concatenate([[0], np.cumsum(counts)])
        self._covisible: dict[int, set[int]] = {}

    def covisible(self, image: int) -> set[int]:
        """Images sharing at least one point with `image`, anywhere in it.

        A far weaker claim than sharing a correspondence inside one patch, and
        that is the point: a solve whose tracks are sparse leaves patches with
        no ground truth at all, so a candidate outside this set is wrong under
        an assumption the sparsity cannot undermine.
        """
        if image not in self._covisible:
            points = np.unique(self.point[self.image == image])
            others: set[int] = set()
            for point in points:
                lo, hi = self.point_start[point], self.point_start[point + 1]
                others.update(int(i) for i in self.obs_image[lo:hi])
            others.discard(int(image))
            self._covisible[image] = others
        return self._covisible[image]

    def point_of(self, image: int, features: np.ndarray) -> np.ndarray:
        """Point index per feature of `image`, -1 where the solve used none."""
        want = (np.int64(image) << np.int64(32)) | features.astype(np.int64)
        pos = np.searchsorted(self.keys, want)
        pos = np.clip(pos, 0, len(self.keys) - 1)
        hit = self.keys[pos] == want
        out = np.where(hit, self.key_point[pos], -1)
        return out

    def correspondences(self, image: int, features: np.ndarray):
        """`(query row, other image, other feature)` for a constellation.

        One row per observation, in another image, of a point the constellation
        also observes. These are what the query should be able to rediscover.
        """
        points = self.point_of(image, features)
        rows, others, other_features = [], [], []
        for slot, point in enumerate(points):
            if point < 0:
                continue
            lo, hi = self.point_start[point], self.point_start[point + 1]
            for o in range(lo, hi):
                if self.obs_image[o] == image:
                    continue
                rows.append(slot)
                others.append(self.obs_image[o])
                other_features.append(self.obs_feature[o])
        return (
            np.asarray(rows, dtype=np.int64),
            np.asarray(others, dtype=np.int64),
            np.asarray(other_features, dtype=np.int64),
        )


# ── Patch sampling ───────────────────────────────────────────────────────────


def grow_radius(xy: np.ndarray, centre: np.ndarray, wanted: int):
    """The smallest radius holding `wanted` of `xy`, and the rows inside it.

    Sorting the distances is the same thing as growing the radius and cheaper
    than stepping it, and it makes the constellation size exact rather than
    approximate wherever the image has enough keypoints.
    """
    d = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1])
    order = np.argsort(d, kind="stable")
    if wanted >= len(order):
        return float(d[order[-1]]), order
    radius = float(d[order[wanted - 1]])
    inside = np.flatnonzero(d <= radius)
    return radius, inside


# ── One measurement ──────────────────────────────────────────────────────────


def image_of_feature(offsets: np.ndarray, feature_ids: np.ndarray) -> np.ndarray:
    """The image each corpus feature ID belongs to."""
    return np.searchsorted(offsets, feature_ids, side="right") - 1


def measure(
    lazy,
    gt: GroundTruth,
    positions,
    offsets,
    image: int,
    rows: np.ndarray,
    knobs: dict,
    warp_tolerance_px: float = 3.0,
) -> dict:
    """Run one query and score it against the ground truth."""
    xy = positions[image][rows]
    ids = (offsets[image] + rows).astype(np.uint32)
    t = time.perf_counter()
    matches = lazy.constellation_query(
        np.ascontiguousarray(xy, dtype=np.float32),
        feature_ids=ids.tolist(),
        image_index=image,
        **knobs,
    )
    elapsed_ms = (time.perf_counter() - t) * 1e3

    gt_rows, gt_images, gt_features = gt.correspondences(image, rows)
    # Ground-truth rows are indexes into `rows`; the query reports the same.
    per_image_counts: dict[int, int] = {}
    for other in gt_images:
        per_image_counts[int(other)] = per_image_counts.get(int(other), 0) + 1
    gt1 = {i for i, c in per_image_counts.items() if c >= 1}
    gt3 = {i for i, c in per_image_counts.items() if c >= 3}
    gt_triples = {
        (int(r), int(i), int(f)) for r, i, f in zip(gt_rows, gt_images, gt_features)
    }

    found, found_inliers, recovered = [], {}, set()
    residuals: list[float] = []
    warp_checked, warp_ok = 0, 0
    for match in matches:
        other = int(match["image_index"])
        found.append(other)
        found_inliers[other] = int(match["inliers"])
        columns = match["inlier_correspondences"]
        fids = np.asarray(columns["feature_id"], dtype=np.int64)
        qidx = np.asarray(columns["query_index"], dtype=np.int64)
        imgs = image_of_feature(offsets, fids)
        feats = fids - offsets[imgs]
        for q, im, fe in zip(qidx, imgs, feats):
            recovered.add((int(q), int(im), int(fe)))
        # Warp accuracy: where the fitted affine puts this patch's ground-truth
        # correspondences, against where that image's keypoints actually are.
        take = gt_images == other
        if take.any():
            affine = np.asarray(match["affine"], dtype=np.float64)
            src = xy[gt_rows[take]].astype(np.float64)
            predicted = src @ affine[:, :2].T + affine[:, 2]
            actual = positions[other][gt_features[take]].astype(np.float64)
            here = np.hypot(*(predicted - actual).T)
            residuals.extend(here.tolist())
            # A warp is trustworthy for this image when the ground truth's own
            # correspondences land where it says they do. Three is the fewest
            # that can disagree with a three-point model.
            if len(here) >= 3:
                warp_checked += 1
                warp_ok += int(np.median(here) <= warp_tolerance_px)

    found_set = set(found)
    false_candidates = [i for i in found if per_image_counts.get(i, 0) == 0]
    covisible = gt.covisible(image)
    never_covisible = [i for i in found if i not in covisible]
    # How thin RANSAC's consensus is: correspondences offered per candidate
    # image against inliers found, which is the inlier ratio its sample count
    # has to be enough for.
    offered = [int(m["correspondences"]) for m in matches]
    true_inliers = [
        int(m["inliers"])
        for m in matches
        if per_image_counts.get(int(m["image_index"]), 0) > 0
    ]

    def ratio(num: int, den: int) -> float | None:
        return float(num) / den if den else None

    return {
        "image": int(image),
        "features": int(len(rows)),
        "gt_correspondences": len(gt_triples),
        "gt_images_1": len(gt1),
        "gt_images_3": len(gt3),
        "found_images": len(found_set),
        "recall_1": ratio(len(found_set & gt1), len(gt1)),
        "recall_3": ratio(len(found_set & gt3), len(gt3)),
        "precision_1": ratio(len(found_set & gt1), len(found_set)),
        "precision_3": ratio(len(found_set & gt3), len(found_set)),
        "correspondence_recall": ratio(len(gt_triples & recovered), len(gt_triples)),
        "residual_median": float(np.median(residuals)) if residuals else None,
        "residual_p90": float(np.percentile(residuals, 90)) if residuals else None,
        "residual_count": len(residuals),
        "false_candidates": len(false_candidates),
        "false_candidate_inliers": [found_inliers[i] for i in false_candidates],
        "false_candidate_images": [int(i) for i in false_candidates],
        "false_candidate_affines": [
            np.asarray(m["affine"]).tolist()
            for m in matches
            if int(m["image_index"]) in set(false_candidates)
        ],
        "never_covisible": len(never_covisible),
        "never_covisible_inliers": [found_inliers[i] for i in never_covisible],
        "warp_checked": warp_checked,
        "warp_ok": warp_ok,
        "warp_ok_rate": ratio(warp_ok, warp_checked),
        "correspondences_median": float(np.median(offered)) if offered else None,
        "true_inliers_median": float(np.median(true_inliers)) if true_inliers else None,
        "ms": elapsed_ms,
    }


# ── Aggregation and reporting ────────────────────────────────────────────────


def summarize(records: list[dict]) -> dict:
    """Medians over patches, plus the sums that ratios of ratios would hide."""

    def med(key):
        values = [r[key] for r in records if r[key] is not None]
        return float(np.median(values)) if values else None

    def mean(key):
        values = [r[key] for r in records if r[key] is not None]
        return float(np.mean(values)) if values else None

    inliers = [i for r in records for i in r["false_candidate_inliers"]]
    strict = [i for r in records for i in r["never_covisible_inliers"]]
    checked = sum(r["warp_checked"] for r in records)
    return {
        "warp_ok_rate": (
            sum(r["warp_ok"] for r in records) / checked if checked else None
        ),
        "warp_checked": checked,
        "never_covisible_mean": mean("never_covisible"),
        "never_covisible_inliers_median": (
            float(np.median(strict)) if strict else None
        ),
        "never_covisible_inliers_max": int(max(strict)) if strict else None,
        "patches": len(records),
        "features_median": med("features"),
        "gt_images_1_median": med("gt_images_1"),
        "gt_images_3_median": med("gt_images_3"),
        "found_images_median": med("found_images"),
        "recall_1": mean("recall_1"),
        "recall_3": mean("recall_3"),
        "precision_1": mean("precision_1"),
        "precision_3": mean("precision_3"),
        "correspondence_recall": mean("correspondence_recall"),
        "residual_median": med("residual_median"),
        "residual_p90": med("residual_p90"),
        "correspondences_median": med("correspondences_median"),
        "true_inliers_median": med("true_inliers_median"),
        "false_candidates_mean": mean("false_candidates"),
        "false_candidate_inliers_median": float(np.median(inliers))
        if inliers
        else None,
        "false_candidate_inliers_max": int(max(inliers)) if inliers else None,
        "ms_median": med("ms"),
    }


def image_paths(sift_paths, kdf_names):
    """The image each `.sift` describes: `<dir>/features/<prefix>/<image>.sift`."""
    return [
        path.parent.parent.parent / path.name.removesuffix(".sift")
        for path, _ in zip(sift_paths, kdf_names)
    ]


def dump_disagreements(records, images, out_dir: Path, limit: int):
    """Crop each false candidate beside the patch it claims to contain.

    The query patch is cut at its own centre and radius; the candidate is cut
    from the box the fitted affine maps that patch into, and resampled to the
    same size. If the two crops show the same piece of surface the query found
    something the solve missed; if they do not, the candidate is wrong.
    """
    try:
        import cv2
    except ImportError:
        print("opencv unavailable; skipping crops")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    side = 256
    for record in records:
        if len(written) >= limit:
            break
        left_image = cv2.imread(str(images[record["image"]]))
        if left_image is None:
            continue
        cx, cy = record["centre"]
        radius = record["radius"]
        for other, inliers, affine in zip(
            record["false_candidate_images"],
            record["false_candidate_inliers"],
            record["false_candidate_affines"],
        ):
            if len(written) >= limit:
                break
            right_image = cv2.imread(str(images[other]))
            if right_image is None:
                continue
            a = np.asarray(affine, dtype=np.float64)
            # Map the patch's bounding box through the warp and cut its hull.
            corners = np.array(
                [
                    [cx - radius, cy - radius],
                    [cx + radius, cy - radius],
                    [cx + radius, cy + radius],
                    [cx - radius, cy + radius],
                ]
            )
            warped = corners @ a[:, :2].T + a[:, 2]
            lo, hi = warped.min(0), warped.max(0)
            crops = [
                crop(
                    left_image, cx - radius, cy - radius, 2 * radius, 2 * radius, side
                ),
                crop(right_image, lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1], side),
            ]
            scale = float(np.sqrt(abs(np.linalg.det(a[:, :2]))))
            name = (
                f"false-{record['image']}-vs-{other}"
                f"-n{record['features']}-in{inliers}-scale{scale:.2f}.jpg"
            )
            cv2.imwrite(str(out_dir / name), np.hstack(crops))
            written.append(
                {
                    "file": name,
                    "query_image": images[record["image"]].name,
                    "candidate_image": images[other].name,
                    "constellation_features": record["features"],
                    "radius": radius,
                    "inliers": inliers,
                    "warp_scale": scale,
                }
            )
    return written


def crop(image, x: float, y: float, width: float, height: float, side: int):
    """A `side` x `side` view of the given box, black outside the image."""
    import cv2

    width, height = max(width, 1.0), max(height, 1.0)
    xs = np.linspace(x, x + width, side)
    ys = np.linspace(y, y + height, side)
    gx, gy = np.meshgrid(xs, ys)
    out = cv2.remap(
        image,
        gx.astype(np.float32),
        gy.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--kdf", required=True)
    p.add_argument("--sfmr", required=True)
    p.add_argument("--features", help="directory of .sift files, when not derivable")
    p.add_argument("--out", required=True, help="raw results, as JSON")
    p.add_argument("--label", default="", help="dataset name, carried into the JSON")
    p.add_argument("--patches", type=int, default=40)
    p.add_argument("--sizes", default=",".join(str(n) for n in DEFAULT_SIZES))
    p.add_argument("--sweep-size", type=int, default=100)
    p.add_argument("--sweep-patches", type=int, default=0, help="0 means --patches")
    p.add_argument("--no-sweeps", action="store_true")
    p.add_argument("--cache-mib", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--threshold", type=float, default=8.0)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--min-correspondences", type=int, default=3)
    p.add_argument("--min-inliers", type=int, default=8)
    p.add_argument("--max-scale", type=float, default=4.0)
    p.add_argument("--dump-disagreements", type=int, default=0)
    p.add_argument("--crops-dir")
    args = p.parse_args()

    if build_profile() != "release":
        raise SystemExit("refusing to measure a debug build of _sfmtool")

    workspace = Path(args.workspace)
    budget = args.cache_mib * MIB
    lazy = LazyKdForest(
        args.kdf,
        cache_bytes=budget,
        max_in_flight_bytes=budget,
        max_chunk_bytes=min(budget, 4 * MIB),
        max_compressed_bytes=64 * MIB,
    )
    table = lazy.image_table()
    if table is None:
        raise SystemExit(f"{args.kdf} carries no SIFT sources")
    kdf_names = list(table["names"])
    sift_paths = resolve_sift_paths(
        workspace, kdf_names, Path(args.features) if args.features else None
    )
    positions = load_keypoints(sift_paths)
    offsets = corpus_offsets(lazy, positions)

    recon = SfmrReconstruction.load(args.sfmr)
    kdf_to_sfmr = align_images(kdf_names, list(recon.image_names))
    gt = GroundTruth(recon, kdf_to_sfmr)
    registered = np.flatnonzero(kdf_to_sfmr >= 0)
    print(
        f"{len(kdf_names)} indexed images, {len(registered)} in the ground truth,"
        f" {lazy.len:,} descriptors",
        flush=True,
    )

    sizes = [int(s) for s in args.sizes.split(",")]
    rng = np.random.default_rng(args.seed)
    # Sample patch centres once, so every size and every sweep point sees the
    # same patches and the comparison is on a fixed observation set.
    patches = []
    while len(patches) < args.patches:
        image = int(registered[rng.integers(len(registered))])
        xy = positions[image]
        if len(xy) < 4:
            continue
        centre = xy[rng.integers(len(xy))]
        patches.append((image, np.asarray(centre, dtype=np.float32)))

    base = {
        "k": args.k,
        "max_leaf_checks": args.budget,
        "threshold_px": args.threshold,
        "iterations": args.iterations,
        "min_correspondences": args.min_correspondences,
        "min_inliers": args.min_inliers,
        "max_scale": args.max_scale,
        "seed": args.seed,
    }

    by_size: dict[int, list[dict]] = {}
    for wanted in sizes:
        records = []
        for image, centre in patches:
            radius, rows = grow_radius(positions[image], centre, wanted)
            record = measure(lazy, gt, positions, offsets, image, rows, base)
            record["radius"] = radius
            record["centre"] = [float(centre[0]), float(centre[1])]
            record["wanted"] = wanted
            records.append(record)
        by_size[wanted] = records
        s = summarize(records)
        print(
            f"N={wanted:>4} actual {s['features_median']:>5.0f}"
            f" recall3 {fmt(s['recall_3'])} prec3 {fmt(s['precision_3'])}"
            f" corr {fmt(s['correspondence_recall'])}"
            f" res {fmt(s['residual_median'])}/{fmt(s['residual_p90'])}px"
            f" corr/img {fmt(s['correspondences_median'])}"
            f" inl {fmt(s['true_inliers_median'])}"
            f" warpok {fmt(s['warp_ok_rate'])} false {fmt(s['false_candidates_mean'])}/{fmt(s['never_covisible_mean'])}"
            f" {s['ms_median']:.0f}ms",
            flush=True,
        )

    # The two entry points must agree: same constellation, same answer.
    agreement = check_at_pixel(lazy, positions, sift_paths, patches[:3], base)

    sweeps: dict[str, dict] = {}
    if not args.no_sweeps:
        subset = patches[: (args.sweep_patches or args.patches)]
        for knob, values in (
            ("k", [16, 32, 64]),
            ("max_leaf_checks", [128, 512, 2048]),
            ("min_inliers", [4, 6, 8, 12]),
            ("threshold_px", [4.0, 8.0, 12.0]),
            ("iterations", [200, 1000, 5000]),
        ):
            sweeps[knob] = {}
            for value in values:
                knobs = dict(base, **{knob: value})
                records = []
                for image, centre in subset:
                    _, rows = grow_radius(positions[image], centre, args.sweep_size)
                    records.append(
                        measure(lazy, gt, positions, offsets, image, rows, knobs)
                    )
                sweeps[knob][str(value)] = {
                    "summary": summarize(records),
                    "records": records,
                }
                s = summarize(records)
                print(
                    f"{knob}={value:<5} recall3 {fmt(s['recall_3'])}"
                    f" prec3 {fmt(s['precision_3'])}"
                    f" corr {fmt(s['correspondence_recall'])}"
                    f" res {fmt(s['residual_median'])}/{fmt(s['residual_p90'])}px"
                    f" corr/img {fmt(s['correspondences_median'])}"
                    f" inl {fmt(s['true_inliers_median'])}"
                    f" warpok {fmt(s['warp_ok_rate'])} false {fmt(s['false_candidates_mean'])}/{fmt(s['never_covisible_mean'])}"
                    f" {s['ms_median']:.0f}ms",
                    flush=True,
                )

    crops = []
    if args.dump_disagreements:
        crops = dump_disagreements(
            by_size[args.sweep_size],
            image_paths(sift_paths, kdf_names),
            Path(args.crops_dir or (Path(args.out).parent / "crops")),
            args.dump_disagreements,
        )

    out = {
        "label": args.label or workspace.name,
        "workspace": str(workspace),
        "kdf": args.kdf,
        "sfmr": args.sfmr,
        "images": len(kdf_names),
        "registered_images": int(len(registered)),
        "descriptors": lazy.len,
        "cache_mib": args.cache_mib,
        "params": base,
        "patches": args.patches,
        "seed": args.seed,
        "at_pixel_agreement": agreement,
        "sizes": {
            str(size): {"summary": summarize(records), "records": records}
            for size, records in by_size.items()
        },
        "sweeps": sweeps,
        "sweep_size": args.sweep_size,
        "crops": crops,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


def fmt(value) -> str:
    return "  n/a" if value is None else f"{value:5.2f}"


def check_at_pixel(lazy, positions, sift_paths, patches, knobs) -> list[dict]:
    """`constellation_at_pixel` against `constellation_query` on the same patch.

    The pixel entry point selects the constellation itself, from the `.sift`
    file, so agreement here is evidence that the radius this script grows and
    the radius the query applies are the same set of features.
    """
    out = []
    for image, centre in patches:
        radius, rows = grow_radius(positions[image], centre, 100)
        found = lazy.constellation_at_pixel(
            str(sift_paths[image]),
            (float(centre[0]), float(centre[1])),
            radius,
            image_index=image,
            **knobs,
        )
        direct = lazy.constellation_query(
            np.ascontiguousarray(positions[image][found["feature_rows"]]),
            feature_ids=np.asarray(found["feature_ids"], dtype=np.uint32).tolist(),
            image_index=image,
            **knobs,
        )
        out.append(
            {
                "image": int(image),
                "rows_match": bool(
                    np.array_equal(np.sort(found["feature_rows"]), np.sort(rows))
                ),
                "matches_agree": bool(
                    [(m["image_index"], m["inliers"]) for m in found["matches"]]
                    == [(m["image_index"], m["inliers"]) for m in direct]
                ),
                "candidates": len(found["matches"]),
            }
        )
    return out


if __name__ == "__main__":
    main()
