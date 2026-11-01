# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Read what the reference-view rule reads on the review cases and a pool of
sampled tracks, for tuning the rule's parameters.

    pixi run python scripts/reference_view/measure.py <dataset> --out <dir>
        [--pool 60] [--all] [--blur-matched] [--localize] [--eligible]

For each review case of the dataset (``cases.json``) the point is found by its
views: the point whose track holds the most of the case's (image, keypoint)
pairs, a keypoint matching within ``--keypoint-tolerance`` px. It is put on
the bench, its rows are made the case's views at the case's keypoints (a row
in an image the case does not have is turned ``out``, a missing one is
added, a moved one is sighted at the case's keypoint), and it is evaluated.
``--pool`` more points of at least three observations are sampled with a fixed
seed and evaluated as they are stored; ``--all`` evaluates every such point
instead of a sample.

Each track is one JSON line in ``<out>/<dataset>.jsonl`` (``-all.jsonl`` with
``--all``) with each row's readings as the rule reads them (coverage, clipped
share, viewing angle, cell deficit, pair ZNCC, the self-similarity semi-axes
and ``max_radius``, the largest radius that reading searched, at which the
semi-major axis is a lower bound, read off the size of its surface), its
``ray``, the unit direction from the patch's centre to the camera centre
(minus the patch's bearing, the same for every row, for a patch at infinity),
which the rule does not read but ``analyze.py`` compares a form of the angle
factor on, its readings per cell of the ZNCC grid's split (``pair_zncc_grid``,
``cell_axes``, the semi-axes of each cell's self-similarity ellipse, and
``cell_off_photo``, how many of each cell's samples are off the photograph),
which ``analyze.py`` reads the cell check from, and what the rule decided at
this build's defaults (cost, terms, exclusion, pick). ``analyze.py`` reads the
files.

- ``--blur-matched`` also reads each row's pair ZNCC from the rows' tiles
  rendered with ``OrientedPatch.render_view_tile``, plain and blur-matched
  (``score_against_bitmap`` with each tile in turn as the bitmap), each the
  median over the other rows.
- ``--localize`` (ground truths) aligns the point's views to each eligible
  row's tile in turn (``PatchCloud.localize_keypoints`` with
  ``reference_images``, the gates off) and records every view's distance from
  the projection of the point at the stored pose.
- ``--eligible`` counts the points of the whole reconstruction for which the
  rule finds no eligible view, by rendering every point's bitmap with the
  stored references cleared (``PatchCloud.render_bitmaps``), and writes
  ``<out>/<dataset>-eligible.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import DATASETS_DIR, REPO, read_image, resolve  # noqa: E402

CASES = Path(__file__).resolve().parent / "cases.json"
READINGS = (
    "coverage",
    "clipped_share",
    "viewing_angle_deg",
    "cell_deficit",
    "pair_zncc",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True, help="directory the output is written to")
    ap.add_argument("--pool", type=int, default=60, help="pool tracks to sample")
    ap.add_argument("--all", action="store_true", help="every point, no cases")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--keypoint-tolerance", type=float, default=None)
    ap.add_argument("--blur-matched", action="store_true")
    ap.add_argument("--localize", action="store_true")
    ap.add_argument("--eligible", action="store_true")
    ap.add_argument(
        "--sfmr",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="read dataset NAME from PATH instead of datasets.py's path",
    )
    args = ap.parse_args()
    out_dir = Path(args.out).resolve()
    for forbidden in (REPO / "test-data", DATASETS_DIR):
        if out_dir.is_relative_to(forbidden.resolve()):
            raise SystemExit(f"refusing to write under {forbidden}")
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = dict(s.split("=", 1) for s in args.sfmr)
    tolerance = args.keypoint_tolerance
    if tolerance is None:
        # The Christmas-tree file is rebuilt with embed_patches, which places
        # the keypoints again (datasets.py); every other file is the one the
        # cases were drawn from, whose keypoints the cases round to 0.01 px.
        tolerance = 1.5 if args.dataset == "xmas" else 0.05
    run(args, resolve(args.dataset, overrides), tolerance, out_dir)


def run(args, sfmr: Path, tolerance: float, out_dir: Path) -> None:
    from sfmtool import bench as B
    from sfmtool.patches import ImagePyramidSet
    from sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction

    name = args.dataset
    t0 = time.time()
    recon = SfmrReconstruction.load(str(sfmr))
    imgs = [read_image(recon.workspace_dir, n) for n in recon.image_names]
    pyramids = ImagePyramidSet(recon, imgs)
    print(name, "images loaded", round(time.time() - t0, 1), "s", flush=True)
    if args.eligible:
        count_eligible(name, recon, pyramids, out_dir)
    edited = EditedReconstruction(recon)
    names = list(recon.image_names)
    resolution = int(recon.patch_bitmap_resolution or 24)
    counts = np.asarray(recon.observation_counts)
    tii = np.asarray(recon.track_image_indexes)
    tpi = np.asarray(recon.track_point_indexes)
    kxy = np.asarray(recon.keypoints_xy, float)
    image_of = {n: i for i, n in enumerate(names)}
    centres = _camera_centres(recon)
    poses = _camera_poses(recon)

    jobs: list[tuple[str, int, dict | None]] = []
    case_points: set[int] = set()
    if not args.all:
        cases = [
            c for c in json.loads(Path(args.cases).read_text()) if c["dataset"] == name
        ]
        for case in cases:
            point, matched = _find_point(case, image_of, tii, tpi, kxy, tolerance)
            if point is None:
                print(name, case["id"], "matches no point", flush=True)
                continue
            if matched < len(case["views"]):
                print(
                    name,
                    case["id"],
                    f"point {point}: {matched} of {len(case['views'])} views match",
                    flush=True,
                )
            jobs.append(("case", point, case))
            case_points.add(point)
    candidates = np.flatnonzero(counts >= 3)
    if args.all:
        jobs += [("all", int(p), None) for p in candidates]
    elif args.pool > 0:
        candidates = np.array([p for p in candidates if int(p) not in case_points])
        rng = np.random.default_rng(args.seed)
        picked = rng.choice(
            candidates, size=min(args.pool, len(candidates)), replace=False
        )
        jobs += [("pool", int(p), None) for p in sorted(picked)]

    stem = f"{name}-all" if args.all else name
    tracks: list[dict] = []
    t0 = time.time()
    with open(out_dir / f"{stem}.jsonl", "w") as out:
        for kind, point, case in jobs:
            try:
                _, track = B.create_track(B.Bench(), edited, point)
                if case is not None:
                    track = _align_rows(B, track, edited, case, image_of)
                track, _ = B.evaluate(track, edited, pyramids)
            except Exception as e:  # noqa: BLE001
                out.write(
                    json.dumps(
                        dict(ds=name, kind=kind, point=point, error=str(e)[:200])
                    )
                    + "\n"
                )
                continue
            record = dict(
                ds=name,
                kind=kind,
                point=point,
                case=case["id"] if case else None,
                resolution=resolution,
                patch_frame=_patch_frame(track.patch_frame),
                rows=_rows(track, names, centres),
            )
            _cell_readings(record, track, poses, pyramids, resolution)
            if args.blur_matched:
                _blur_matched_pairs(record, poses, pyramids, resolution)
            tracks.append(record)
            if not args.localize:
                out.write(json.dumps(record) + "\n")
        if args.localize:
            _localize(recon, poses, pyramids, tracks, resolution)
            for record in tracks:
                out.write(json.dumps(record) + "\n")
    print(
        name, "done", len(tracks), "tracks", round(time.time() - t0, 1), "s", flush=True
    )


def _find_point(case, image_of, tii, tpi, kxy, tolerance):
    """The point whose track holds the most of the case's views, and how many."""
    votes: dict[int, int] = {}
    for image_name, x, y in case["views"]:
        image = image_of.get(image_name)
        if image is None:
            continue
        rows = np.flatnonzero(tii == image)
        near = rows[np.linalg.norm(kxy[rows] - [x, y], axis=1) < tolerance]
        for p in {int(p) for p in tpi[near]}:
            votes[p] = votes.get(p, 0) + 1
    if not votes:
        return None, 0
    point = max(votes, key=lambda p: (votes[p], p == case.get("point")))
    return point, votes[point]


def _align_rows(B, track, edited, case, image_of):
    """The track with its rows made the case's views at the case's keypoints."""
    want = {image_of[n]: (x, y) for n, x, y in case["views"] if n in image_of}
    seen = set()
    for row, obs in enumerate(track.observations):
        image = obs["image"]
        if image not in want:
            track, _ = B.set_verdict(track, row, "out")
            continue
        seen.add(image)
        keypoint = (obs.get("track") or {}).get("keypoint")
        x, y = want[image]
        if keypoint is None or np.hypot(keypoint[0] - x, keypoint[1] - y) > 0.01:
            track, _ = B.sight_observation(track, edited, row, [x, y])
    for image, (x, y) in want.items():
        if image not in seen:
            track, report = B.add_observation(track, image, [x, y])
            track, _ = B.set_verdict(track, report["observation"], "in")
    return track


def _patch_frame(patch_frame):
    if patch_frame is None:
        return None
    return {
        k: (np.asarray(v, float).tolist() if k != "w" else float(v))
        for k, v in patch_frame.items()
    }


def _camera_centres(recon):
    """Each image's camera centre in world coordinates, ``-Rᵀ t``."""
    q = np.asarray(recon.quaternions_wxyz, float)
    t = np.asarray(recon.translations, float)
    return np.array([-_rotation(qi).T @ ti for qi, ti in zip(q, t)])


def _ray(patch_frame, centre):
    """The unit direction from the patch's centre to a camera centre. For a
    patch at infinity (``w = 0``) it is minus the patch's bearing, the same
    for every camera."""
    if patch_frame is None:
        return None
    c = np.asarray(patch_frame["center"], float)
    d = -c if patch_frame["w"] == 0.0 else centre - c
    n = np.linalg.norm(d)
    return None if n == 0.0 else (d / n).tolist()


def _rows(track, names, centres):
    rows = []
    patch_frame = _patch_frame(track.patch_frame)
    for row, obs in enumerate(track.observations):
        m = obs.get("track") or {}
        ellipse = (m.get("zncc_self_similarity_ellipse") or {}).get("grid_px")
        axes = None if ellipse is None else [float(a) for a in ellipse["axes"]]
        surface = m.get("zncc_self_similarity_surface")
        standing = m.get("reference_view")
        keypoint = m.get("keypoint")
        rows.append(
            dict(
                row=row,
                image=int(obs["image"]),
                image_name=names[obs["image"]],
                verdict=obs["verdict"],
                keypoint=None if keypoint is None else [float(v) for v in keypoint],
                ray=_ray(patch_frame, centres[int(obs["image"])]),
                semi_major=None if axes is None else axes[0],
                semi_minor=None if axes is None else axes[1],
                # The surface spans the (2r + 1)² square of shifts searched.
                max_radius=None
                if axes is None or surface is None
                else (np.asarray(surface).shape[0] - 1) / 2,
                **{k: (None if m.get(k) is None else float(m[k])) for k in READINGS},
                rule=None
                if standing is None
                else dict(
                    is_reference=bool(standing["is_reference"]),
                    cost=standing["cost"],
                    terms=standing["terms"],
                    costliest_term=standing["costliest_term"],
                    excluded_by=standing["excluded_by"],
                ),
            )
        )
    return rows


def _cell_readings(record, track, poses, pyramids, resolution):
    """Each rule row's readings per cell of the ZNCC grid's split, which the
    cell check reads: ``pair_zncc_grid``, ``cell_axes`` (the semi-axes of each
    cell's self-similarity ellipse, grid px, ``[row][col][major, minor]``) and
    ``cell_off_photo`` (how many of each cell's samples the warp places off
    the photograph, read from the row's tile rendered again with
    ``OrientedPatch.render_view_tile``)."""
    patch = _oriented_patch(record["patch_frame"]) if record["patch_frame"] else None
    cut = [0, resolution // 3, resolution - resolution // 3, resolution]
    for r in record["rows"]:
        m = track.observations[r["row"]].get("track") or {}
        grid = m.get("pair_zncc_grid")
        if grid is not None:
            r["pair_zncc_grid"] = _nan_to_none(np.asarray(grid, float))
        ellipses = m.get("zncc_self_similarity_ellipse_grid")
        if ellipses is not None:
            r["cell_axes"] = _nan_to_none(np.asarray(ellipses["axes"], float))
        if patch is None or r["rule"] is None or r["keypoint"] is None:
            continue
        camera, pose = poses[r["image"]]
        tile = patch.render_view_tile(
            camera,
            pose,
            pyramids,
            image_index=r["image"],
            keypoint=tuple(r["keypoint"]),
            resolution=resolution,
            sampler="per_view",
        )
        off = ~np.asarray(tile["valid"])
        r["cell_off_photo"] = [
            [int(off[cut[i] : cut[i + 1], cut[j] : cut[j + 1]].sum()) for j in range(3)]
            for i in range(3)
        ]


def _nan_to_none(a):
    """A float array as nested lists, ``None`` where it is not finite."""
    if a.ndim == 0:
        return float(a) if np.isfinite(a) else None
    return [_nan_to_none(x) for x in a]


def _oriented_patch(patch_frame):
    from sfmtool.patches import OrientedPatch

    u = np.asarray(patch_frame["u_halfvec"], float)
    v = np.asarray(patch_frame["v_halfvec"], float)
    half = [float(np.linalg.norm(u)), float(np.linalg.norm(v))]
    if patch_frame["w"] == 0.0:
        patch = OrientedPatch.from_infinity_direction(
            patch_frame["center"], (v / half[1]).tolist(), half
        )
        if not np.allclose(patch.u_axis, u / half[0], atol=1e-9):
            return None
        return patch
    return OrientedPatch(
        patch_frame["center"], (u / half[0]).tolist(), (v / half[1]).tolist(), half
    )


def _camera_poses(recon):
    """Each image's ``(camera, world-to-camera pose)``.

    Each getter builds a new array on every access, so they are read once here
    rather than for each row."""
    from sfmtool.geometry import RigidTransform

    cameras = recon.cameras
    return [
        (
            cameras[int(c)],
            RigidTransform.from_wxyz_translation(q.tolist(), t.tolist()),
        )
        for c, q, t in zip(
            np.asarray(recon.camera_indexes),
            np.asarray(recon.quaternions_wxyz),
            np.asarray(recon.translations),
        )
    ]


def _blur_matched_pairs(record, poses, pyramids, resolution):
    """Each rule row's pair ZNCC from the rows' own tiles, plain and
    blur-matched, as ``py_pair_zncc`` and ``bm_pair_zncc``.

    A pair is read with the sharper tile as the bitmap, so the sharper one is
    blurred to the other's sharpness where ``score_against_bitmap`` would blur
    it; where neither direction blurs, the plain score is used. Each is the
    median over the other rows of the finite pairs."""
    from sfmtool.patches import score_against_bitmap

    rows = [
        r for r in record["rows"] if r["rule"] is not None and r["keypoint"] is not None
    ]
    patch = _oriented_patch(record["patch_frame"]) if record["patch_frame"] else None
    if patch is None or len(rows) < 2:
        return
    tiles, valid = [], []
    for r in rows:
        camera, pose = poses[r["image"]]
        tile = patch.render_view_tile(
            camera,
            pose,
            pyramids,
            image_index=r["image"],
            keypoint=tuple(r["keypoint"]),
            resolution=resolution,
            sampler="per_view",
        )
        # The tile must be the one the bench read.
        if r["coverage"] is not None and abs(tile["coverage"] - r["coverage"]) > 1e-9:
            record["blur_matched_error"] = "tile differs from the bench's"
            return
        tiles.append(tile["samples"][..., :3])
        valid.append(tile["valid"])
    tiles = np.ascontiguousarray(np.stack(tiles))
    valid = np.ascontiguousarray(np.stack(valid))
    k = len(rows)
    plain = np.full((k, k), np.nan)
    blurred = np.full((k, k), np.nan)
    sigma = np.zeros((k, k))
    for i in range(k):
        bitmap = np.dstack([tiles[i], np.where(valid[i], 255, 0).astype(np.uint8)])
        s = score_against_bitmap(
            np.ascontiguousarray(bitmap), tiles, valid=valid, reference=i
        )
        plain[i] = s["plain_zncc"]
        blurred[i] = s["blur_matched_zncc"]
        sigma[i] = s["blur_sigma"]
    for a, r in enumerate(rows):
        p, b = [], []
        for c in range(k):
            if c == a:
                continue
            p.append(plain[a, c])
            if sigma[a, c] > 0:
                b.append(blurred[a, c])
            elif sigma[c, a] > 0:
                b.append(blurred[c, a])
            else:
                b.append(plain[a, c])
        r["py_pair_zncc"] = _median(p)
        r["bm_pair_zncc"] = _median(b)


def _median(values):
    v = [x for x in values if np.isfinite(x)]
    return float(np.median(v)) if v else None


def _project(camera, q, t, xyzw):
    rot = _rotation(q)
    xc = rot @ xyzw[:3] + xyzw[3] * t
    px = camera.ray_to_pixel_batch((xc / np.linalg.norm(xc))[None])[0]
    return px if np.all(np.isfinite(px)) else None


def _rotation(q):
    """The rotation matrix of a ``wxyz`` unit quaternion."""
    w_, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w_), 2 * (x * z + y * w_)],
            [2 * (x * y + z * w_), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w_)],
            [2 * (x * z - y * w_), 2 * (y * z + x * w_), 1 - 2 * (x * x + y * y)],
        ]
    )


def _localize(recon, poses, pyramids, tracks, resolution):
    """For each track, align its views to each eligible row's tile in turn and
    record each view's distance from the point's projection, as
    ``record["localized"][reference image] = {image: error px}``.

    One ``localize_keypoints`` call per pass, the pass naming each point's
    next eligible row as its reference. The gates are off (relative and
    absolute score, shift, member radius), so every view but a grazing one is
    kept; the views start at the rows' keypoints."""
    cloud = recon.patches
    positions = np.asarray(recon.positions_xyzw, float)
    quaternions = np.asarray(recon.quaternions_wxyz)
    translations = np.asarray(recon.translations)
    plans = {}
    for record in tracks:
        rows = [
            r
            for r in record["rows"]
            if r["rule"] is not None and r["keypoint"] is not None
        ]
        eligible = [r["image"] for r in rows if r["rule"]["excluded_by"] is None]
        if len(rows) < 2 or not eligible:
            continue
        plans[record["point"]] = (record, rows, eligible)
        record["localized"] = {}
    passes = max((len(e) for _, _, e in plans.values()), default=0)
    for k in range(passes):
        points = [p for p, (_, _, e) in plans.items() if len(e) > k]
        results = cloud.localize_keypoints(
            recon,
            pyramids,
            patch_views={p: [r["image"] for r in plans[p][1]] for p in points},
            patch_keypoints={p: [r["keypoint"] for r in plans[p][1]] for p in points},
            reference_images={p: plans[p][2][k] for p in points},
            point_indexes=points,
            max_shift_px=1e9,
            min_relative_zncc=0.0,
            min_absolute_zncc=0.0,
            max_member_zncc_self_similarity_radius=0.0,
            resolution=resolution,
        )
        for res in results:
            p = int(res["point_index"])
            record, _, eligible = plans[p]
            if (
                res["reference_image"] is None
                or int(res["reference_image"]) != eligible[k]
            ):
                continue
            errors = {}
            for image, keypoint in zip(res["views"], res["keypoints"]):
                image = int(image)
                gt = _project(
                    poses[image][0],
                    quaternions[image],
                    translations[image],
                    positions[p],
                )
                if gt is not None:
                    errors[str(int(image))] = float(
                        np.linalg.norm(np.asarray(keypoint) - gt)
                    )
            record["localized"][str(eligible[k])] = errors


def count_eligible(name, recon, pyramids, out_dir):
    """Count the points with two or more observations for which the rule finds
    no eligible view, with every stored reference cleared so the rule picks for
    every point."""
    n = len(np.asarray(recon.observation_counts))
    cleared = recon.clone_with_changes(reference_observations=np.full(n, -1, np.int32))
    t0 = time.time()
    _, references = cleared.patches.render_bitmaps(cleared, pyramids)
    counts = np.asarray(recon.observation_counts)
    references = np.asarray(references)
    result = dict(
        ds=name,
        points=int(n),
        with_two_views=int((counts >= 2).sum()),
        no_eligible_view=int(((references < 0) & (counts >= 2)).sum()),
        seconds=round(time.time() - t0, 2),
    )
    (out_dir / f"{name}-eligible.json").write_text(json.dumps(result) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
