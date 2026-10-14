# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the cell-registered grid-plane normal with today's Grid Plane Normal.

At each finite ground-truth point seen in two or more images, the point is put
on the bench at its ground-truth position, sightings and half-extent, its patch
turned to the mean viewing direction (as ``normal_diag.py``'s ``mean_view``),
and fitted once with the bench ``fit``. From that one fitted track two steps
are run:

- ``old``: ``finite_difference_normal`` with ``layout="grid"``, 2x2 pieces and
  no overlap, Track View's default Grid Plane Normal;
- ``new``: ``grid_plane_normal`` at its defaults.

The angle of each normal to the ground-truth normal is folded to 0..90 degrees
(both turned to face the first sighting's camera, as in ``normal_diag.py``).
A fit leaves the normal where it was, so the fitted track's error is the
mean-view start's, the baseline both steps start from.

    pixi run -e test python scripts/track_at_pixel/normal_sources/grid_plane_eval.py \\
        DATASET --out DIR [--limit N] [--seed S] [--start ground_truth] [--no-fit]

``DATASET`` is a name ``dataset.py`` knows (``seoul_bull``) or the path of a
ground-truth ``.sfmr``; ``prepare`` copies it into its cache workspace. Rows go
to ``DIR/<dataset>.jsonl``, one per point, and the summary is printed. Each row
carries what a later render of the point needs: the ground-truth position and
normal, the start normal, the fitted frame, each step's resulting frame, and
the reference and ``in`` sightings with their keypoints; ``DIR/cameras.json``
holds each image's name, centre, world-from-camera rotation and lens summary,
and the image directory. With
``--limit`` the first ``N`` points are run, or with ``--seed`` a random sample
of ``N``. Two controls change the start, each adding a suffix to the file
name: ``--start ground_truth`` (``_from_gt``) starts both steps from the
ground-truth normal instead of the mean view, which shows how far each moves
off it, and ``--no-fit`` (``_no_fit``) skips the fit, so both steps start from
the ground-truth position and keypoints.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from context import DatasetContext  # noqa: E402
from dataset import prepare  # noqa: E402
from normal_diag import angle, face, unit  # noqa: E402


def patch_normal(track):
    frame = track.placement
    return unit(np.cross(frame["u_halfvec"], frame["v_halfvec"]))


def frame(track):
    """The track's patch as plain lists, for a later render of the point."""
    f = track.placement
    return {
        "center": np.asarray(f["center"], float).tolist(),
        "u_halfvec": np.asarray(f["u_halfvec"], float).tolist(),
        "v_halfvec": np.asarray(f["v_halfvec"], float).tolist(),
        "normal": patch_normal(track).tolist(),
    }


def sightings(track):
    """The reference observation and the ``in`` observations, image and keypoint."""

    def entry(i):
        o = track.observation(int(i))
        kp = o.get("track", {}).get("keypoint")
        return {
            "observation": int(i),
            "image": int(o["image"]),
            "keypoint": None if kp is None else np.asarray(kp, float).tolist(),
        }

    ref = track.reference_observation
    return {
        "reference": None if ref is None else entry(ref),
        "in": [entry(i) for i in track.in_observations],
    }


def write_cameras(ds, prepared, out):
    """``cameras.json``: each image's name, pose and lens, and where they are."""
    images = []
    for name, cam in zip(ds.image_names, ds.cameras):
        images.append(
            {
                "name": name,
                "center": cam.center.tolist(),
                "world_from_camera_rotation": cam.R.T.tolist(),
                "width": int(cam.intrinsics.width),
                "height": int(cam.intrinsics.height),
                "focal": cam.focal,
            }
        )
    doc = {
        "dataset": prepared.name,
        "ground_truth": str(prepared.ground_truth),
        "image_dir": str(prepared.workspace),
        "source_image_dir": str(prepared.ground_truth.parent),
        "images": images,
    }
    with open(out / "cameras.json", "w") as sink:
        json.dump(doc, sink, indent=1)


def run_point(ds, edited, p, *, fit=True, start="mean_view"):
    """One row: the start, the fit, and each step from the fitted track."""
    from sfmtool import bench as B

    images = ds.point_images[p]
    xyz = ds.point_xyz[p]
    to_cam = unit(ds.cameras[int(images[0])].center - xyz)
    gt = face(ds.point_normal[p], to_cam)
    mv = unit(sum(unit(ds.cameras[int(i)].center - xyz) for i in images))
    if start == "ground_truth":
        mv = gt.copy()
    row = {
        "point": int(p),
        "views": int(len(images)),
        "half_extent": float(ds.point_half[p]),
        "gt_vs_mean_view_deg": angle(face(mv, to_cam), gt),
        "gt_position": xyz.tolist(),
        "gt_normal": gt.tolist(),
    }

    _, track = B.create_track(B.Bench(), edited, int(p))
    # Turn by the least rotation: ask for the mean view on the side the patch
    # already faces, so the turn is the angle between the two (under 90).
    if mv @ patch_normal(track) < 0:
        mv = -mv
    try:
        track, tilt = B.tilt_patch(track, edited, tuple(mv))
        row["start_tilt_stopped"] = tilt["stopped"] is not None
        t0 = time.perf_counter()
        if fit:
            track, fitted = B.fit(track, edited, ds.pyramids)
        row["fit_ms"] = 1000 * (time.perf_counter() - t0)
    except ValueError as e:
        row["status"] = "start_refused"
        row["reason"] = str(e)
        return row
    if fit and fitted["at_infinity"]:
        row["status"] = "fit_at_infinity"
        return row
    row["fit_shift_half_extents"] = float(
        np.linalg.norm(np.asarray(track.position) - xyz) / ds.point_half[p]
    )
    row["status"] = "ok"
    row["start_deg"] = angle(face(patch_normal(track), to_cam), gt)
    row["start_normal"] = patch_normal(track).tolist()
    row["fitted_frame"] = frame(track)
    row["sightings"] = sightings(track)

    t0 = time.perf_counter()
    try:
        old, report = B.finite_difference_normal(
            track, edited, ds.pyramids, layout="grid", pieces=2, overlap=0.0
        )
        row["old_ms"] = 1000 * (time.perf_counter() - t0)
        est = report["estimate"]
        row["old_ok"] = True
        row["old_fitted"] = est["fitted"]
        row["old_both_axes"] = est["both_axes"]
        row["old_turned_deg"] = report["tilt"]["degrees"]
        row["old_deg"] = angle(face(patch_normal(old), to_cam), gt)
        row["old_frame"] = frame(old)
    except ValueError as e:
        row["old_ms"] = 1000 * (time.perf_counter() - t0)
        row["old_ok"] = False
        row["old_reason"] = str(e)
        row["old_deg"] = row["start_deg"]  # a refusal leaves the normal

    t0 = time.perf_counter()
    try:
        new, report = B.grid_plane_normal(track, edited, ds.pyramids)
    except ValueError as e:
        row["new_ms"] = 1000 * (time.perf_counter() - t0)
        row["new_outcome"] = "refused"
        row["new_reason"] = str(e)
        row["new_deg"] = row["start_deg"]
        return row
    row["new_ms"] = 1000 * (time.perf_counter() - t0)
    cells = report["estimate"]
    row["new_outcome"] = cells["outcome"]
    row["new_stop"] = cells["stop"]
    row["new_determinacy"] = cells["determinacy"]
    row["new_working_resolution"] = cells["working_resolution"]
    row["new_in_plane"] = cells["in_plane"]
    row["new_iterations"] = cells["iterations"]
    row["new_score_before"] = cells["score_before"]
    row["new_score_after"] = cells["score_after"]
    row["new_cell_status"] = cells["cell_status"]
    counts = {}
    for grid in cells["view_cells"]:
        for line in grid:
            for word in line:
                counts[word] = counts.get(word, 0) + 1
    row["new_view_cell_counts"] = counts
    row["new_turned_deg"] = report["tilt"]["degrees"]
    row["new_deg"] = angle(face(patch_normal(new), to_cam), gt)
    row["new_frame"] = frame(new)
    return row


def stats(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return "    n=0"
    p50, p75 = np.percentile(a, [50, 75])
    return (
        f"n={a.size:4d}  median {p50:5.1f}  p75 {p75:5.1f}  "
        f"<10 {np.mean(a < 10):4.0%}  <20 {np.mean(a < 20):4.0%}"
    )


def summarize(name, rows):
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n== {name}: {len(rows)} points, {len(ok)} finite after the fit")
    for status in sorted({r["status"] for r in rows} - {"ok"}):
        print(f"   {status}: {sum(r['status'] == status for r in rows)}")
    print("all points (error to the ground-truth normal, degrees)")
    print(f"  start              {stats(r['start_deg'] for r in ok)}")
    print(f"  old 2x2 grid       {stats(r['old_deg'] for r in ok)}")
    print(f"  new grid_plane     {stats(r['new_deg'] for r in ok)}")
    old_ok = [r for r in ok if r["old_ok"]]
    print(f"old step: {len(old_ok)} turned, {len(ok) - len(old_ok)} refused")
    if old_ok:
        print(f"  old, where it ran  {stats(r['old_deg'] for r in old_ok)}")
        print(f"  start, same points {stats(r['start_deg'] for r in old_ok)}")
    print("new step outcomes")
    for outcome in ("turned", "undetermined", "unstable", "refused"):
        n = sum(r["new_outcome"] == outcome for r in ok)
        print(f"  {outcome:12s} {n:4d}  {n / max(len(ok), 1):5.1%}")
    stops = {}
    for r in ok:
        key = r.get("new_stop", "refused")
        stops[key] = stops.get(key, 0) + 1
    print("  stops: " + ", ".join(f"{k} {v}" for k, v in sorted(stops.items())))
    turned = [r for r in ok if r["new_outcome"] == "turned"]
    print(f"points the new step turned ({len(turned)})")
    print(f"  start              {stats(r['start_deg'] for r in turned)}")
    print(f"  old 2x2 grid       {stats(r['old_deg'] for r in turned)}")
    print(f"  new grid_plane     {stats(r['new_deg'] for r in turned)}")
    print(
        "median shift of the fitted centre from the ground truth: "
        f"{np.median([r['fit_shift_half_extents'] for r in ok]):.3f} half-extents"
    )
    print(
        "median ms per call: fit {:.0f}, old {:.0f}, new {:.0f}".format(
            np.median([r["fit_ms"] for r in ok]),
            np.median([r["old_ms"] for r in ok]),
            np.median([r["new_ms"] for r in ok]),
        )
    )
    resolutions = {}
    for r in ok:
        k = r.get("new_working_resolution")
        if k is not None:
            resolutions[k] = resolutions.get(k, 0) + 1
    print(
        "new working resolution: "
        + ", ".join(f"{k}: {v}" for k, v in sorted(resolutions.items()))
    )
    print("examples")
    for outcome in ("undetermined", "unstable", "refused"):
        ids = [r["point"] for r in ok if r["new_outcome"] == outcome][:10]
        if ids:
            print(f"  {outcome}: {ids}")
    worst = sorted(turned, key=lambda r: -r["new_deg"])[:10]
    if worst:
        print(
            "  worst turned: "
            + ", ".join(f"{r['point']} ({r['new_deg']:.1f})" for r in worst)
        )
    refused = [r["point"] for r in ok if not r["old_ok"]][:10]
    if refused:
        print(f"  old refused: {refused}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--start",
        choices=("mean_view", "ground_truth"),
        default="mean_view",
        help="the normal both steps start from; ground_truth is a control",
    )
    ap.add_argument(
        "--no-fit",
        action="store_true",
        help="run both steps from the turned ground-truth track, unfitted",
    )
    args = ap.parse_args()
    from sfmtool.reconstruction import EditedReconstruction

    prepared = prepare(args.dataset, None, quiet=True)
    ds = DatasetContext(prepared)
    rec = ds.recon
    # One edited reconstruction for every point, so the fit measures the
    # reconstruction's noise level once.
    edited = EditedReconstruction(rec)
    points = [
        p
        for p in range(rec.point_count)
        if ds.point_w[p] != 0 and len(ds.point_images[p]) >= 2
    ]
    if args.limit and args.seed is not None:
        rng = np.random.default_rng(args.seed)
        points = sorted(rng.choice(points, size=args.limit, replace=False).tolist())
    elif args.limit:
        points = points[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    write_cameras(ds, prepared, args.out)
    suffix = ("_from_gt" if args.start == "ground_truth" else "") + (
        "_no_fit" if args.no_fit else ""
    )
    path = args.out / f"{prepared.name}{suffix}.jsonl"
    rows = []
    t0 = time.perf_counter()
    with open(path, "w") as sink:
        for k, p in enumerate(points):
            row = run_point(ds, edited, p, fit=not args.no_fit, start=args.start)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            if k % 50 == 0:
                print(f"{k}/{len(points)}", flush=True)
    print(f"{len(points)} points in {time.perf_counter() - t0:.0f} s, rows in {path}")
    summarize(prepared.name, rows)


if __name__ == "__main__":
    main()
