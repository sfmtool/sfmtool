# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the cell-registered and grown-plane normals with today's Grid Plane Normal.

At each finite ground-truth point seen in two or more images, the point is put
on the bench at its ground-truth position, sightings and half-extent, its patch
turned to the mean viewing direction (as ``normal_diag.py``'s ``mean_view``),
and fitted once with the bench ``fit``. From that one fitted track two steps,
or three, are run:

- ``old``: ``finite_difference_normal`` with ``layout="grid"``, 2x2 pieces and
  no overlap, Track View's default Grid Plane Normal (``--old-pieces`` sets the
  pieces per axis); ``--also-old-pieces N``, repeated, runs it again at ``N``
  pieces per axis into ``old{N}_*`` fields beside the ``old_*`` ones;
- ``new``: ``grid_plane_normal`` at its defaults, or with the keywords
  ``--new-opt`` gives;
- ``grown``, with ``--grown``: ``grown_plane_normal`` at its defaults, or with
  the keywords ``--grown-opt`` gives (which implies ``--grown``).

The angle of each normal to the ground-truth normal is folded to 0..90 degrees
(both turned to face the first sighting's camera, as in ``normal_diag.py``).
A fit leaves the normal where it was, so the fitted track's error is the
mean-view start's, the baseline every step starts from.

    pixi run -e test python scripts/track_at_pixel/normal_sources/grid_plane_eval.py \\
        DATASET --out DIR [--limit N] [--seed S] [--start ground_truth] [--no-fit] \\
        [--old-pieces N] [--also-old-pieces N ...] [--new-opt KEY=VALUE ...] \\
        [--grown] [--grown-opt KEY=VALUE ...]

``DATASET`` is a name ``dataset.py`` knows (``seoul_bull``) or the path of a
ground-truth ``.sfmr``; ``prepare`` copies it into its cache workspace. Rows go
to ``DIR/<dataset>.jsonl``, one per point, and the summary is printed. Each row
carries what a later render of the point needs (the ground-truth position and
normal, the start normal, the fitted frame, each step's resulting frame, the
reference and ``in`` sightings with their keypoints) and the new step's
predicted precision, views' baseline and per-render trace, each proposed normal
with its error; ``DIR/cameras.json`` holds each image's name, centre,
world-from-camera rotation and lens summary, and the workspace directories the
image names are relative to: the cache's and the source's.
``--new-opt KEY=VALUE``, repeated, passes a keyword to ``grid_plane_normal``
(``max_iterations=8``, ``acceptance=none``, ``cell_shift_bound_px=4``, ...),
and ``--grown-opt KEY=VALUE`` one to ``grown_plane_normal``
(``min_held_zncc=0.6``, ``window_fractions=0.125,0.5``, ...); a value with a
comma is read as a list of numbers.

The grown step's fields are ``grown_deg`` (its error to the ground truth),
``grown_outcome`` (``"turned"``, ``"held"`` when the plane found was a large
turn on weak evidence and the step kept the start, ``"unchanged"`` when no
view joined, or ``"refused"``), ``grown_found_deg`` (the error of the plane
found, whether or not the patch was turned to it), ``grown_turn_deg`` (the
turn from the start to that plane) and ``grown_turn_sigma_ratio`` (that turn
over ``grown_sigma_weak_deg``), each ``None`` when no view joined,
``grown_sigma_weak_deg`` and
``grown_sigma_strong_deg`` (every kept estimate's information summed) and
``grown_discounted_sigma_weak_deg`` (each kept pair's information halved),
each ``None`` when no view joined, ``grown_low_confidence``,
``grown_joined``, ``grown_left_out``, ``grown_shift_only``, ``grown_matched``
(views grown against the reference) and ``grown_candidates`` (the views
beside the reference it could try), ``grown_pairs`` (each pair's images,
angle, kept flag, refusal and own plane's error ``own_deg``),
``grown_picks`` (the selection's picks), ``grown_view_kinds`` (a count of
each per-view word),
``grown_left_out_reasons``, ``grown_written``, ``grown_working_resolution``,
``grown_window_half_widths``, ``grown_turned_deg``, ``grown_ms`` (the call's
wall time), ``grown_text``, ``grown_frame`` and ``grown_trace``, one entry
per try with the error of the combined plane after it (``combined_deg``) and
each attempt's own plane's error (``own_deg``) to the ground truth. The summary adds the grown
step's errors, outcomes, confidence, views joined, and on how many points it
beats ``old``.

With ``--limit`` the first ``N`` points are run, or with ``--seed`` a random
sample of ``N``. Two controls change the start, each adding a suffix to the
file name: ``--start ground_truth`` (``_from_gt``) starts every step from the
ground-truth normal instead of the mean view, which shows how far each moves
off it, and ``--no-fit`` (``_no_fit``) skips the fit, so every step starts from
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
    frame = track.patch_frame
    return unit(np.cross(frame["u_halfvec"], frame["v_halfvec"]))


def frame(track):
    """The track's patch as plain lists, for a later render of the point."""
    f = track.patch_frame
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
        "source_image_dir": str(prepared.source_workspace),
        "images": images,
    }
    with open(out / "cameras.json", "w") as sink:
        json.dump(doc, sink, indent=1)


def run_old(row, prefix, track, edited, ds, pieces, to_cam, gt):
    """Today's Grid Plane Normal at ``pieces`` per axis into ``{prefix}_*``."""
    from sfmtool import bench as B

    t0 = time.perf_counter()
    try:
        old, report = B.finite_difference_normal(
            track, edited, ds.pyramids, layout="grid", pieces=pieces, overlap=0.0
        )
        row[f"{prefix}_ms"] = 1000 * (time.perf_counter() - t0)
        est = report["estimate"]
        row[f"{prefix}_ok"] = True
        row[f"{prefix}_fitted"] = est["fitted"]
        row[f"{prefix}_both_axes"] = est["both_axes"]
        row[f"{prefix}_turned_deg"] = report["tilt"]["degrees"]
        row[f"{prefix}_deg"] = angle(face(patch_normal(old), to_cam), gt)
        row[f"{prefix}_frame"] = frame(old)
    except ValueError as e:
        row[f"{prefix}_ms"] = 1000 * (time.perf_counter() - t0)
        row[f"{prefix}_ok"] = False
        row[f"{prefix}_reason"] = str(e)
        row[f"{prefix}_deg"] = row["start_deg"]  # a refusal leaves the normal


def start_track(ds, edited, p, *, fit=True, start="mean_view"):
    """The point's start: its row so far, and the track every step runs from.

    The point is put on the bench at its ground-truth position, sightings and
    half-extent, turned to ``start`` (the mean viewing direction, or the
    ground-truth normal), and fitted once unless ``fit`` is off. Returns
    ``(row, track, to_cam, gt)``, with ``track`` ``None`` when the start was
    refused or the fit put the point at infinity (``row["status"]`` says
    which); ``to_cam`` is the unit direction to the first sighting's camera and
    ``gt`` the ground-truth normal turned to face it.
    """
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
        return row, None, to_cam, gt
    if fit and fitted["at_infinity"]:
        row["status"] = "fit_at_infinity"
        return row, None, to_cam, gt
    row["fit_shift_half_extents"] = float(
        np.linalg.norm(np.asarray(track.position) - xyz) / ds.point_half[p]
    )
    row["status"] = "ok"
    row["start_deg"] = angle(face(patch_normal(track), to_cam), gt)
    row["start_normal"] = patch_normal(track).tolist()
    row["fitted_frame"] = frame(track)
    row["sightings"] = sightings(track)
    return row, track, to_cam, gt


def run_new(row, track, edited, ds, new_options, to_cam, gt):
    """``grid_plane_normal`` with ``new_options`` into ``new_*``; its report.

    Returns the step's report, or ``None`` when it refused.
    """
    from sfmtool import bench as B

    t0 = time.perf_counter()
    try:
        new, report = B.grid_plane_normal(
            track, edited, ds.pyramids, **(new_options or {})
        )
    except ValueError as e:
        row["new_ms"] = 1000 * (time.perf_counter() - t0)
        row["new_outcome"] = "refused"
        row["new_reason"] = str(e)
        row["new_deg"] = row["start_deg"]
        return None
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
    row["new_normal_precision_deg"] = cells["normal_precision_deg"]
    row["new_view_baseline_deg"] = cells["view_baseline_deg"]
    trace = []
    for entry in cells["trace"]:
        proposed = entry["proposed_normal"]
        if proposed is not None:
            proposed = np.asarray(proposed, float)
        trace.append(
            {
                "score": entry["score"],
                "determinacy": entry["determinacy"],
                "in_plane": entry["in_plane"],
                "normal_precision_deg": entry["normal_precision_deg"],
                "gated": entry["gated"],
                "proposed_normal": None if proposed is None else proposed.tolist(),
                "proposed_deg": None
                if proposed is None
                else angle(face(unit(proposed), to_cam), gt),
                "largest_move_px": entry["largest_move_px"],
                "taken": entry["taken"],
            }
        )
    row["new_trace"] = trace
    row["new_deg"] = angle(face(patch_normal(new), to_cam), gt)
    row["new_frame"] = frame(new)
    return report


def run_grown(row, track, edited, ds, grown_options, to_cam, gt):
    """``grown_plane_normal`` with ``grown_options`` into ``grown_*``; its report.

    Returns the step's report, or ``None`` when it refused.
    """
    from sfmtool import bench as B

    def error(normal):
        return angle(face(unit(np.asarray(normal, float)), to_cam), gt)

    t0 = time.perf_counter()
    try:
        grown, report = B.grown_plane_normal(
            track, edited, ds.pyramids, **(grown_options or {})
        )
    except ValueError as e:
        row["grown_ms"] = 1000 * (time.perf_counter() - t0)
        row["grown_outcome"] = "refused"
        row["grown_reason"] = str(e)
        row["grown_deg"] = row["start_deg"]
        return None
    row["grown_ms"] = 1000 * (time.perf_counter() - t0)
    seq = report["estimate"]
    row["grown_outcome"] = {
        "turned": "turned",
        "weak_turn": "held",
        "none_joined": "unchanged",
    }[seq["outcome"]]
    row["grown_text"] = report["text"]
    found = seq["normal"]
    row["grown_found_deg"] = None if found is None else error(found)
    row["grown_turn_deg"] = seq["turn_deg"]
    row["grown_turn_sigma_ratio"] = seq["turn_sigma_ratio"]
    row["grown_working_resolution"] = seq["working_resolution"]
    row["grown_window_half_widths"] = seq["window_half_widths"]
    row["grown_joined"] = seq["joined"]
    row["grown_left_out"] = seq["left_out"]
    row["grown_shift_only"] = seq["shift_only"]
    row["grown_matched"] = seq["matched"]
    row["grown_candidates"] = sum(
        v["kind"] not in ("reference", "not_attempted") for v in seq["views"]
    )
    row["grown_written"] = seq["written"]
    precision = seq["precision"]
    row["grown_sigma_weak_deg"] = (
        None if precision is None else precision["sigma_weak_deg"]
    )
    row["grown_sigma_strong_deg"] = (
        None if precision is None else precision["sigma_strong_deg"]
    )
    row["grown_discounted_sigma_weak_deg"] = (
        None if precision is None else precision["discounted_sigma_weak_deg"]
    )
    row["grown_low_confidence"] = (
        None if precision is None else precision["low_confidence"]
    )
    row["grown_pairs"] = [
        {
            "images": pair["images"],
            "angle_deg": pair["angle_deg"],
            "zncc": pair["zncc"],
            "sigma_deg": pair["sigma_deg"],
            "distance": pair["distance"],
            "kept": pair["kept"],
            "refusal": pair["refusal"],
            "own_deg": error(pair["normal"]),
        }
        for pair in seq["pairs"]
    ]
    row["grown_picks"] = [
        {
            "image": pick["image"],
            "angle_deg": pick["angle_deg"],
            "scale_px": pick["scale_px"],
            "predicted_weak": pick["predicted_weak"],
            "matched": pick["matched"],
            "sigma_after_deg": pick["sigma_after_deg"],
        }
        for pick in seq["picks"]
    ]
    kinds = {}
    for view in seq["views"]:
        kinds[view["kind"]] = kinds.get(view["kind"], 0) + 1
    row["grown_view_kinds"] = kinds
    row["grown_left_out_reasons"] = [
        v["reason"] for v in seq["views"] if v["kind"] == "left_out"
    ]
    trace = []
    for entry in seq["trace"]:
        combined = entry["combined_normal"]
        trace.append(
            {
                "observation": entry["observation"],
                "image": entry["image"],
                "angle_deg": entry["angle_deg"],
                "outcome": entry["outcome"],
                "combined_deg": None if combined is None else error(combined),
                "combined_sigma_deg": entry["combined_sigma_deg"],
                "attempts": [
                    {
                        "start": a["start"],
                        "stage_zncc": a["stage_zncc"].tolist(),
                        "shift": a["shift"].tolist(),
                        "own_deg": error(a["normal"]),
                        "grid_zncc": a["grid_zncc"],
                        "sigma_deg": a["sigma_deg"],
                        "refusal": a["refusal"],
                    }
                    for a in entry["attempts"]
                ],
            }
        )
    row["grown_trace"] = trace
    row["grown_turned_deg"] = report["tilt"]["degrees"]
    row["grown_deg"] = angle(face(patch_normal(grown), to_cam), gt)
    row["grown_frame"] = frame(grown)
    return report


def run_point(
    ds,
    edited,
    p,
    *,
    fit=True,
    start="mean_view",
    new_options=None,
    old_pieces=2,
    also_old_pieces=(),
    grown_options=None,
):
    """One row: the start, the fit, and each step from the fitted track.

    ``grown_options`` ``None`` leaves the grown step out; a dict, empty for
    its defaults, runs it.
    """
    row, track, to_cam, gt = start_track(ds, edited, p, fit=fit, start=start)
    if track is None:
        return row

    run_old(row, "old", track, edited, ds, old_pieces, to_cam, gt)
    for pieces in also_old_pieces:
        run_old(row, f"old{pieces}", track, edited, ds, pieces, to_cam, gt)

    run_new(row, track, edited, ds, new_options, to_cam, gt)
    if grown_options is not None:
        run_grown(row, track, edited, ds, grown_options, to_cam, gt)
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


def extra_old_keys(rows):
    """The ``old{N}`` prefixes ``--also-old-pieces`` added to the rows."""
    keys = set()
    for r in rows:
        keys |= {
            k[: -len("_deg")] for k in r if k.startswith("old") and k.endswith("_deg")
        }
    keys -= {"old", "old_turned"}
    return sorted(k for k in keys if not k.endswith("_turned"))


def summarize(name, rows):
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n== {name}: {len(rows)} points, {len(ok)} finite after the fit")
    for status in sorted({r["status"] for r in rows} - {"ok"}):
        print(f"   {status}: {sum(r['status'] == status for r in rows)}")
    print("all points (error to the ground-truth normal, degrees)")
    print(f"  start              {stats(r['start_deg'] for r in ok)}")
    print(f"  old grid           {stats(r['old_deg'] for r in ok)}")
    for key in extra_old_keys(rows):
        print(f"  {key:18s} {stats(r.get(key + '_deg') for r in ok)}")
    print(f"  new grid_plane     {stats(r['new_deg'] for r in ok)}")
    grown = any("grown_deg" in r for r in ok)
    if grown:
        print(f"  grown_plane        {stats(r['grown_deg'] for r in ok)}")
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
    print(f"  old grid           {stats(r['old_deg'] for r in turned)}")
    for key in extra_old_keys(rows):
        print(f"  {key:18s} {stats(r.get(key + '_deg') for r in turned)}")
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
        + (f", grown {np.median([r['grown_ms'] for r in ok]):.0f}" if grown else "")
    )
    if grown:
        summarize_grown(ok)
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


def summarize_grown(ok):
    """The grown step's outcomes, precision and views, and how it does against old."""
    print("grown step outcomes")
    for outcome in ("turned", "held", "unchanged", "refused"):
        n = sum(r["grown_outcome"] == outcome for r in ok)
        print(f"  {outcome:12s} {n:4d}  {n / max(len(ok), 1):5.1%}")
    held = [r for r in ok if r["grown_outcome"] == "held"]
    if held:
        print(f"  held, kept start   {stats(r['grown_deg'] for r in held)}")
        print(f"  held, plane found  {stats(r['grown_found_deg'] for r in held)}")
    turned = [r for r in ok if r["grown_outcome"] == "turned"]
    low = [r for r in turned if r["grown_low_confidence"]]
    print(f"  low confidence {len(low)} of {len(turned)} turned")
    confident = [r for r in turned if not r["grown_low_confidence"]]
    print(f"  confident          {stats(r['grown_deg'] for r in confident)}")
    print(f"  low confidence     {stats(r['grown_deg'] for r in low)}")
    joined = sum(r.get("grown_joined", 0) for r in ok)
    tried = sum(r.get("grown_joined", 0) + r.get("grown_left_out", 0) for r in ok)
    print(f"  views joined {joined} of {tried} tried")
    beats = sum(r["grown_deg"] < r["old_deg"] for r in ok)
    print(f"  grown beats old grid on {beats} of {len(ok)} points")
    worst = sorted(ok, key=lambda r: -r["grown_deg"])[:10]
    print(
        "  worst grown: "
        + ", ".join(f"{r['point']} ({r['grown_deg']:.1f})" for r in worst)
    )


def parse_option(text):
    """``key=value`` with the value read as an int, a float, ``None``, a list of
    floats when it holds a comma (``window_fractions=0.125,0.5``) or a word."""
    key, _, value = text.partition("=")
    if not key or not _:
        raise SystemExit(f"an option takes key=value, not {text!r}")
    if "," in value:
        try:
            return key, [float(v) for v in value.split(",") if v]
        except ValueError:
            raise SystemExit(f"{key} takes numbers separated by commas, not {value!r}")
    for kind in (int, float):
        try:
            return key, kind(value)
        except ValueError:
            pass
    return key, None if value == "None" else value


def add_grown_arguments(ap):
    """``--grown`` and ``--grown-opt``, shared with ``grid_plane_detail.py``."""
    ap.add_argument(
        "--grown",
        action="store_true",
        help="also run grown_plane_normal, into grown_* fields",
    )
    ap.add_argument(
        "--grown-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a grown_plane_normal keyword, repeated (min_held_zncc=0.6, "
        "max_agreement_distance=4, ...); implies --grown",
    )


def grown_options_of(args):
    """The grown step's keywords, or ``None`` when it is not run."""
    if not (args.grown or args.grown_opt):
        return None
    return dict(parse_option(o) for o in args.grown_opt)


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
        help="the normal every step starts from; ground_truth is a control",
    )
    ap.add_argument(
        "--no-fit",
        action="store_true",
        help="run every step from the turned ground-truth track, unfitted",
    )
    ap.add_argument(
        "--old-pieces",
        type=int,
        default=2,
        help="pieces per axis of today's Grid Plane Normal in the old_* fields",
    )
    ap.add_argument(
        "--also-old-pieces",
        type=int,
        action="append",
        default=[],
        metavar="N",
        help="also run today's step at N pieces per axis, into old{N}_* fields",
    )
    ap.add_argument(
        "--new-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a grid_plane_normal keyword, repeated (max_iterations=8, "
        "acceptance=none, cell_shift_bound_px=4, ...)",
    )
    add_grown_arguments(ap)
    args = ap.parse_args()
    new_options = dict(parse_option(o) for o in args.new_opt)
    grown_options = grown_options_of(args)
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
            row = run_point(
                ds,
                edited,
                p,
                fit=not args.no_fit,
                start=args.start,
                new_options=new_options,
                old_pieces=args.old_pieces,
                also_old_pieces=args.also_old_pieces,
                grown_options=grown_options,
            )
            row["new_options"] = new_options
            if grown_options is not None:
                row["grown_options"] = grown_options
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            if k % 50 == 0:
                print(f"{k}/{len(points)}", flush=True)
    print(f"{len(points)} points in {time.perf_counter() - t0:.0f} s, rows in {path}")
    summarize(prepared.name, rows)


if __name__ == "__main__":
    main()
