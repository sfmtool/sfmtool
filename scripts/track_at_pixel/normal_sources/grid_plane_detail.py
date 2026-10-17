# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Record everything ``grid_plane_normal`` does at each render, for a few points.

For each point named, the track is built exactly as ``grid_plane_eval.py``
builds it (the point put on the bench at its ground-truth position, sightings
and half-extent, turned to the start's normal and fitted once), and
``grid_plane_normal`` is run with ``record_detail=True`` from each of two
starts:

- ``mean_view``: the evaluation's start, the mean viewing direction;
- ``ground_truth``: the stored ground-truth normal (``--start ground_truth``
  in the evaluation).

Today's Grid Plane Normal (``finite_difference_normal`` with
``layout="grid"``) is run from the same track at 2x2 and 3x3 pieces into the
``old_*`` and ``old3_*`` fields, as the evaluation names them. With
``--grown``, ``grown_plane_normal`` is run from the same track too, into the
evaluation's ``grown_*`` fields (its error, ``σ``, confidence, views joined
and left out, wall time and per-try trace), with its detail phases recorded
(``record_phases=True``); ``--grown-opt KEY=VALUE``, repeated, passes it a
keyword and implies ``--grown``.

    pixi run -e test python scripts/track_at_pixel/normal_sources/grid_plane_detail.py \\
        DATASET POINT [POINT ...] --out DIR [--label NAME] \\
        [--start mean_view|ground_truth ...] [--new-opt KEY=VALUE ...] \\
        [--grown] [--grown-opt KEY=VALUE ...]

``DATASET`` is a name ``dataset.py`` knows (``seoul_bull``) or the path of a
ground-truth ``.sfmr``, as for ``grid_plane_eval.py``; ``prepare`` copies it
into its cache workspace. A ``POINT`` is a point index or a
``pt3d_<hash>_<index>`` id, whose hash must be the first eight hex digits of
the ``.sfmr``'s content hash. ``--new-opt KEY=VALUE``, repeated, passes a
keyword to ``grid_plane_normal``, as in the evaluation. ``--start``, repeated,
picks the starts (both by default).

Output goes to ``DIR/<start>/<label>/``, with ``<label>`` ``defaults`` when no
``--new-opt`` or ``--grown-opt`` is given and otherwise the options joined,
the grown step's after ``grown-`` (or ``--label``):

- ``point_<index>.json``: the evaluation's row for the point (``old_*``,
  ``old3_*``, ``new_*`` and, with ``--grown``, ``grown_*`` fields, the trace
  with each proposal's error to the ground truth), plus ``point_id``,
  ``start``, ``new_options``, with ``--grown`` ``grown_options``,
  ``grown_phases`` (the step's recorded phases, as the binding gives them) and
  ``grown_phase_ms`` (each phase name's summed ``ms`` and ``count``), and
  ``detail``, the step's detail record with every array replaced by
  ``{"npz": KEY, "shape": [...], "dtype": ...}``;
- ``point_<index>.npz``: those arrays under their ``KEY`` (the path into the
  record, such as ``detail/0/views/1/tile``), and ``patch_bitmap``, the
  point's stored ``(R, R, 4)`` uint8 bitmap, where the ``.sfmr`` stores
  bitmaps.

``DIR/cameras.json`` holds each image's name, pose and lens, as the
evaluation writes it. With ``--grown``, the grown step's sentence and its
time per phase name (summed, slowest first, each with its share of the step;
a phase's time includes the phases inside it) are printed for each point as
it finishes, and at the end a table for each start: the
point, the errors of the start, ``old``, ``new`` and the grown step, the grown
step's weakest ``σ`` (every kept estimate's information summed, and with each
kept pair's halved) and confidence, its views joined of those tried, the
views it grew against the reference of those beside the reference, the pairs
kept of those matched, and its wall time.

The 16 hand-picked points of the grown-plane draft are ``dino_dog_toy`` (the
candidate ground truth, given by its path) ``0 1 528 672 810 933``,
``seoul_bull`` ``120 50 236 32``, and ``kerry_park``
(``test-data/images/kerry_park/kerry_park_ground_truth.sfmr``)
``85 200 228 342 91 242``.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from context import DatasetContext  # noqa: E402
from dataset import prepare  # noqa: E402
from grid_plane_eval import (  # noqa: E402
    add_grown_arguments,
    grown_options_of,
    parse_option,
    run_grown,
    run_new,
    run_old,
    start_track,
    write_cameras,
)

POINT_ID = re.compile(r"^pt3d_([0-9a-fA-F]{8})_(\d+)$")
STARTS = ("mean_view", "ground_truth")


def parse_points(specs, recon):
    """Point indices from indices or ``pt3d_<hash>_<index>`` ids."""
    hash8 = recon.content_xxh128[:8].lower()
    points = []
    for spec in specs:
        match = POINT_ID.match(spec)
        if match:
            if match.group(1).lower() != hash8:
                raise SystemExit(
                    f"{spec} names a reconstruction with content hash "
                    f"{match.group(1)}, but this one's begins {hash8}"
                )
            index = int(match.group(2))
        elif spec.isdigit():
            index = int(spec)
        else:
            raise SystemExit(f"not a point index or pt3d_<hash>_<index> id: {spec!r}")
        if index >= recon.point_count:
            raise SystemExit(f"point {index} is past the {recon.point_count} points")
        points.append(index)
    return points, hash8


def split_arrays(value, key, arrays):
    """``value`` with every array moved into ``arrays`` under its path."""
    if isinstance(value, np.ndarray):
        arrays[key] = value
        return {"npz": key, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {k: split_arrays(v, f"{key}/{k}", arrays) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [split_arrays(v, f"{key}/{i}", arrays) for i, v in enumerate(value)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(ds, edited, p, start, new_options, grown_options=None):
    """The evaluation's row for ``p`` from ``start``, with the step's detail.

    ``grown_options`` ``None`` leaves the grown step out.
    """
    row, track, to_cam, gt = start_track(ds, edited, p, start=start)
    if track is None:
        return row, None
    run_old(row, "old", track, edited, ds, 2, to_cam, gt)
    run_old(row, "old3", track, edited, ds, 3, to_cam, gt)
    report = run_new(
        row, track, edited, ds, {**new_options, "record_detail": True}, to_cam, gt
    )
    if grown_options is not None:
        grown = run_grown(
            row, track, edited, ds, {**grown_options, "record_phases": True}, to_cam, gt
        )
        if grown is not None:
            row["grown_phases"] = grown["phases"]
            row["grown_phase_ms"] = phase_breakdown(grown["phases"])
    return row, None if report is None else report["estimate"]["detail"]


def phase_breakdown(phases):
    """Each phase name's summed time, ms, and how many times it ran.

    A phase's time includes the phases inside it, so the names do not add up
    to the step's.
    """
    out = {}
    for phase in phases:
        entry = out.setdefault(phase["name"], {"ms": 0.0, "count": 0})
        entry["ms"] += 1000 * phase["duration_s"]
        entry["count"] += 1
    return out


def breakdown_lines(breakdown):
    """The breakdown as lines, slowest first, each with its share of the step."""
    total = breakdown.get("grown plane normal", {}).get("ms", 0.0)
    lines = []
    for name, entry in sorted(breakdown.items(), key=lambda kv: -kv[1]["ms"]):
        share = 100 * entry["ms"] / total if total > 0 else float("nan")
        lines.append(
            f"    {name:<20s} {entry['ms']:9.1f} ms {share:5.1f}%  x{entry['count']}"
        )
    return lines


GROWN_HEADER = " point  start    old    new  grown  sigma  sig_b conf joined matched pairs       ms"


def grown_line(row):
    """One table line: the grown step on a point, beside the others."""

    def deg(value):
        return "     -" if value is None else f"{value:6.2f}"

    confidence = {None: "-", True: "low", False: "ok"}[row.get("grown_low_confidence")]
    tried = row.get("grown_joined", 0) + row.get("grown_left_out", 0)
    pairs = row.get("grown_pairs", [])
    kept = sum(p["kept"] for p in pairs)
    return (
        f"{row['point']:6d} {deg(row.get('start_deg'))} {deg(row.get('old_deg'))} "
        f"{deg(row.get('new_deg'))} {deg(row.get('grown_deg'))} "
        f"{deg(row.get('grown_sigma_weak_deg'))} "
        f"{deg(row.get('grown_discounted_sigma_weak_deg'))} {confidence:>4s} "
        f"{row.get('grown_joined', 0):3d}/{tried:<3d} "
        f"{row.get('grown_matched', 0):3d}/{row.get('grown_candidates', 0):<3d} "
        f"{kept:2d}/{len(pairs):<2d} "
        f"{row.get('grown_ms', float('nan')):8.0f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("points", nargs="+", help="point indices or pt3d ids")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default=None, help="the options' directory name")
    ap.add_argument(
        "--start",
        action="append",
        choices=STARTS,
        default=None,
        help="a start to run from, repeated (both by default)",
    )
    ap.add_argument(
        "--new-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a grid_plane_normal keyword, repeated",
    )
    add_grown_arguments(ap)
    args = ap.parse_args()
    new_options = dict(parse_option(o) for o in args.new_opt)
    if "record_detail" in new_options:
        raise SystemExit("record_detail is always on here")
    grown_options = grown_options_of(args)
    names = [f"{k}={v}" for k, v in new_options.items()]
    if grown_options:
        # As written, so a list keeps its commas and no brackets or spaces.
        names.append("grown-" + ",".join(args.grown_opt))
    label = args.label or (",".join(names) or "defaults")
    from sfmtool.reconstruction import EditedReconstruction

    prepared = prepare(args.dataset, None, quiet=True)
    ds = DatasetContext(prepared)
    rec = ds.recon
    points, hash8 = parse_points(args.points, rec)
    for p in points:
        if ds.point_w[p] == 0 or len(ds.point_images[p]) < 2:
            raise SystemExit(
                f"point {p} is at infinity or seen in fewer than two images"
            )
    # One edited reconstruction for every point, as in the evaluation.
    edited = EditedReconstruction(rec)
    bitmaps = rec.patch_bitmaps

    args.out.mkdir(parents=True, exist_ok=True)
    write_cameras(ds, prepared, args.out)
    for start in args.start or STARTS:
        directory = args.out / start / label
        directory.mkdir(parents=True, exist_ok=True)
        lines = []
        for p in points:
            row, detail = run(ds, edited, p, start, new_options, grown_options)
            row["point_id"] = f"pt3d_{hash8}_{p}"
            row["start"] = start
            row["new_options"] = new_options
            if grown_options is not None:
                row["grown_options"] = grown_options
            arrays = {}
            row["detail"] = split_arrays(detail, "detail", arrays)
            if bitmaps is not None:
                arrays["patch_bitmap"] = np.asarray(bitmaps[p])
            np.savez_compressed(directory / f"point_{p}.npz", **arrays)
            with open(directory / f"point_{p}.json", "w") as sink:
                json.dump(row, sink, indent=1)
            print(
                f"{start} {label} {row['point_id']}: {row.get('new_outcome')} "
                f"{row.get('new_stop')}, {row.get('new_iterations')} renders",
                flush=True,
            )
            if grown_options is not None:
                print(f"  grown: {row.get('grown_text', row['status'])}", flush=True)
                if "grown_phase_ms" in row:
                    for line in breakdown_lines(row["grown_phase_ms"]):
                        print(line, flush=True)
                if row["status"] == "ok":
                    lines.append(grown_line(row))
        if lines:
            print(f"\n{prepared.name} from {start}, errors to the ground truth (deg)")
            print(GROWN_HEADER)
            print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
