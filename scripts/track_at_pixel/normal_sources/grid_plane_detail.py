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
``old_*`` and ``old3_*`` fields, as the evaluation names them.

    pixi run -e test python scripts/track_at_pixel/normal_sources/grid_plane_detail.py \\
        DATASET POINT [POINT ...] --out DIR [--label NAME] \\
        [--start mean_view|ground_truth ...] [--new-opt KEY=VALUE ...]

``DATASET`` is a name ``dataset.py`` knows (``seoul_bull``) or the path of a
ground-truth ``.sfmr``, as for ``grid_plane_eval.py``; ``prepare`` copies it
into its cache workspace. A ``POINT`` is a point index or a
``pt3d_<hash>_<index>`` id, whose hash must be the first eight hex digits of
the ``.sfmr``'s content hash. ``--new-opt KEY=VALUE``, repeated, passes a
keyword to ``grid_plane_normal``, as in the evaluation. ``--start``, repeated,
picks the starts (both by default).

Output goes to ``DIR/<start>/<label>/``, with ``<label>`` ``defaults`` when no
``--new-opt`` is given and otherwise the options joined (or ``--label``):

- ``point_<index>.json``: the evaluation's row for the point (``old_*``,
  ``old3_*`` and ``new_*`` fields, the trace with each proposal's error to the
  ground truth), plus ``point_id``, ``start``, ``new_options`` and
  ``detail``, the step's detail record with every array replaced by
  ``{"npz": KEY, "shape": [...], "dtype": ...}``;
- ``point_<index>.npz``: those arrays under their ``KEY`` (the path into the
  record, such as ``detail/0/views/1/tile``), and ``patch_bitmap``, the
  point's stored ``(R, R, 4)`` uint8 bitmap, where the ``.sfmr`` stores
  bitmaps.

``DIR/cameras.json`` holds each image's name, pose and lens, as the
evaluation writes it.
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
    parse_option,
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


def run(ds, edited, p, start, new_options):
    """The evaluation's row for ``p`` from ``start``, with the step's detail."""
    row, track, to_cam, gt = start_track(ds, edited, p, start=start)
    if track is None:
        return row, None
    run_old(row, "old", track, edited, ds, 2, to_cam, gt)
    run_old(row, "old3", track, edited, ds, 3, to_cam, gt)
    report = run_new(
        row, track, edited, ds, {**new_options, "record_detail": True}, to_cam, gt
    )
    return row, None if report is None else report["estimate"]["detail"]


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
    args = ap.parse_args()
    new_options = dict(parse_option(o) for o in args.new_opt)
    if "record_detail" in new_options:
        raise SystemExit("record_detail is always on here")
    label = args.label or (
        ",".join(f"{k}={v}" for k, v in new_options.items()) or "defaults"
    )
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
        for p in points:
            row, detail = run(ds, edited, p, start, new_options)
            row["point_id"] = f"pt3d_{hash8}_{p}"
            row["start"] = start
            row["new_options"] = new_options
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


if __name__ == "__main__":
    main()
