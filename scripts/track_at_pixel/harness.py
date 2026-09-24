# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Leave-one-track-out evaluation of track-at-pixel candidates.

For each ground-truth point under test: remove it from the reconstruction, and
for every image it was observed in, ask the candidate to build a track at that
observation's pixel. Each query yields one row -- either the failure's stage and
reason, or the built track scored against the removed one (see ``metrics.py``).

    pixi run -e test python scripts/track_at_pixel/harness.py \\
        --dataset seoul_bull --candidate baseline --points 20

Rows go to ``<out>/rows.jsonl`` and a summary is printed and written beside
them. ``--opt key=value`` overrides a candidate default (values parsed as
JSON, else kept as strings), so a variant needs no new file.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sfmtool._sfmtool import bench  # noqa: E402
from sfmtool._sfmtool.reconstruction import EditedReconstruction  # noqa: E402

from api import TrackAtPixelError  # noqa: E402
from context import DatasetContext  # noqa: E402
from dataset import default_cache_dir, prepare  # noqa: E402
from metrics import ground_truth_reading, score  # noqa: E402


def parse_opts(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def choose_points(ds: DatasetContext, args) -> list[int]:
    if args.point_ids:
        return [int(p) for p in args.point_ids.split(",")]
    lengths = np.asarray([len(i) for i in ds.point_images])
    ok = lengths >= args.min_track_length
    if not args.include_infinity:
        ok &= ds.point_w != 0
    candidates = np.flatnonzero(ok)
    rng = np.random.default_rng(args.seed)
    if args.points and args.points < len(candidates):
        candidates = np.sort(rng.choice(candidates, args.points, replace=False))
    return [int(p) for p in candidates]


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


SUMMARY_METRICS = [
    "query_keypoint_offset_px",
    "query_projection_offset_px",
    "position_err_angle_deg",
    "position_err_rel_depth",
    "position_err_in_gt_halves",
    "normal_err_deg",
    "half_extent_ratio",
    "texel_scale_min",
    "texel_scale_median",
    "texel_scale_max",
    "gt_texel_scale_min",
    "gt_texel_scale_max",
    "texel_scale_aniso_max",
    "image_recall",
    "image_precision",
    "view_precision",
    "kp_err_median_px",
    "zncc_median",
    "gt_zncc_median",
    "zncc_median_delta",
    "n_in",
    "n_gt",
    "seconds",
]


def summarize(rows: list[dict]) -> str:
    lines = []
    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    lines.append(f"queries: {n}   built: {len(ok)} ({100 * len(ok) / max(n, 1):.0f}%)")
    good = [r for r in ok if r.get("good")]
    lines.append(
        f"good: {len(good)} ({100 * len(good) / max(n, 1):.0f}% of queries)   "
        f"built but not good: {len(ok) - len(good)}"
    )
    misses = Counter(f for r in ok for f in r.get("good_failures", []))
    if misses:
        lines.append(
            "  not good because: "
            + ", ".join(f"{k} {v}" for k, v in misses.most_common())
        )
    # Group refusals by their sentence with the numbers masked, so "sits 2.6 px"
    # and "sits 5.0 px" count as one kind of refusal.
    failures = Counter(
        f"{r['stage']}: {re.sub(r'[0-9]+(\.[0-9]+)?', '#', r['reason'])[:100]}"
        for r in rows
        if r["status"] != "ok"
    )
    by_stage = Counter(r["stage"] for r in rows if r["status"] != "ok")
    if by_stage:
        lines.append(
            "failures by stage: "
            + ", ".join(f"{k} {v}" for k, v in by_stage.most_common())
        )
        for text, count in failures.most_common(8):
            lines.append(f"  {count:4d}  {text}")
    if ok:
        lines.append("")
        lines.append(
            f"{'metric (built tracks)':32s} {'n':>5s} {'p10':>9s} {'median':>9s} {'p90':>9s}"
        )
        for key in SUMMARY_METRICS:
            vals = np.asarray(
                [r[key] for r in ok if r.get(key) is not None], dtype=float
            )
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            p10, p50, p90 = np.percentile(vals, [10, 50, 90])
            lines.append(f"{key:32s} {vals.size:5d} {p10:9.3f} {p50:9.3f} {p90:9.3f}")
        frac = np.mean(
            [bool(r.get("at_infinity")) != bool(r.get("gt_at_infinity")) for r in ok]
        )
        lines.append(f"{'finite/infinity disagreement':32s} {frac:9.1%}")
    return "\n".join(lines)


def save_tracks(built, out: Path, prepared, candidate: str, options: dict) -> Path:
    """Write the returned tracks as ``tracks.sfmr`` in the run directory.

    A row's ``output_point`` is its track's point index in this file.
    """
    recon, _, _ = built.materialize()
    path = out / "tracks.sfmr"
    recon.save(
        path,
        operation="track_at_pixel",
        tool_options={
            "dataset": prepared.name,
            "ground_truth": str(prepared.ground_truth),
            "candidate": candidate,
            "options": json.dumps(options),
        },
    )
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dataset",
        default="seoul_bull",
        help="dataset name or a ground-truth .sfmr path",
    )
    ap.add_argument("--candidate", default="baseline", help="module under candidates/")
    ap.add_argument(
        "--opt", action="append", default=[], help="candidate option key=value"
    )
    ap.add_argument(
        "--points", type=int, default=0, help="sample this many points (0 = all)"
    )
    ap.add_argument(
        "--point-ids", default="", help="comma-separated point indexes to test"
    )
    ap.add_argument("--min-track-length", type=int, default=2)
    ap.add_argument(
        "--include-infinity", action="store_true", help="also test points at infinity"
    )
    ap.add_argument(
        "--max-queries-per-point", type=int, default=0, help="0 = every observation"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--matches",
        type=Path,
        default=None,
        help="cluster-patches .matches to hand the candidate (default: built "
        "from the dataset's .kdf in the cache workspace)",
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=None, help=f"default {default_cache_dir()}"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="run directory (default <cache>/runs/<candidate>-<time>)",
    )
    ap.add_argument(
        "--raise",
        dest="reraise",
        action="store_true",
        help="re-raise unexpected exceptions",
    )
    args = ap.parse_args(argv)

    candidate = importlib.import_module(f"candidates.{args.candidate}")
    options = parse_opts(args.opt)

    prepared = prepare(args.dataset, args.cache_dir, matches=args.matches)
    t0 = time.perf_counter()
    ds = DatasetContext(prepared)
    print(
        f"loaded {prepared.name}: {ds.recon.image_count} images, {ds.recon.point_count} points "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    out = (
        args.out
        or prepared.workspace
        / "runs"
        / f"{args.candidate}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                "dataset": prepared.name,
                "candidate": args.candidate,
                "options": options,
                "matches": str(prepared.matches),
                "defaults": getattr(candidate, "DEFAULTS", {}),
                "argv": sys.argv[1:],
            },
            indent=2,
            default=str,
        )
    )

    points = choose_points(ds, args)
    rows = []
    # Every returned track, committed into the ground truth's cameras with none
    # of its points, so the file loads beside the ground truth in the viewer.
    built = EditedReconstruction(
        ds.recon.filter_points_by_mask(np.zeros(ds.recon.point_count, dtype=bool))
    )
    with open(out / "rows.jsonl", "w") as sink:
        for pi, point in enumerate(points):
            gt_read = ground_truth_reading(ds, point)
            ctx = ds.holdout(point)
            images = ds.point_images[point]
            order = range(len(images))
            if args.max_queries_per_point:
                order = list(order)[: args.max_queries_per_point]
            for j in order:
                image = int(images[j])
                pixel = ds.point_keypoints[point][j]
                row = {"point": point, "image": image, "pixel": pixel.tolist()}
                start = time.perf_counter()
                try:
                    result = candidate.build_track(ctx, image, pixel, options)
                    row["status"] = "ok"
                    row.update(score(ds, point, image, pixel, result, gt_read))
                    row["diagnostics"] = result.diagnostics
                    try:
                        built, committed = bench.commit(
                            built, result.track, node="tracks.sfmr"
                        )
                        row["output_point"] = committed["point"]
                    except ValueError as e:
                        row["output_error"] = str(e)
                except TrackAtPixelError as e:
                    row.update(
                        status="refused",
                        stage=e.stage,
                        reason=e.reason,
                        diagnostics=e.diagnostics,
                    )
                except Exception as e:  # a candidate bug, not a refusal
                    if args.reraise:
                        raise
                    row.update(
                        status="crashed",
                        stage="exception",
                        reason=f"{type(e).__name__}: {e}",
                        traceback=traceback.format_exc(),
                    )
                row["seconds"] = time.perf_counter() - start
                rows.append(row)
                sink.write(json.dumps(_jsonable(row)) + "\n")
                sink.flush()
            n_built = sum(r["status"] == "ok" for r in rows if r["point"] == point)
            print(
                f"[{pi + 1}/{len(points)}] point {point}: {n_built}/{len(order)} built"
            )

    summary = summarize(rows)
    (out / "summary.txt").write_text(summary + "\n")
    print()
    print(summary)
    print(f"\nrows: {out / 'rows.jsonl'}")
    if built.point_count == 0:
        # A .sfmr with no points cannot be written; there is nothing to compare.
        print("tracks: none built, so no tracks.sfmr was written")
        return 0
    sfmr = save_tracks(built, out, prepared, args.candidate, options)
    print(f"tracks: {sfmr} ({built.point_count} points)")
    print(f"compare: pixi run gui -- {prepared.ground_truth} {sfmr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
