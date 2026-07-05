#!/usr/bin/env python3
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Validate the D-optimal refinement-basis cap (``--refine-max-views``).

The harness required by ``specs/core/patch-normal-refine-view-subset.md``: run
``sfm embed-patches`` on a given ``.sfmr`` once per ``--refine-max-views`` value
(``0`` = the all-views baseline), each under ``SFMTOOL_PROFILE=1``, and report
per run:

- **Wall time** of every ``refine_patch_cloud_normals`` pass (parsed from the
  profile blocks; round 1 is uncapped by design, rounds 2+ carry the cap) plus
  the end-to-end wall time, and the subset/fallback patch counters.
- **Normal agreement vs. the baseline**: per-surviving-point angular delta
  between the run's patch normal and the baseline's (points matched by their
  homogeneous position, which embed-patches carries through unchanged) —
  mean / median / p95 degrees.
- **Output shape**: point and observation counts (the lossless claim — the cap
  shrinks only the refinement basis, never the observation set).
- **Quality**: the surviving points' reprojection-error distribution
  (mean / p95).

Acceptance target (see the spec): at ``K = 5`` the round-2 refine wall time
drops substantially (aim >= 2x) while the median normal delta vs. baseline
stays on the order of a degree and the reproj-error p95 does not regress.

Example (a solved workspace's sift_files reconstruction; ~15 min/run on the
250-image profiling dataset — budget accordingly):

  pixi run python scripts/validate_refine_subset.py \\
      /path/to/solve.sfmr --sweep 0,3,5,8 --out-dir /tmp/refine_subset
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROFILE_PASS_RE = re.compile(
    r"refine_patch_cloud_normals: (\d+) patches, wall ([0-9.]+)s"
)
SUBSET_COUNTERS_RE = re.compile(r"view-subsets (\d+)\s+subset-fallbacks (\d+)")


def run_embed_patches(
    sfm: list[str], input_sfmr: Path, output: Path, k: int, extra: list[str]
) -> dict:
    """One ``sfm embed-patches`` run with the cap ``k``, profiled. Returns the
    parsed profile passes, counters, and end-to-end wall time."""
    env = dict(os.environ, SFMTOOL_PROFILE="1")
    cmd = [
        *sfm,
        "embed-patches",
        str(input_sfmr),
        str(output),
        "--refine-max-views",
        str(k),
        *extra,
    ]
    print(f"\n=== K={k}: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"embed-patches failed for K={k} (exit {proc.returncode})")
    # The profile blocks go to stderr; one block per refine_normals pass
    # (round 1 first, then rounds 2+ — only the latter carry the cap).
    passes = [
        {"patches": int(m.group(1)), "wall_s": float(m.group(2))}
        for m in PROFILE_PASS_RE.finditer(proc.stderr)
    ]
    subsets, fallbacks = 0, 0
    for m in SUBSET_COUNTERS_RE.finditer(proc.stderr):
        subsets += int(m.group(1))
        fallbacks += int(m.group(2))
    return {
        "k": k,
        "refine_passes": passes,
        "view_subsets": subsets,
        "subset_fallbacks": fallbacks,
        "end_to_end_s": wall,
    }


def load_points(path: Path) -> dict:
    """Per-point position -> (normal, reproj error) map plus the shape counts.

    embed-patches carries positions through unchanged (it only culls points), so
    the rounded homogeneous position is a stable cross-run point identity.
    """
    from sfmtool._sfmtool import SfmrReconstruction

    recon = SfmrReconstruction.load(str(path))
    cloud = recon.patches
    if cloud is None:
        raise SystemExit(f"{path} carries no patch frames")
    pos = np.asarray(recon.positions_xyzw, dtype=np.float64)
    errors = np.asarray(recon.errors, dtype=np.float64)
    by_pos = {}
    for i, pid in enumerate(np.asarray(cloud.point_indexes)):
        key = tuple(np.round(pos[int(pid)], 9))
        by_pos[key] = (
            np.asarray(cloud[i].normal, dtype=np.float64),
            float(errors[int(pid)]),
        )
    return {
        "points": recon.point_count,
        "observations": recon.observation_count,
        "by_pos": by_pos,
        "errors": errors,
    }


def compare_to_baseline(base: dict, run: dict) -> dict:
    """Angular normal deltas (deg) over the points surviving in both runs."""
    deltas = []
    for key, (n, _) in run["by_pos"].items():
        b = base["by_pos"].get(key)
        if b is None:
            continue
        dot = float(np.clip(abs(np.dot(n, b[0])), -1.0, 1.0))
        deltas.append(np.degrees(np.arccos(dot)))
    deltas = np.asarray(deltas)
    if deltas.size == 0:
        return {"matched": 0}
    return {
        "matched": int(deltas.size),
        "normal_delta_mean_deg": float(deltas.mean()),
        "normal_delta_median_deg": float(np.median(deltas)),
        "normal_delta_p95_deg": float(np.percentile(deltas, 95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_sfmr", type=Path, help="sift_files .sfmr to convert")
    ap.add_argument(
        "--sweep",
        default="0,3,5,8",
        help="comma-separated --refine-max-views values (0 = baseline, first)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where the per-K outputs + report.json go (default: alongside input)",
    )
    ap.add_argument(
        "--sfm",
        default=None,
        help="sfm executable (default: 'sfm' on PATH — run under pixi)",
    )
    ap.add_argument(
        "extra",
        nargs="*",
        help="extra args forwarded to embed-patches (after '--')",
    )
    args = ap.parse_args()

    sweep = [int(k) for k in args.sweep.split(",")]
    if 0 not in sweep:
        sweep.insert(0, 0)  # the baseline every other run is compared against
    sweep.sort()

    sfm_bin = args.sfm or shutil.which("sfm")
    if sfm_bin is None:
        raise SystemExit("no 'sfm' on PATH — run via `pixi run python scripts/...`")
    out_dir = args.out_dir or args.input_sfmr.parent / "refine_subset_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    outputs: dict[int, Path] = {}
    for k in sweep:
        output = out_dir / f"embedded_k{k}.sfmr"
        row = run_embed_patches([sfm_bin], args.input_sfmr, output, k, args.extra)
        outputs[k] = output
        report.append(row)

    base = load_points(outputs[0])
    for row in report:
        run = load_points(outputs[row["k"]])
        row["points"] = run["points"]
        row["observations"] = run["observations"]
        row["reproj_error_mean"] = float(run["errors"].mean())
        row["reproj_error_p95"] = float(np.percentile(run["errors"], 95))
        row.update(compare_to_baseline(base, run))

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_dir / 'report.json'}")
    hdr = (
        f"{'K':>3} {'end2end_s':>10} {'refine_walls_s':>28} {'pts':>7} {'obs':>8} "
        f"{'dN_med':>7} {'dN_p95':>7} {'err_mean':>9} {'err_p95':>8}"
    )
    print(hdr)
    for row in report:
        walls = "+".join(f"{p['wall_s']:.1f}" for p in row["refine_passes"])
        print(
            f"{row['k']:>3} {row['end_to_end_s']:>10.1f} {walls:>28} "
            f"{row['points']:>7} {row['observations']:>8} "
            f"{row.get('normal_delta_median_deg', 0.0):>7.3f} "
            f"{row.get('normal_delta_p95_deg', 0.0):>7.3f} "
            f"{row['reproj_error_mean']:>9.4f} {row['reproj_error_p95']:>8.4f}"
        )


if __name__ == "__main__":
    main()
