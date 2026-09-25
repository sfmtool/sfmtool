# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tables from the harness's JSON lines.

One table per (dataset, pose, measurement), one row per strategy:

- ``recall``: tracks the image was in (that survived its removal) that it
  rejoined, over how many there were.
- ``err med / p90``: distance of a rejoined keypoint from the original, px.
- ``>1px``, ``>2px``: rejoined keypoints that far from the original (the
  precision problems on known tracks).
- ``extra``: accepted tracks the image was not in, in total.
- ``x zncc``, ``x off``: their median ZNCC and median distance from the
  projection at the pose, px.
- ``x new res med / p90``: the new observation's reprojection residual after
  retriangulating the point with it included, px.
- ``x bad``: extra tracks whose new observation's residual exceeds 2 px.
- ``x worse``: extra tracks whose largest residual over all observations grew
  by more than 1 px when the new observation was included.
- ``k new res med``: the same new-observation residual for rejoined known
  tracks, the baseline an extra track is compared with.
- ``gt far``: of the ``>2px`` rejoined keypoints, those whose original keypoint
  itself sits more than 2 px from the point's projection at the pose.

Usage::

    pixi run -e test python scripts/add_image_to_tracks/summarize.py <out>/<dataset>.jsonl ...
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def pct(values, q):
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.percentile(v, q)) if v else float("nan")


def load(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def header(rows):
    by_image = {}
    for r in rows:
        by_image[(r["dataset"], r["image"])] = r
    out = defaultdict(list)
    for (ds, _), r in by_image.items():
        out[ds].append(r)
    lines = []
    for ds, images in out.items():
        acc = [r for r in images if r["resect_accepted"]]
        rel = [r["center_error"] / r["scene_scale"] for r in acc if r["scene_scale"]]
        lines.append(
            f"**{ds}**: {len(images)} images, {sum(r['known'] for r in images)} known "
            f"observations to recover, {sum(r['lost'] for r in images)} points lost on "
            f"removal (fewer than two observations left); resection accepted "
            f"{len(acc)}/{len(images)} "
            f"({sum(r['resect_rotation_only'] for r in acc)} rotation-only), rotation error "
            f"median {pct([r['rotation_deg'] for r in acc], 50):.3f} deg / max "
            f"{pct([r['rotation_deg'] for r in acc], 100):.3f} deg, centre error median "
            f"{pct(rel, 50):.4f} / max {pct(rel, 100):.4f} of the scene scale."
        )
        refused = [r for r in images if not r["resect_accepted"]]
        for r in refused:
            lines.append(f"  - image {r['image']} refused: {r['resect_refusal']}")
    return "\n".join(lines)


def table(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["pose"], r["measurement"], r["strategy"])].append(r)
    by_table = defaultdict(list)
    for (ds, pose, meas, strat), rs in groups.items():
        known = sum(r["known"] for r in rs)
        rec = sum(r["known_recovered"] for r in rs)
        err = [e for r in rs for e in r["known_err_px"]]
        extra = [x for r in rs for x in r["extra"]]
        kres = [x for r in rs for x in r.get("known_new_residual_px", [])]
        xres = [
            x["new_residual_px"] for x in extra if x.get("new_residual_px") is not None
        ]
        worse = [
            x["max_residual_px"] - x["base_max_residual_px"]
            for x in extra
            if x.get("base_max_residual_px") is not None
        ]
        gt_far = sum(
            e > 2 and o > 2
            for r in rs
            for e, o in zip(r["known_err_px"], r.get("known_gt_offset_px", []))
        )
        by_table[(ds, pose, meas)].append(
            [
                strat,
                f"{rec}/{known} ({100 * rec / max(known, 1):.1f}%)",
                f"{pct(err, 50):.3f} / {pct(err, 90):.3f}",
                str(sum(e > 1 for e in err)),
                str(sum(e > 2 for e in err)),
                str(len(extra)),
                f"{pct([x['zncc'] for x in extra], 50):.3f}",
                f"{pct([x['offset_px'] for x in extra], 50):.2f}",
                f"{pct(xres, 50):.2f} / {pct(xres, 90):.2f}",
                str(sum(x > 2 for x in xres)),
                str(sum(w > 1 for w in worse)),
                f"{pct(kres, 50):.2f}",
                str(gt_far),
            ]
        )
    cols = [
        "strategy",
        "recall",
        "err med / p90",
        ">1px",
        ">2px",
        "extra",
        "x zncc",
        "x off",
        "x new res med / p90",
        "x bad",
        "x worse",
        "k new res med",
        "gt far",
    ]
    out = []
    for (ds, pose, meas), body in sorted(by_table.items()):
        out.append(f"\n### {ds}, {pose}, measurement `{meas}`\n")
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "---|" * len(cols))
        for row in body:
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main():
    rows = load(sys.argv[1:])
    print(header(rows))
    print(table(rows))


if __name__ == "__main__":
    main()
