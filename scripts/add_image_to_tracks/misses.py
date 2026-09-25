# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Why the known tracks an image was in are not all rejoined.

Runs the operation's defaults (or the keywords given with ``--kwargs``) at the
resected and the ground-truth pose for every image, and writes one JSON line per
known track that was refused, with the numbers the refusal was made on and
two readings of the original observation: its keypoint's distance from the
point's projection at the pose (``gt_offset_px``), and the distance from it to
the keypoint the operation found (``found_vs_gt_px``). Where the two keypoints
agree, the refused ZNCC is the original sighting's own.

Usage::

    pixi run -e test python scripts/add_image_to_tracks/misses.py \\
        --dataset kerry_park --cache <dir> --out <dir> [--kwargs '{"position_k": 5}']
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import prepare  # noqa: E402
from harness import remove_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()
    kwargs = json.loads(args.kwargs)

    from sfmtool._sfmtool.geometry import resect_images
    from sfmtool._sfmtool.patches import ImagePyramidSet
    from sfmtool._sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction
    from sfmtool._workspace_image import read_workspace_image

    ds = prepare(args.dataset, args.cache)
    gt = SfmrReconstruction.load(ds.sfmr)
    names = list(gt.image_names)
    pyramids = ImagePyramidSet(
        gt, [read_workspace_image(gt.workspace_dir, n) for n in names]
    )
    gt_pts = np.asarray(gt.track_point_indexes)
    gt_imgs = np.asarray(gt.track_image_indexes)
    gt_kps = np.asarray(gt.keypoints_xy, dtype=float)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.dataset}.{args.tag}.misses.jsonl"
    totals = {}
    with open(path, "w") as out:
        for image, name in enumerate(names):
            reduced, gt_to_new, known_gt, _ = remove_image(gt, image)
            known = {}
            for g in known_gt:
                row = np.flatnonzero((gt_pts == g) & (gt_imgs == image))[0]
                known[int(gt_to_new[g])] = gt_kps[row]
            resected, _ = resect_images(
                reduced, [name], cluster_patches_path=str(ds.matches)
            )
            for pose, recon in (("gt_pose", reduced), ("resected", resected)):
                _, rep = EditedReconstruction(recon).add_image_to_tracks(
                    image, pyramids, **kwargs
                )
                c = rep["candidates"]
                t = totals.setdefault(pose, {"known": 0, "rejoined": 0})
                for k, p in enumerate(c["point"]):
                    p = int(p)
                    if p not in known:
                        continue
                    t["known"] += 1
                    if c["accepted"][k]:
                        t["rejoined"] += 1
                        continue
                    proj = c["projection"][k]
                    kp = c["keypoint"][k]
                    rec = {
                        "image": image,
                        "pose": pose,
                        "point": p,
                        "refusal": c["refusal"][k],
                        "references": len(c["references"][k]),
                        "zncc": float(c["zncc"][k]),
                        "bar": float(c["bar"][k]),
                        "judged": float(c["judged"][k]),
                        "offset_px": float(c["offset_px"][k]),
                        "bound_px": rep["position_bound_px"],
                        "sigma_pos": float(c["sigma_pos"][k]),
                        "peak_zncc": float(c["peak_zncc"][k]),
                        "gt_offset_px": float(np.hypot(*(known[p] - proj)))
                        if np.all(np.isfinite(proj))
                        else None,
                        "found_vs_gt_px": float(np.hypot(*(kp - known[p])))
                        if np.all(np.isfinite(kp))
                        else None,
                        "ref_loo_min": min(c["reference_loo_zncc"][k], default=None),
                        "ref_loo_median": float(np.median(c["reference_loo_zncc"][k]))
                        if c["reference_loo_zncc"][k]
                        else None,
                    }
                    out.write(json.dumps(rec) + "\n")
    print(json.dumps(totals))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
