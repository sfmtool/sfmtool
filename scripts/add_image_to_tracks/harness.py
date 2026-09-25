# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Leave-one-image-out evaluation of Add Image to Tracks.

For each image of a ground-truth reconstruction:

1. Remove the image from the ground truth: delete all of its observations, and
   the points that are left with fewer than two (those cannot come back, and
   are counted as lost). The image keeps its ground-truth pose.
2. Resect it back in with ``geometry.resect_images`` over the tracks and the
   dataset's cluster-patches file, as the viewer's *Resect Image* does. An
   image with no observations left is posed from the clusters alone.
3. Run ``EditedReconstruction.add_image_to_tracks`` with every strategy, once
   at the resected pose and once at the ground-truth pose, so the error the pose
   brings and the error the matching brings can be told apart.
4. Measure, against the observations the image had in the ground truth:
   recall of the tracks it was in (that still exist), the distance of each
   recovered keypoint from the original, the recovered keypoints that landed
   far from it, and the tracks it joins that it was not in, with their
   reprojection error at the pose, their ZNCC and the residuals of the point
   retriangulated with the new observation included.

The rules only filter candidates the operation has measured, so for each
measurement setting (template, sub-pixel step, localizability gate) the
retriangulation of every measured candidate is computed once and every
strategy's accepted set is looked up in it.

Writes one JSON line per (image, pose, measurement, strategy) to
``<out>/<dataset>.jsonl``; ``summarize.py`` turns those into tables.

Usage::

    pixi run -e test python scripts/add_image_to_tracks/harness.py \\
        --dataset seoul_bull --cache <dir> --out <dir> [--images 0,3,5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import prepare  # noqa: E402
from strategies import MEASUREMENTS, STRATEGIES  # noqa: E402


def rotation_from_wxyz(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class Geometry:
    """Poses, lenses and tracks of one reconstruction, for retriangulation."""

    def __init__(self, recon):
        self.cams = recon.cameras
        self.cam_idx = np.asarray(recon.camera_indexes)
        self.R = [rotation_from_wxyz(q) for q in np.asarray(recon.quaternions_wxyz)]
        self.t = [np.asarray(t, dtype=float) for t in np.asarray(recon.translations)]
        self.C = [-R.T @ t for R, t in zip(self.R, self.t)]
        self.xyzw = np.asarray(recon.positions_xyzw)
        pts = np.asarray(recon.track_point_indexes)
        imgs = np.asarray(recon.track_image_indexes)
        kps = np.asarray(recon.keypoints_xy, dtype=float)
        order = np.argsort(pts, kind="stable")
        self.obs = {}
        bounds = np.searchsorted(pts[order], np.arange(len(self.xyzw) + 1))
        for p in range(len(self.xyzw)):
            rows = order[bounds[p] : bounds[p + 1]]
            self.obs[p] = [(int(imgs[r]), kps[r]) for r in rows]

    def ray(self, image: int, uv) -> np.ndarray:
        cam = self.cams[self.cam_idx[image]]
        d = self.R[image].T @ np.asarray(cam.pixel_to_ray(float(uv[0]), float(uv[1])))
        return d / np.linalg.norm(d)

    def project(self, image: int, X) -> np.ndarray | None:
        cam = self.cams[self.cam_idx[image]]
        pc = self.R[image] @ X + self.t[image]
        px = cam.ray_to_pixel([float(pc[0]), float(pc[1]), float(pc[2])])
        return None if px is None else np.asarray(px)

    def retriangulate(self, point: int, image: int, keypoint) -> dict | None:
        """The point triangulated from its observations plus ``(image, keypoint)``.

        A least-squares ray intersection, then the reprojection residual of
        every observation at that position. ``None`` for a point at infinity.
        """
        if self.xyzw[point, 3] == 0.0:
            return None
        base = self.obs[point]
        obs = base + [(image, np.asarray(keypoint, dtype=float))]
        X = self._intersect(obs)
        X0 = self._intersect(base)
        if X is None or X0 is None:
            return None
        res = self._residuals(obs, X)
        res0 = self._residuals(base, X0)
        old = self.xyzw[point, :3]
        depth = np.median([np.linalg.norm(old - self.C[i]) for i, _ in obs])
        return {
            "new_residual_px": res[-1],
            "max_residual_px": max(res),
            "base_max_residual_px": max(res0),
            "median_residual_px": float(np.median(res)),
            "move": float(np.linalg.norm(X - old)),
            "move_rel": float(np.linalg.norm(X - old) / depth) if depth > 0 else np.nan,
        }

    def _intersect(self, obs):
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for i, uv in obs:
            d = self.ray(i, uv)
            P = np.eye(3) - np.outer(d, d)
            A += P
            b += P @ self.C[i]
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

    def _residuals(self, obs, X):
        res = []
        for i, uv in obs:
            px = self.project(i, X)
            res.append(np.inf if px is None else float(np.hypot(*(px - uv))))
        return res


def remove_image(gt, image: int):
    """``gt`` without ``image``'s observations and the points left under two.

    Returns ``(recon, gt_to_new, known_gt, lost)``: the reduced reconstruction,
    the map from ground-truth point index to its index there (``-1`` for a
    dropped point), the ground-truth points the image observed that survive,
    and how many it observed that do not.
    """
    pts = np.asarray(gt.track_point_indexes)
    imgs = np.asarray(gt.track_image_indexes)
    n = gt.point_count
    remaining = np.bincount(pts[imgs != image], minlength=n)
    observed = np.zeros(n, bool)
    observed[pts[imgs == image]] = True
    keep_point = remaining >= 2
    lost = int(np.sum(observed & ~keep_point))
    reduced = gt.filter_points_by_mask(keep_point)
    gt_to_new = np.full(n, -1, np.int64)
    gt_to_new[keep_point] = np.arange(int(keep_point.sum()))

    r_pts = np.asarray(reduced.track_point_indexes)
    r_imgs = np.asarray(reduced.track_image_indexes)
    keep = r_imgs != image
    changes = dict(
        track_image_indexes=np.ascontiguousarray(r_imgs[keep], dtype=np.uint32),
        track_feature_indexes=np.zeros(int(keep.sum()), np.uint32),
        track_point_indexes=np.ascontiguousarray(r_pts[keep], dtype=np.uint32),
        keypoints_xy=np.ascontiguousarray(
            np.asarray(reduced.keypoints_xy)[keep], dtype=np.float32
        ),
    )
    conf = reduced.observation_confidence
    if conf is not None:
        changes["observation_confidence"] = np.ascontiguousarray(np.asarray(conf)[keep])
    recon = reduced.clone_with_changes(**changes)
    known_gt = np.flatnonzero(observed & keep_point)
    return recon, gt_to_new, known_gt, lost


def pose_error(gt, recon, image: int) -> dict:
    Rg = rotation_from_wxyz(np.asarray(gt.quaternions_wxyz)[image])
    Rr = rotation_from_wxyz(np.asarray(recon.quaternions_wxyz)[image])
    tg = np.asarray(gt.translations)[image]
    tr = np.asarray(recon.translations)[image]
    cos = np.clip((np.trace(Rg.T @ Rr) - 1) / 2, -1, 1)
    return {
        "rotation_deg": float(np.degrees(np.arccos(cos))),
        "center_error": float(np.linalg.norm(-Rg.T @ tg + Rr.T @ tr)),
    }


def evaluate_image(ctx, image: int, strategies, measurements, out):
    from sfmtool._sfmtool.geometry import resect_images
    from sfmtool._sfmtool.reconstruction import EditedReconstruction

    gt = ctx["gt"]
    name = ctx["names"][image]
    reduced, gt_to_new, known_gt, lost = remove_image(gt, image)
    gt_pts = np.asarray(gt.track_point_indexes)
    gt_imgs = np.asarray(gt.track_image_indexes)
    gt_kps = np.asarray(gt.keypoints_xy, dtype=float)
    known = {}
    for g in known_gt:
        row = np.flatnonzero((gt_pts == g) & (gt_imgs == image))[0]
        known[int(gt_to_new[g])] = gt_kps[row]

    t0 = time.perf_counter()
    resected, rep = resect_images(
        reduced, [name], cluster_patches_path=str(ctx["matches"])
    )
    r = rep["images"][0]
    header = {
        "dataset": ctx["dataset"],
        "image": image,
        "image_name": name,
        "known": len(known),
        "lost": lost,
        "points": reduced.point_count,
        "resect_accepted": bool(r["accepted"]),
        "resect_refusal": r["refusal"],
        "resect_rotation_only": bool(r["rotation_only"]),
        "resect_inliers": int(r["inliers"]),
        "resect_correspondences": int(r["correspondences"]),
        "resect_cluster_correspondences": int(r["cluster_correspondences"]),
        "scene_scale": r["scene_scale"],
        "resect_seconds": time.perf_counter() - t0,
    }
    header.update(pose_error(gt, resected, image))
    if resected.point_count != reduced.point_count:
        raise SystemExit("the resection removed points; the index maps would not hold")

    poses = [("gt_pose", reduced)]
    if r["accepted"]:
        poses.append(("resected", resected))
    for pose_name, recon in poses:
        geom = Geometry(recon)
        edited = EditedReconstruction(recon)
        for meas_name, meas in measurements.items():
            if (
                meas.get("template") == "stored_bitmap"
                and recon.patch_bitmap_resolution is None
            ):
                continue
            # Every measured candidate, judged by nothing, to retriangulate once.
            _, everything = edited.add_image_to_tracks(
                image,
                ctx["pyramids"],
                rule="fixed",
                min_zncc=-2.0,
                min_keypoint_separation_px=0.0,
                **meas,
            )
            cands = everything["candidates"]
            tri = {}
            for k, p in enumerate(cands["point"]):
                kp = cands["keypoint"][k]
                if np.all(np.isfinite(kp)):
                    tri[int(p)] = geom.retriangulate(int(p), image, kp)
            for strat_name, strat in strategies.items():
                t1 = time.perf_counter()
                _, report = edited.add_image_to_tracks(
                    image, ctx["pyramids"], **meas, **strat
                )
                seconds = time.perf_counter() - t1
                out.write(
                    json.dumps(
                        record(
                            header,
                            pose_name,
                            meas_name,
                            strat_name,
                            report,
                            known,
                            tri,
                            seconds,
                        )
                    )
                    + "\n"
                )
                out.flush()


def record(
    header, pose_name, meas_name, strat_name, report, known, tri, seconds
) -> dict:
    c = report["candidates"]
    accepted = np.asarray(c["accepted"])
    points = np.asarray(c["point"])
    keypoints = np.asarray(c["keypoint"])
    zncc = np.asarray(c["zncc"])
    offset = np.asarray(c["offset_px"])
    nrefs = np.array([len(x) for x in c["references"]])
    rec = dict(header)
    rec.update(
        pose=pose_name,
        measurement=meas_name,
        strategy=strat_name,
        seconds=seconds,
        candidates=int(len(points)),
        accepted=int(report["accepted"]),
        refusal_counts=report["refusal_counts"],
        pooled_bar=report["pooled_bar"],
        position_bound_px=report["position_bound_px"],
    )
    projection = np.asarray(c["projection"])
    known_err, known_refused, known_refs, known_gt_offset = [], {}, [], []
    extra = []
    for k, p in enumerate(points):
        p = int(p)
        if p in known:
            if accepted[k]:
                known_err.append(float(np.hypot(*(keypoints[k] - known[p]))))
                known_refs.append(int(nrefs[k]))
                known_gt_offset.append(float(np.hypot(*(known[p] - projection[k]))))
            else:
                reason = c["refusal"][k]
                known_refused[reason] = known_refused.get(reason, 0) + 1
        elif accepted[k]:
            t = tri.get(p)
            extra.append(
                {
                    "zncc": float(zncc[k]),
                    "offset_px": float(offset[k]),
                    "refs": int(nrefs[k]),
                    "max_residual_px": None if t is None else t["max_residual_px"],
                    "base_max_residual_px": None
                    if t is None
                    else t["base_max_residual_px"],
                    "new_residual_px": None if t is None else t["new_residual_px"],
                    "move_rel": None if t is None else t["move_rel"],
                }
            )
    known_tri = [
        tri[int(p)]
        for k, p in enumerate(points)
        if int(p) in known and accepted[k] and tri.get(int(p)) is not None
    ]
    rec.update(
        known_recovered=len(known_err),
        known_err_px=known_err,
        known_refs=known_refs,
        known_gt_offset_px=known_gt_offset,
        known_new_residual_px=[t["new_residual_px"] for t in known_tri],
        known_refused=known_refused,
        known_max_residual_px=[t["max_residual_px"] for t in known_tri],
        extra=extra,
    )
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--cache", type=Path, required=True, help="where prepared datasets live"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--images", default="all", help="'all' or a comma list of indexes")
    ap.add_argument("--measurements", default="all")
    ap.add_argument("--strategies", default="all")
    args = ap.parse_args()

    from sfmtool._sfmtool.patches import ImagePyramidSet
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction
    from sfmtool._workspace_image import read_workspace_image

    ds = prepare(args.dataset, args.cache)
    gt = SfmrReconstruction.load(ds.sfmr)
    names = list(gt.image_names)
    images = [read_workspace_image(gt.workspace_dir, n) for n in names]
    ctx = {
        "dataset": args.dataset,
        "gt": gt,
        "names": names,
        "matches": ds.matches,
        "pyramids": ImagePyramidSet(gt, images),
    }
    which = (
        range(len(names))
        if args.images == "all"
        else [int(s) for s in args.images.split(",")]
    )
    measurements = (
        MEASUREMENTS
        if args.measurements == "all"
        else {k: MEASUREMENTS[k] for k in args.measurements.split(",")}
    )
    strategies = (
        STRATEGIES
        if args.strategies == "all"
        else {k: STRATEGIES[k] for k in args.strategies.split(",")}
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.dataset}.jsonl"
    with open(path, "w") as out:
        for image in which:
            t0 = time.perf_counter()
            evaluate_image(ctx, image, strategies, measurements, out)
            print(
                f"{args.dataset} image {image} ({names[image]}): {time.perf_counter() - t0:.1f} s"
            )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
