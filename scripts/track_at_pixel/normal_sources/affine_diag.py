# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Normal from the cluster-patches members' affine shapes, versus the GT normal.

For each finite GT point, the cluster with a kept/reference member within
``--match-px`` of the GT keypoint in the most GT images is taken. Its members'
refined shapes give measured local affines A_ab = S_b S_a^-1 between images;
a plane with normal n through the GT point predicts A_ab(n) = J_b J_a^-1,
where J_i is the projection's Jacobian over the plane's tangent basis. The
normal minimising the log-affine residual is found by a coarse grid over the
hemisphere facing the cameras, then Nelder-Mead.

    pixi run -e test python scripts/track_at_pixel/normal_sources/affine_diag.py DATASET
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context import DatasetContext  # noqa: E402
from dataset import prepare  # noqa: E402


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def tangent(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(np.cross(n, a))
    return e1, np.cross(n, e1)


def jac(cam, X, t1, t2, h):
    cols = []
    for t in (t1, t2):
        p = cam.project(X + h * t)
        m = cam.project(X - h * t)
        if p is None or m is None:
            return None
        cols.append((p - m) / (2 * h))
    return np.stack(cols, 1)


def n_from_angles(base, e1, e2, a):
    return (
        unit(base + np.tan(a[0]) * e1 + np.tan(a[1]) * e2)
        if max(abs(a[0]), abs(a[1])) < 1.45
        else None
    )


def residual(n, X, cams, shapes, h):
    t1, t2 = tangent(n)
    Js = []
    for c in cams:
        J = jac(c, X, t1, t2, h)
        if J is None or abs(np.linalg.det(J)) < 1e-12:
            return 1e3
        Js.append(J)
    r = 0.0
    k = 0
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            A_pred = Js[j] @ np.linalg.inv(Js[i])
            A_obs = shapes[j] @ np.linalg.inv(shapes[i])
            # compare in a scale-free way: A_pred^-1 A_obs should be identity
            M = np.linalg.inv(A_pred) @ A_obs
            s = np.sqrt(abs(np.linalg.det(M)))
            if s <= 0:
                return 1e3
            r += np.sum((M / s - np.eye(2)) ** 2) + np.log(s) ** 2
            k += 1
    return r / max(k, 1)


def affine_normal(X, cams, shapes, facing, h):
    e1, e2 = tangent(facing)
    best = None
    grid = np.radians(np.arange(-70, 71, 10))
    for a in grid:
        for b in grid:
            n = n_from_angles(facing, e1, e2, (a, b))
            if n is None:
                continue
            r = residual(n, X, cams, shapes, h)
            if best is None or r < best[0]:
                best = (r, (a, b))

    def f(a):
        n = n_from_angles(facing, e1, e2, a)
        return 1e3 if n is None else residual(n, X, cams, shapes, h)

    x, fx = np.asarray(best[1], float), best[0]
    step = np.radians(5.0)
    while step > np.radians(0.1):
        moved = False
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            y = x + step * np.asarray(d)
            fy = f(y)
            if fy < fx:
                x, fx, moved = y, fy, True
                break
        if not moved:
            step /= 2
    return n_from_angles(facing, e1, e2, x), fx


def main():
    ds_name = sys.argv[1]
    match_px = 2.0
    prepared = prepare(ds_name, None, quiet=True)
    ds = DatasetContext(prepared)
    rec = ds.recon
    errs, errs_mv, nviews, t = [], [], [], 0.0
    for p in range(rec.point_count):
        if ds.point_w[p] == 0 or len(ds.point_images[p]) < 2:
            continue
        X = ds.point_xyz[p]
        images = [int(i) for i in ds.point_images[p]]
        q = images[0]
        ctx = ds.holdout(p, empty=True)
        cands = ctx.clusters_near(q, ds.point_keypoints[p][0], match_px)
        best = None
        for c in cands:
            ms = [
                m
                for m in c["members"]
                if m["status"] in ("reference", "kept") and m["image"] >= 0
            ]
            by_img = {}
            for m in ms:
                if m["image"] not in by_img or m["zncc"] > by_img[m["image"]]["zncc"]:
                    by_img[m["image"]] = m
            if len(by_img) >= 2 and (best is None or len(by_img) > len(best)):
                best = by_img
        if best is None:
            continue
        imgs = sorted(best)
        cams = [ds.cameras[i] for i in imgs]
        shapes = [best[i]["shape"] for i in imgs]
        cq = ds.cameras[q]
        facing = unit(sum(unit(ds.cameras[i].center - X) for i in imgs))
        gt = ds.point_normal[p]
        h = 0.01 * np.linalg.norm(cq.center - X)
        t0 = time.perf_counter()
        n, _ = affine_normal(X, cams, shapes, facing, h)
        t += time.perf_counter() - t0
        e = np.degrees(np.arccos(np.clip(abs(n @ gt), -1, 1)))
        errs.append(e)
        errs_mv.append(np.degrees(np.arccos(np.clip(abs(facing @ gt), -1, 1))))
        nviews.append(len(imgs))
    errs, errs_mv, nviews = map(np.asarray, (errs, errs_mv, nviews))
    print(
        f"{prepared.name}: {len(errs)} points with a matching cluster, {t / max(len(errs), 1) * 1000:.1f} ms each"
    )
    for name, a in (("affine", errs), ("mean_view(cluster imgs)", errs_mv)):
        print(
            f"{name:26s} p25 {np.percentile(a, 25):5.1f} med {np.median(a):5.1f} p75 {np.percentile(a, 75):5.1f} <10 {np.mean(a < 10):.2f} <20 {np.mean(a < 20):.2f}"
        )
    for k in (2, 3, 4):
        sel = nviews >= k if k == 4 else nviews == k
        if sel.any():
            print(
                f"  views {'>=' if k == 4 else '=='}{k}: n {sel.sum():4d} affine med {np.median(errs[sel]):5.1f} mean_view med {np.median(errs_mv[sel]):5.1f}"
            )


if __name__ == "__main__":
    main()
