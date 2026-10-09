# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Empty-pass normals from pseudo-neighbours: nearby clusters, triangulated and
photometrically refined, then averaged.

For each finite GT point (at its GT position, queried from its first image):
the clusters with a kept/reference member within R px of the pixel are
triangulated from their members (midpoint of rays, least squares); those whose
depth is within tol of the point's depth are photometrically refined from the
mean viewing direction over their own member images, and their normals are
combined. Reported against the GT normal, with the single-patch photometric
estimate at the point itself for comparison.

    pixi run -e test python scripts/track_at_pixel/normal_sources/cluster_nb_diag.py DATASET
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


def triangulate(cams, pixels):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, px in zip(cams, pixels):
        d = unit(c.ray(px))
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c.center
    if np.linalg.cond(A) > 1e8:
        return None
    return np.linalg.solve(A, b)


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))


def main():
    from sfmtool import patches

    prepared = prepare(sys.argv[1], None, quiet=True)
    ds = DatasetContext(prepared)
    rec = ds.recon
    views = patches.CameraViews(
        list(rec.cameras),
        np.asarray(rec.quaternions_wxyz, float),
        np.asarray(rec.translations, float),
        np.asarray(rec.camera_indexes, np.uint32),
    )

    def photo(X, images, half, init, **kw):
        e1, e2 = tangent(init)
        cloud = patches.PatchCloud.from_halfvec_arrays(
            np.asarray([e1 * half], np.float32),
            np.asarray([e2 * half], np.float32),
            np.asarray([X], np.float64),
        )
        try:
            r = cloud.refine_normals(
                views,
                ds.pyramids,
                view_indices=[list(images)],
                use_stored_keypoints=False,
                min_views=2,
                **kw,
            )
        except Exception:  # noqa: BLE001
            return None, None
        return unit(r["normal"][0]), float(r["photoconsistency"][0])

    res = {}
    counts = []
    t_tot = 0.0
    pts = [
        p
        for p in range(rec.point_count)
        if ds.point_w[p] != 0 and len(ds.point_images[p]) >= 2
    ]
    for p in pts:
        X = ds.point_xyz[p]
        images = [int(i) for i in ds.point_images[p]]
        q = images[0]
        pixel = ds.point_keypoints[p][0]
        cq = ds.cameras[q]
        depth = cq.depth(X)
        gt = ds.point_normal[p]
        half = float(ds.point_half[p])
        mv = unit(sum(unit(ds.cameras[i].center - X) for i in images))
        ctx = ds.holdout(p, empty=True)
        n0, pc0 = photo(X, images, half, mv)
        if n0 is not None and (not np.all(np.isfinite(n0)) or not np.isfinite(pc0)):
            n0 = None
        res.setdefault("single_photo", []).append(
            ang(n0, gt) if n0 is not None else np.nan
        )
        res.setdefault("mean_view", []).append(ang(mv, gt))
        t0 = time.perf_counter()
        for R in (20.0, 40.0):
            nbs = []
            for c in ctx.clusters_near(q, pixel, R):
                ms = [
                    m
                    for m in c["members"]
                    if m["status"] in ("reference", "kept") and m["image"] >= 0
                ]
                by = {}
                for m in ms:
                    if m["image"] not in by or m["zncc"] > by[m["image"]]["zncc"]:
                        by[m["image"]] = m
                if len(by) < 2:
                    continue
                imgs = sorted(by)
                Y = triangulate(
                    [ds.cameras[i] for i in imgs], [by[i]["position"] for i in imgs]
                )
                if Y is None:
                    continue
                dY = cq.depth(Y)
                if dY <= 0 or abs(dY / depth - 1) > 0.15:
                    continue
                # half-size: the member's scale in the query image, back-projected
                mq = c["member"]
                s = np.sqrt(abs(np.linalg.det(mq["shape"])))
                hY = 2.0 * s * dY / cq.focal
                mvY = unit(sum(unit(ds.cameras[i].center - Y) for i in imgs))
                nY, pcY = photo(Y, imgs, hY, mvY)
                if nY is None or not np.all(np.isfinite(nY)) or not np.isfinite(pcY):
                    continue
                to_cam = unit(cq.center - Y)
                nY = nY if nY @ to_cam >= 0 else -nY
                nbs.append((nY, pcY, c["distance_px"], len(imgs)))
            if R == 40.0:
                counts.append(len(nbs))
            to_cam = unit(cq.center - X)
            f = lambda n: n if n @ to_cam >= 0 else -n  # noqa: E731
            if nbs:
                w = np.asarray([max(pc, 0.05) / (1 + d) for _, pc, d, _ in nbs])
                nm = unit(sum(wi * n for wi, (n, *_rest) in zip(w, nbs)))
                M = sum(wi * np.outer(n, n) for wi, (n, *_r) in zip(w, nbs))
                npc = f(np.linalg.eigh(M)[1][:, -1])
                withself = (
                    nm
                    if n0 is None
                    else unit(
                        sum(wi * n for wi, (n, *_r) in zip(w, nbs))
                        + max(pc0, 0.05) * f(n0)
                    )
                )
                # median-ish: the neighbour normal closest to all others
                if len(nbs) >= 3:
                    Ns = np.asarray([n for n, *_ in nbs])
                    cost = [np.sum(np.arccos(np.clip(Ns @ n, -1, 1))) for n in Ns]
                    nmed = Ns[int(np.argmin(cost))]
                else:
                    nmed = nm
            else:
                nm = npc = withself = nmed = n0
            for name, n in (
                (f"cnb_mean_r{R:g}", nm),
                (f"cnb_pca_r{R:g}", npc),
                (f"cnb_mean+self_r{R:g}", withself),
                (f"cnb_medoid_r{R:g}", nmed),
            ):
                res.setdefault(name, []).append(ang(n, gt) if n is not None else np.nan)
        t_tot += time.perf_counter() - t0
    print(
        f"{prepared.name}: {len(pts)} points, {t_tot / len(pts) * 1000:.0f} ms/pt, pseudo-neighbours@40: median {np.median(counts)}, zero for {np.mean(np.asarray(counts) == 0):.2f}"
    )
    for k, v in res.items():
        a = np.asarray(v, float)
        a = a[np.isfinite(a)]
        print(
            f"{k:24s} n {a.size:4d} med {np.median(a):5.1f} p25 {np.percentile(a, 25):5.1f} p75 {np.percentile(a, 75):5.1f} <10 {np.mean(a < 10):.2f} <20 {np.mean(a < 20):.2f}"
        )


if __name__ == "__main__":
    main()
