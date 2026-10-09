# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""How well does each normal source recover the ground-truth normal?

Isolated from the rest of the pipeline: at each GT point's own position and
sightings, estimate a normal every way we know and measure its angle to the
GT normal (both turned to face the query camera, so the error is 0..90 deg).

    pixi run -e test python scripts/track_at_pixel/normal_sources/normal_diag.py DATASET [--limit N]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context import DatasetContext  # noqa: E402
from dataset import prepare  # noqa: E402


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def face(n, to_cam):
    if n is None:
        return None
    return n if n @ to_cam >= 0 else -n


def angle(a, b):
    if a is None or b is None:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))


def tangent(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(np.cross(n, a))
    return e1, np.cross(n, e1)


def principal(ns, w):
    M = sum(wi * np.outer(n, n) for n, wi in zip(ns, w))
    vals, vecs = np.linalg.eigh(M)
    return vecs[:, -1]


def plane_fit(xyz, w=None):
    xyz = np.asarray(xyz)
    if len(xyz) < 3:
        return None
    w = np.ones(len(xyz)) if w is None else np.asarray(w)
    c = (w[:, None] * xyz).sum(0) / w.sum()
    d = xyz - c
    M = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)
    vals, vecs = np.linalg.eigh(M)
    if vals[1] < 1e-12 * max(vals[2], 1e-300):
        return None
    return vecs[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--photo", default="1,2,3")
    args = ap.parse_args()
    from sfmtool import patches

    prepared = prepare(args.dataset, None, quiet=True)
    ds = DatasetContext(prepared)
    rec = ds.recon
    views = patches.CameraViews(
        list(rec.cameras),
        np.asarray(rec.quaternions_wxyz, float),
        np.asarray(rec.translations, float),
        np.asarray(rec.camera_indexes, np.uint32),
    )
    pts = [
        p
        for p in range(rec.point_count)
        if ds.point_w[p] != 0 and len(ds.point_images[p]) >= 2
    ]
    if args.limit:
        pts = pts[: args.limit]
    scales = [float(s) for s in args.photo.split(",") if s]
    errs: dict[str, list] = {}
    times: dict[str, float] = {}

    def rec_err(name, n, gt, t0=None):
        errs.setdefault(name, []).append(angle(n, gt))
        if t0 is not None:
            times[name] = times.get(name, 0.0) + time.perf_counter() - t0

    def photo(name, init, xyz, half, images, gt, to_cam, **kw):
        t0 = time.perf_counter()
        if init is None:
            rec_err(name, None, gt)
            return None
        e1, e2 = tangent(init)
        cloud = patches.PatchCloud.from_halfvec_arrays(
            np.asarray([e1 * half], np.float32),
            np.asarray([e2 * half], np.float32),
            np.asarray([xyz], np.float64),
        )
        try:
            res = cloud.refine_normals(
                views,
                ds.pyramids,
                view_indices=[list(map(int, images))],
                use_stored_keypoints=False,
                min_views=2,
                **kw,
            )
            n = face(unit(res["normal"][0]), to_cam)
        except Exception as ex:  # noqa: BLE001
            print("photo failed", ex)
            n = None
        rec_err(name, n, gt, t0)
        return n

    for k, p in enumerate(pts):
        ctx = ds.holdout(p)
        images = ds.point_images[p]
        image = int(images[0])
        pixel = ds.point_keypoints[p][0]
        cam = ds.cameras[image]
        xyz = ds.point_xyz[p]
        to_cam = unit(cam.center - xyz)
        gt = face(ds.point_normal[p], to_cam)
        half = float(ds.point_half[p])
        depth = cam.depth(xyz)

        rec_err("facing_query", to_cam, gt)
        mv = face(
            unit(sum(unit(ds.cameras[int(i)].center - xyz) for i in images)), to_cam
        )
        rec_err("mean_view", mv, gt)

        # 2D neighbours (the shipped prior).
        for r, kk in ((40, 10), (20, 5), (60, 20)):
            t0 = time.perf_counter()
            near = [
                o
                for o in ctx.observations_near(image, pixel, r)
                if o["depth"] and abs(o["depth"] / depth - 1) < 0.15
            ][:kk]
            n = None
            if near:
                w = np.asarray([1 / (1 + o["distance_px"]) for o in near])
                n = face(
                    unit(
                        (w[:, None] * [face(o["normal"], to_cam) for o in near]).sum(0)
                    ),
                    to_cam,
                )
            rec_err(f"nb2d_mean_r{r}k{kk}", n, gt, t0)
            if r == 40:
                nb2d = n
                # the shipped one averages raw (unflipped) normals
                if near:
                    n_raw = face(
                        unit((w[:, None] * [o["normal"] for o in near]).sum(0)), to_cam
                    )
                else:
                    n_raw = None
                rec_err("nb2d_shipped", n_raw, gt)
                n = None
                if near:
                    n = face(
                        principal([face(o["normal"], to_cam) for o in near], w), to_cam
                    )
                rec_err("nb2d_principal_r40", n, gt)
                # plane through neighbours' 3D positions
                P = [ds.point_xyz[o["point"]] for o in near]
                rec_err(
                    "nb2d_plane_r40",
                    face(plane_fit(P) if len(P) >= 3 else None, to_cam),
                    gt,
                )

        # 3D neighbours
        for kk in (6, 12, 24):
            t0 = time.perf_counter()
            near3 = ctx.points_near(xyz, k=kk)
            n = None
            if near3:
                w = np.asarray(
                    [1 / (1 + o["distance"] / max(half, 1e-9)) for o in near3]
                )
                n = face(
                    unit(
                        (w[:, None] * [face(o["normal"], to_cam) for o in near3]).sum(0)
                    ),
                    to_cam,
                )
            rec_err(f"nb3d_mean_k{kk}", n, gt, t0)
            P = [o["position"] for o in near3]
            rec_err(
                f"nb3d_plane_k{kk}",
                face(plane_fit(P) if len(P) >= 3 else None, to_cam),
                gt,
            )

        # Photometric, from several inits and sizes.
        for s in scales:
            photo(f"photo_mv_x{s:g}", mv, xyz, half * s, images, gt, to_cam)
            photo(f"photo_nb2d_x{s:g}", nb2d, xyz, half * s, images, gt, to_cam)
            photo(
                f"photo_nb2d_x{s:g}_narrow",
                nb2d,
                xyz,
                half * s,
                images,
                gt,
                to_cam,
                angular_range_deg=10.0,
            )
        photo("photo_gt_x1", gt, xyz, half, images, gt, to_cam)
        photo("photo_gt_x2", gt, xyz, half * 2, images, gt, to_cam)
        if k % 50 == 0:
            print(f"{k}/{len(pts)}", flush=True)

    print(f"\n{prepared.name}: {len(pts)} finite points")
    print(
        f"{'estimator':32s} {'n':>5s} {'p25':>6s} {'med':>6s} {'p75':>6s} {'<10':>6s} {'<20':>6s} {'ms':>6s}"
    )
    for name, v in errs.items():
        a = np.asarray(v, float)
        ok = a[np.isfinite(a)]
        if ok.size == 0:
            continue
        p25, p50, p75 = np.percentile(ok, [25, 50, 75])
        ms = 1000 * times.get(name, 0.0) / len(a)
        print(
            f"{name:32s} {ok.size:5d} {p25:6.1f} {p50:6.1f} {p75:6.1f} "
            f"{np.mean(ok < 10):6.2f} {np.mean(ok < 20):6.2f} {ms:6.1f}"
        )


if __name__ == "__main__":
    main()
