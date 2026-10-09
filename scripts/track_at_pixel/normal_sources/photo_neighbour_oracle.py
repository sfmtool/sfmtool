# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Would averaging photometric normals over true neighbours beat one estimate?

An oracle, not a candidate. Every finite GT point gets a photometric normal at
its GT position and views, with ``renormal``'s settings. Then each query
(every observation of every point, as the harness makes them) is given the
tight-to-wide neighbour chain of ``renormal``, but reading the neighbours'
*photometric* normals instead of their GT ones, with and without the point's
own estimate. If that is much closer to the GT normal than the point's own
photometric normal, better pseudo-neighbours in the empty pass would pay; if
not, the photometric error is shared across a surface and averaging cannot
remove it.

    pixi run -e test python scripts/track_at_pixel/normal_sources/photo_neighbour_oracle.py DATASET
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context import DatasetContext  # noqa: E402
from dataset import prepare  # noqa: E402

CHAIN = [
    (10.0, 5, 0.05),
    (20.0, 5, 0.10),
    (30.0, 8, 0.15),
    (40.0, 12, 0.30),
]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def tangent(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(np.cross(n, a))
    return e1, np.cross(n, e1)


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
    photo = {}
    for p in range(rec.point_count):
        if ds.point_w[p] == 0 or len(ds.point_images[p]) < 2:
            continue
        X = ds.point_xyz[p]
        imgs = [int(i) for i in ds.point_images[p]]
        mv = unit(sum(unit(ds.cameras[i].center - X) for i in imgs))
        half = 2.0 * float(ds.point_half[p])
        e1, e2 = tangent(mv)
        cloud = patches.PatchCloud.from_halfvec_arrays(
            np.asarray([e1 * half], np.float32),
            np.asarray([e2 * half], np.float32),
            np.asarray([X], np.float64),
        )
        try:
            r = cloud.refine_normals(
                views,
                ds.pyramids,
                view_indices=[imgs],
                use_stored_keypoints=False,
                min_views=2,
                angular_range_deg=45.0,
                fronto_prior_weight=0.05,
            )
        except Exception:  # noqa: BLE001
            continue
        n = unit(r["normal"][0])
        if np.all(np.isfinite(n)):
            photo[p] = n

    errs = {
        "own photometric": [],
        "neighbours' photometric": [],
        "neighbours' + own": [],
        "neighbours' GT (as in the full pass)": [],
    }
    for p, own in photo.items():
        gt = ds.point_normal[p]
        ctx = ds.holdout(p)
        for j, image in enumerate(ds.point_images[p]):
            image = int(image)
            pixel = ds.point_keypoints[p][j]
            cam = ds.cameras[image]
            to_cam = unit(cam.center - ds.point_xyz[p])
            depth = cam.depth(ds.point_xyz[p])

            def face(n):
                return n if n @ to_cam >= 0 else -n

            allnear = [
                o for o in ctx.observations_near(image, pixel, 40.0) if o["depth"]
            ]
            chosen = []
            for radius, k, tol in CHAIN:
                chosen = [
                    o
                    for o in allnear
                    if o["distance_px"] <= radius
                    and abs(o["depth"] / depth - 1) < tol
                    and o["point"] in photo
                ][:k]
                if chosen:
                    break
            errs["own photometric"].append(ang(own, gt))
            if not chosen:
                for key in list(errs)[1:]:
                    errs[key].append(np.nan)
                continue
            w = np.asarray([1 / (1 + o["distance_px"]) ** 2 for o in chosen])

            def principal(ns, ws):
                M = sum(wi * np.outer(n, n) for n, wi in zip(ns, ws))
                return np.linalg.eigh(M)[1][:, -1]

            nb_ph = [face(photo[o["point"]]) for o in chosen]
            nb_gt = [face(o["normal"]) for o in chosen]
            errs["neighbours' photometric"].append(ang(principal(nb_ph, w), gt))
            errs["neighbours' + own"].append(
                ang(principal(nb_ph + [face(own)], list(w) + [w.max()]), gt)
            )
            errs["neighbours' GT (as in the full pass)"].append(
                ang(principal(nb_gt, w), gt)
            )

    print(f"{prepared.name}: {len(photo)} points with a photometric normal")
    own = np.asarray(errs["own photometric"])
    covered = np.isfinite(np.asarray(errs["neighbours' photometric"]))
    print(f"queries {own.size}, with a neighbour {covered.sum()}")
    for key, v in errs.items():
        a = np.asarray(v, float)[covered]
        credit = np.mean(np.clip(1 - a / 30, 0, 1))
        print(
            f"{key:38s} over queries with a neighbour: median {np.median(a):5.1f} credit {credit:.3f}"
        )


if __name__ == "__main__":
    main()
