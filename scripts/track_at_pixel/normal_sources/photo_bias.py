# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Is the photometric normal's error against the ground truth systematic?

At each finite GT point's position and views (each point weighted by its
track length, as the harness's queries weight it), the photometric normal is
found with ``renormal``'s settings, and compared with the GT normal and with
simple references: the mean viewing direction, and the world axes. Blends of
the photometric normal toward the mean viewing direction are scored, as is
snapping it to the nearest of the reconstruction's dominant normal directions.

    pixi run -e test python scripts/track_at_pixel/normal_sources/photo_bias.py DATASET
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context import DatasetContext  # noqa: E402
from dataset import prepare  # noqa: E402


FRONTO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05


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
    from sfmtool._sfmtool import patches

    prepared = prepare(sys.argv[1], None, quiet=True)
    ds = DatasetContext(prepared)
    rec = ds.recon
    views = patches.CameraViews(
        list(rec.cameras),
        np.asarray(rec.quaternions_wxyz, float),
        np.asarray(rec.translations, float),
        np.asarray(rec.camera_indexes, np.uint32),
    )
    rows = []
    for p in range(rec.point_count):
        if ds.point_w[p] == 0 or len(ds.point_images[p]) < 2:
            continue
        X = ds.point_xyz[p]
        imgs = [int(i) for i in ds.point_images[p]]
        gt = ds.point_normal[p]
        mv = unit(sum(unit(ds.cameras[i].center - X) for i in imgs))
        gt = gt if gt @ mv >= 0 else -gt
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
                fronto_prior_weight=FRONTO,
            )
        except Exception:  # noqa: BLE001
            continue
        ph = unit(r["normal"][0])
        if not np.all(np.isfinite(ph)):
            continue
        ph = ph if ph @ mv >= 0 else -ph
        rows.append((len(imgs), gt, mv, ph, float(r["photoconsistency"][0])))
    w = np.asarray([r[0] for r in rows], float)
    GT = np.asarray([r[1] for r in rows])
    MV = np.asarray([r[2] for r in rows])
    PH = np.asarray([r[3] for r in rows])

    def report(name, N):
        e = np.asarray([ang(a, b) for a, b in zip(N, GT)])
        order = np.argsort(e)
        cw = np.cumsum(w[order]) / w.sum()
        med = e[order][np.searchsorted(cw, 0.5)]
        credit = float((w * np.clip(1 - e / 30, 0, 1)).sum() / w.sum())
        print(f"{name:34s} query-weighted median {med:5.1f}  credit {credit:.3f}")

    print(f"{prepared.name}: {len(rows)} points, {int(w.sum())} queries")
    report("mean view", MV)
    report("photometric", PH)
    for a in (0.25, 0.5, 0.75):
        report(
            f"photo blended {a:.2f} toward mean view",
            [unit((1 - a) * p + a * m) for p, m in zip(PH, MV)],
        )
    for b in (0.25, 0.5, 0.75, 1.0):
        report(
            f"photo tilted {b:.2f} further from mean view",
            [unit(p + b * (p - m)) for p, m in zip(PH, MV)],
        )
    # Snapping to the vertical ("up") or into the horizontal plane. The up
    # direction is estimated from the cameras: the mean of each camera's
    # image-up axis in the world (-y of the camera frame, as rows grow down).
    cam_up = unit(sum(-c.R.T @ np.array([0.0, 1.0, 0.0]) for c in ds.cameras))
    print(
        f"camera-derived up vs world z: {ang(cam_up, np.array([0.0, 0.0, 1.0])):.1f} deg"
    )
    for up_name, up in (("world z", np.array([0.0, 0.0, 1.0])), ("camera up", cam_up)):
        for t in (10, 15, 20, 25):

            def snap(n, t=t, up=up):
                a = ang(n, up)
                if a < t:
                    return up if n @ up >= 0 else -up
                if a > 90 - t:
                    return unit(n - (n @ up) * up)
                return n

            report(f"snap {up_name} within {t}", [snap(p) for p in PH])
        near = np.asarray([ang(g, up) for g in GT])
        print(
            f"  GT within 10 deg of {up_name}: {np.mean(near < 10):.2f}; within 10 deg of horizontal: {np.mean(near > 80):.2f}"
        )
    # How the GT sits relative to the mean view and the photometric normal.
    tilt_gt = [ang(g, m) for g, m in zip(GT, MV)]
    tilt_ph = [ang(p, m) for p, m in zip(PH, MV)]
    print(
        f"tilt from mean view: GT median {np.median(tilt_gt):.1f}, photometric {np.median(tilt_ph):.1f}"
    )
    # World-axis structure of the GT normals.
    for axis, v in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
        a = np.asarray([ang(g, np.asarray(v, float)) for g in GT])
        print(
            f"GT normal angle to world {axis}: p10 {np.percentile(a, 10):5.1f} median {np.median(a):5.1f} p90 {np.percentile(a, 90):5.1f}"
        )
    # Dominant directions: principal axes of the GT normals' scatter.
    M = (GT[:, :, None] * GT[:, None, :]).sum(0)
    vals, vecs = np.linalg.eigh(M)
    print("GT normal scatter eigenvalues", np.round(vals / vals.sum(), 3))


if __name__ == "__main__":
    main()
