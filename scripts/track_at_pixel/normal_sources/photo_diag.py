# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tune photometric normal refinement for the empty pass (no neighbours).

At each finite GT point's position, with its GT views, refine_normals is run
from the mean viewing direction under several configurations; the angle to
the GT normal is reported.

    pixi run -e test python scripts/track_at_pixel/normal_sources/photo_diag.py DATASET
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context import DatasetContext  # noqa: E402
from dataset import prepare  # noqa: E402

CONFIGS = {
    "default": {},
    "range45": {"angular_range_deg": 45.0},
    "range60_steps9": {"angular_range_deg": 60.0, "init_steps": 9},
    "mean_obj": {"objective": "mean"},
    "aniso": {"sampler": "anisotropic"},
    "nocache": {"cache": "off"},
    "res32": {"resolution": 32},
    "res16": {"resolution": 16},
    "fronto0.05": {"fronto_prior_weight": 0.05},
    "fronto0.2": {"fronto_prior_weight": 0.2},
    "obliq2": {"obliquity_weight_power": 2.0},
    "uniform": {"window": "uniform"},
    "range45_fronto0.05": {"angular_range_deg": 45.0, "fronto_prior_weight": 0.05},
    "range45_obliq2": {"angular_range_deg": 45.0, "obliquity_weight_power": 2.0},
    "range35": {"angular_range_deg": 35.0},
    "range45_nocache_aniso": {
        "angular_range_deg": 45.0,
        "cache": "off",
        "sampler": "anisotropic",
    },
}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def tangent(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(np.cross(n, a))
    return e1, np.cross(n, e1)


def main():
    from sfmtool import patches

    prepared = prepare(sys.argv[1], None, quiet=True)
    scales = [1.0, 0.7, 1.5]
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
    errs, times = {}, {}
    for p in pts:
        X = ds.point_xyz[p]
        images = [int(i) for i in ds.point_images[p]]
        gt = ds.point_normal[p]
        mv = unit(sum(unit(ds.cameras[i].center - X) for i in images))
        half = float(ds.point_half[p])
        e1, e2 = tangent(mv)
        for s in scales:
            for name, kw in CONFIGS.items():
                if s != 1.0 and name not in ("default", "range45"):
                    continue
                key = f"{name}@x{s:g}"
                cloud = patches.PatchCloud.from_halfvec_arrays(
                    np.asarray([e1 * half * s], np.float32),
                    np.asarray([e2 * half * s], np.float32),
                    np.asarray([X], np.float64),
                )
                t0 = time.perf_counter()
                try:
                    res = cloud.refine_normals(
                        views,
                        ds.pyramids,
                        view_indices=[images],
                        use_stored_keypoints=False,
                        min_views=2,
                        **kw,
                    )
                    n = unit(res["normal"][0])
                    e = float(np.degrees(np.arccos(np.clip(abs(n @ gt), -1, 1))))
                except Exception as ex:  # noqa: BLE001
                    print(key, ex)
                    e = np.nan
                times[key] = times.get(key, 0) + time.perf_counter() - t0
                errs.setdefault(key, []).append(e)
        errs.setdefault("mean_view", []).append(
            float(np.degrees(np.arccos(np.clip(abs(mv @ gt), -1, 1))))
        )
    print(f"{prepared.name}: {len(pts)} points")
    for k, v in errs.items():
        a = np.asarray(v)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        print(
            f"{k:30s} med {np.median(a):5.1f} p25 {np.percentile(a, 25):5.1f} p75 {np.percentile(a, 75):5.1f} <10 {np.mean(a < 10):.2f} <20 {np.mean(a < 20):.2f} ms {1000 * times.get(k, 0) / len(v):6.1f}"
        )


if __name__ == "__main__":
    main()
