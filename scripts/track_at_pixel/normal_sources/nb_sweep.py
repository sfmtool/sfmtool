# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep the neighbour-normal estimator's parameters against the GT normal.

Each GT point is queried from every image it is seen in (as the harness does),
with the point itself held out; the estimator uses the observations near the
pixel whose depth is within ``tol`` of the GT depth.

    pixi run -e test python scripts/track_at_pixel/normal_sources/nb_sweep.py DATASET
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context import DatasetContext  # noqa: E402
from dataset import prepare  # noqa: E402


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def main():
    prepared = prepare(sys.argv[1], None, quiet=True)
    ds = DatasetContext(prepared)
    radii = [10, 15, 20, 30, 40]
    ks = [3, 5, 8, 12]
    tols = [0.05, 0.1, 0.15, 0.3, 10.0]
    weights = ["inv", "inv2", "uniform"]
    modes = ["mean", "principal"]
    errs = {c: [] for c in itertools.product(radii, ks, tols, weights, modes)}
    n_q = 0
    for p in range(ds.recon.point_count):
        if ds.point_w[p] == 0 or len(ds.point_images[p]) < 2:
            continue
        ctx = ds.holdout(p)
        gt = ds.point_normal[p]
        for j, image in enumerate(ds.point_images[p]):
            image = int(image)
            pixel = ds.point_keypoints[p][j]
            cam = ds.cameras[image]
            to_cam = unit(cam.center - ds.point_xyz[p])
            depth = cam.depth(ds.point_xyz[p])
            allnear = ctx.observations_near(image, pixel, max(radii))
            n_q += 1
            for r, k, tol, w, mode in errs:
                near = [
                    o
                    for o in allnear
                    if o["distance_px"] <= r
                    and o["depth"]
                    and abs(o["depth"] / depth - 1) < tol
                ][:k]
                if not near:
                    errs[(r, k, tol, w, mode)].append(np.nan)
                    continue
                d = np.asarray([o["distance_px"] for o in near])
                wt = {
                    "inv": 1 / (1 + d),
                    "inv2": 1 / (1 + d) ** 2,
                    "uniform": np.ones_like(d),
                }[w]
                ns = np.asarray(
                    [
                        o["normal"] if o["normal"] @ to_cam >= 0 else -o["normal"]
                        for o in near
                    ]
                )
                if mode == "mean":
                    n = unit((wt[:, None] * ns).sum(0))
                else:
                    M = (wt[:, None, None] * ns[:, :, None] * ns[:, None, :]).sum(0)
                    n = np.linalg.eigh(M)[1][:, -1]
                errs[(r, k, tol, w, mode)].append(
                    float(np.degrees(np.arccos(np.clip(abs(n @ gt), -1, 1))))
                    if n is not None
                    else np.nan
                )
    chains = {
        "shipped r40k10": [(40, 12, 0.15, "inv", "mean")],
        "renormal r20>r40": [
            (20, 5, 0.15, "inv", "mean"),
            (40, 5, 0.15, "inv", "mean"),
        ],
        "r10t05>r20>r40": [
            (10, 5, 0.05, "inv2", "principal"),
            (20, 5, 0.15, "inv", "principal"),
            (40, 5, 0.15, "inv", "principal"),
        ],
        "r10t05>r15t1>r20>r40>r40t3": [
            (10, 5, 0.05, "inv2", "principal"),
            (15, 5, 0.1, "inv2", "principal"),
            (20, 5, 0.15, "inv", "principal"),
            (40, 12, 0.15, "inv", "principal"),
            (40, 12, 0.3, "inv", "principal"),
        ],
        "r10t05>r20t1>r30t15>r40t3": [
            (10, 5, 0.05, "inv2", "principal"),
            (20, 5, 0.1, "inv2", "principal"),
            (30, 8, 0.15, "inv2", "principal"),
            (40, 12, 0.3, "inv2", "principal"),
        ],
    }
    print("chains (credit over ALL queries, missing = 0; coverage; median)")
    for name, chain in chains.items():
        out = np.full(n_q, np.nan)
        for c in chain:
            a = np.asarray(errs[c], float)
            fill = np.isnan(out) & np.isfinite(a)
            out[fill] = a[fill]
        credit = np.mean(np.where(np.isfinite(out), np.clip(1 - out / 30, 0, 1), 0))
        print(
            f"  {name:32s} {credit:.3f} {np.mean(np.isfinite(out)):.2f} {np.nanmedian(out):5.1f}"
        )
    # Score: over all queries, credit max(0, 1 - err/30), missing = 0 (as a
    # stand-in for "no neighbour estimate": the caller then falls back).
    rows = []
    for c, v in errs.items():
        a = np.asarray(v, float)
        cover = np.mean(np.isfinite(a))
        credit = np.nanmean(np.where(np.isfinite(a), np.clip(1 - a / 30, 0, 1), np.nan))
        rows.append((credit, cover, np.nanmedian(a), c))
    rows.sort(key=lambda t: -t[0])
    print(f"{prepared.name}: {n_q} queries")
    print(
        "top by mean credit over covered queries (credit, coverage, median err, config)"
    )
    for credit, cover, med, c in rows[:15]:
        print(f"{credit:.3f} {cover:.2f} {med:5.1f} {c}")
    for c in [
        (40, 12, 0.15, "inv", "mean"),
        (20, 5, 0.15, "inv", "mean"),
        (20, 5, 0.15, "inv", "principal"),
    ]:
        credit, cover, med, _ = next(t for t in rows if t[3] == c)
        print("ref", c, f"{credit:.3f} {cover:.2f} {med:5.1f}")


if __name__ == "__main__":
    main()
