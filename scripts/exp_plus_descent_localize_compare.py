#!/usr/bin/env python
"""Compare exhaustive vs plus_descent keypoint-localization on two
already-written embed-patches outputs.

Reads two ``embedded_patches`` ``.sfmr`` files (same input, different
``--localize-search-strategy``), joins observations by ``(world position,
image index)``, and reports the per-observation keypoint shift.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline_sfmr", help="exhaustive embed-patches output")
    ap.add_argument("variant_sfmr", help="plus_descent embed-patches output")
    args = ap.parse_args()

    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    a = SfmrReconstruction.load(args.baseline_sfmr)
    b = SfmrReconstruction.load(args.variant_sfmr)

    def per_obs(recon):
        """Yield (world_x, world_y, world_z, image_index, kp_x, kp_y) per
        observation."""
        positions = np.asarray(recon.positions)
        track_pids = np.asarray(recon.track_point_indexes, dtype=np.uint32)
        track_images = np.asarray(recon.track_image_indexes, dtype=np.uint32)
        keypoints = np.asarray(recon.keypoints_xy, dtype=np.float64)
        out = {}
        for i in range(len(track_pids)):
            pid = int(track_pids[i])
            img = int(track_images[i])
            pos = positions[pid]
            key = (round(float(pos[0]), 4),
                   round(float(pos[1]), 4),
                   round(float(pos[2]), 4),
                   img)
            out[key] = (float(keypoints[i, 0]), float(keypoints[i, 1]))
        return out

    map_a = per_obs(a)
    map_b = per_obs(b)
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    print(f"baseline obs={len(map_a)}, variant obs={len(map_b)}, "
          f"matched={len(common)}", file=sys.stderr)

    if not common:
        sys.exit("no matched observations — different recons?")

    shifts = np.array([
        [map_b[k][0] - map_a[k][0], map_b[k][1] - map_a[k][1]]
        for k in common
    ])
    dist = np.linalg.norm(shifts, axis=1)
    print(f"per-observation keypoint shift (source-image px):")
    print(f"  n            = {len(dist)}")
    print(f"  mean         = {dist.mean():.4f}")
    print(f"  median       = {np.median(dist):.4f}")
    print(f"  p90          = {np.percentile(dist, 90):.4f}")
    print(f"  p99          = {np.percentile(dist, 99):.4f}")
    print(f"  max          = {dist.max():.4f}")
    print(f"  frac <= 0.05 = {(dist <= 0.05).mean():.4f}")
    print(f"  frac <= 0.5  = {(dist <= 0.5).mean():.4f}")
    print(f"  frac <= 1.0  = {(dist <= 1.0).mean():.4f}")


if __name__ == "__main__":
    main()
