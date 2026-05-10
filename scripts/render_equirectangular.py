#!/usr/bin/env python
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Render an equirectangular panorama from an SfM reconstruction.

Pipeline:
  1. Build a SphericalTileRig sized for the target equirect resolution.
  2. Build a float32 PerSphericalTileSourceStack (rotation-only) over the
     reconstruction's source images.
  3. Run refine_photometric_ransac to get per-source log_gain plus the
     per-row primary cluster mask.
  4. Collapse the stack into a per-tile consensus atlas via the per-pixel
     median of the gain-corrected primary cluster.
  5. Resample the atlas through an Equirectangular CameraIntrinsics.
  6. Save the result as a PNG.

Usage:
    pixi run python scripts/render_equirectangular.py RECON.sfmr -o panorama.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from sfmtool._sfmtool import (
    CameraIntrinsics,
    PerSphericalTileSourceStack,
    RotQuaternion,
    SfmrReconstruction,
    SphericalTileRig,
    refine_photometric_ransac,
)


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _resolve_image_dir(recon_path: Path, first_image: str) -> Path | None:
    # Try the .sfmr's directory and one level up (the typical workspace layout
    # is workspace_root/sfmr/file.sfmr with workspace_root/images/ alongside).
    bases = [recon_path.parent, recon_path.parent.parent]
    sub_names = ["test_17_image", "imgs", "images", ""]
    for base in bases:
        for sub in sub_names:
            cand = base / sub if sub else base
            if (cand / first_image).exists():
                return cand
    return None


def render(args: argparse.Namespace) -> int:
    out_w = args.width
    out_h = args.width // 2

    print(f"Loading reconstruction {args.reconstruction}")
    recon = SfmrReconstruction.load(args.reconstruction)
    cameras = recon.cameras
    camera_indexes = recon.camera_indexes
    quats = recon.quaternions_wxyz
    image_names = recon.image_names
    print(f"  {len(image_names)} images, {len(cameras)} cameras")

    image_dir = args.image_dir
    if image_dir is None:
        image_dir = _resolve_image_dir(args.reconstruction, image_names[0])
        if image_dir is None:
            print(
                f"Could not auto-locate image directory for {image_names[0]}. "
                f"Pass --image-dir explicitly.",
                file=sys.stderr,
            )
            return 1

    arc_per_pixel = 2 * np.pi / out_w
    print(
        f"Building rig: n={args.n_tiles}, arc_per_pixel={arc_per_pixel:.6f} "
        f"(= 2pi/{out_w})"
    )
    rig = SphericalTileRig(n=args.n_tiles, arc_per_pixel=arc_per_pixel, seed=args.seed)
    raw_ps = rig.patch_size
    rig.set_patch_size(_next_pow2(raw_ps))
    atlas_w, atlas_h = rig.atlas_size
    print(
        f"  patch_size: {raw_ps} -> {rig.patch_size} (next pow2), "
        f"atlas: {atlas_w}x{atlas_h}"
    )

    print(f"Loading {len(image_names)} images from {image_dir}")
    sources = []
    for i, name in enumerate(image_names):
        cam = cameras[camera_indexes[i]]
        q = RotQuaternion(quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3])
        bgr = cv2.imread(str(image_dir / name), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Could not read {image_dir / name}", file=sys.stderr)
            return 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sources.append((cam, q, rgb))

    t0 = time.perf_counter()
    stack = PerSphericalTileSourceStack.build_rotation_only(
        rig, sources, dtype="float32"
    )
    print(
        f"Stack: {stack.total_contrib_rows} rows across {stack.n_tiles} tiles "
        f"({time.perf_counter() - t0:.2f}s)"
    )

    t0 = time.perf_counter()
    out = refine_photometric_ransac(stack)
    primary_count = int(out.primary_mask.sum())
    print(
        f"Photometric RANSAC: {primary_count}/{stack.total_contrib_rows} primary rows, "
        f"log_gain std {float(out.log_gain.std()):.4f} "
        f"({time.perf_counter() - t0:.2f}s)"
    )

    t0 = time.perf_counter()
    atlas = stack.primary_consensus_atlas(rig, out.primary_mask, out.log_gain)
    nan_pixels = np.isnan(atlas).any(axis=-1) if atlas.ndim == 3 else np.isnan(atlas)
    print(
        f"Consensus atlas: {atlas.shape}, "
        f"{float(nan_pixels.mean()) * 100:.1f}% NaN pixels "
        f"({time.perf_counter() - t0:.2f}s)"
    )

    # resample_atlas is NaN-aware: it skips NaN samples in the k-NN blend
    # and renormalises remaining weights, so we keep the consensus's NaN
    # holes intact through the resampling step. NaN flips to black only at
    # the final u8 cast for PNG output.
    atlas = atlas.astype(np.float32)

    eq_cam = CameraIntrinsics(
        model="EQUIRECTANGULAR",
        width=out_w,
        height=out_h,
        params={
            "focal_length_x": out_w / (2 * np.pi),
            "focal_length_y": out_h / np.pi,
            "principal_point_x": out_w / 2.0,
            "principal_point_y": out_h / 2.0,
        },
    )

    t0 = time.perf_counter()
    identity = RotQuaternion(1.0, 0.0, 0.0, 0.0)
    pano = rig.resample_atlas(atlas, eq_cam, identity, k=args.k)
    print(f"Resampled to equirectangular {pano.shape} ({time.perf_counter() - t0:.2f}s)")

    pano = np.where(np.isnan(pano), 0.0, pano)
    pano_u8 = np.clip(pano, 0, 255).astype(np.uint8)
    bgr_out = cv2.cvtColor(pano_u8, cv2.COLOR_RGB2BGR)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), bgr_out)
    print(f"Wrote {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reconstruction", type=Path, help=".sfmr file path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("panorama.png"),
        help="Output PNG path (default: panorama.png)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing source images (default: auto-detected near the .sfmr)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2160,
        help="Equirectangular output width in pixels; height = width / 2 (default: 2160)",
    )
    parser.add_argument(
        "--n-tiles",
        type=int,
        default=320,
        help="Number of spherical tiles in the rig (default: 320)",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Rig relaxer seed (default: 1234)"
    )
    parser.add_argument(
        "-k",
        type=int,
        default=4,
        help="k-NN tile blending in resample_atlas (default: 4)",
    )
    return render(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
