#!/usr/bin/env python
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Render an equirectangular panorama from an SfM reconstruction.

Pipeline:
  1. Build a SphericalTileRig sized for the target equirect resolution.
  2. Build a float32 PerSphericalTileSourceStack (rotation-only) over the
     reconstruction's source images.
  3. Run refine_photometric_ransac to get the per-row primary cluster mask.
  4. Collapse the stack into a per-tile consensus atlas via the per-pixel
     median of the primary cluster.
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


def _resample_and_save(
    stack: PerSphericalTileSourceStack,
    rig: SphericalTileRig,
    mask: np.ndarray,
    eq_cam: CameraIntrinsics,
    k: int,
    out_path: Path,
    label: str,
) -> dict[str, float]:
    t0 = time.perf_counter()
    atlas = stack.primary_consensus_atlas(rig, mask)
    t_atlas = time.perf_counter() - t0

    t1 = time.perf_counter()
    identity = RotQuaternion(1.0, 0.0, 0.0, 0.0)
    pano = rig.resample_atlas(atlas, eq_cam, identity, k=k)
    t_resample = time.perf_counter() - t1

    t1 = time.perf_counter()
    pano = np.where(np.isnan(pano), 0.0, pano)
    t_nan = time.perf_counter() - t1

    t1 = time.perf_counter()
    pano_u8 = np.clip(pano, 0, 255).astype(np.uint8)
    t_clip = time.perf_counter() - t1

    t1 = time.perf_counter()
    bgr = cv2.cvtColor(pano_u8, cv2.COLOR_RGB2BGR)
    t_cvt = time.perf_counter() - t1

    t1 = time.perf_counter()
    cv2.imwrite(str(out_path), bgr)
    t_write = time.perf_counter() - t1

    total = time.perf_counter() - t0
    print(
        f"  [{label}] {int(mask.sum())} rows -> {out_path.name} {total:.2f}s "
        f"(atlas {t_atlas:.2f} resample {t_resample:.2f} "
        f"nan {t_nan:.2f} clip {t_clip:.2f} cvt {t_cvt:.2f} write {t_write:.2f})"
    )
    return {
        "atlas": t_atlas,
        "resample": t_resample,
        "nan": t_nan,
        "clip": t_clip,
        "cvt": t_cvt,
        "write": t_write,
        "total": total,
    }


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
        rig, sources, dtype=args.dtype
    )
    print(
        f"Stack: {stack.total_contrib_rows} rows across {stack.n_tiles} tiles, "
        f"dtype={stack.dtype} ({time.perf_counter() - t0:.2f}s)"
    )

    t0 = time.perf_counter()
    out = refine_photometric_ransac(stack)
    primary_count = int(out.primary_mask.sum())
    print(
        f"Photometric RANSAC: {primary_count}/{stack.total_contrib_rows} primary rows "
        f"({time.perf_counter() - t0:.2f}s)"
    )

    t0 = time.perf_counter()
    atlas = stack.primary_consensus_atlas(rig, out.primary_mask)
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
    print(
        f"Resampled to equirectangular {pano.shape} ({time.perf_counter() - t0:.2f}s)"
    )

    pano = np.where(np.isnan(pano), 0.0, pano)
    pano_u8 = np.clip(pano, 0, 255).astype(np.uint8)
    bgr_out = cv2.cvtColor(pano_u8, cv2.COLOR_RGB2BGR)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), bgr_out)
    print(f"Wrote {args.output}")

    if args.debug:
        debug_dir = args.output.parent / f"{args.output.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug mode: writing per-cluster + per-source panoramas to {debug_dir}")

        primary_mask = out.primary_mask
        secondary_mask = out.secondary_mask
        outliers_mask = ~(primary_mask | secondary_mask)
        src_id = stack.src_id()

        # primary.png is identical to the main output — copy rather than recompute.
        cv2.imwrite(str(debug_dir / "primary.png"), bgr_out)
        print(f"  [primary] {int(primary_mask.sum())} rows -> primary.png (copied)")

        _resample_and_save(
            stack,
            rig,
            secondary_mask,
            eq_cam,
            args.k,
            debug_dir / "secondary.png",
            "secondary",
        )
        _resample_and_save(
            stack,
            rig,
            outliers_mask,
            eq_cam,
            args.k,
            debug_dir / "outliers.png",
            "outliers",
        )

        n_sources = len(image_names)
        idx_width = max(3, len(str(n_sources - 1)))
        per_source_timings: list[dict[str, float]] = []
        for s in range(n_sources):
            mask = primary_mask & (src_id == np.uint32(s))
            if not mask.any():
                continue
            stem = Path(image_names[s]).stem
            fname = f"source_{s:0{idx_width}d}_{stem}.png"
            per_source_timings.append(
                _resample_and_save(
                    stack,
                    rig,
                    mask,
                    eq_cam,
                    args.k,
                    debug_dir / fname,
                    f"source {s}/{n_sources - 1}",
                )
            )

        if per_source_timings:
            n = len(per_source_timings)
            keys = ["atlas", "resample", "nan", "clip", "cvt", "write", "total"]
            sums = {k: sum(t[k] for t in per_source_timings) for k in keys}
            print(f"Per-source timing summary over {n} sources:")
            for k in keys:
                total_s = sums[k]
                mean_ms = (total_s / n) * 1000.0
                pct = (total_s / sums["total"]) * 100.0 if sums["total"] > 0 else 0.0
                print(
                    f"  {k:>9}: total {total_s:7.2f}s  mean {mean_ms:6.1f}ms  ({pct:5.1f}%)"
                )

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
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Write per-cluster (primary/secondary/outliers) and per-source "
            "panoramas to {output_stem}_debug/ next to the main output."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float32",
        help=(
            "Stack storage dtype. 'float16' halves stack memory (level 0 is "
            "the bulk) at the cost of ~3 decimal digits of precision. "
            "Default: 'float32'."
        ),
    )
    return render(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
