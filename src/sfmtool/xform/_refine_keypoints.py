# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Subpixel keypoint refinement as an ``sfm xform`` operation.

Like ``RefineNormalsTransform``, ``RefineKeypointsTransform`` is a *modifier*:
it removes no points, drops no views, and moves no 3D geometry — it only
rewrites the per-observation ``keypoints_xy`` values to the sub-pixel locations
that best match the point's reference render (forward-additive ECC
Gauss–Newton, one pass; never worse than the seed). The reference observation
is the one the point stores, or the reference-view rule's pick where it stores
none, and its keypoint is not moved; a point that stored none records the pick.
With ``bitmaps`` each refined point records the reference its views were
aligned to, so a stored one that does not render at its keypoint gives way to
``-1`` and the fused mean; without, a point keeps the reference it stores.
The track structure (``track_image_indexes`` / ``track_point_indexes`` /
``observation_counts``) is untouched. Because the refinement is photometric it
reads the workspace's source images (``workspace_dir / image_name``), the same
way ``--refine-normals`` does; a missing image is a hard error.

The operation requires an ``embedded_patches`` reconstruction (enforced by
``apply_transforms``): the refiner seeds each view from the recon's stored
inline keypoint and refines over the stored per-point patch frame
(``recon.patches``). Keypoint *localization* (basin search that can drop views)
is a heavier, structural operation and is intentionally not this op.

See ``specs/cli/reconstruction/xform/refine-keypoints-command.md`` and
``specs/core/patch/keypoint-subpixel-refinement.md``.
"""

import numpy as np

from .._sfmtool.reconstruction import SfmrReconstruction
from ._images import load_workspace_images
from .._patch_compaction import (
    reference_images_by_point,
    reference_observations_from_images,
    render_from_references,
    stored_reference_images,
)
from ._patch_params import validate_patch_params


class RefineKeypointsTransform:
    """Refine per-observation 2D keypoints to sub-pixel, in place.

    All knobs default to the ``PatchCloud.refine_keypoints`` binding defaults,
    so the two layers cannot drift. The point count, positions, poses, cameras,
    and track structure are unchanged; only ``keypoints_xy`` is rewritten
    (points at infinity are refined like finite ones, not skipped; a
    guard-failed view keeps its seed).

    Requires an ``embedded_patches`` reconstruction (enforced by
    ``apply_transforms``): the refiner seeds from the stored per-observation
    keypoints and reuses the stored per-point patch frame, both of which only
    that source carries. Convert first with ``--to-embedded-patches``.
    """

    # Precondition checked per-step by `apply_transforms` (see `_apply.py`).
    required_feature_source = "embedded_patches"

    def __init__(
        self,
        *,
        # whether to render and persist the per-point RGBA patch bitmaps.
        # Defaults on so a refined reconstruction is self-contained (carries its
        # per-point patch textures and can display them without re-rendering);
        # pass ``bitmaps=false`` on intermediate stages to skip the render.
        bitmaps: bool = True,
        # forwarded to PatchCloud.refine_keypoints
        resolution: int = 24,
        window: str = "gaussian_disk",
        window_sigma: float = 0.6,
        sampler: str = "per_view",
        robust_iters: int = 3,
        max_gn_steps: int = 10,
        convergence_px: float = 0.01,
        max_offset_px: float = 2.0,
    ):
        if resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {resolution}")
        validate_patch_params(window=window, window_sigma=window_sigma, sampler=sampler)
        if robust_iters < 1:
            raise ValueError(f"robust_iters must be >= 1, got {robust_iters}")
        if max_gn_steps < 1:
            raise ValueError(f"max_gn_steps must be >= 1, got {max_gn_steps}")
        if convergence_px <= 0:
            raise ValueError(f"convergence_px must be positive, got {convergence_px}")
        if max_offset_px <= 0:
            raise ValueError(f"max_offset_px must be positive, got {max_offset_px}")
        if not isinstance(bitmaps, bool):
            raise ValueError(f"bitmaps must be a bool, got {bitmaps!r}")

        self.bitmaps = bitmaps
        self.resolution = resolution
        self.window = window
        self.window_sigma = window_sigma
        self.sampler = sampler
        self.robust_iters = robust_iters
        self.max_gn_steps = max_gn_steps
        self.convergence_px = convergence_px
        self.max_offset_px = max_offset_px

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        images = load_workspace_images(recon)

        # The reconstruction is embedded_patches (enforced by apply_transforms), so
        # the patch frame is already stored — read it back as the cloud rather than
        # rebuilding it. With view_sets=None and starting_keypoints=None the binding
        # seeds every view from the recon's stored inline keypoint and refines each
        # point's full track — "refine the keypoints already here, in place" —
        # against its stored reference observation where it has one.
        cloud = recon.patches
        if cloud is None:
            raise ValueError(
                "reconstruction has no patch frames to refine keypoints over; "
                "expected embedded_patches (run `sfm xform --to-embedded-patches` "
                "first)"
            )
        print(
            f"  Refining keypoints for {recon.point_count} points "
            f"across {recon.image_count} images..."
        )

        result = cloud.refine_keypoints(
            recon,
            images,
            resolution=self.resolution,
            window=self.window,
            window_sigma=self.window_sigma,
            sampler=self.sampler,
            robust_iters=self.robust_iters,
            max_gn_steps=self.max_gn_steps,
            convergence_px=self.convergence_px,
            max_offset_px=self.max_offset_px,
            render_bitmaps=self.bitmaps,
            reference_images=reference_images_by_point(recon),
        )

        # The refiner changes no view membership, so write back by copying the
        # stored keypoints and overwriting only the refined observations —
        # never touching track_image_indexes / track_point_indexes /
        # observation_counts. Each observation is keyed by (point, image); with
        # `stride = image_count` a single int encodes that pair uniquely, so the
        # scatter is a vectorized searchsorted rather than a per-observation
        # Python loop over the (potentially millions of) track rows.
        pt = np.asarray(recon.track_point_indexes).astype(np.int64)
        im = np.asarray(recon.track_image_indexes).astype(np.int64)
        kxy = np.asarray(recon.keypoints_xy, dtype=np.float32).reshape(-1, 2).copy()
        stride = np.int64(recon.image_count)
        track_key = pt * stride + im
        order = np.argsort(track_key, kind="stable")
        sorted_keys = track_key[order]

        # Flatten every point's kept views + refined keypoints into parallel
        # arrays (one numpy op per point, not per observation).
        res_pids, res_views, res_kpts = [], [], []
        for d in result:
            views = np.asarray(d["views"], dtype=np.int64)
            if views.size == 0:
                continue
            res_pids.append(
                np.full(views.shape[0], int(d["point_index"]), dtype=np.int64)
            )
            res_views.append(views)
            res_kpts.append(np.asarray(d["keypoints"], dtype=np.float32).reshape(-1, 2))

        if res_pids and sorted_keys.size:
            res_key = np.concatenate(res_pids) * stride + np.concatenate(res_views)
            res_kpts = np.concatenate(res_kpts, axis=0)
            pos = np.searchsorted(sorted_keys, res_key)
            # Keep only refined observations that map to a real track row (views
            # are drawn from the track, so this holds; the guard is defensive).
            in_range = pos < sorted_keys.size
            matched = np.zeros(res_key.shape[0], dtype=bool)
            matched[in_range] = sorted_keys[pos[in_range]] == res_key[in_range]
            kxy[order[pos[matched]]] = res_kpts[matched]

        # The refiner keeps keypoints strictly in-frame in f64, but the f32 the
        # format stores can round a near-edge value up to exactly width/height,
        # which the writer's `< width` (f32) check then rejects — failing the
        # whole save. Clamp each stored keypoint to the largest in-frame f32 for
        # its image's camera (same clamp as the embed-patches pipeline).
        cam_idx = np.asarray(recon.camera_indexes)
        cams = recon.cameras
        img_w = np.array([cams[int(c)].width for c in cam_idx], dtype=np.float32)
        img_h = np.array([cams[int(c)].height for c in cam_idx], dtype=np.float32)
        zero = np.float32(0.0)
        u_max = np.nextafter(img_w[im], zero)
        v_max = np.nextafter(img_h[im], zero)
        kxy[:, 0] = np.clip(kxy[:, 0], zero, u_max)
        kxy[:, 1] = np.clip(kxy[:, 1], zero, v_max)

        self._print_summary(result)

        # The reference observation each point's views were aligned to: its
        # stored one where that is in the track and renders, else the
        # reference-view rule's pick, or none (-1) where they were aligned to
        # a fused mean or to nothing. Each refined point records that one, so
        # its keypoints, bitmap and reference agree; a point the refiner did
        # not return keeps what it stores. A stored reference the refiner
        # used keeps its own observation, even in an image the point is
        # observed twice in.
        reference_images = np.full(recon.point_count, -1, dtype=np.int64)
        refined = np.zeros(recon.point_count, dtype=bool)
        for d in result:
            pid = int(d["point_index"])
            refined[pid] = True
            if d.get("reference_image") is not None:
                reference_images[pid] = int(d["reference_image"])
        aligned = reference_observations_from_images(recon, reference_images)
        stored = recon.reference_observations
        if stored is None:
            references = aligned
        else:
            stored = np.asarray(stored, dtype=np.int32)
            same = ~refined | (reference_images == stored_reference_images(recon))
            references = np.where(same, stored, aligned).astype(np.int32)

        # With `bitmaps`, also persist the per-point stored bitmaps rendered at
        # the final refined keypoints, each the tile of the reference
        # observation its views were aligned to. The stored frame is unchanged
        # (keypoints moved, not the surfel), so re-persisting it keeps the
        # recon consistent and lets the bitmaps attach to it.
        if self.bitmaps:
            npoints = recon.point_count
            bitmaps = np.zeros(
                (npoints, self.resolution, self.resolution, 4), dtype=np.uint8
            )
            n_filled = 0
            for d in result:
                bmp = d.get("bitmap")
                if bmp is not None:
                    pid = int(d["point_index"])
                    bitmaps[pid] = np.asarray(bmp, dtype=np.uint8)
                    n_filled += 1
            print(
                f"  Saving {len(result)} patches and {n_filled} bitmaps "
                f"to the reconstruction"
            )
            # Every bitmap with a reference is rendered again from the stored
            # (f32) keypoints, so dropping and adding the bitmaps later gives
            # the same bytes; a point at -1 keeps the refiner's fused mean.
            moved = recon.clone_with_changes(
                keypoints_xy=kxy, patches=cloud, reference_observations=references
            )
            bitmaps, references = render_from_references(
                moved,
                images,
                bitmaps,
                references,
                resolution=self.resolution,
                sampler=self.sampler,
            )
            return moved.clone_with_changes(
                patch_bitmaps=bitmaps, reference_observations=references
            )
        # Stored bitmaps were rendered at the old keypoints, so they go, as
        # ``--refine-normals bitmaps=false`` drops them. With no bitmap written
        # there is nothing for the reference to agree with, so a point that
        # stores a reference keeps it, even one nothing was aligned to, and a
        # point at -1 takes the one its views were aligned to.
        stored = recon.reference_observations
        if stored is not None:
            stored = np.asarray(stored, dtype=np.int32)
            references = np.where(stored >= 0, stored, aligned).astype(np.int32)
        else:
            references = aligned
        return recon.clone_with_changes(
            keypoints_xy=kxy, patch_bitmaps=None, reference_observations=references
        )

    def _print_summary(self, result: list[dict]) -> None:
        """One-line ``xform``-style summary over the views actually scored.

        A view that could not be scored carries a NaN score (its keypoint kept
        the seed); the statistics are over the finitely-scored views, the
        reference observation's included. ``offsets_px`` is in patch-grid px, a
        relative signal of how far the refiner moved each keypoint.
        """
        offsets: list[np.ndarray] = []
        for d in result:
            scores = np.asarray(d["zncc"], dtype=np.float64)
            off = np.asarray(d["offsets_px"], dtype=np.float64)
            scored = np.isfinite(scores)
            offsets.append(off[scored])
        all_offsets = (
            np.concatenate(offsets) if offsets else np.empty(0, dtype=np.float64)
        )
        n = int(all_offsets.size)
        if n == 0:
            print("  Refined 0 keypoints (no view could be scored)")
            return
        mean_offset = float(np.abs(all_offsets).mean())
        print(
            f"  Refined {n} keypoints (mean |offset| {mean_offset:.3f} patch-grid px)"
        )

    def description(self) -> str:
        # This string is also reused as the operation name in the precondition
        # error. `bitmaps` gates whether the RGBA textures are rendered.
        bitmaps = ", bitmaps" if self.bitmaps else ""
        return f"Refine keypoints (sampler={self.sampler}{bitmaps})"
