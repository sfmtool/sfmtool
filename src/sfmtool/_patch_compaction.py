# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Compaction glue: assemble pre-computed patch-keypoint results into an
``embedded_patches`` reconstruction.

This is the *write* tail of the
[sift-based -> patch-based pipeline](../../specs/core/sift-to-patch-reconstruction.md):
given a reconstruction, its refined :class:`PatchCloud`, and the per-point
keypoint-localization results, :func:`compact_to_embedded_patches` culls
under-supported points (and, given a validity mask, points with no valid
consensus bitmap), renumbers the survivors into a dense point set, and emits a
valid ``embedded_patches`` :class:`SfmrReconstruction` (inline ``keypoints_xy``,
per-point patch frame + optional bitmaps, ``image_file_hashes``,
``feature_source = "embedded_patches"``).

:func:`image_file_hashes_from_images` / :func:`image_file_hashes_from_sift`
compute the per-image identity hashes an ``embedded_patches`` file stores in
place of the ``.sift`` link. The :mod:`sfmtool._embed_patches` orchestration and
the ``xform`` localize-keypoints op both write back through this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sfmtool._sfmtool import (
    PatchCloud,
    SfmrReconstruction,
)


def image_file_hashes_from_images(recon: SfmrReconstruction) -> list[bytes]:
    """The per-image ``image_file_hashes`` an ``embedded_patches`` file needs,
    computed directly from the workspace image bytes (XXH128).

    This is the image-identity hash an ``embedded_patches`` file stores in place of
    the ``.sift`` link — the same value an image's ``.sift`` records as
    ``image_file_xxh128`` (see ``specs/formats/sfmr-file-format.md``). Each entry is
    the 16-byte little-endian digest expected by
    :meth:`SfmrReconstruction.clone_with_changes`.
    """
    from sfmtool.sift.file import xxh128_of_file

    ws = Path(recon.workspace_dir)
    return [bytes.fromhex(xxh128_of_file(ws / name)) for name in recon.image_names]


def image_file_hashes_from_sift(recon: SfmrReconstruction) -> list[bytes]:
    """The per-image ``image_file_hashes`` read from each image's ``.sift``
    ``image_file_xxh128`` metadata (hex → 16 bytes).

    This is the image-identity hash recorded when the features were extracted —
    the same value :func:`image_file_hashes_from_images` would recompute, but read
    straight from the ``.sift`` rather than re-hashing the (potentially large)
    image bytes. It matches the value ``SfmrReconstruction.to_embedded_patches``
    reads internally; :func:`embed_patches` now sources its hashes from the
    converted recon (``embedded.image_file_hashes``), so this is retained as a
    standalone helper rather than a pipeline step.

    Raises:
        FileNotFoundError: naming the image whose ``.sift`` cannot be resolved (the
            hash source the format requires for an ``embedded_patches`` file).
    """
    from sfmtool.sift.file import SiftReader, get_sift_path_from_recon

    hashes: list[bytes] = []
    for name in recon.image_names:
        sift_path = get_sift_path_from_recon(recon, name)
        try:
            meta = SiftReader(sift_path).metadata
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"no resolvable .sift for image {name!r} (needed for "
                f"image_file_hashes): {sift_path}"
            ) from e
        hashes.append(bytes.fromhex(meta["image_file_xxh128"]))
    return hashes


def compact_to_embedded_patches(
    recon: SfmrReconstruction,
    cloud: PatchCloud,
    localizations: list[dict[str, Any]],
    image_file_hashes: list[bytes],
    *,
    patch_bitmaps: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    min_views: int = 2,
) -> SfmrReconstruction:
    """Compact per-point keypoint-localization results into an ``embedded_patches``
    reconstruction.

    Args:
        recon: The source reconstruction (provides camera poses/intrinsics, image
            names, and the per-point geometry to carry over).
        cloud: The refined patch cloud (its ``point_indexes`` index ``recon``'s
            points); supplies each surviving point's ``(u, v)`` frame.
        localizations: The per-point dicts returned by
            :meth:`PatchCloud.localize_keypoints` — each ``{point_index, views,
            keypoints, ...}`` with the kept image indices and refined keypoints.
        image_file_hashes: One 16-byte XXH128 per image (see
            :func:`image_file_hashes_from_images`), parallel to ``recon.image_names``.
        patch_bitmaps: Optional ``(point_count, R, R, 4)`` uint8 reference textures
            scattered per source point (the pipeline sources these from
            ``refine_keypoints(render_bitmaps=True)``, fused at the final
            keypoints; ``refine_normals(render_bitmaps=True)``'s ``bitmaps`` fit
            too); culled to the survivors and stored as the patch bitmaps.
        valid: Optional bool mask per source point (parallel to ``recon``'s
            points): ``True`` where the keypoint refiner produced a valid
            cross-view consensus bitmap for the point. When given, a ``False``
            point is **dropped** — uniformly for finite and infinity points (a
            culled point would otherwise be kept with an all-black bitmap).
            ``None`` skips the validity cull (``min_views`` still applies).
        min_views: Drop a point whose kept-view count is below this.

    Returns:
        A new ``embedded_patches`` :class:`SfmrReconstruction`, ready to ``save()``.

    Raises:
        ValueError: if ``image_file_hashes`` is not parallel to the images, or no
            point survives the ``min_views`` / validity cull.
    """
    n_images = recon.image_count
    if len(image_file_hashes) != n_images:
        raise ValueError(
            f"image_file_hashes has {len(image_file_hashes)} entries, "
            f"expected one per image ({n_images})"
        )
    if min_views < 1:
        # Every kept point needs at least one observation (the format requires
        # observation_counts >= 1); a track of 1 is already degenerate, so the
        # meaningful floor is >= 1 here and the pipeline default is 2.
        raise ValueError(f"min_views must be >= 1, got {min_views}")

    cloud_pids = np.asarray(cloud.point_indexes)
    pid_to_cloud = {int(p): i for i, p in enumerate(cloud_pids)}
    loc_by_pid = {int(loc["point_index"]): loc for loc in localizations}
    valid_arr = None if valid is None else np.asarray(valid, dtype=bool)

    # Survivors: points with a patch, at least `min_views` kept observations, and
    # (when a validity mask is given) a valid consensus bitmap — the same rule for
    # finite and infinity points — in ascending source-point order (so the
    # renumbering is deterministic).
    survivors = sorted(
        pid
        for pid, loc in loc_by_pid.items()
        if pid in pid_to_cloud
        and len(np.asarray(loc["views"])) >= min_views
        and (valid_arr is None or bool(valid_arr[pid]))
    )
    if not survivors:
        raise ValueError(
            f"no point survived the min_views={min_views} / validity cull "
            f"({len(loc_by_pid)} points localized)"
        )

    # Per-point arrays, sliced to survivors in the new dense order. `positions_xyzw`
    # (homogeneous) carries each point's `w`, so a point at infinity (`w == 0`)
    # stays at infinity through the rebuild; `centers` (the Euclidean position, or
    # the unit direction for an infinity point) is only the patch-frame anchor for
    # `from_halfvec_arrays`.
    positions_xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64)[survivors]
    centers = np.asarray(recon.positions, dtype=np.float64)[survivors]
    colors = np.asarray(recon.colors, dtype=np.uint8)[survivors]
    errors = np.asarray(recon.errors, dtype=np.float32)[survivors]
    normals = (
        np.asarray(recon.normals, dtype=np.float32)[survivors]
        if recon.has_normals
        else None
    )

    # Surviving patch frame as half-extent vectors, then a renumbered cloud whose
    # point_indexes are the new dense indices (0..len(survivors)-1).
    p_new = len(survivors)
    u_xyz = np.zeros((p_new, 3), dtype=np.float32)
    v_xyz = np.zeros((p_new, 3), dtype=np.float32)
    for new_id, old_id in enumerate(survivors):
        patch = cloud[pid_to_cloud[old_id]]
        he = patch.half_extent
        u_xyz[new_id] = np.asarray(patch.u_axis, dtype=np.float64) * he[0]
        v_xyz[new_id] = np.asarray(patch.v_axis, dtype=np.float64) * he[1]
    culled_cloud = PatchCloud.from_halfvec_arrays(u_xyz, v_xyz, centers)
    # from_halfvec_arrays drops zero-`u` rows; every survivor must keep a frame, or
    # the cloud's point_indexes would no longer be the dense 0..P_new-1 the scatter
    # below relies on (and the survivor would be left frameless).
    if len(culled_cloud) != p_new:
        raise ValueError(
            "a surviving point has a degenerate (zero-length) patch u-axis; "
            "cannot build its embedded_patches frame"
        )

    new_bitmaps = None
    if patch_bitmaps is not None:
        new_bitmaps = np.ascontiguousarray(
            np.asarray(patch_bitmaps, dtype=np.uint8)[survivors]
        )

    # Flat, point-then-image-sorted observations and parallel keypoints.
    track_image_indexes: list[int] = []
    track_point_indexes: list[int] = []
    keypoints: list[np.ndarray] = []
    for new_id, old_id in enumerate(survivors):
        loc = loc_by_pid[old_id]
        views = np.asarray(loc["views"], dtype=np.uint32)
        kpts = np.asarray(loc["keypoints"], dtype=np.float32).reshape(-1, 2)
        for j in np.argsort(views, kind="stable"):
            track_image_indexes.append(int(views[j]))
            track_point_indexes.append(new_id)
            keypoints.append(kpts[j])

    keypoints_xy = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    track_image_indexes_arr = np.asarray(track_image_indexes, dtype=np.uint32)
    track_point_indexes_arr = np.asarray(track_point_indexes, dtype=np.uint32)

    # The localizer keeps keypoints strictly in-frame in f64, but the f32 the format
    # stores can round a near-edge value up to exactly the width/height, which the
    # writer's `< width` (f32) check then rejects — failing the whole save. Clamp
    # each stored keypoint to the largest in-frame f32 for its image's camera.
    cam_idx = np.asarray(recon.camera_indexes)
    cams = recon.cameras
    img_w = np.array([cams[int(c)].width for c in cam_idx], dtype=np.float32)
    img_h = np.array([cams[int(c)].height for c in cam_idx], dtype=np.float32)
    zero = np.float32(0.0)
    u_max = np.nextafter(img_w[track_image_indexes_arr], zero)
    v_max = np.nextafter(img_h[track_image_indexes_arr], zero)
    keypoints_xy[:, 0] = np.clip(keypoints_xy[:, 0], zero, u_max)
    keypoints_xy[:, 1] = np.clip(keypoints_xy[:, 1], zero, v_max)
    # feature_indexes are ignored for embedded_patches but must accompany the other
    # two track arrays in clone_with_changes.
    track_feature_indexes = np.zeros(len(track_image_indexes), dtype=np.uint32)

    # `positions` first so the point set is resized before the per-point arrays and
    # the patch frame are applied (clone_with_changes processes kwargs in order).
    kwargs: dict[str, Any] = {
        "positions": positions_xyzw,
        "colors": colors,
        "errors": errors,
    }
    if normals is not None:
        kwargs["normals"] = normals
    kwargs["patches"] = culled_cloud
    kwargs["feature_source"] = "embedded_patches"
    kwargs["image_file_hashes"] = list(image_file_hashes)
    kwargs["keypoints_xy"] = keypoints_xy
    kwargs["track_image_indexes"] = track_image_indexes_arr
    kwargs["track_feature_indexes"] = track_feature_indexes
    kwargs["track_point_indexes"] = track_point_indexes_arr
    if new_bitmaps is not None:
        kwargs["patch_bitmaps"] = new_bitmaps

    return recon.clone_with_changes(**kwargs)
