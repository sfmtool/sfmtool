# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Compaction glue: assemble pre-computed patch-keypoint results into an
``embedded_patches`` reconstruction.

This is the *write* tail of the
[sift-based → patch-based pipeline](../../specs/core/sift-to-patch-reconstruction.md):
given a reconstruction, its refined :class:`PatchCloud`, and the per-point
keypoint-localization results, :func:`compact_to_embedded_patches` culls
under-supported points, renumbers the survivors into a dense point set, and emits
a valid ``embedded_patches`` :class:`SfmrReconstruction` (inline ``keypoints_xy``,
per-point patch frame + optional bitmaps, ``image_file_hashes``,
``feature_source = "embedded_patches"``).

:func:`embed_patches` runs the whole pipeline end to end: it converts to the
baseline embedded form, photometrically refines each point's patch normal,
selects + vets the view set per point, congeals the keypoints, then hands the
results to :func:`compact_to_embedded_patches`. The ``sfm embed-patches`` CLI is
a thin wrapper over it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sfmtool._sfmtool import PatchCloud, SfmrReconstruction


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
    min_views: int = 2,
) -> SfmrReconstruction:
    """Compact per-point keypoint-localization results into an ``embedded_patches``
    reconstruction.

    Args:
        recon: The source reconstruction (provides camera poses/intrinsics, image
            names, and the per-point geometry to carry over).
        cloud: The refined patch cloud (its ``point_ids`` index ``recon``'s points);
            supplies each surviving point's ``(u, v)`` frame.
        localizations: The per-point dicts returned by
            :meth:`PatchCloud.localize_keypoints` — each ``{point_id, views,
            keypoints, ...}`` with the kept image indices and refined keypoints.
        image_file_hashes: One 16-byte XXH128 per image (see
            :func:`image_file_hashes_from_images`), parallel to ``recon.image_names``.
        patch_bitmaps: Optional ``(point_count, R, R, 4)`` uint8 reference textures
            scattered per source point (e.g. ``refine_normals(render_bitmaps=True)``'s
            ``bitmaps``); culled to the survivors and stored as the patch bitmaps.
        min_views: Drop a point whose kept-view count is below this.

    Returns:
        A new ``embedded_patches`` :class:`SfmrReconstruction`, ready to ``save()``.

    Raises:
        ValueError: if ``image_file_hashes`` is not parallel to the images, or no
            point survives the ``min_views`` cull.
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

    cloud_pids = np.asarray(cloud.point_ids)
    pid_to_cloud = {int(p): i for i, p in enumerate(cloud_pids)}
    loc_by_pid = {int(loc["point_id"]): loc for loc in localizations}

    # Survivors: points with a patch and at least `min_views` kept observations,
    # in ascending source-point order (so the renumbering is deterministic).
    survivors = sorted(
        pid
        for pid, loc in loc_by_pid.items()
        if pid in pid_to_cloud and len(np.asarray(loc["views"])) >= min_views
    )
    if not survivors:
        raise ValueError(
            f"no point survived the min_views={min_views} cull "
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
    # point_ids are the new dense indices (0..len(survivors)-1).
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
    # the cloud's point_ids would no longer be the dense 0..P_new-1 the scatter
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
    track_point_ids: list[int] = []
    keypoints: list[np.ndarray] = []
    for new_id, old_id in enumerate(survivors):
        loc = loc_by_pid[old_id]
        views = np.asarray(loc["views"], dtype=np.uint32)
        kpts = np.asarray(loc["keypoints"], dtype=np.float32).reshape(-1, 2)
        for j in np.argsort(views, kind="stable"):
            track_image_indexes.append(int(views[j]))
            track_point_ids.append(new_id)
            keypoints.append(kpts[j])

    keypoints_xy = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    track_image_indexes_arr = np.asarray(track_image_indexes, dtype=np.uint32)
    track_point_ids_arr = np.asarray(track_point_ids, dtype=np.uint32)

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
    kwargs["track_point_ids"] = track_point_ids_arr
    if new_bitmaps is not None:
        kwargs["patch_bitmaps"] = new_bitmaps

    return recon.clone_with_changes(**kwargs)


def _refine_subpixel(
    cloud: PatchCloud,
    embedded: SfmrReconstruction,
    images: list[np.ndarray],
    localizations: list[dict[str, Any]],
    *,
    mode: str,
    resolution: int,
) -> list[dict[str, Any]]:
    """Run :meth:`PatchCloud.refine_keypoints` seeded at ``localizations``'s
    per-view keypoints, and splice the refined source-px keypoints back into the
    localizer's per-point dicts (preserving the kept-view membership, order, and
    every other field — only the per-view ``keypoints`` array is replaced).

    Per-point view sets and seeds are derived from the localizer's output so the
    refiner sees exactly the same membership the localizer chose; a point the
    localizer dropped (or never localized) keeps its localization dict
    unchanged. ``mode`` selects the variant:

    - ``"lk"`` — per-sweep consensus, ``max_outer_sweeps = 1`` (the simple variant)
    - ``"lk_per_move"`` — per-move (Gauss–Seidel) incremental consensus,
      ``max_outer_sweeps = 5`` (the variant from #142)
    """
    if mode == "lk":
        kwargs = dict(max_outer_sweeps=1, consensus_refresh="per_sweep")
    elif mode == "lk_per_move":
        kwargs = dict(max_outer_sweeps=5, consensus_refresh="per_move")
    else:
        raise ValueError(
            f"unknown subpixel mode {mode!r} (expected 'none', 'lk', or 'lk_per_move')"
        )

    # Build per-point view sets + starting keypoints parallel to each other (the
    # refiner reads `starting_keypoints[pid][k]` as the seed for the k'th view
    # of `view_sets[pid]`, in order — so the two MUST be built in the same loop).
    view_sets: dict[int, list[int]] = {}
    seeds: dict[int, list[list[float]]] = {}
    for loc in localizations:
        pid = int(loc["point_id"])
        views = np.asarray(loc["views"], dtype=np.uint32).tolist()
        kpts = np.asarray(loc["keypoints"], dtype=np.float64).reshape(-1, 2)
        if not views:
            continue
        view_sets[pid] = views
        seeds[pid] = [[float(p[0]), float(p[1])] for p in kpts]

    if not view_sets:
        return localizations

    refined = cloud.refine_keypoints(
        embedded,
        images,
        view_sets=view_sets,
        starting_keypoints=seeds,
        point_ids=list(view_sets.keys()),
        resolution=resolution,
        **kwargs,
    )

    # Splice the refined keypoints back into each point's localization dict.
    # The refiner returns views in input order and never changes membership
    # (the only drop is the projection gate — a view in which `project_i(X_p)`
    # fails to land in frame — which the localizer already filtered out). If
    # that *does* happen here (a different image was somehow rejected by the
    # refiner's gate), we fall back to the localizer's keypoint for any view
    # the refiner didn't return — preserving the compaction-side membership.
    refined_by_pid = {int(r["point_id"]): r for r in refined}
    out: list[dict[str, Any]] = []
    for loc in localizations:
        pid = int(loc["point_id"])
        r = refined_by_pid.get(pid)
        if r is None:
            out.append(loc)
            continue
        r_views = np.asarray(r["views"], dtype=np.uint32)
        r_kpts = np.asarray(r["keypoints"], dtype=np.float64).reshape(-1, 2)
        l_views = np.asarray(loc["views"], dtype=np.uint32)
        l_kpts = np.asarray(loc["keypoints"], dtype=np.float64).reshape(-1, 2)
        # Map refiner's per-view keypoints by image index, then walk the
        # localizer's view order to keep the membership identical.
        r_map = {int(v): r_kpts[i] for i, v in enumerate(r_views.tolist())}
        new_kpts = np.array(
            [r_map.get(int(v), l_kpts[i]) for i, v in enumerate(l_views.tolist())],
            dtype=np.float64,
        ).reshape(-1, 2)
        new_loc = dict(loc)
        new_loc["keypoints"] = new_kpts
        out.append(new_loc)
    return out


def embed_patches(
    recon: SfmrReconstruction,
    images: list[np.ndarray],
    *,
    min_relative_zncc: float = 0.7,
    patch_size: float = 10.0,
    max_shift_px: float = 3.0,
    min_views: int = 2,
    max_iters: int = 5,
    search: float = 6.0,
    resolution: int = 24,
    search_resolution_multiplier: float = 1.0,
    subpixel: str = "none",
) -> SfmrReconstruction:
    """Convert a ``sift_files`` reconstruction to ``embedded_patches``, running the
    full photometric pipeline (see
    ``specs/core/sift-to-patch-reconstruction.md``).

    0. **Convert to the baseline ``embedded_patches`` form** with a single call to
       the Rust ``SfmrReconstruction.to_embedded_patches`` — the only step that
       reads the ``.sift`` files: it gives each point a mean-viewing ``(u, v)``
       frame, copies each observation's SIFT detection keypoint inline, and reads
       each image's hash from the ``.sift`` metadata. Everything below runs
       ``embedded_patches → embedded_patches``.
    1. **Refine each normal** photometrically, anchoring every view on its stored
       (SIFT) keypoint rather than the reprojected point center
       (``use_stored_keypoints``), and render each point's reference bitmap. Points
       at infinity keep their fixed tangent-sphere frame untouched.
    2. **Select the view set** per point: the track plus other views that
       geometrically see the surfel and clear ``min_relative_zncc`` against a
       track-seeded template.
    3. **Project + congeal** each view's keypoint to sub-pixel, dropping views that
       won't co-register (grazing, out-of-frame, ``max_shift_px``, low LOO ZNCC).
    4. **Cull + compact**: drop points left below ``min_views`` and renumber the
       survivors into a valid ``embedded_patches`` reconstruction.

    Args:
        recon: A ``sift_files`` reconstruction (the caller validates this).
        images: One full-resolution image per ``recon.image_names`` entry, matching
            each camera's dimensions (e.g. via ``read_workspace_image``).
        patch_size: Surfel size — the full patch edge (feature-size multiples),
            halved to the library half-extent and passed to ``to_embedded_patches``.
        min_relative_zncc, max_shift_px, min_views, max_iters, search: The pipeline
            knobs documented in ``specs/cli/embed-patches-command.md``.
        resolution: The ``R × R`` patch grid the kernels render/score on.
        search_resolution_multiplier: ``m`` for the discrete cross-view search in
            :meth:`PatchCloud.localize_keypoints` (step 3). ``1.0`` (default) is the
            no-op; ``> 1`` runs the supersampled grid (cost grows ~``m²``) — see
            ``specs/core/keypoint-localization-search-cache.md``.
        subpixel: Optional photometric sub-pixel pass applied to the localizer's
            output (step 3.5 in the pipeline). One of:
            ``"none"`` (default) — no refinement; the localizer's keypoints are
            used as-is.
            ``"lk"`` — LK / ECC Gauss–Newton refinement seeded at the localizer's
            keypoints; per-sweep consensus, ``max_outer_sweeps = 1``.
            ``"lk_per_move"`` — same but the per-move (Gauss–Seidel) incremental
            consensus variant with ``max_outer_sweeps = 5``.

    Returns:
        A new ``embedded_patches`` :class:`SfmrReconstruction`, ready to ``save()``.
    """
    half_extent = patch_size / 2.0

    # 0. The single `.sift`-consuming step: baseline embedded conversion. It sizes
    #    each point's mean-viewing frame by SIFT feature scale, copies the SIFT
    #    detection keypoints inline, and reads the image hashes from `.sift`
    #    metadata. Its frame, keypoints, and hashes are all consumed below.
    embedded = recon.to_embedded_patches(
        normal="mean_viewing", extent="feature_size", extent_value=half_extent
    )

    # 1. Take the patch cloud `to_embedded_patches` built (mean-viewing frames
    #    sized by SIFT feature scale — read from the embedded recon's stored frames,
    #    so no second `.sift` read), then refine each normal over the embedded recon,
    #    anchoring every view on its stored SIFT keypoint (use_stored_keypoints)
    #    instead of the reprojected center; render each point's reference bitmap.
    cloud = embedded.patches
    if cloud is None:
        raise ValueError("to_embedded_patches produced no patch frames to refine")
    refine = cloud.refine_normals(
        embedded,
        images,
        resolution=resolution,
        use_stored_keypoints=True,
        render_bitmaps=True,
    )

    # 2. Expand + vet the view set per point.
    selections = cloud.select_views(
        embedded, images, min_relative_zncc=min_relative_zncc, resolution=resolution
    )
    view_sets = {
        int(s["point_id"]): np.asarray(s["admitted"]).tolist() for s in selections
    }

    # 3. Project starting keypoints (implicit) and congeal them, dropping views that
    #    won't co-register in-loop.
    localizations = cloud.localize_keypoints(
        embedded,
        images,
        view_sets=view_sets,
        max_iters=max_iters,
        search=search,
        max_shift_px=max_shift_px,
        min_relative_zncc=min_relative_zncc,
        resolution=resolution,
        search_resolution_multiplier=search_resolution_multiplier,
    )

    # 3.5. Optional photometric sub-pixel refinement, seeded at the localizer's
    #      kept keypoints. The localizer's output IS the precondition that lets
    #      the local refiner work — it puts each view in the basin of the true
    #      optimum (≲ 1 px). The refiner moves the per-view keypoints; the
    #      kept-view membership stays exactly as the localizer chose it.
    if subpixel != "none":
        localizations = _refine_subpixel(
            cloud,
            embedded,
            images,
            localizations,
            mode=subpixel,
            resolution=resolution,
        )

    # 4. Cull under-supported points and compact into an embedded_patches recon. The
    #    image hashes already live on the embedded recon (set in step 0), so there
    #    is no second `.sift` read; the original `recon` carries the per-point
    #    geometry (its point indexing matches `embedded` one-to-one).
    hashes = embedded.image_file_hashes
    return compact_to_embedded_patches(
        recon,
        cloud,
        localizations,
        hashes,
        patch_bitmaps=refine.get("bitmaps"),
        min_views=min_views,
    )
