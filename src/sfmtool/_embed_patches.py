# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Compaction glue: assemble pre-computed patch-keypoint results into an
``embedded_patches`` reconstruction.

This is the *write* tail of the
[sift-based → patch-based pipeline](../../specs/core/sift-to-patch-reconstruction.md):
given a reconstruction, its refined :class:`PatchCloud`, and the per-point
keypoint-localization results, :func:`compact_to_embedded_patches` culls
under-supported points (and, given a validity mask, points with no valid
consensus bitmap), renumbers the survivors into a dense point set, and emits
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

import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from sfmtool._sfmtool import (
    ImagePyramidSet,
    PatchCloud,
    ProgressCounter,
    SfmrReconstruction,
)


@contextmanager
def _timed_step(log: Any, label: str):
    """Log ``label`` at the start of a step and its elapsed wall time on
    completion, so each blocking Rust pass shows a start line (and a "done"
    line proving it advanced) rather than one silent block. A no-op when
    ``log`` is ``None``."""
    if log is None:
        yield
        return
    log(label)
    t0 = time.perf_counter()
    yield
    log(f"    ...done ({time.perf_counter() - t0:.1f}s)")


def _progress_poll_loop(
    log: Any,
    read: Any,
    total: int,
    stop: threading.Event,
    interval: float,
) -> None:
    """Until ``stop`` is set, every ``interval`` seconds read the current work
    count via ``read()`` and echo a ``done/total`` line through ``log``. Split
    out from :func:`_poll_progress` so the reporting logic is unit-testable with
    an injected ``read`` (the live path reads a :class:`ProgressCounter`)."""
    while not stop.wait(interval):
        done = read()
        # `done` can momentarily read 0 before the first patches finish, and
        # equal `total` just as the pass ends; only report genuine mid-pass
        # progress so we never print a redundant 0%/100% around the done line.
        if 0 < done < total:
            log(f"    {done}/{total} patches ({100 * done // total}%)")


@contextmanager
def _poll_progress(log: Any, total: int, *, interval: float = 5.0):
    """Yield a :class:`ProgressCounter` and, while the body runs, echo its
    ``value``/``total`` from a background thread every ``interval`` seconds.

    The counted Rust pass runs with the GIL released (``py.detach``), so the
    poller thread can read the shared counter and report intra-pass progress
    that would otherwise be one opaque blocking step. Yields ``None`` (no
    counter) when ``log`` is ``None`` or ``total`` is trivial, so the caller
    passes ``progress=None`` and the pass runs uninstrumented."""
    if log is None or total <= 0:
        yield None
        return
    counter = ProgressCounter()
    stop = threading.Event()
    poller = threading.Thread(
        target=_progress_poll_loop,
        args=(log, lambda: counter.value, total, stop, interval),
        daemon=True,
    )
    poller.start()
    try:
        yield counter
    finally:
        stop.set()
        poller.join(timeout=interval)


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


def _refine_subpixel(
    cloud: PatchCloud,
    embedded: SfmrReconstruction,
    images: list[np.ndarray] | ImagePyramidSet,
    localizations: list[dict[str, Any]],
    *,
    sweeps: int,
    resolution: int,
    render_bitmaps: bool = False,
    progress: Any = None,
) -> tuple[list[dict[str, Any]], np.ndarray | None, np.ndarray | None]:
    """Run :meth:`PatchCloud.refine_keypoints` seeded at ``localizations``'s
    per-view keypoints, and splice the refined source-px keypoints back into the
    localizer's per-point dicts (preserving the kept-view membership, order, and
    every other field — only the per-view ``keypoints`` array is replaced).

    Per-point view sets and seeds are derived from the localizer's output so the
    refiner sees exactly the same membership the localizer chose; a point the
    localizer dropped (or never localized) keeps its localization dict unchanged.
    ``sweeps`` is the LK/ECC Gauss–Newton ``max_outer_sweeps`` (>= 1), always with
    the per-sweep consensus. ``sweeps == 0`` moves no keypoint (the input
    localizations are returned as is); combined with ``render_bitmaps`` it still
    runs the refiner **render-only** (``max_gn_steps=0``, seeds kept) so the
    bitmaps/validity below are produced at the localizer's own keypoints.

    Returns:
        ``(localizations, bitmaps, valid)``. With ``render_bitmaps=True``,
        ``bitmaps`` is a ``(point_count, R, R, 4)`` uint8 array of consensus
        (representative) textures fused at the FINAL per-view keypoints, scattered
        per source-point index (zero rows where no valid consensus), and ``valid``
        the parallel bool mask — the culled-point signal
        :func:`compact_to_embedded_patches` drops on, uniform for finite and
        infinity points. With ``render_bitmaps=False`` both are ``None``.
    """
    if sweeps < 1 and not render_bitmaps:
        return localizations, None, None
    kwargs: dict[str, Any] = dict(
        max_outer_sweeps=max(sweeps, 1), consensus_refresh="per_sweep"
    )
    if sweeps < 1:
        # Render-only: keep every seed (no GN step) but still fuse the
        # representative bitmaps + validity at those seeds.
        kwargs["max_gn_steps"] = 0
    if render_bitmaps:
        kwargs["render_bitmaps"] = True

    # Build per-point view sets + starting keypoints parallel to each other (the
    # refiner reads `starting_keypoints[pid][k]` as the seed for the k'th view
    # of `view_sets[pid]`, in order — so the two MUST be built in the same loop).
    view_sets: dict[int, list[int]] = {}
    seeds: dict[int, list[list[float]]] = {}
    for loc in localizations:
        pid = int(loc["point_index"])
        views = np.asarray(loc["views"], dtype=np.uint32).tolist()
        kpts = np.asarray(loc["keypoints"], dtype=np.float64).reshape(-1, 2)
        if not views:
            continue
        view_sets[pid] = views
        seeds[pid] = [[float(p[0]), float(p[1])] for p in kpts]

    if not view_sets:
        # Nothing to refine (and nothing that could hold a consensus bitmap).
        return localizations, None, None

    refined = cloud.refine_keypoints(
        embedded,
        images,
        view_sets=view_sets,
        starting_keypoints=seeds,
        point_indexes=list(view_sets.keys()),
        resolution=resolution,
        progress=progress,
        **kwargs,
    )

    # Scatter the per-point consensus bitmaps (fused at the final keypoints) and
    # the parallel validity mask per SOURCE point index — zero rows / False where
    # the refiner produced no valid consensus (the culled-point signal).
    bitmaps: np.ndarray | None = None
    valid: np.ndarray | None = None
    if render_bitmaps:
        n_points = embedded.point_count
        bitmaps = np.zeros((n_points, resolution, resolution, 4), dtype=np.uint8)
        valid = np.zeros(n_points, dtype=bool)
        for r in refined:
            bm = r.get("bitmap")
            if bm is not None:
                pid = int(r["point_index"])
                bitmaps[pid] = np.asarray(bm, dtype=np.uint8)
                valid[pid] = True

    if sweeps < 1:
        # Render-only pass: the localizer's keypoints are used as is.
        return localizations, bitmaps, valid

    # Splice the refined keypoints back into each point's localization dict.
    # The refiner returns views in input order and never changes membership
    # (the only drop is the projection gate — a view in which `project_i(X_p)`
    # fails to land in frame — which the localizer already filtered out). If
    # that *does* happen here (a different image was somehow rejected by the
    # refiner's gate), we fall back to the localizer's keypoint for any view
    # the refiner didn't return — preserving the compaction-side membership.
    refined_by_pid = {int(r["point_index"]): r for r in refined}
    out: list[dict[str, Any]] = []
    for loc in localizations:
        pid = int(loc["point_index"])
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
    return out, bitmaps, valid


def _localizations_from_recon(recon: SfmrReconstruction) -> list[dict[str, Any]]:
    """Rebuild the per-point localization dicts (``point_index``, ``views``,
    ``keypoints``) from an ``embedded_patches`` recon's inline tracks — the seed a
    later round's sub-pixel refinement starts from once the discrete localizer has
    run (round 1 only). Membership is exactly the recon's current track set."""
    pt = np.asarray(recon.track_point_indexes)
    im = np.asarray(recon.track_image_indexes, dtype=np.uint32)
    kxy = np.asarray(recon.keypoints_xy, dtype=np.float64).reshape(-1, 2)
    by_pid: dict[int, dict[str, list]] = {}
    for k in range(len(pt)):
        d = by_pid.setdefault(int(pt[k]), {"views": [], "keypoints": []})
        d["views"].append(int(im[k]))
        d["keypoints"].append(kxy[k])
    return [
        {
            "point_index": pid,
            "views": np.asarray(d["views"], dtype=np.uint32),
            "keypoints": np.asarray(d["keypoints"], dtype=np.float64).reshape(-1, 2),
        }
        for pid, d in sorted(by_pid.items())
    ]


def _patch_normals(cloud: PatchCloud) -> np.ndarray:
    """Per-patch unit normal (``u × v``, normalized) for every patch, as an
    ``(N, 3)`` array — the quantity ``refine_normals`` moves. Used only to report
    per-round normal evolution."""
    return np.asarray(
        [np.asarray(cloud[i].normal, np.float64) for i in range(len(cloud))]
    )


def _mean_angle_deg(n0: np.ndarray, n1: np.ndarray) -> float:
    """Mean angle (degrees) between corresponding rows of two ``(N, 3)`` unit-
    normal arrays."""
    if n0.size == 0:
        return 0.0
    dots = np.clip(np.sum(n0 * n1, axis=1), -1.0, 1.0)
    return float(np.degrees(np.arccos(dots)).mean())


def _camera_centers(recon: SfmrReconstruction) -> np.ndarray:
    """World-space camera centers ``(n_images, 3)`` — ``-Rᵀ t`` for the
    ``x_cam = R x_world + t`` pose of each image."""
    from sfmtool._sfmtool.geometry import RigidTransform

    quats = np.asarray(recon.quaternions_wxyz, np.float64)
    trans = np.asarray(recon.translations, np.float64)
    centers = np.empty((len(quats), 3), np.float64)
    for i in range(len(quats)):
        rot = np.asarray(
            RigidTransform.from_wxyz_translation(
                quats[i].tolist(), trans[i].tolist()
            ).to_rotation_matrix(),
            np.float64,
        )
        centers[i] = -rot.T @ trans[i]
    return centers


def _drop_grazing_observations(
    localizations: list[dict[str, Any]],
    cloud: PatchCloud,
    centers: np.ndarray,
    positions: np.ndarray,
    at_infinity: np.ndarray,
    max_obliquity_deg: float,
) -> tuple[list[dict[str, Any]], int]:
    """Drop observations whose view direction is more than ``max_obliquity_deg``
    off the surfel normal (``|v̂·n| < cos(max_obliquity_deg)``). A grazing view
    renders as a cross-view-consistent but degenerate smear that would otherwise
    bias the consensus and pull the normal toward grazing over subsequent rounds.
    Returns ``(pruned_localizations, n_dropped)``. ``90°`` is a no-op.

    ``at_infinity`` (bool per point index) flags points at infinity, which are
    left untouched: their ``positions`` row is a unit *direction*, not a location,
    so the ``centers - position`` view vector (and its obliquity) is meaningless —
    and the Rust normal refinement leaves their fixed tangent-sphere frame alone,
    so there is nothing to grade the view against."""
    if max_obliquity_deg >= 90.0:
        return localizations, 0
    cos_min = float(np.cos(np.radians(max_obliquity_deg)))
    pids = np.asarray(cloud.point_indexes)
    normals = {
        int(pids[i]): np.asarray(cloud[i].normal, np.float64) for i in range(len(cloud))
    }
    out: list[dict[str, Any]] = []
    dropped = 0
    for loc in localizations:
        pid = int(loc["point_index"])
        n = normals.get(pid)
        views = np.asarray(loc["views"], dtype=np.uint32)
        kpts = np.asarray(loc["keypoints"], dtype=np.float64).reshape(-1, 2)
        if n is None or len(views) == 0 or bool(at_infinity[pid]):
            out.append(loc)
            continue
        d = centers[views.astype(np.intp)] - positions[pid]
        nrm = np.linalg.norm(d, axis=1)
        valid = nrm > 1e-9
        cos = np.zeros(len(views))
        cos[valid] = np.abs((d[valid] / nrm[valid, None]) @ n)
        keep = valid & (cos >= cos_min)
        if keep.all():
            out.append(loc)
            continue
        dropped += int((~keep).sum())
        new_loc = dict(loc)
        new_loc["views"] = views[keep]
        new_loc["keypoints"] = kpts[keep]
        out.append(new_loc)
    return out, dropped


def _mean_keypoint_shift(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> float:
    """Mean per-observation keypoint displacement (source px) between two
    localization lists, matched by ``(point_index, image_index)``."""
    after_by_pid = {int(loc["point_index"]): loc for loc in after}
    shifts: list[float] = []
    for lb in before:
        la = after_by_pid.get(int(lb["point_index"]))
        if la is None:
            continue
        b_views = np.asarray(lb["views"], dtype=np.uint32).tolist()
        b_kpts = np.asarray(lb["keypoints"], dtype=np.float64).reshape(-1, 2)
        a_views = np.asarray(la["views"], dtype=np.uint32).tolist()
        a_kpts = np.asarray(la["keypoints"], dtype=np.float64).reshape(-1, 2)
        a_map = {int(v): a_kpts[i] for i, v in enumerate(a_views)}
        for i, v in enumerate(b_views):
            a = a_map.get(int(v))
            if a is not None:
                shifts.append(float(np.hypot(a[0] - b_kpts[i, 0], a[1] - b_kpts[i, 1])))
    return float(np.mean(shifts)) if shifts else 0.0


def embed_patches(
    recon: SfmrReconstruction,
    images: list[np.ndarray],
    *,
    min_relative_zncc: float = 0.7,
    patch_size: float = 5.0,
    max_shift_px: float = 3.0,
    min_views: int = 2,
    max_iters: int = 5,
    search: float = 6.0,
    resolution: int = 24,
    search_resolution_multiplier: float = 1.0,
    subpixel: int = 1,
    rounds: int = 2,
    max_obliquity_deg: float = 80.0,
    obliquity_weight_power: float = 2.0,
    fronto_prior_weight: float = 0.05,
    max_refine_views: int = 8,
    localize_search_strategy: str = "plus_descent",
    progress: Any = None,
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
       (``use_stored_keypoints``). Points at infinity keep their fixed
       tangent-sphere frame untouched.
    2. **Select the view set** per point: the track plus other views that
       geometrically see the surfel and clear ``min_relative_zncc`` against a
       track-seeded template.
    3. **Project + congeal** each view's keypoint to sub-pixel, dropping views that
       won't co-register (grazing, out-of-frame, ``max_shift_px``, low LOO ZNCC).
       The final round's sub-pixel pass also fuses each point's **consensus
       bitmap** at the final keypoints (points at infinity included — they render
       through the same ``w``-aware path) and reports per-point validity.
    4. **Cull + compact**: drop points left below ``min_views`` **and** points the
       sub-pixel pass produced no valid consensus bitmap for (the culled-point
       signal, uniform for finite and infinity points), then renumber the
       survivors into a valid ``embedded_patches`` reconstruction carrying those
       bitmaps.

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
        subpixel: LK/ECC Gauss–Newton ``max_outer_sweeps`` for the photometric
            sub-pixel keypoint refinement applied in each round (per-sweep
            consensus). ``0`` disables the keypoint movement (the localizer's
            keypoints are used as is; the final round still runs a render-only
            pass to fuse the consensus bitmaps + validity at those keypoints);
            ``>= 1`` runs it with that many sweeps.
        rounds: Number of (normal-refinement, keypoint-refinement) rounds. Round 1
            runs the SIFT-anchored normal refine, the discrete localizer (the
            seed), then the sub-pixel keypoint refine. Each subsequent round
            re-refines every normal against the *previous* round's keypoints, then
            re-refines the keypoints against the new normals — a fixed-point
            alternation. The per-point view set can only shrink across rounds (the
            grazing-observation drop below); it is never expanded after round 1.
        max_obliquity_deg: After **each** round's normal refinement, drop every
            observation viewing its surfel more than this off the (just-refined)
            normal (``< 90`` enables the filter). Grazing views render as
            cross-view-consistent but degenerate smears; the low-parallax
            degeneracy tilts a normal toward grazing gradually across rounds, so a
            view only crosses the threshold once the tilt reaches it — pruning each
            round chases the tilt and culls a surfel that has gone fully edge-on
            rather than letting it settle into a smear.
        obliquity_weight_power: Exponent ``p`` of the multiplicative obliquity
            view-weight ``|v̂·n|^p`` folded into the robust normal-refinement
            consensus (use A). ``0.0`` disables it; ``2.0`` (default) is the
            ``cos²θ`` foreshortening weight that softly down-weights oblique views —
            a continuous complement to the hard ``max_obliquity_deg`` cut.
        fronto_prior_weight: Weight ``λ`` of the additive fronto-parallel prior
            ``λ·mean_v (v̂·n)²`` on each candidate normal during refinement (use B).
            ``0.0`` disables it; the small default (``0.05``) pulls a low-parallax
            (flat-``Φ``) normal toward facing the cameras instead of drifting to a
            photometrically-equivalent tilt, without overriding a normal that real
            parallax constrains.
        max_refine_views: When ``> 0``, cap the **round-2+ normal-refinement
            basis** at this many views per point — the D-optimal geometric pick of
            the most normal-informative views (see
            ``specs/core/patch-normal-refine-view-subset.md``). Applied only to the
            fine-tuning rounds, whose view set is the ``select_views``-expanded one;
            the round-1 (raw-track) refine is untouched. Lossless for the output:
            only the refinement basis shrinks — every observation stays, and the
            consensus bitmaps are still fused over the full view set. ``0`` uses
            all views (disables the cap); the default is ``8`` (cuts roughly a
            third off end-to-end time on large view sets — the round-2+ refine
            pass itself ~5x — at the cost of a different, not necessarily worse,
            normal on high-view points).
        progress: Optional callable (e.g. ``click.echo``) that receives a per-round
            summary line reporting the mean normal change (deg) and mean keypoint
            shift (px); when given, those metrics are computed each round.

    Returns:
        A new ``embedded_patches`` :class:`SfmrReconstruction`, ready to ``save()``.
    """
    log = progress if callable(progress) else None
    half_extent = patch_size / 2.0

    # Decode every source image into its full pyramid ONCE. Each kernel call
    # below (six on a default two-round run) previously rebuilt all the
    # pyramids from the numpy list on entry; the shared set removes that
    # per-call marshalling cost without changing any pyramid content.
    with _timed_step(log, f"  building image pyramids ({len(images)} imgs)..."):
        pyramids = ImagePyramidSet(recon, images)

    # 0. The single `.sift`-consuming step: baseline embedded conversion. It sizes
    #    each point's mean-viewing frame by SIFT feature scale, copies the SIFT
    #    detection keypoints inline, and reads the image hashes from `.sift`
    #    metadata. Its frame, keypoints, and hashes are all consumed below.
    with _timed_step(
        log,
        f"  round 1/{rounds}: converting sift→patches "
        f"({recon.point_count} pts, {len(images)} imgs)...",
    ):
        embedded = recon.to_embedded_patches(
            normal="mean_viewing", extent="feature_size", extent_value=half_extent
        )

    # 1. Refine each normal over the embedded recon, anchoring every view on its
    #    stored SIFT keypoint (use_stored_keypoints) instead of the reprojected
    #    center. (Reference bitmaps are NOT rendered here — the final round's
    #    sub-pixel pass fuses them at the final keypoints, step 3.5.)
    cloud = embedded.patches
    if cloud is None:
        raise ValueError("to_embedded_patches produced no patch frames to refine")
    n_before = _patch_normals(cloud) if log else None
    with (
        _timed_step(log, f"  round 1/{rounds}: refining normals ({len(cloud)} pts)..."),
        _poll_progress(log, len(cloud)) as counter,
    ):
        cloud.refine_normals(
            embedded,
            pyramids,
            resolution=resolution,
            use_stored_keypoints=True,
            obliquity_weight_power=obliquity_weight_power,
            fronto_prior_weight=fronto_prior_weight,
            progress=counter,
        )

    # 2. Expand + vet the view set per point (round 1 only; membership is fixed
    #    afterwards).
    with (
        _timed_step(log, f"  round 1/{rounds}: selecting views ({len(cloud)} pts)..."),
        _poll_progress(log, len(cloud)) as counter,
    ):
        selections = cloud.select_views(
            embedded,
            pyramids,
            min_relative_zncc=min_relative_zncc,
            resolution=resolution,
            progress=counter,
        )
    view_sets = {
        int(s["point_index"]): np.asarray(s["admitted"]).tolist() for s in selections
    }

    # 3. Discrete localizer (the seed): project starting keypoints and congeal them,
    #    dropping views that won't co-register in-loop. Runs once, in round 1.
    with (
        _timed_step(
            log, f"  round 1/{rounds}: localizing keypoints ({len(cloud)} pts)..."
        ),
        _poll_progress(log, len(cloud)) as counter,
    ):
        localizations = cloud.localize_keypoints(
            embedded,
            pyramids,
            view_sets=view_sets,
            max_iters=max_iters,
            search=search,
            max_shift_px=max_shift_px,
            min_relative_zncc=min_relative_zncc,
            resolution=resolution,
            search_resolution_multiplier=search_resolution_multiplier,
            search_strategy=localize_search_strategy,
            progress=counter,
        )

    # 3.5. Sub-pixel keypoint refinement, seeded at the localizer's kept keypoints
    #      (the localizer put each view in the basin; the LK refiner sharpens it).
    #      The FINAL round's pass also fuses each point's consensus bitmap at the
    #      final keypoints and reports per-point validity — the reference textures
    #      and the culled-point drop signal the compaction consumes (with
    #      subpixel=0 the pass is render-only: seeds kept, bitmaps still fused).
    seed_loc = localizations
    with (
        _timed_step(
            log, f"  round 1/{rounds}: sub-pixel keypoint refine ({len(cloud)} pts)..."
        ),
        _poll_progress(log, len(cloud)) as counter,
    ):
        localizations, bitmaps, valid = _refine_subpixel(
            cloud,
            embedded,
            pyramids,
            localizations,
            sweeps=subpixel,
            resolution=resolution,
            render_bitmaps=rounds == 1,
            progress=counter,
        )
    if log:
        ndeg = _mean_angle_deg(n_before, _patch_normals(cloud))
        kpx = _mean_keypoint_shift(seed_loc, localizations)
        log(
            f"  round 1/{rounds}: normal Δ {ndeg:.3f}°, keypoint Δ {kpx:.3f}px (vs seed)"
        )

    # After round 1: drop grazing observations against the refined normal, so the
    # subsequent rounds' consensus is not dragged toward a degenerate grazing smear.
    localizations, n_dropped = _drop_grazing_observations(
        localizations,
        cloud,
        _camera_centers(embedded),
        np.asarray(embedded.positions, np.float64),
        np.asarray(embedded.point_is_at_infinity),
        max_obliquity_deg,
    )
    if log and n_dropped:
        log(
            f"  dropped {n_dropped} grazing obs (> {max_obliquity_deg:.0f} deg off normal)"
        )

    # Rounds 2..N: compact the current state into a self-contained embedded recon
    # (keeping every localized point, min_views=1), re-refine its normals against
    # the carried-forward keypoints, then re-refine the keypoints against the new
    # normals. Each iteration's (recon, cloud, localizations) are mutually
    # consistent in the compacted dense indexing.
    work_recon, work_cloud, work_loc = recon, cloud, localizations
    hashes = embedded.image_file_hashes
    if log and max_refine_views > 0 and rounds > 1:
        log(
            f"  rounds 2+: normal-refinement basis capped at the "
            f"{max_refine_views} most-informative views per point (D-optimal)"
        )
    for r in range(2, rounds + 1):
        # Intermediate recons carry no bitmaps — nothing reads them; the final
        # bitmaps are fused by the last round's sub-pixel pass below.
        emb_r = compact_to_embedded_patches(
            work_recon, work_cloud, work_loc, hashes, min_views=1
        )
        cloud_r = emb_r.patches
        n_before = _patch_normals(cloud_r) if log else None
        with (
            _timed_step(
                log, f"  round {r}/{rounds}: refining normals ({len(cloud_r)} pts)..."
            ),
            _poll_progress(log, len(cloud_r)) as counter,
        ):
            cloud_r.refine_normals(
                emb_r,
                pyramids,
                resolution=resolution,
                use_stored_keypoints=True,
                obliquity_weight_power=obliquity_weight_power,
                fronto_prior_weight=fronto_prior_weight,
                # The round-2+ view set is the select_views-expanded one; the
                # D-optimal cap (0 = off) trims the refinement basis only —
                # membership (and the fused bitmaps) still span every view.
                max_refine_views=max_refine_views,
                progress=counter,
            )
        base_loc = _localizations_from_recon(emb_r)
        # Re-prune grazing observations against THIS round's refined normal: the
        # low-parallax degeneracy tilts a normal toward grazing gradually over
        # rounds, so a view that was near-frontal at round 1 only crosses the
        # threshold now. Pruning each round chases the tilt and culls a surfel that
        # has gone fully edge-on rather than letting it settle into a smear.
        base_loc, n_dropped = _drop_grazing_observations(
            base_loc,
            cloud_r,
            _camera_centers(emb_r),
            np.asarray(emb_r.positions, np.float64),
            np.asarray(emb_r.point_is_at_infinity),
            max_obliquity_deg,
        )
        with (
            _timed_step(
                log,
                f"  round {r}/{rounds}: sub-pixel keypoint refine ({len(cloud_r)} pts)...",
            ),
            _poll_progress(log, len(cloud_r)) as counter,
        ):
            loc_r, bitmaps, valid = _refine_subpixel(
                cloud_r,
                emb_r,
                pyramids,
                base_loc,
                sweeps=subpixel,
                resolution=resolution,
                render_bitmaps=r == rounds,
                progress=counter,
            )
        if log:
            ndeg = _mean_angle_deg(n_before, _patch_normals(cloud_r))
            kpx = _mean_keypoint_shift(base_loc, loc_r)
            drop_note = f", dropped {n_dropped} grazing obs" if n_dropped else ""
            log(
                f"  round {r}/{rounds}: normal Δ {ndeg:.3f}°, "
                f"keypoint Δ {kpx:.3f}px{drop_note}"
            )
        work_recon, work_cloud, work_loc = emb_r, cloud_r, loc_r
        hashes = emb_r.image_file_hashes

    # 4. Cull under-supported points (the real min_views) plus every point the
    #    final sub-pixel pass produced no valid consensus bitmap for (finite and
    #    infinity alike), and compact into the final embedded_patches recon. The
    #    stored bitmaps are the final-keypoint consensus textures from that pass.
    with _timed_step(log, "  compacting survivors into embedded_patches..."):
        result = compact_to_embedded_patches(
            work_recon,
            work_cloud,
            work_loc,
            hashes,
            patch_bitmaps=bitmaps,
            valid=valid,
            min_views=min_views,
        )
    return result
