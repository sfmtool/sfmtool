# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ``sift_files → embedded_patches`` orchestration.

:func:`embed_patches` runs the whole photometric pipeline (see
[the pipeline spec](../../specs/core/sift-to-patch-reconstruction.md)): it
converts to the baseline embedded form, photometrically refines each point's
patch normal, selects + vets the view set per point, congeals the keypoints
(with sub-pixel refinement), then hands the results to
:func:`sfmtool._patch_compaction.compact_to_embedded_patches` — the write tail
that culls under-supported points and renumbers the survivors into a valid
``embedded_patches`` :class:`SfmrReconstruction`. The ``sfm embed-patches`` CLI
is a thin wrapper over it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sfmtool._patch_compaction import compact_to_embedded_patches
from sfmtool._progress import _poll_progress, _timed_step
from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.patches import ImagePyramidSet, PatchCloud


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


def _cull_by_localizability(
    cloud: PatchCloud,
    recon: SfmrReconstruction,
    bitmaps: np.ndarray | None,
    localizations: list[dict[str, Any]],
    max_keypoint_uncertainty: float,
    *,
    sigma_noise: float = 3.0,
) -> tuple[list[dict[str, Any]], int]:
    """Drop poorly-localized points from ``localizations`` — those whose predicted
    keypoint position uncertainty ``σ_pos`` (**patch-grid px**) exceeds
    ``max_keypoint_uncertainty`` (``τ``). See ``specs/core/patch-localizability.md``.

    ``σ_pos`` is scored from each point's cross-view consensus ``bitmaps`` (scattered
    per source point) — the same scorer the ``xform`` filter uses. It is measured in
    grid px (the intrinsic, resolution-independent unit) rather than source px so a
    fixed ``τ`` culls a comparable fraction across datasets of different resolution.
    A point that cannot be scored (empty consensus) has ``σ_pos = NaN``; ``NaN > τ``
    is ``False``, so it is kept (benefit of the doubt). Because localizability is
    intrinsic and per-point independent, removing a point here has no feedback on
    any survivor's consensus, so the cull is safe to run early. Returns
    ``(kept_localizations, n_culled)``.
    """
    if bitmaps is None or not localizations:
        return localizations, 0
    result = cloud.score_localizability(recon, bitmaps, sigma_noise=sigma_noise)
    sigma = np.asarray(result["sigma_pos_grid"], dtype=float)
    kept: list[dict[str, Any]] = []
    culled = 0
    for loc in localizations:
        pid = int(loc["point_index"])
        s = sigma[pid] if pid < len(sigma) else np.nan
        if s > max_keypoint_uncertainty:  # NaN -> False -> kept
            culled += 1
        else:
            kept.append(loc)
    return kept, culled


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
    max_keypoint_uncertainty: float = 0.35,
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
        max_keypoint_uncertainty: Cull points whose predicted keypoint position
            uncertainty ``σ_pos`` (**patch-grid px**) exceeds this ``τ``, **early**
            — right after round 1's localize + sub-pixel refine, before the
            multi-round refinement (see ``specs/core/patch-localizability.md``).
            ``σ_pos`` is the noise-normalized weak-axis structure-tensor uncertainty
            of each point's round-1 consensus (the aperture/flat blind spot the
            cross-view agreement gate misses), in grid px so a fixed ``τ`` transfers
            across resolutions. Enabling it forces the round-1 consensus render. The
            cut is a conservative tail threshold that self-limits — it removes
            egregious points where a dataset has them and little where it doesn't
            (~1-3% on well-textured sets, more where a poorly-localized tail
            exists). ``0`` (or a non-positive value) disables the cull.
        progress: Optional callable (e.g. ``click.echo``) that receives a per-round
            summary line reporting the mean normal change (deg) and mean keypoint
            shift (px); when given, those metrics are computed each round.

    Returns:
        A new ``embedded_patches`` :class:`SfmrReconstruction`, ready to ``save()``.
    """
    log = progress if callable(progress) else None
    half_extent = patch_size / 2.0
    cull_localizability = max_keypoint_uncertainty and max_keypoint_uncertainty > 0

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
        # Round-1 consensus bitmaps are needed for the final compaction on a
        # single-round run AND for the early localizability cull (below) on any
        # run; render them whenever either consumer is active.
        localizations, bitmaps, valid = _refine_subpixel(
            cloud,
            embedded,
            pyramids,
            localizations,
            sweeps=subpixel,
            resolution=resolution,
            render_bitmaps=rounds == 1 or bool(cull_localizability),
            progress=counter,
        )
    if log:
        ndeg = _mean_angle_deg(n_before, _patch_normals(cloud))
        kpx = _mean_keypoint_shift(seed_loc, localizations)
        log(
            f"  round 1/{rounds}: normal Δ {ndeg:.3f}°, keypoint Δ {kpx:.3f}px (vs seed)"
        )

    # Early localizability cull: drop points whose round-1 consensus predicts a
    # keypoint position uncertainty above τ, before the multi-round refinement that
    # dominates cost. Intrinsic + per-point independent, so an early cull has no
    # feedback on the survivors' consensus (unlike view-dropping). Removing a point
    # from `localizations` propagates cleanly through the compaction renumbering, so
    # culled points are absent from every later round and the output.
    if cull_localizability:
        localizations, n_culled = _cull_by_localizability(
            cloud, embedded, bitmaps, localizations, max_keypoint_uncertainty
        )
        if log and n_culled:
            log(
                f"  culled {n_culled} poorly-localized points "
                f"(keypoint uncertainty > {max_keypoint_uncertainty:.2f} grid px)"
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
