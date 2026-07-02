# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test for ``PatchCloud.refine_keypoints`` (the continuous ECC /
Gauss-Newton subpixel refiner) against a real reconstruction.

Builds a patch cloud from a solved reconstruction and refines the per-view
keypoints over the real ``.sift``-derived patches and source images — the
multi-view rendering + continuous solve the Rust unit tests exercise only
synthetically. See ``specs/core/keypoint-subpixel-refinement.md``.

The refiner is a *local* refiner that changes no view membership and is never
worse than the seed; the checks here are the binding's array contract, the
unchanged view set, and the never-worse-than-seed guarantee (compared against a
zero-step "seed only" run on the same data).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from sfmtool._sfmtool import PatchCloud, SfmrReconstruction


def _load_images(recon) -> list[np.ndarray]:
    import cv2  # heavy module, only needed by this integration test

    ws = recon.workspace_dir
    images = []
    for name in recon.image_names:
        bgr = cv2.imread(os.path.join(ws, name), cv2.IMREAD_COLOR)
        assert bgr is not None, f"could not read {name}"
        images.append(np.ascontiguousarray(bgr))
    return images


def _sample_point_ids(cloud, n: int = 200, seed: int = 0) -> list[int]:
    ids = np.asarray(cloud.point_indexes)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ids, size=min(n, len(ids)), replace=False)).tolist()


def _rotation_matrices(recon) -> np.ndarray:
    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = w * w + x * x + y * y + z * z
    s = np.where(n > 0, 2.0 / n, 0.0)
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - s * (y * y + z * z)
    R[:, 0, 1] = s * (x * y - z * w)
    R[:, 0, 2] = s * (x * z + y * w)
    R[:, 1, 0] = s * (x * y + z * w)
    R[:, 1, 1] = 1 - s * (x * x + z * z)
    R[:, 1, 2] = s * (y * z - x * w)
    R[:, 2, 0] = s * (x * z - y * w)
    R[:, 2, 1] = s * (y * z + x * w)
    R[:, 2, 2] = 1 - s * (x * x + y * y)
    return R


def _project(recon, point_xyz: np.ndarray, image_idx: int):
    R = _rotation_matrices(recon)[image_idx]
    t = np.asarray(recon.translations, dtype=np.float64)[image_idx]
    x_cam = R @ point_xyz + t
    if x_cam[2] <= 0:
        return None
    cam = recon.cameras[int(np.asarray(recon.camera_indexes)[image_idx])]
    return np.asarray(cam.project(x_cam[0] / x_cam[2], x_cam[1] / x_cam[2]))


def test_refine_keypoints_array_contract_and_never_worse(seoul_bull_workspace: Path):
    # `refine_keypoints` requires starting keypoints (a local refiner needs to
    # seed in the basin of the true optimum, and the projection isn't a "real"
    # keypoint). Embed first so the recon carries inline per-observation
    # keypoints that the default seeding can use.
    recon = SfmrReconstruction.load(seoul_bull_workspace).to_embedded_patches(
        extent_value=5.0
    )
    images = _load_images(recon)
    cloud = recon.patches
    assert cloud is not None and len(cloud) > 0

    sample = _sample_point_ids(cloud)
    view_sets = {
        int(r["point_index"]): np.asarray(r["admitted"]).tolist()
        for r in cloud.select_views(recon, images, point_indexes=sample, resolution=12)
    }

    common = dict(
        recon=recon,
        images=images,
        view_sets=view_sets,
        point_indexes=sample,
        resolution=12,
    )
    # Seed-only baseline (no GN steps) and the full continuous refine.
    seed = cloud.refine_keypoints(**common, max_gn_steps=0)
    refined = cloud.refine_keypoints(**common, max_gn_steps=10)

    assert {int(r["point_index"]) for r in refined} == set(sample)
    seed_by_pid = {int(r["point_index"]): r for r in seed}
    positions = np.asarray(recon.positions, dtype=np.float64)

    moved_any = False
    improved_any = False
    for r in refined:
        pid = int(r["point_index"])
        views = np.asarray(r["views"], dtype=np.int64)
        kpts = np.asarray(r["keypoints"], dtype=np.float64)
        offs = np.asarray(r["offsets_px"], dtype=np.float64)
        scores = np.asarray(r["scores"], dtype=np.float64)

        # Parallel arrays, no duplicates, indices in range.
        assert kpts.shape == (len(views), 2)
        assert offs.shape == (len(views),)
        assert scores.shape == (len(views),)
        assert len(set(views.tolist())) == len(views)
        assert np.all(views >= 0) and np.all(views < len(images))

        # View set is UNCHANGED versus the seed run (no membership change, same order).
        sviews = np.asarray(seed_by_pid[pid]["views"], dtype=np.int64)
        assert np.array_equal(views, sviews), (
            f"point {pid}: refine changed the view set/order"
        )

        # The reported offset is the keypoint's distance from the projection.
        for k, image_idx in enumerate(views.tolist()):
            proj = _project(recon, positions[pid], image_idx)
            assert proj is not None
            assert np.allclose(np.linalg.norm(kpts[k] - proj), offs[k], atol=1e-6)

        # Never worse than the seed: every finite refined score is >= its seed score.
        sscores = np.asarray(seed_by_pid[pid]["scores"], dtype=np.float64)
        both_finite = np.isfinite(scores) & np.isfinite(sscores)
        assert np.all(scores[both_finite] >= sscores[both_finite] - 1e-6), (
            f"point {pid}: a refined score dropped below the seed"
        )
        if np.any(scores[both_finite] > sscores[both_finite] + 1e-4):
            improved_any = True

        skpts = np.asarray(seed_by_pid[pid]["keypoints"], dtype=np.float64)
        if skpts.shape == kpts.shape and np.any(
            np.linalg.norm(kpts - skpts, axis=1) > 1e-3
        ):
            moved_any = True

    # The refiner must actually do something on real data.
    assert moved_any, "refinement never moved any keypoint"
    assert improved_any, "refinement never improved any view's ECC score"


def test_refine_keypoints_defaults_to_track(seoul_bull_workspace: Path):
    """With no view_sets, each point refines over its track; the view set is the
    track's views, in order, unchanged by the local refiner."""
    recon = SfmrReconstruction.load(seoul_bull_workspace).to_embedded_patches(
        extent_value=5.0
    )
    images = _load_images(recon)
    cloud = recon.patches
    assert cloud is not None
    point_ids = np.asarray(recon.track_point_indexes)
    image_idxs = np.asarray(recon.track_image_indexes)
    tracks: dict[int, set[int]] = {}
    for pid, im in zip(point_ids.tolist(), image_idxs.tolist()):
        tracks.setdefault(int(pid), set()).add(int(im))

    sample = _sample_point_ids(cloud, n=120)
    results = cloud.refine_keypoints(recon, images, point_indexes=sample, resolution=12)
    for r in results:
        pid = int(r["point_index"])
        views = set(np.asarray(r["views"], dtype=np.int64).tolist())
        assert views.issubset(tracks[pid])


def test_refine_keypoints_empty_view_set_yields_empty_arrays(
    seoul_bull_workspace: Path,
):
    recon = SfmrReconstruction.load(seoul_bull_workspace).to_embedded_patches(
        extent_value=5.0
    )
    images = _load_images(recon)
    cloud = recon.patches
    assert cloud is not None
    pid = int(np.asarray(cloud.point_indexes)[0])
    res = cloud.refine_keypoints(
        recon, images, view_sets={pid: []}, point_indexes=[pid], resolution=12
    )
    assert len(res) == 1
    assert np.asarray(res[0]["views"]).shape == (0,)
    assert np.asarray(res[0]["keypoints"]).shape == (0, 2)
    assert np.asarray(res[0]["offsets_px"]).shape == (0,)
    assert np.asarray(res[0]["scores"]).shape == (0,)


def test_refine_keypoints_rejects_out_of_range_view_index(seoul_bull_workspace: Path):
    import pytest

    recon = SfmrReconstruction.load(seoul_bull_workspace).to_embedded_patches(
        extent_value=5.0
    )
    images = _load_images(recon)
    cloud = recon.patches
    assert cloud is not None
    pid = int(np.asarray(cloud.point_indexes)[0])
    bad = {pid: [0, len(images)]}
    with pytest.raises(ValueError):
        cloud.refine_keypoints(
            recon, images, view_sets=bad, point_indexes=[pid], resolution=12
        )


def test_refine_keypoints_rejects_sift_files_recon_without_starting_keypoints(
    seoul_bull_workspace: Path,
):
    """The strict requirement: a local refiner needs starting keypoints. A
    sift-files recon without explicit `starting_keypoints` errors fast — before
    any pyramid decode — so callers don't accidentally refine off the bare
    projection."""
    import pytest

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    assert recon.feature_source == "sift_files"
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    # Pass no images so the test verifies the require-check fires before any
    # pyramid decode (an empty list would otherwise produce a count mismatch).
    with pytest.raises(
        ValueError,
        match="refine_keypoints requires starting keypoints",
    ):
        cloud.refine_keypoints(recon, [], resolution=12)


def test_refine_keypoints_honors_starting_keypoints(seoul_bull_workspace: Path):
    """``starting_keypoints`` shifts the GN seed off the recon's default
    inline stored keypoint: a refinement seeded ~0.5 px away lands somewhere
    subtly different from the stored-seeded refinement, because GN's basin
    is *local* — the seed direction biases which side of the photometric
    peak it climbs from.

    The check is the binding contract: a different seed produces a different
    refined keypoint for at least one view, and ``len(seeds) != len(views)``
    is rejected.
    """
    import pytest

    recon = SfmrReconstruction.load(seoul_bull_workspace).to_embedded_patches(
        extent_value=5.0
    )
    images = _load_images(recon)
    cloud = recon.patches
    assert cloud is not None
    sample = _sample_point_ids(cloud, n=50)
    view_sets = {
        int(r["point_index"]): np.asarray(r["admitted"]).tolist()
        for r in cloud.select_views(recon, images, point_indexes=sample, resolution=12)
    }
    common = dict(
        recon=recon,
        images=images,
        view_sets=view_sets,
        point_indexes=sample,
        resolution=12,
        max_gn_steps=10,
    )

    # Baseline: seeds default to the recon's inline stored keypoints (the
    # SIFT detections an embedded_patches recon carries inline).
    baseline = cloud.refine_keypoints(**common)
    baseline_by_pid = {int(r["point_index"]): r for r in baseline}
    stored_xy = np.asarray(recon.keypoints_xy, dtype=np.float64)
    track_pids = np.asarray(recon.track_point_indexes, dtype=np.int64)
    track_imgs = np.asarray(recon.track_image_indexes, dtype=np.int64)
    kp_by_obs = {
        (int(p), int(i)): stored_xy[k]
        for k, (p, i) in enumerate(zip(track_pids, track_imgs))
    }
    positions = np.asarray(recon.positions, dtype=np.float64)

    # Build seeds offset 0.5 px in +x from each view's stored keypoint (or
    # the projection for views without a stored observation, the same
    # per-view fallback the recon-default uses). Only the subset of points
    # with at least 2 views (the only ones the refiner actually GN-steps)
    # is in scope; for the rest the refiner short-circuits to "no
    # consensus" and the seed change is invisible.
    seeds: dict[int, list[list[float]]] = {}
    seeded_pids: list[int] = []
    for r in baseline:
        pid = int(r["point_index"])
        views = view_sets[pid]
        if len(views) < 2:
            continue
        per_view = []
        for image_idx in views:
            stored = kp_by_obs.get((pid, int(image_idx)))
            if stored is not None:
                per_view.append([float(stored[0]) + 0.5, float(stored[1])])
                continue
            proj = _project(recon, positions[pid], image_idx)
            if proj is None:
                # The projection gate would drop this view anyway; seed at
                # something arbitrary, the refiner won't read it.
                per_view.append([0.0, 0.0])
            else:
                per_view.append([float(proj[0]) + 0.5, float(proj[1])])
        seeds[pid] = per_view
        seeded_pids.append(pid)

    assert seeded_pids, "no multi-view point to seed-shift"

    shifted = cloud.refine_keypoints(**common, starting_keypoints=seeds)

    # At least one point's refined keypoints actually move when the seed is
    # shifted. This is the binding contract (seeds are honored), not a
    # quantitative claim about which local optimum the refiner finds.
    moved_any = False
    for r in shifted:
        pid = int(r["point_index"])
        if pid not in seeds:
            continue
        b = baseline_by_pid[pid]
        b_kpts = np.asarray(b["keypoints"], dtype=np.float64).reshape(-1, 2)
        s_kpts = np.asarray(r["keypoints"], dtype=np.float64).reshape(-1, 2)
        if b_kpts.shape == s_kpts.shape and np.any(
            np.linalg.norm(b_kpts - s_kpts, axis=1) > 1e-3
        ):
            moved_any = True
            break
    assert moved_any, (
        "shifting the GN seed by 0.5 px never changed any refined keypoint"
    )

    # Length mismatch is rejected up front (per-point seeds must be parallel
    # to that point's view set).
    bad_pid = seeded_pids[0]
    bad_seeds = {bad_pid: seeds[bad_pid][:-1]}  # one short
    with pytest.raises(
        ValueError,
        match=r"starting_keypoints\[\d+\] has \d+ seeds but the view set has \d+ views",
    ):
        cloud.refine_keypoints(**common, starting_keypoints=bad_seeds)

    # Unknown pid (not a point in this patch cloud) is rejected with a clear
    # message — silently passing it through hid bugs in callers that built
    # seed maps off a stale recon.
    max_pid = int(np.asarray(cloud.point_indexes).max())
    unknown_pid = max_pid + 1000
    with pytest.raises(
        ValueError,
        match=r"starting_keypoints\[\d+\] is not a point in this patch cloud",
    ):
        cloud.refine_keypoints(
            **common,
            starting_keypoints={unknown_pid: [[0.0, 0.0]]},
        )

    # A pid that exists in the cloud but is excluded by ``point_ids`` is also
    # rejected (rather than producing the confusing "has K seeds but view set
    # has 0 views" message the cleared set would otherwise yield).
    excluded_pid = next(
        int(p) for p in np.asarray(cloud.point_indexes).tolist() if p not in sample
    )
    with pytest.raises(
        ValueError, match=r"starting_keypoints\[\d+\] is excluded by point_indexes"
    ):
        cloud.refine_keypoints(
            **common,
            starting_keypoints={excluded_pid: [[0.0, 0.0]]},
        )


def test_refine_keypoints_default_seeds_from_embedded_recon(
    seoul_bull_workspace: Path,
):
    """On an embedded_patches recon, the default (``starting_keypoints=None``)
    seeds each view at that observation's inline keypoint — not the projection.
    A zero-step "seed only" refinement reproduces the recon's stored keypoints
    bit-for-bit, confirming the seed source.
    """
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    emb = recon.to_embedded_patches(extent_value=5.0)
    images = _load_images(emb)
    cloud = emb.patches
    assert cloud is not None

    stored_xy = np.asarray(emb.keypoints_xy, dtype=np.float64)
    track_pids = np.asarray(emb.track_point_indexes, dtype=np.int64)
    track_imgs = np.asarray(emb.track_image_indexes, dtype=np.int64)
    kp_by_obs = {
        (int(p), int(i)): stored_xy[k]
        for k, (p, i) in enumerate(zip(track_pids, track_imgs))
    }

    sample = _sample_point_ids(cloud, n=50)
    seed_only = cloud.refine_keypoints(
        emb, images, point_indexes=sample, resolution=12, max_gn_steps=0
    )

    # Every (point_id, view) the seed-only refinement reports lands at the
    # inline stored keypoint — the default seed for an embedded_patches recon
    # (projection only for views with no inline observation, which is none of
    # the per-track views here).
    checked = 0
    for r in seed_only:
        pid = int(r["point_index"])
        views = np.asarray(r["views"], dtype=np.int64)
        kpts = np.asarray(r["keypoints"], dtype=np.float64).reshape(-1, 2)
        for k, img in enumerate(views.tolist()):
            stored = kp_by_obs.get((pid, int(img)))
            if stored is None:
                continue  # view added by some non-track admission; skip
            assert np.allclose(kpts[k], stored, atol=1e-9), (
                f"seed-only refinement at (pid={pid}, image={img}) returned "
                f"{kpts[k].tolist()}, expected stored {stored.tolist()}"
            )
            checked += 1
    assert checked > 0, "no (pid, view) pair had a stored keypoint to check"


def test_refine_keypoints_default_falls_back_to_projection_for_non_track_view(
    seoul_bull_workspace: Path,
):
    """The third branch of the seed-source ladder: on an embedded_patches recon,
    a view that's in the refiner's view set but NOT in the point's SIFT track
    has no inline stored keypoint, so its seed falls back to the projection
    ``project_i(X_p)``. Track views in the same call still seed at their
    inline keypoint (covered by the test above); this one pins the per-view
    fall-through within the recon-default branch.
    """
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    emb = recon.to_embedded_patches(extent_value=5.0)
    images = _load_images(emb)
    cloud = emb.patches
    assert cloud is not None

    stored_xy = np.asarray(emb.keypoints_xy, dtype=np.float64)
    track_pids = np.asarray(emb.track_point_indexes, dtype=np.int64)
    track_imgs = np.asarray(emb.track_image_indexes, dtype=np.int64)
    track_by_pid: dict[int, set[int]] = {}
    for p, i in zip(track_pids.tolist(), track_imgs.tolist()):
        track_by_pid.setdefault(int(p), set()).add(int(i))
    kp_by_obs = {
        (int(p), int(i)): stored_xy[k]
        for k, (p, i) in enumerate(zip(track_pids, track_imgs))
    }
    positions = np.asarray(emb.positions, dtype=np.float64)

    # Find a (pid, non_track_img) pair where the refiner will actually keep
    # the view. ``_project`` only checks ``z > 0``, but the refiner also gates
    # on front-facing + projection-in-frame; the cleanest way to get a view
    # that survives all the refiner's guards is to ask ``select_views`` for
    # the photometrically-vetted candidate set and pick a non-track admitted
    # view. ``select_views`` admits the SIFT track plus extra views that
    # pass its visibility + ZNCC checks — guaranteed visible to the refiner.
    sample = _sample_point_ids(cloud, n=120)
    selections = cloud.select_views(emb, images, point_indexes=sample, resolution=12)
    pid, non_track_img, projection = None, None, None
    for sel in selections:
        candidate_pid = int(sel["point_index"])
        admitted = set(np.asarray(sel["admitted"], dtype=np.int64).tolist())
        candidates = admitted - track_by_pid.get(candidate_pid, set())
        for img in sorted(candidates):
            proj = _project(emb, positions[candidate_pid], img)
            if proj is None:
                continue
            pid, non_track_img, projection = candidate_pid, img, proj
            break
        if pid is not None:
            break
    assert pid is not None, (
        "select_views admitted no non-track candidate view on the sampled points; "
        "either pick a larger sample or another dataset"
    )

    # Construct a view set that mixes one track view (seeded at its stored
    # keypoint) and the non-track view (which must seed at the projection).
    track_img = next(iter(track_by_pid[pid]))
    view_sets = {pid: [track_img, non_track_img]}
    seed_only = cloud.refine_keypoints(
        emb,
        images,
        view_sets=view_sets,
        point_indexes=[pid],
        resolution=12,
        max_gn_steps=0,
    )
    [r] = seed_only
    views_out = np.asarray(r["views"], dtype=np.int64).tolist()
    kpts_out = np.asarray(r["keypoints"], dtype=np.float64).reshape(-1, 2)
    seed_by_img = dict(zip(views_out, kpts_out))

    # Track view: seeded at the inline stored keypoint.
    assert np.allclose(
        seed_by_img[track_img], kp_by_obs[(pid, track_img)], atol=1e-9
    ), (
        f"track view {track_img} of pid {pid} should seed at the stored keypoint "
        f"{kp_by_obs[(pid, track_img)].tolist()}, got {seed_by_img[track_img].tolist()}"
    )
    # Non-track view: no stored keypoint; per-view fall-through to projection.
    assert np.allclose(seed_by_img[non_track_img], projection, atol=1e-6), (
        f"non-track view {non_track_img} of pid {pid} should seed at the projection "
        f"{projection.tolist()}, got {seed_by_img[non_track_img].tolist()}"
    )


def test_refine_keypoints_explicit_seeds_override_recon_default_on_embedded(
    seoul_bull_workspace: Path,
):
    """The override branch of the seed-source ladder on an embedded_patches
    recon: explicit ``starting_keypoints`` for a point silence the recon-default
    (the inline stored keypoint) and the refiner takes the explicit seed
    instead. Verified by a ``max_gn_steps=0`` "seed only" refine: the result
    keypoints for overridden points must equal the override, not the stored
    inline keypoint.
    """
    recon = SfmrReconstruction.load(seoul_bull_workspace).to_embedded_patches(
        extent_value=5.0
    )
    images = _load_images(recon)
    cloud = recon.patches
    assert cloud is not None

    stored_xy = np.asarray(recon.keypoints_xy, dtype=np.float64)
    track_pids = np.asarray(recon.track_point_indexes, dtype=np.int64)
    track_imgs = np.asarray(recon.track_image_indexes, dtype=np.int64)
    kp_by_obs = {
        (int(p), int(i)): stored_xy[k]
        for k, (p, i) in enumerate(zip(track_pids, track_imgs))
    }
    track_by_pid: dict[int, list[int]] = {}
    for p, i in zip(track_pids.tolist(), track_imgs.tolist()):
        track_by_pid.setdefault(int(p), []).append(int(i))

    # Pick a point with at least two track views, build explicit seeds that
    # are 0.5 px in +x off the stored keypoints.
    pid = next(p for p, views in track_by_pid.items() if len(views) >= 2)
    views = track_by_pid[pid]
    override = [
        [float(kp_by_obs[(pid, img)][0] + 0.5), float(kp_by_obs[(pid, img)][1])]
        for img in views
    ]

    seed_only = cloud.refine_keypoints(
        recon,
        images,
        view_sets={pid: views},
        point_indexes=[pid],
        starting_keypoints={pid: override},
        resolution=12,
        max_gn_steps=0,
    )
    [r] = seed_only
    views_out = np.asarray(r["views"], dtype=np.int64).tolist()
    kpts_out = np.asarray(r["keypoints"], dtype=np.float64).reshape(-1, 2)
    for k, img in enumerate(views_out):
        stored = kp_by_obs[(pid, img)]
        expected_override = np.array(
            [float(stored[0] + 0.5), float(stored[1])], dtype=np.float64
        )
        assert np.allclose(kpts_out[k], expected_override, atol=1e-9), (
            f"override should win over recon-default at (pid={pid}, image={img}): "
            f"expected {expected_override.tolist()}, got {kpts_out[k].tolist()} "
            f"(stored is {stored.tolist()})"
        )
        assert not np.allclose(kpts_out[k], stored, atol=1e-3), (
            f"override at (pid={pid}, image={img}) silently fell back to stored"
        )


def test_refine_keypoints_render_bitmaps_returns_consensus_bitmaps(
    seoul_bull_workspace: Path,
):
    """``render_bitmaps=True`` adds a per-point ``bitmap``: an ``(R, R, 4)`` uint8
    consensus texture fused at the final keypoints for a point with a valid
    cross-view consensus, or ``None`` for one without (fewer than two usable
    views — the culled-point signal). Without the flag the key is absent, so the
    existing return shape is preserved."""
    recon = SfmrReconstruction.load(seoul_bull_workspace).to_embedded_patches(
        extent_value=5.0
    )
    images = _load_images(recon)
    cloud = recon.patches
    assert cloud is not None
    sample = _sample_point_ids(cloud, n=40)

    plain = cloud.refine_keypoints(recon, images, point_indexes=sample, resolution=12)
    assert all("bitmap" not in r for r in plain)

    results = cloud.refine_keypoints(
        recon, images, point_indexes=sample, resolution=12, render_bitmaps=True
    )
    got_bitmap = False
    for r in results:
        assert "bitmap" in r
        bm = r["bitmap"]
        if len(np.asarray(r["views"])) < 2:
            assert bm is None, "no cross-view consensus below two views"
            continue
        if bm is None:
            continue  # a view can drop out at its final offset (render gate)
        bm = np.asarray(bm)
        assert bm.shape == (12, 12, 4)
        assert bm.dtype == np.uint8
        if bm[..., 3].any():
            got_bitmap = True
    assert got_bitmap, "no sampled point produced a nonzero consensus bitmap"
