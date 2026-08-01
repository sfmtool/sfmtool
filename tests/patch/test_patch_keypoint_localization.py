# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test for ``PatchCloud.localize_keypoints`` against real
reconstructions.

Builds a patch cloud from a solved reconstruction, selects each point's view set
photometrically, then congeals the per-view keypoints over the real
``.sift``-derived patches and source images — the multi-view rendering + shift
search the Rust unit tests can't exercise without on-disk images. See
``specs/core/patch-keypoint-localization.md``.

``seoul_bull`` is the convex case; ``kerry_park`` (a back-to-back fisheye rig) is
the non-convex / cluttered stress case, exercised for shape and sanity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.patches import PatchCloud

from .conftest import load_images, rotation_matrices, sample_point_ids


def _project(recon, point_xyz: np.ndarray, image_idx: int):
    """Project a 3D point into image `image_idx`; ``None`` if behind the camera."""
    R = rotation_matrices(recon)[image_idx]
    t = np.asarray(recon.translations, dtype=np.float64)[image_idx]
    x_cam = R @ point_xyz + t
    # Canonical cameras look down -Z: in front means z < 0, and normalized image
    # coords divide by the depth -z (> 0 in front).
    if x_cam[2] >= 0:
        return None
    cam = recon.cameras[int(np.asarray(recon.camera_indexes)[image_idx])]
    return np.asarray(cam.project(x_cam[0] / -x_cam[2], x_cam[1] / -x_cam[2]))


def _view_sets_from_selection(cloud, recon, images, sample):
    """Photometric view sets keyed by point id, for the sampled points."""
    sel = cloud.select_views(recon, images, point_indexes=sample, resolution=12)
    return {int(r["point_index"]): np.asarray(r["admitted"]).tolist() for r in sel}


def test_localize_keypoints_convex_dataset(seoul_bull_workspace: Path):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    assert len(cloud) > 0

    sample = sample_point_ids(cloud)
    view_sets = _view_sets_from_selection(cloud, recon, images, sample)

    max_shift_px = 3.0
    results = cloud.localize_keypoints(
        recon,
        images,
        view_sets=view_sets,
        point_indexes=sample,
        resolution=12,
        max_shift_px=max_shift_px,
    )

    assert {int(r["point_index"]) for r in results} == set(sample)
    positions = np.asarray(recon.positions, dtype=np.float64)

    refined_any = False
    for r in results:
        pid = int(r["point_index"])
        views = np.asarray(r["views"], dtype=np.int64)
        kpts = np.asarray(r["keypoints"], dtype=np.float64)
        offs = np.asarray(r["offsets_px"], dtype=np.float64)
        loo = np.asarray(r["loo_zncc"], dtype=np.float64)

        # Parallel arrays, no duplicates, indices in range, kept ⊆ input set.
        assert kpts.shape == (len(views), 2)
        assert offs.shape == (len(views),)
        assert loo.shape == (len(views),)
        assert len(set(views.tolist())) == len(views)
        assert np.all(views >= 0) and np.all(views < len(images))
        assert set(views.tolist()).issubset(set(view_sets[pid]))

        for k, image_idx in enumerate(views.tolist()):
            # The reported offset is the keypoint's distance from the projection.
            proj = _project(recon, positions[pid], image_idx)
            assert proj is not None
            assert np.allclose(np.linalg.norm(kpts[k] - proj), offs[k], atol=1e-6)
            if offs[k] > 1e-3:
                refined_any = True
        # The absolute-shift gate is enforced whenever more than two views survive;
        # the two-view leave-one-out floor can retain a larger-shift pair (the spec
        # stops dropping once only two views remain).
        if len(views) > 2:
            assert np.all(offs <= max_shift_px + 1e-6), (
                f"point {pid}: a kept view exceeds max_shift_px with >2 views: {offs}"
            )
        # With two or more kept views, the leave-one-out ZNCC is finite.
        if len(views) >= 2:
            assert np.all(np.isfinite(loo))

    # The point of the algorithm: at least one keypoint actually moved.
    assert refined_any, "congealing never moved any keypoint"


def test_localize_keypoints_nonconvex_fisheye_rig(kerry_park_workspace: Path):
    """Shape / sanity on the non-convex kerry_park fisheye-rig stress case."""
    recon = SfmrReconstruction.load(kerry_park_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    assert len(cloud) > 0

    sample = sample_point_ids(cloud, n=120)
    results = cloud.localize_keypoints(
        recon, images, point_indexes=sample, resolution=12
    )

    assert {int(r["point_index"]) for r in results} == set(sample)
    for r in results:
        views = np.asarray(r["views"], dtype=np.int64)
        kpts = np.asarray(r["keypoints"], dtype=np.float64)
        offs = np.asarray(r["offsets_px"], dtype=np.float64)
        assert kpts.shape == (len(views), 2)
        assert offs.shape == (len(views),)
        assert len(set(views.tolist())) == len(views)
        assert np.all(views >= 0) and np.all(views < len(images))


def test_localize_keypoints_defaults_to_track(seoul_bull_workspace: Path):
    """With no view_sets, each point congeals over its track; kept ⊆ track."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )

    # The track view set per point.
    point_ids = np.asarray(recon.track_point_indexes)
    image_idxs = np.asarray(recon.track_image_indexes)
    tracks: dict[int, set[int]] = {}
    for pid, im in zip(point_ids.tolist(), image_idxs.tolist()):
        tracks.setdefault(int(pid), set()).add(int(im))

    sample = sample_point_ids(cloud, n=120)
    results = cloud.localize_keypoints(
        recon, images, point_indexes=sample, resolution=12
    )

    for r in results:
        pid = int(r["point_index"])
        views = set(np.asarray(r["views"], dtype=np.int64).tolist())
        assert views.issubset(tracks[pid]), (
            f"point {pid}: kept {sorted(views)} not within track {sorted(tracks[pid])}"
        )


def _tracks(recon) -> dict[int, set[int]]:
    point_ids = np.asarray(recon.track_point_indexes)
    image_idxs = np.asarray(recon.track_image_indexes)
    out: dict[int, set[int]] = {}
    for pid, im in zip(point_ids.tolist(), image_idxs.tolist()):
        out.setdefault(int(pid), set()).add(int(im))
    return out


def test_localize_keypoints_view_sets_override_is_honored(seoul_bull_workspace: Path):
    """A strict-subset override is actually applied — not silently replaced by the
    track. We find a point whose view ``v`` survives congealing over its full track,
    then re-run that point with ``v`` removed from the override and assert ``v`` is
    gone (and a point left uncovered keeps falling back to its track)."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    tracks = _tracks(recon)
    sample = sample_point_ids(cloud, n=120)

    # Baseline: localize over the full track per point.
    base = cloud.localize_keypoints(
        recon,
        images,
        view_sets={pid: sorted(tracks[pid]) for pid in sample},
        point_indexes=sample,
        resolution=12,
    )
    # Pick a point with >=3 track views where some kept view can be dropped while
    # leaving >=2 — so the override stays above the two-view floor.
    target_pid, drop_view = None, None
    for r in base:
        pid = int(r["point_index"])
        kept = np.asarray(r["views"], dtype=np.int64).tolist()
        if len(kept) >= 3 and len(tracks[pid]) >= 3:
            target_pid, drop_view = pid, kept[0]
            break
    assert target_pid is not None, "no suitable multi-view point in the sample"

    override = sorted(tracks[target_pid] - {drop_view})
    res = cloud.localize_keypoints(
        recon,
        images,
        view_sets={target_pid: override},
        point_indexes=[target_pid],
        resolution=12,
    )
    assert len(res) == 1
    got = set(np.asarray(res[0]["views"], dtype=np.int64).tolist())
    # The dropped view was kept under the full track but must be absent now, and the
    # result must stay within the override — proving the override was applied.
    assert drop_view not in got, (
        f"override not honored: {drop_view} still kept {sorted(got)}"
    )
    assert got.issubset(set(override))

    # A point left out of view_sets entirely still localizes over its track.
    uncovered = next(p for p in sample if p != target_pid)
    res2 = cloud.localize_keypoints(
        recon, images, view_sets={}, point_indexes=[uncovered], resolution=12
    )
    got2 = set(np.asarray(res2[0]["views"], dtype=np.int64).tolist())
    assert got2.issubset(tracks[uncovered])


def test_localize_keypoints_grazing_cutoff_drops_views(kerry_park_workspace: Path):
    """A strict min_grazing_cos pre-filters oblique views, so far fewer view-tiles
    survive than under a permissive cutoff (the binding plumbs the param through)."""
    recon = SfmrReconstruction.load(kerry_park_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    sample = sample_point_ids(cloud, n=120)

    def total_kept(min_grazing_cos: float) -> int:
        res = cloud.localize_keypoints(
            recon,
            images,
            point_indexes=sample,
            resolution=12,
            min_grazing_cos=min_grazing_cos,
        )
        return sum(len(r["views"]) for r in res)

    permissive = total_kept(0.0)
    strict = total_kept(0.99)  # only near-fronto views survive
    assert strict < permissive, (
        f"strict grazing cutoff should drop views: strict={strict} permissive={permissive}"
    )


def test_localize_keypoints_empty_view_set_yields_empty_arrays(
    seoul_bull_workspace: Path,
):
    """An empty view_set override yields a well-formed empty result (the binding's
    explicit (0, 2) keypoints array, not a column-inference failure)."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    pid = int(np.asarray(cloud.point_indexes)[0])
    res = cloud.localize_keypoints(
        recon, images, view_sets={pid: []}, point_indexes=[pid], resolution=12
    )
    assert len(res) == 1
    assert np.asarray(res[0]["views"]).shape == (0,)
    assert np.asarray(res[0]["keypoints"]).shape == (0, 2)
    assert np.asarray(res[0]["offsets_px"]).shape == (0,)
    assert np.asarray(res[0]["loo_zncc"]).shape == (0,)


def test_localize_keypoints_rejects_out_of_range_view_index(
    seoul_bull_workspace: Path,
):
    """An out-of-range image index in view_sets is a clean ValueError, not a panic."""
    import pytest

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    pid = int(np.asarray(cloud.point_indexes)[0])
    bad = {pid: [0, len(images)]}  # len(images) is one past the last valid index
    with pytest.raises(ValueError):
        cloud.localize_keypoints(
            recon, images, view_sets=bad, point_indexes=[pid], resolution=12
        )


def _selection(cloud, recon, images, sample):
    """The full ``select_views`` output keyed by point id, for the sampled points."""
    sel = cloud.select_views(recon, images, point_indexes=sample, resolution=12)
    return {int(r["point_index"]): r for r in sel}


def test_select_views_reports_the_track_view_count(seoul_bull_workspace: Path):
    """``track_view_count`` splits ``admitted`` into track views then vetted
    candidates — the provenance split the localizer's basis pick consumes."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    sample = sample_point_ids(cloud, n=60)

    by_pid: dict[int, set[int]] = {}
    for pid, img in zip(
        np.asarray(recon.track_point_indexes).tolist(),
        np.asarray(recon.track_image_indexes).tolist(),
    ):
        by_pid.setdefault(int(pid), set()).add(int(img))

    for pid, r in _selection(cloud, recon, images, sample).items():
        adm = np.asarray(r["admitted"]).tolist()
        t = int(r["track_view_count"])
        assert 0 <= t <= len(adm)
        # The leading `t` entries are exactly the point's (deduped) track views.
        assert set(adm[:t]) <= by_pid.get(pid, set())
        assert set(adm[t:]).isdisjoint(by_pid.get(pid, set()))


def test_localize_keypoints_basis_cap_off_matches_uncapped(seoul_bull_workspace: Path):
    """``basis_max_views=0`` (and a cap above the view count) is the uncapped
    path, byte-for-byte, even with scores supplied. The default is the capped
    ``8``, so the uncapped reference passes ``basis_max_views=0`` explicitly."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    sample = sample_point_ids(cloud, n=80)
    sel = _selection(cloud, recon, images, sample)
    view_sets = {pid: np.asarray(r["admitted"]).tolist() for pid, r in sel.items()}
    scores = {
        pid: np.asarray(r["scores"], dtype=np.float64).tolist()
        for pid, r in sel.items()
    }
    counts = {pid: int(r["track_view_count"]) for pid, r in sel.items()}

    common = dict(
        view_sets=view_sets, point_indexes=sample, resolution=12, max_shift_px=3.0
    )
    base = cloud.localize_keypoints(recon, images, basis_max_views=0, **common)
    off = cloud.localize_keypoints(
        recon,
        images,
        basis_max_views=0,
        view_scores=scores,
        track_view_counts=counts,
        **common,
    )
    wide = cloud.localize_keypoints(
        recon,
        images,
        basis_max_views=512,
        view_scores=scores,
        track_view_counts=counts,
        **common,
    )
    for other in (off, wide):
        assert len(other) == len(base)
        for a, b in zip(base, other):
            assert int(a["point_index"]) == int(b["point_index"])
            assert np.array_equal(np.asarray(a["views"]), np.asarray(b["views"]))
            assert np.array_equal(
                np.asarray(a["keypoints"]), np.asarray(b["keypoints"])
            )
            assert np.all(np.asarray(b["is_basis"]))


def test_localize_keypoints_basis_cap_keeps_the_contract(seoul_bull_workspace: Path):
    """A biting cap still localizes every admitted view it does not gate out, in
    input order, with parallel arrays and a well-formed basis/tail split."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    sample = sample_point_ids(cloud, n=80)
    sel = _selection(cloud, recon, images, sample)
    view_sets = {pid: np.asarray(r["admitted"]).tolist() for pid, r in sel.items()}
    scores = {
        pid: np.asarray(r["scores"], dtype=np.float64).tolist()
        for pid, r in sel.items()
    }
    counts = {pid: int(r["track_view_count"]) for pid, r in sel.items()}

    k = 3
    res = cloud.localize_keypoints(
        recon,
        images,
        view_sets=view_sets,
        point_indexes=sample,
        resolution=12,
        basis_max_views=k,
        view_scores=scores,
        track_view_counts=counts,
    )
    assert {int(r["point_index"]) for r in res} == set(sample)
    saw_tail = False
    for r in res:
        pid = int(r["point_index"])
        views = np.asarray(r["views"], dtype=np.int64)
        is_basis = np.asarray(r["is_basis"], dtype=bool)
        assert is_basis.shape == views.shape
        assert np.asarray(r["keypoints"]).shape == (len(views), 2)
        assert np.asarray(r["loo_zncc"]).shape == (len(views),)
        assert set(views.tolist()).issubset(set(view_sets[pid]))
        # Kept views stay in the input view-set order.
        order = {v: i for i, v in enumerate(view_sets[pid])}
        got = [order[v] for v in views.tolist()]
        assert got == sorted(got), f"point {pid} lost the input order"
        # Never more than K basis members.
        assert int(is_basis.sum()) <= max(k, 2)
        if len(view_sets[pid]) > k and (~is_basis).any():
            saw_tail = True
    assert saw_tail, "no point exercised the tail path"


def test_localize_keypoints_rejects_mismatched_view_scores(seoul_bull_workspace: Path):
    """A ``view_scores`` list that is not parallel to the point's view set is a
    clean ValueError rather than a silently mis-ranked basis."""
    import pytest

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    pid = int(np.asarray(cloud.point_indexes)[0])
    with pytest.raises(ValueError):
        cloud.localize_keypoints(
            recon,
            images,
            view_sets={pid: [0, 1]},
            view_scores={pid: [0.5]},
            point_indexes=[pid],
            resolution=12,
            basis_max_views=2,
        )


def test_localize_keypoints_rejects_unknown_basis_pick(seoul_bull_workspace: Path):
    import pytest

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    pid = int(np.asarray(cloud.point_indexes)[0])
    with pytest.raises(ValueError):
        cloud.localize_keypoints(
            recon,
            images,
            view_sets={pid: [0, 1]},
            point_indexes=[pid],
            resolution=12,
            basis_pick="nearest",
        )


def test_localize_keypoints_chunked_with_whole_cloud_view_scores(
    seoul_bull_workspace: Path,
):
    """The natural caller pattern: run ``select_views`` once over the whole
    cloud, then localize in chunks with ``point_indexes``. The score map still
    covers every point, so the parallel-length check must run against each
    point's own view set — not against the sets ``point_indexes`` cleared."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    sample = sample_point_ids(cloud, n=60)
    sel = _selection(cloud, recon, images, sample)
    view_sets = {pid: np.asarray(r["admitted"]).tolist() for pid, r in sel.items()}
    scores = {
        pid: np.asarray(r["scores"], dtype=np.float64).tolist()
        for pid, r in sel.items()
    }
    counts = {pid: int(r["track_view_count"]) for pid, r in sel.items()}

    # Localize in two chunks, passing the WHOLE score map each time.
    halves = [sample[: len(sample) // 2], sample[len(sample) // 2 :]]
    chunked = []
    for chunk in halves:
        chunked.extend(
            cloud.localize_keypoints(
                recon,
                images,
                view_sets=view_sets,
                view_scores=scores,
                track_view_counts=counts,
                point_indexes=chunk,
                resolution=12,
                basis_max_views=3,
            )
        )

    one_shot = cloud.localize_keypoints(
        recon,
        images,
        view_sets=view_sets,
        view_scores=scores,
        track_view_counts=counts,
        point_indexes=sample,
        resolution=12,
        basis_max_views=3,
    )

    assert {int(r["point_index"]) for r in chunked} == set(sample)
    by_pid = {int(r["point_index"]): r for r in chunked}
    for r in one_shot:
        c = by_pid[int(r["point_index"])]
        assert np.array_equal(np.asarray(r["views"]), np.asarray(c["views"]))
        assert np.array_equal(np.asarray(r["keypoints"]), np.asarray(c["keypoints"]))
        assert np.array_equal(np.asarray(r["is_basis"]), np.asarray(c["is_basis"]))
