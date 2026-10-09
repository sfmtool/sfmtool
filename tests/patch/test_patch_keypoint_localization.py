# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration test for ``PatchCloud.localize_keypoints`` against real
reconstructions.

Builds a patch cloud from a real reconstruction, selects each point's view set
photometrically, then congeals the per-view keypoints over the real
``.sift``-derived patches and source images — the multi-view rendering + shift
search the Rust unit tests can't exercise without on-disk images. See
``specs/core/patch/patch-keypoint-localization.md``.

``seoul_bull`` is the convex case; ``kerry_park`` (a back-to-back fisheye rig) is
the non-convex / cluttered stress case, exercised for shape and sanity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.patches import PatchCloud

from ..conftest import points_with_oblique_view
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
        zncc = np.asarray(r["zncc"], dtype=np.float64)
        ref = r["reference_image"]

        # Parallel arrays, no duplicates, indices in range, kept ⊆ input set.
        assert kpts.shape == (len(views), 2)
        assert offs.shape == (len(views),)
        assert zncc.shape == (len(views),)
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
        # The absolute-shift gate holds for every kept view but the reference,
        # which it does not judge.
        others = views != (-1 if ref is None else int(ref))
        assert np.all(offs[others] <= max_shift_px + 1e-6), (
            f"point {pid}: a kept view exceeds max_shift_px: {offs}"
        )
        # Aligned to a reference: it is kept with a score of 1, and every other
        # kept view was scored against its render.
        if ref is not None:
            assert int(ref) in views.tolist()
            assert zncc[~others] == 1.0
            assert np.all(np.isfinite(zncc))

    # The point of the algorithm: at least one keypoint actually moved.
    assert refined_any, "the alignment never moved any keypoint"


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


def test_localize_keypoints_view_sets_override_is_honored(
    seoul_bull_workspace: Path,
):
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
    # A cutoff can only drop a view that is oblique to the patch normal, so
    # sample from the points that have one (the fixture guarantees they exist).
    # An unrestricted 120-point draw can land entirely on narrow-baseline tracks
    # -- every ray within ~8 deg of the mean viewing direction, which 0.99 does
    # not cut -- and then strict == permissive on a correct reconstruction.
    sample = sample_point_ids(cloud, n=120, restrict_to=points_with_oblique_view(recon))
    assert sample, (
        "no cloud point has an observation more than 10 deg off its mean viewing "
        "direction; kerry_park_workspace_once guarantees MIN_OBLIQUE_POINTS of them"
    )

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
    assert np.asarray(res[0]["zncc"]).shape == (0,)
    assert res[0]["reference_image"] is None


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


def test_select_views_reports_the_track_view_count(
    seoul_bull_workspace: Path,
):
    """``track_view_count`` splits ``admitted`` into track views then vetted
    candidates."""
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


def _seeded_run(cloud, recon, images, view_sets, sample, seeds):
    return {
        int(r["point_index"]): r
        for r in cloud.localize_keypoints(
            recon,
            images,
            view_sets=view_sets,
            starting_keypoints=seeds,
            point_indexes=sample,
            resolution=12,
        )
    }


def test_localize_keypoints_per_view_optional_seeds(
    seoul_bull_workspace: Path,
):
    """``starting_keypoints`` takes a per-view ``None``: that view seeds at the
    point's projection while its siblings keep their explicit seeds. An all-``None``
    table therefore reproduces the unseeded run, and displacing one view's seed
    moves only what a seed can move."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    sample = sample_point_ids(cloud, n=80)
    view_sets = _view_sets_from_selection(cloud, recon, images, sample)
    sample = [pid for pid in sample if len(view_sets.get(pid, [])) >= 2]
    assert sample, "no multi-view point in the sample"
    positions = np.asarray(recon.positions, dtype=np.float64)

    unseeded = _seeded_run(cloud, recon, images, view_sets, sample, None)

    # 1. Every view unseeded via the table == no table at all.
    all_none = _seeded_run(
        cloud,
        recon,
        images,
        view_sets,
        sample,
        {pid: [None] * len(view_sets[pid]) for pid in sample},
    )
    for pid, r in unseeded.items():
        assert np.array_equal(
            np.asarray(all_none[pid]["views"]), np.asarray(r["views"])
        )
        assert np.array_equal(
            np.asarray(all_none[pid]["keypoints"]), np.asarray(r["keypoints"])
        )

    # 2. A mixed table: the first view of each point is seeded a few px off its
    #    projection, every other view is left at `None`. The seeds must be read
    #    (some point's result moves) and the un-seeded views must still be
    #    localized (no point loses its whole view set to a length/parse error).
    mixed: dict[int, list[list[float] | None]] = {}
    for pid in sample:
        views = view_sets[pid]
        seeds: list[list[float] | None] = [None] * len(views)
        p = _project(recon, positions[pid], int(views[0]))
        if p is not None:
            seeds[0] = [float(p[0]) + 2.0, float(p[1])]
        mixed[pid] = seeds
    seeded = _seeded_run(cloud, recon, images, view_sets, sample, mixed)

    assert set(seeded) == set(unseeded)
    moved = 0
    for pid, r in seeded.items():
        base = unseeded[pid]
        kb = {
            int(v): np.asarray(base["keypoints"])[i]
            for i, v in enumerate(np.asarray(base["views"]).tolist())
        }
        for i, v in enumerate(np.asarray(r["views"]).tolist()):
            b = kb.get(int(v))
            if b is None:
                moved += 1
                continue
            if np.hypot(*(np.asarray(r["keypoints"])[i] - b)) > 0.1:
                moved += 1
    assert moved > 0, "a displaced per-view seed changed nothing"


def test_localize_keypoints_rejects_mismatched_starting_keypoints(
    seoul_bull_workspace: Path,
):
    """The parallel-length check is unchanged by the per-view ``None``: a seed
    list that is not parallel to the point's view set is still a clean
    ValueError."""
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
            starting_keypoints={pid: [None]},
            point_indexes=[pid],
            resolution=12,
        )


def test_localize_keypoints_chunked_with_whole_cloud_reference_images(
    seoul_bull_workspace: Path,
):
    """The natural caller pattern: one ``reference_images`` map over the whole
    cloud, localized in chunks with ``point_indexes``. Each chunk reads only its
    own points' entries, so the chunks give what one call gives."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    sample = sample_point_ids(cloud, n=60)
    view_sets = _view_sets_from_selection(cloud, recon, images, sample)
    references = {pid: views[-1] for pid, views in view_sets.items() if views}

    halves = [sample[: len(sample) // 2], sample[len(sample) // 2 :]]
    chunked = []
    for chunk in halves:
        chunked.extend(
            cloud.localize_keypoints(
                recon,
                images,
                view_sets=view_sets,
                reference_images=references,
                point_indexes=chunk,
                resolution=12,
            )
        )
    one_shot = cloud.localize_keypoints(
        recon,
        images,
        view_sets=view_sets,
        reference_images=references,
        point_indexes=sample,
        resolution=12,
    )

    assert {int(r["point_index"]) for r in chunked} == set(sample)
    by_pid = {int(r["point_index"]): r for r in chunked}
    for r in one_shot:
        c = by_pid[int(r["point_index"])]
        assert np.array_equal(np.asarray(r["views"]), np.asarray(c["views"]))
        assert np.array_equal(np.asarray(r["keypoints"]), np.asarray(c["keypoints"]))
        assert r["reference_image"] == c["reference_image"]


def _stored_seeds(recon, pid: int) -> tuple[list[int], list[list[float]]]:
    """A point's track images and the keypoints stored for them, in track
    order, from an ``embedded_patches`` recon."""
    pts = np.asarray(recon.track_point_indexes)
    rows = np.flatnonzero(pts == pid)
    imgs = np.asarray(recon.track_image_indexes)[rows].astype(int).tolist()
    kxy = np.asarray(recon.keypoints_xy, dtype=np.float64)[rows]
    return imgs, kxy.tolist()


def _distinct_track_points(recon, cloud, n: int) -> list[int]:
    """Up to ``n`` cloud points with at least three observations, each in a
    different image."""
    pts = np.asarray(recon.track_point_indexes)
    imgs = np.asarray(recon.track_image_indexes)
    out = []
    for pid in np.asarray(cloud.point_indexes).tolist():
        views = imgs[pts == pid]
        if len(views) >= 3 and len(set(views.tolist())) == len(views):
            out.append(int(pid))
        if len(out) == n:
            break
    return out


def test_reference_keypoint_is_not_moved(seoul_bull_workspace: Path):
    """The reference observation named in ``reference_images`` is the one the
    views are aligned to: ``localize_keypoints`` and ``refine_keypoints`` both
    report it, keep it, and return its keypoint exactly as it was given."""
    recon = SfmrReconstruction.load(seoul_bull_workspace).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )
    images = load_images(recon)
    cloud = recon.patches
    pids = _distinct_track_points(recon, cloud, 12)
    assert pids

    view_sets, seeds, references = {}, {}, {}
    for k, pid in enumerate(pids):
        imgs, kpts = _stored_seeds(recon, pid)
        view_sets[pid], seeds[pid] = imgs, kpts
        # Not always the first view, so the reference is not just the default.
        references[pid] = imgs[k % len(imgs)]

    localized = cloud.localize_keypoints(
        recon,
        images,
        view_sets=view_sets,
        starting_keypoints=seeds,
        reference_images=references,
        point_indexes=pids,
        resolution=12,
        # The grazing pre-filter can turn the named reference away, and the
        # rule then picks another; off, every named reference is used.
        min_grazing_cos=0.0,
    )
    refined = cloud.refine_keypoints(
        recon,
        images,
        view_sets=view_sets,
        starting_keypoints=seeds,
        reference_images=references,
        point_indexes=pids,
        resolution=12,
        render_bitmaps=True,
    )
    for name, results, score_key in [
        ("localize_keypoints", localized, "zncc"),
        ("refine_keypoints", refined, "scores"),
    ]:
        assert {int(r["point_index"]) for r in results} == set(pids)
        moved_any = False
        for r in results:
            pid = int(r["point_index"])
            ref = references[pid]
            assert r["reference_image"] == ref, name
            views = np.asarray(r["views"]).astype(int).tolist()
            assert ref in views, name
            k = views.index(ref)
            given = seeds[pid][view_sets[pid].index(ref)]
            got = np.asarray(r["keypoints"], dtype=np.float64)[k]
            assert np.array_equal(got, np.asarray(given)), (name, pid, got, given)
            assert np.asarray(r[score_key])[k] == 1.0, name
            for j, v in enumerate(views):
                if v != ref:
                    seed = np.asarray(seeds[pid][view_sets[pid].index(v)])
                    moved_any |= not np.array_equal(
                        np.asarray(r["keypoints"], dtype=np.float64)[j], seed
                    )
        assert moved_any, f"{name} moved no view but the reference either"


def test_reference_outside_the_view_set_falls_back_to_the_rule(
    seoul_bull_workspace: Path,
):
    """A ``reference_images`` entry whose image is not in the point's view set
    is set aside, and the reference-view rule picks, exactly as with no entry."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=5.0
    )
    sample = sample_point_ids(cloud, n=20)
    view_sets = _view_sets_from_selection(cloud, recon, images, sample)
    outside = {
        pid: next(i for i in range(len(images)) if i not in views)
        for pid, views in view_sets.items()
    }
    common = dict(view_sets=view_sets, point_indexes=sample, resolution=12)
    ruled = cloud.localize_keypoints(recon, images, **common)
    fallback = cloud.localize_keypoints(
        recon, images, reference_images=outside, **common
    )
    for a, b in zip(ruled, fallback):
        assert a["reference_image"] == b["reference_image"]
        assert np.array_equal(np.asarray(a["keypoints"]), np.asarray(b["keypoints"]))


# The member self-similarity gate's bar in the flat-member test: the default,
# set explicitly so the test reads the bar it judges by.
MEMBER_GATE = 2.5


def _healthy_two_view_point(recon, cloud, images) -> tuple[int, list[int]]:
    """A cloud point and two of its observing views that survive the default
    gates, with the member gate at ``MEMBER_GATE``, on the real imagery — the
    control the flat-member case is measured against. Not every real pair does
    (some are already refused), so this scans rather than taking the first
    two-view point it finds."""
    track_pids = np.asarray(recon.track_point_indexes)
    track_imgs = np.asarray(recon.track_image_indexes)
    for candidate in np.asarray(cloud.point_indexes).tolist():
        views = np.unique(track_imgs[track_pids == candidate])
        if len(views) < 2:
            continue
        pid, pair = int(candidate), [int(views[0]), int(views[1])]
        res = cloud.localize_keypoints(
            recon,
            images,
            view_sets={pid: pair},
            reference_images={pid: pair[0]},
            point_indexes=[pid],
            max_member_zncc_self_similarity_radius=MEMBER_GATE,
        )
        if [int(v) for v in np.asarray(res[0]["views"])] == pair:
            return pid, pair
    raise AssertionError("no two-view point survives the default gates")


def test_localize_keypoints_flat_member_culls_a_two_view_point(
    seoul_bull_workspace: Path,
):
    """A two-view point whose second member renders a textureless tile is left
    below ``min_views`` with the member gate at ``MEMBER_GATE``, and keeps both
    views once the two absolute gates are switched off.

    The first member is the reference, so the flat member is the one aligned
    to it. ``max_member_zncc_self_similarity_radius`` judges the flat tile on
    its own content, before it is scored: a flat tile matches itself at every
    shift, so its ZNCC self-similarity radius reads the largest shift searched.
    """
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    images = load_images(recon)
    cloud = PatchCloud.from_reconstruction(
        recon, normal="mean_viewing", extent_value=11.0
    )
    pid, pair = _healthy_two_view_point(recon, cloud, images)

    def kept(imgs, **kwargs) -> list[int]:
        res = cloud.localize_keypoints(
            recon,
            imgs,
            view_sets={pid: pair},
            reference_images={pid: pair[0]},
            point_indexes=[pid],
            **kwargs,
        )
        return [int(v) for v in np.asarray(res[0]["views"])]

    # Blank the second member's source image: its tile then carries no gradient
    # at all, so it matches itself at every shift.
    flat = list(images)
    flat[pair[1]] = np.full_like(images[pair[1]], 128)

    assert len(kept(flat, max_member_zncc_self_similarity_radius=MEMBER_GATE)) < 2, (
        "the flat member must be dropped, leaving the point below min_views"
    )
    assert (
        kept(flat, min_absolute_zncc=0.0, max_member_zncc_self_similarity_radius=0.0)
        == pair
    ), "with both absolute gates at 0 the flat member survives, as it used to"
