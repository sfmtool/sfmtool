# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Binding tests for ``patches.spawn_candidate_tracks`` and the
``starting_keypoints`` parameter of ``PatchCloud.localize_keypoints``.

Both run on a fully synthetic scene — pinhole cameras looking down world ``+z``
at a textured plane — so the geometric truth is known exactly and no dataset
fixture is needed. See ``specs/core/candidate-track-spawning.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sfmtool._sfmtool.geometry import CameraIntrinsics
from sfmtool._sfmtool.patches import (
    CameraViews,
    ImagePyramidSet,
    PatchCloud,
    spawn_candidate_tracks,
)

# ---------------------------------------------------------------------------
# The synthetic scene
# ---------------------------------------------------------------------------

PLANE_Z = 4.0
IMG_W, IMG_H = 320, 240
FOCAL = 260.0
HALF_EXTENT = 0.25
RES = 20
# Cameras with real parallax about the patch, well clear of a degenerate solve.
CENTERS = np.array(
    [
        [0.55, 0.10, 0.0],
        [-0.50, -0.15, 0.0],
        [0.05, 0.60, 0.0],
        [-0.20, -0.55, 0.30],
    ],
    dtype=np.float64,
)
# World-to-camera rotation of a `+z`-looking view: 180 degrees about x.
ROT = np.diag([1.0, -1.0, -1.0])


def _texture(x, y):
    """Broadband, non-periodic plane texture with a unique local peak."""
    return (
        127.5
        + 46.0 * np.sin(x * 17.0)
        + 38.0 * np.cos(y * 23.0)
        + 22.0 * np.sin((x + y) * 31.0)
        + 14.0 * np.cos((x - 2.0 * y) * 7.3)
    )


def _render(center):
    """What a pinhole at ``center`` sees of the textured plane, as RGB uint8."""
    col, row = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
    dx = (col + 0.5 - IMG_W / 2.0) / FOCAL
    dy = (row + 0.5 - IMG_H / 2.0) / FOCAL
    lam = PLANE_Z - center[2]
    gray = _texture(center[0] + lam * dx, center[1] + lam * dy)
    gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))


def _camera():
    return CameraIntrinsics(
        "PINHOLE",
        IMG_W,
        IMG_H,
        {
            "focal_length_x": FOCAL,
            "focal_length_y": FOCAL,
            "principal_point_x": IMG_W / 2.0,
            "principal_point_y": IMG_H / 2.0,
        },
    )


def _views():
    quats = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (len(CENTERS), 1))
    trans = np.ascontiguousarray(-(ROT @ CENTERS.T).T)
    return CameraViews([_camera()], quats, trans)


def _project(image_index: int, point_xyz: np.ndarray) -> np.ndarray:
    """The pixel a world point lands on in one view (in front by construction)."""
    x_cam = ROT @ point_xyz + (-(ROT @ CENTERS[image_index]))
    assert x_cam[2] < 0.0, "test points must be in front of every camera"
    cam = _camera()
    return np.asarray(cam.project(x_cam[0] / -x_cam[2], x_cam[1] / -x_cam[2]))


# The parent's frame, matching `OrientedPatch::from_center_normal` with the
# normal pointing back at the cameras and world +y as the up hint.
U_AXIS = np.array([-1.0, 0.0, 0.0])
V_AXIS = np.array([0.0, 1.0, 0.0])


def _cloud(center=(0.0, 0.0, PLANE_Z)):
    """A one-patch cloud on the plane, centred where the caller asks."""
    return PatchCloud.from_halfvec_arrays(
        np.ascontiguousarray((U_AXIS * HALF_EXTENT)[None, :], dtype=np.float32),
        np.ascontiguousarray((V_AXIS * HALF_EXTENT)[None, :], dtype=np.float32),
        np.ascontiguousarray(np.asarray(center, dtype=np.float64)[None, :]),
    )


def _true_center(du: float, dv: float) -> np.ndarray:
    """The world centre a ``(du, dv)`` request off the parent asks for."""
    return (
        np.array([0.0, 0.0, PLANE_Z])
        + U_AXIS * (du * HALF_EXTENT)
        + V_AXIS * (dv * HALF_EXTENT)
    )


@pytest.fixture(scope="module")
def scene():
    views = _views()
    images = [_render(c) for c in CENTERS]
    return views, ImagePyramidSet(views, images)


def _spawn(scene, offsets, view_sets=None, parents=None, **kw):
    views, pyramids = scene
    offsets = np.ascontiguousarray(np.asarray(offsets, dtype=np.float64).reshape(-1, 2))
    n = len(offsets)
    if parents is None:
        parents = np.zeros(n, dtype=np.uint32)
    if view_sets is None:
        view_sets = [[0, 1, 2, 3]] * n
    kw.setdefault("resolution", RES)
    return spawn_candidate_tracks(
        views, pyramids, _cloud(), parents, offsets, view_sets, **kw
    )


# ---------------------------------------------------------------------------
# The dict surface
# ---------------------------------------------------------------------------


class TestSpawnDictSurface:
    def test_keys_shapes_and_dtypes(self, scene):
        out = _spawn(scene, [[2.0, 0.0], [0.0, -2.0], [400.0, 400.0]])
        assert set(out) == {
            "status",
            "positions",
            "requested_centers",
            "reproj_rms_px",
            "n_views",
            "obs_offsets",
            "obs_view_indexes",
            "obs_keypoints_xy",
        }
        n = 3
        assert out["status"].shape == (n,) and out["status"].dtype == np.uint8
        assert out["positions"].shape == (n, 3)
        assert out["positions"].dtype == np.float64
        assert out["requested_centers"].shape == (n, 3)
        assert out["reproj_rms_px"].shape == (n,)
        assert out["reproj_rms_px"].dtype == np.float64
        assert out["n_views"].shape == (n,) and out["n_views"].dtype == np.uint32
        assert out["obs_offsets"].shape == (n + 1,)
        assert out["obs_offsets"].dtype == np.uint32
        n_obs = len(out["obs_view_indexes"])
        assert out["obs_view_indexes"].dtype == np.uint32
        assert out["obs_keypoints_xy"].shape == (n_obs, 2)

    def test_csr_offsets_are_consistent(self, scene):
        out = _spawn(scene, [[2.0, 0.0], [400.0, 400.0], [0.0, 2.0]])
        offs = out["obs_offsets"].astype(np.int64)
        assert offs[0] == 0
        assert np.all(np.diff(offs) >= 0)
        assert offs[-1] == len(out["obs_view_indexes"])
        for i in range(len(out["status"])):
            block = out["obs_view_indexes"][offs[i] : offs[i + 1]]
            assert np.all(np.diff(block.astype(np.int64)) > 0), "ascending view order"

    def test_empty_batch(self, scene):
        views, pyramids = scene
        out = spawn_candidate_tracks(
            views,
            pyramids,
            _cloud(),
            np.zeros(0, np.uint32),
            np.zeros((0, 2), np.float64),
            [],
        )
        assert out["status"].shape == (0,)
        assert out["positions"].shape == (0, 3)
        assert out["obs_keypoints_xy"].shape == (0, 2)
        assert out["obs_offsets"].tolist() == [0]

    def test_non_contiguous_offsets_are_accepted(self, scene):
        """A Fortran-ordered / sliced offsets array gives the same answer."""
        base = np.asfortranarray(np.array([[2.0, 0.0], [0.0, -2.0]], np.float64))
        views, pyramids = scene
        out = spawn_candidate_tracks(
            views,
            pyramids,
            _cloud(),
            np.zeros(2, np.uint32),
            base,
            [[0, 1, 2, 3]] * 2,
            resolution=RES,
        )
        ref = _spawn(scene, [[2.0, 0.0], [0.0, -2.0]])
        assert out["status"].tolist() == ref["status"].tolist()
        np.testing.assert_allclose(
            out["requested_centers"], ref["requested_centers"], atol=1e-12
        )


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class TestSpawnBehaviour:
    def test_candidate_on_the_plane_spawns_where_it_was_asked(self, scene):
        out = _spawn(scene, [[2.0, 0.0]])
        assert out["status"][0] == 0, f"status {out['status'][0]}"
        assert out["n_views"][0] == 4
        truth = _true_center(2.0, 0.0)
        np.testing.assert_allclose(out["requested_centers"][0], truth, atol=1e-12)
        assert np.linalg.norm(out["positions"][0] - truth) < 0.25 * HALF_EXTENT
        assert 0.0 < out["reproj_rms_px"][0] < 0.5
        # The observations are the four keypoints that produced that position.
        assert out["obs_view_indexes"].tolist() == [0, 1, 2, 3]
        for k, image_index in enumerate(out["obs_view_indexes"].tolist()):
            expected = _project(image_index, truth)
            assert np.linalg.norm(out["obs_keypoints_xy"][k] - expected) < 2.0

    def test_candidate_off_every_image_reports_too_few_views(self, scene):
        out = _spawn(scene, [[400.0, 400.0]])
        assert out["status"][0] == 1
        assert out["n_views"][0] == 0
        assert np.all(np.isnan(out["positions"][0]))
        assert np.isnan(out["reproj_rms_px"][0])
        assert out["obs_offsets"].tolist() == [0, 0]

    def test_empty_view_set_reports_too_few_views(self, scene):
        out = _spawn(scene, [[2.0, 0.0]], view_sets=[[]])
        assert out["status"][0] == 1
        assert out["n_views"][0] == 0

    def test_min_views_floor_above_the_survivors(self, scene):
        out = _spawn(scene, [[2.0, 0.0]], min_views=5)
        assert out["status"][0] == 1

    def test_unreachable_reprojection_gate(self, scene):
        achieved = float(_spawn(scene, [[2.0, 0.0]])["reproj_rms_px"][0])
        assert achieved > 0.0
        out = _spawn(scene, [[2.0, 0.0]], max_reproj_rms_px=achieved / 10.0)
        assert out["status"][0] == 3
        assert np.all(np.isnan(out["positions"][0]))
        # A high-reproj casualty keeps its observations, so it can be diagnosed.
        assert out["n_views"][0] == 4
        assert len(out["obs_view_indexes"]) == 4

    def test_discrete_only_still_spawns(self, scene):
        out = _spawn(scene, [[2.0, 0.0]], subpixel_sweeps=0)
        assert out["status"][0] == 0
        truth = _true_center(2.0, 0.0)
        assert np.linalg.norm(out["positions"][0] - truth) < 0.5 * HALF_EXTENT

    def test_batch_matches_the_candidates_run_alone(self, scene):
        offsets = [[2.0, 0.0], [-2.0, 0.5], [0.0, 2.0]]
        batched = _spawn(scene, offsets)
        for i, off in enumerate(offsets):
            alone = _spawn(scene, [off])
            assert batched["status"][i] == alone["status"][0]
            assert batched["n_views"][i] == alone["n_views"][0]
            np.testing.assert_allclose(
                batched["positions"][i], alone["positions"][0], atol=1e-12
            )
        assert int((batched["status"] == 0).sum()) >= 2

    def test_reconstruction_free_scene_accepts_an_image_list(self, scene):
        """The scene/imagery arguments take the same shapes as the sibling
        kernels: a raw image list works where an ImagePyramidSet does."""
        views = _views()
        images = [_render(c) for c in CENTERS]
        out = spawn_candidate_tracks(
            views,
            images,
            _cloud(),
            np.zeros(1, np.uint32),
            np.array([[2.0, 0.0]]),
            [[0, 1, 2, 3]],
            resolution=RES,
        )
        ref = _spawn(scene, [[2.0, 0.0]])
        assert out["status"][0] == ref["status"][0]
        np.testing.assert_allclose(out["positions"][0], ref["positions"][0], atol=1e-12)


# ---------------------------------------------------------------------------
# Malformed inputs
# ---------------------------------------------------------------------------


class TestSpawnValidation:
    def test_parent_out_of_range(self, scene):
        with pytest.raises(ValueError, match="out of range for the cloud"):
            _spawn(scene, [[2.0, 0.0]], parents=np.array([7], np.uint32))

    def test_offsets_row_count_mismatch(self, scene):
        views, pyramids = scene
        with pytest.raises(ValueError, match="offsets_uv has 2 rows"):
            spawn_candidate_tracks(
                views,
                pyramids,
                _cloud(),
                np.zeros(1, np.uint32),
                np.zeros((2, 2), np.float64),
                [[0, 1]],
            )

    def test_offsets_column_count(self, scene):
        views, pyramids = scene
        with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
            spawn_candidate_tracks(
                views,
                pyramids,
                _cloud(),
                np.zeros(1, np.uint32),
                np.zeros((1, 3), np.float64),
                [[0, 1]],
            )

    def test_view_sets_length_mismatch(self, scene):
        with pytest.raises(ValueError, match="view_sets has 2 entries"):
            _spawn(scene, [[2.0, 0.0]], view_sets=[[0, 1], [0, 1]])

    def test_view_index_out_of_range(self, scene):
        with pytest.raises(ValueError, match="out of range for this scene"):
            _spawn(scene, [[2.0, 0.0]], view_sets=[[0, 1, 9]])


# ---------------------------------------------------------------------------
# `localize_keypoints(starting_keypoints=...)`
# ---------------------------------------------------------------------------

# Displacement of the wrong-position cloud, in patch-grid px along the parent's
# u axis. Four grid px is comfortably inside the localizer's 6 grid-px search
# window and well outside its sub-pixel precision, so neither bound is close.
DISPLACE_GRID_PX = 4.0
WPP = 2.0 * HALF_EXTENT / RES
SRC_PX_PER_GRID = WPP * FOCAL / PLANE_Z


def _localize(scene, cloud, seeds=None, **kw):
    views, pyramids = scene
    return cloud.localize_keypoints(
        views,
        pyramids,
        view_sets={0: [0, 1, 2, 3]},
        resolution=RES,
        max_shift_px=12.0,
        starting_keypoints=seeds,
        **kw,
    )[0]


class TestLocalizeStartingKeypoints:
    def test_default_is_unchanged_when_omitted(self, scene):
        """The new parameter is additive: not passing it localizes exactly as
        passing ``None`` does."""
        cloud = _cloud()
        a = _localize(scene, cloud)
        views, pyramids = scene
        b = cloud.localize_keypoints(
            views,
            pyramids,
            view_sets={0: [0, 1, 2, 3]},
            resolution=RES,
            max_shift_px=12.0,
        )[0]
        assert a["views"].tolist() == b["views"].tolist()
        np.testing.assert_array_equal(a["keypoints"], b["keypoints"])

    def test_seeds_at_the_projection_reproduce_the_default(self, scene):
        """When the cloud position is already right, seeding at the true image
        locations changes nothing."""
        cloud = _cloud()
        truth = np.array([0.0, 0.0, PLANE_Z])
        default = _localize(scene, cloud)
        seeds = [_project(i, truth).tolist() for i in default["views"].tolist()]
        seeded = _localize(scene, cloud, seeds={0: seeds})
        assert seeded["views"].tolist() == default["views"].tolist()
        np.testing.assert_allclose(seeded["keypoints"], default["keypoints"], atol=1e-6)

    def test_seeds_recover_a_point_whose_cloud_position_is_displaced(self, scene):
        """The case the default seeding cannot reach: the cloud row's centre is
        wrong, but the caller's seeds point at the true image locations. Seeded,
        localization stays on the evidence; unseeded, it congeals around the
        wrong projection instead."""
        truth = np.array([0.0, 0.0, PLANE_Z])
        displaced = truth + U_AXIS * (DISPLACE_GRID_PX * WPP)
        cloud = _cloud(displaced)

        default = _localize(scene, cloud)
        assert default["views"].tolist() == [0, 1, 2, 3]
        seeds = [_project(i, truth).tolist() for i in default["views"].tolist()]

        seeded = _localize(scene, cloud, seeds={0: seeds})
        assert seeded["views"].tolist() == [0, 1, 2, 3]

        seed_arr = np.asarray(seeds, dtype=np.float64)
        seeded_err = np.linalg.norm(seeded["keypoints"] - seed_arr, axis=1)
        default_err = np.linalg.norm(default["keypoints"] - seed_arr, axis=1)
        displacement_px = DISPLACE_GRID_PX * SRC_PX_PER_GRID

        assert seeded_err.max() < 1.5, (
            f"seeded localization should land on the seeds, got {seeded_err}"
        )
        assert default_err.min() > 0.5 * displacement_px, (
            "projection seeding should stay near the displaced projection, "
            f"got {default_err} (displacement {displacement_px:.2f} px)"
        )
        # And the two paths genuinely disagree, by about the displacement.
        assert default_err.min() > 3.0 * seeded_err.max()

    def test_seed_count_must_match_the_view_set(self, scene):
        cloud = _cloud()
        with pytest.raises(ValueError, match="has 2 seeds but the view set has 4"):
            _localize(scene, cloud, seeds={0: [[10.0, 10.0], [11.0, 11.0]]})

    def test_seeds_for_an_unknown_point(self, scene):
        cloud = _cloud()
        with pytest.raises(ValueError, match="not a point in this patch cloud"):
            _localize(scene, cloud, seeds={7: [[10.0, 10.0]]})

    def test_seeds_excluded_by_point_indexes(self, scene):
        cloud = _cloud()
        with pytest.raises(ValueError, match="excluded by point_indexes"):
            _localize(
                scene,
                cloud,
                seeds={0: [[10.0, 10.0]]},
                point_indexes=[],
            )
