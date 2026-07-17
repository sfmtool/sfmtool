# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``CameraViews`` value object and ``PatchCloud.from_tracks``.

``CameraViews`` lets a caller drive the patch kernels
(:meth:`PatchCloud.localize_keypoints`, ``refine_keypoints``, ``refine_normals``,
``select_views``) and :class:`ImagePyramidSet` from in-memory posed views instead
of a full reconstruction, and ``PatchCloud.from_tracks`` builds a cloud from
arrays. See ``specs/core/patch-cloud.md`` ("Patch operations without a
reconstruction"). The equivalence tests prove a reconstruction and an equivalent
``CameraViews`` + explicit view sets produce identical patch-kernel output.
"""

import numpy as np
import pytest

from sfmtool._sfmtool import (
    CameraViews,
    ImagePyramidSet,
    PatchCloud,
    SfmrReconstruction,
)
from sfmtool._sfmtool.geometry import CameraIntrinsics


def _pinhole(width=16, height=16, f=20.0):
    return CameraIntrinsics(
        "PINHOLE",
        width,
        height,
        {
            "focal_length_x": f,
            "focal_length_y": f,
            "principal_point_x": width / 2.0,
            "principal_point_y": height / 2.0,
        },
    )


def _identity_views(n=2, camera=None):
    """``n`` identity-rotation views translated along +X."""
    cam = camera or _pinhole()
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    trans = np.zeros((n, 3), np.float64)
    trans[:, 0] = np.arange(n) * 0.5
    return CameraViews([cam], quats, trans)


def _tiny_cloud(views):
    """A 3-point ``from_tracks`` cloud in front of the identity views."""
    positions = np.array(
        [
            [0.0, 0.0, -3.0, 1.0],
            [0.2, 0.1, -3.0, 1.0],
            [-0.1, 0.2, -3.0, 1.0],
        ],
        np.float64,
    )
    tpi = np.array([0, 0, 1, 1, 2, 2], np.uint32)
    tii = np.array([0, 1, 0, 1, 0, 1], np.uint32)
    return PatchCloud.from_tracks(
        views,
        positions,
        tpi,
        tii,
        normal="mean_viewing",
        extent="fixed",
        extent_value=0.1,
    )


# ---------------------------------------------------------------------------
# CameraViews construction + validation
# ---------------------------------------------------------------------------


class TestCameraViewsConstruction:
    def test_len_and_default_camera_index(self):
        views = _identity_views(3)
        assert len(views) == 3

    def test_explicit_camera_indexes(self):
        cam_a = _pinhole(16, 16)
        cam_b = _pinhole(32, 24)
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
        trans = np.zeros((2, 3), np.float64)
        views = CameraViews([cam_a, cam_b], quats, trans, np.array([1, 0], np.uint32))
        assert len(views) == 2

    def test_empty_cameras_rejected(self):
        quats = np.array([[1.0, 0.0, 0.0, 0.0]])
        trans = np.zeros((1, 3))
        with pytest.raises(ValueError, match="non-empty"):
            CameraViews([], quats, trans)

    def test_bad_quaternion_shape(self):
        cam = _pinhole()
        with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
            CameraViews([cam], np.zeros((2, 3)), np.zeros((2, 3)))

    def test_translations_shape_mismatch(self):
        cam = _pinhole()
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
        with pytest.raises(ValueError, match="translations_xyz"):
            CameraViews([cam], quats, np.zeros((3, 3)))

    def test_non_unit_quaternion_rejected(self):
        cam = _pinhole()
        quats = np.array([[2.0, 0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="unit quaternion"):
            CameraViews([cam], quats, np.zeros((1, 3)))

    def test_camera_index_out_of_range(self):
        cam = _pinhole()
        quats = np.array([[1.0, 0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="out of range"):
            CameraViews([cam], quats, np.zeros((1, 3)), np.array([5], np.uint32))

    def test_camera_indexes_length_mismatch(self):
        cam = _pinhole()
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
        with pytest.raises(ValueError, match="length 2"):
            CameraViews([cam], quats, np.zeros((2, 3)), np.array([0], np.uint32))


# ---------------------------------------------------------------------------
# ImagePyramidSet from CameraViews
# ---------------------------------------------------------------------------


class TestImagePyramidSetFromViews:
    def test_build_from_views(self):
        views = _identity_views(2)
        images = [np.zeros((16, 16, 3), np.uint8), np.zeros((16, 16, 3), np.uint8)]
        pyr = ImagePyramidSet(views, images)
        assert len(pyr) == 2

    def test_dimension_mismatch_rejected(self):
        views = _identity_views(2)
        images = [np.zeros((16, 16, 3), np.uint8), np.zeros((8, 8, 3), np.uint8)]
        with pytest.raises(ValueError, match="camera is 16x16"):
            ImagePyramidSet(views, images)

    def test_image_count_mismatch_rejected(self):
        views = _identity_views(2)
        images = [np.zeros((16, 16, 3), np.uint8)]
        with pytest.raises(ValueError, match="parallel to the scene"):
            ImagePyramidSet(views, images)


# ---------------------------------------------------------------------------
# PatchCloud.from_tracks validation
# ---------------------------------------------------------------------------


class TestFromTracksValidation:
    def test_builds_and_indexes_positions(self):
        views = _identity_views(2)
        cloud = _tiny_cloud(views)
        assert len(cloud) == 3
        assert list(np.asarray(cloud.point_indexes)) == [0, 1, 2]

    def test_infinity_row_gets_frame(self):
        views = _identity_views(2)
        positions = np.array([[0.0, 0.0, -3.0, 1.0], [0.0, 0.0, -1.0, 0.0]], np.float64)
        tpi = np.array([0, 0, 1, 1], np.uint32)
        tii = np.array([0, 1, 0, 1], np.uint32)
        cloud = PatchCloud.from_tracks(
            views,
            positions,
            tpi,
            tii,
            normal="mean_viewing",
            extent="fixed",
            extent_value=0.1,
        )
        assert len(cloud) == 2
        assert cloud[0].w == 1.0
        assert cloud[1].w == 0.0

    def test_nondecreasing_point_indexes_required(self):
        views = _identity_views(2)
        positions = np.array([[0.0, 0.0, -3.0, 1.0], [0.2, 0.0, -3.0, 1.0]], np.float64)
        tpi = np.array([1, 0], np.uint32)
        tii = np.array([0, 1], np.uint32)
        with pytest.raises(ValueError, match="nondecreasing"):
            PatchCloud.from_tracks(
                views, positions, tpi, tii, extent="fixed", extent_value=0.1
            )

    def test_every_point_needs_an_observation(self):
        views = _identity_views(2)
        positions = np.array([[0.0, 0.0, -3.0, 1.0], [0.2, 0.0, -3.0, 1.0]], np.float64)
        # Only point 0 is observed; point 1 has no observation.
        tpi = np.array([0, 0], np.uint32)
        tii = np.array([0, 1], np.uint32)
        with pytest.raises(ValueError, match="point 1 has no observation"):
            PatchCloud.from_tracks(
                views, positions, tpi, tii, extent="fixed", extent_value=0.1
            )

    def test_feature_size_requires_scales(self):
        views = _identity_views(2)
        positions = np.array([[0.0, 0.0, -3.0, 1.0]], np.float64)
        tpi = np.array([0, 0], np.uint32)
        tii = np.array([0, 1], np.uint32)
        with pytest.raises(ValueError, match="keypoint_scales is required"):
            PatchCloud.from_tracks(views, positions, tpi, tii, extent="feature_size")

    def test_stored_normal_requires_normals(self):
        views = _identity_views(2)
        positions = np.array([[0.0, 0.0, -3.0, 1.0]], np.float64)
        tpi = np.array([0, 0], np.uint32)
        tii = np.array([0, 1], np.uint32)
        with pytest.raises(ValueError, match="normals is required"):
            PatchCloud.from_tracks(
                views,
                positions,
                tpi,
                tii,
                normal="stored",
                extent="fixed",
                extent_value=0.1,
            )

    def test_feature_size_nan_scale_counts_as_unreadable(self):
        views = _identity_views(2)
        positions = np.array([[0.0, 0.0, -3.0, 1.0]], np.float64)
        tpi = np.array([0, 0], np.uint32)
        tii = np.array([0, 1], np.uint32)
        scales = np.array([np.nan, np.nan], np.float64)
        with pytest.raises(ValueError, match="usable keypoint scale"):
            PatchCloud.from_tracks(
                views,
                positions,
                tpi,
                tii,
                keypoint_scales=scales,
                extent="feature_size",
            )

    def test_image_index_out_of_range(self):
        views = _identity_views(2)
        positions = np.array([[0.0, 0.0, -3.0, 1.0]], np.float64)
        tpi = np.array([0, 0], np.uint32)
        tii = np.array([0, 5], np.uint32)
        with pytest.raises(ValueError, match="out of range for 2 views"):
            PatchCloud.from_tracks(
                views, positions, tpi, tii, extent="fixed", extent_value=0.1
            )


# ---------------------------------------------------------------------------
# Required per-patch view arguments in CameraViews mode
# ---------------------------------------------------------------------------


class TestViewsModeRequiredArguments:
    @pytest.fixture
    def scene(self):
        views = _identity_views(2)
        cloud = _tiny_cloud(views)
        images = [np.zeros((16, 16, 3), np.uint8), np.zeros((16, 16, 3), np.uint8)]
        return views, cloud, images

    def test_localize_requires_view_sets(self, scene):
        views, cloud, images = scene
        with pytest.raises(ValueError, match="view_sets is required"):
            cloud.localize_keypoints(views, images)

    def test_refine_normals_requires_view_indices(self, scene):
        views, cloud, images = scene
        with pytest.raises(ValueError, match="view_indices is required"):
            cloud.refine_normals(views, images)

    def test_select_views_requires_candidate_views(self, scene):
        views, cloud, images = scene
        with pytest.raises(ValueError, match="candidate_views is required"):
            cloud.select_views(views, images)

    def test_refine_keypoints_requires_view_sets(self, scene):
        views, cloud, images = scene
        with pytest.raises(ValueError, match="view_sets is required"):
            cloud.refine_keypoints(views, images)

    def test_localize_runs_with_view_sets(self, scene):
        views, cloud, images = scene
        view_sets = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
        results = cloud.localize_keypoints(views, images, view_sets=view_sets)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# End-to-end equivalence against a reconstruction
# ---------------------------------------------------------------------------


def _view_sets_from_recon(recon):
    """``point_index -> [image_index, ...]`` in track order (the same lists
    ``view_indices_from_reconstruction`` derives internally)."""
    tpi = np.asarray(recon.track_point_indexes)
    tii = np.asarray(recon.track_image_indexes)
    sets: dict[int, list[int]] = {}
    for pid, img in zip(tpi, tii):
        sets.setdefault(int(pid), []).append(int(img))
    return sets


def _views_from_recon(recon):
    return CameraViews(
        recon.cameras,
        np.asarray(recon.quaternions_wxyz, np.float64),
        np.asarray(recon.translations, np.float64),
        np.asarray(recon.camera_indexes, np.uint32),
    )


def _assert_clouds_equal(a, b):
    assert list(np.asarray(a.point_indexes)) == list(np.asarray(b.point_indexes))
    assert len(a) == len(b)
    for i in range(len(a)):
        pa, pb = a[i], b[i]
        np.testing.assert_allclose(pa.center, pb.center, atol=1e-12)
        np.testing.assert_allclose(pa.u_axis, pb.u_axis, atol=1e-12)
        np.testing.assert_allclose(pa.v_axis, pb.v_axis, atol=1e-12)
        np.testing.assert_allclose(pa.half_extent, pb.half_extent, atol=1e-12)
        assert pa.w == pb.w


class TestReconstructionEquivalence:
    def test_from_tracks_matches_from_reconstruction(self, seoul_bull_workspace):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        # PixelRadius + mean_viewing need no `.sift` scales or stored normals, so
        # the two clouds are geometry-only and must be patch-for-patch identical.
        kwargs = dict(
            normal="mean_viewing",
            extent="pixel_radius",
            extent_value=4.0,
            pixel_reduce="min",
        )
        cloud_recon = PatchCloud.from_reconstruction(recon, **kwargs)
        views = _views_from_recon(recon)
        cloud_tracks = PatchCloud.from_tracks(
            views,
            np.asarray(recon.positions_xyzw, np.float64),
            np.asarray(recon.track_point_indexes, np.uint32),
            np.asarray(recon.track_image_indexes, np.uint32),
            **kwargs,
        )
        _assert_clouds_equal(cloud_recon, cloud_tracks)

    def test_localize_keypoints_matches_across_modes(self, seoul_bull_workspace):
        from sfmtool.xform._images import load_workspace_images

        recon = SfmrReconstruction.load(seoul_bull_workspace)
        images = load_workspace_images(recon)

        kwargs = dict(
            normal="mean_viewing",
            extent="pixel_radius",
            extent_value=4.0,
            pixel_reduce="min",
        )
        cloud = PatchCloud.from_reconstruction(recon, **kwargs)
        views = _views_from_recon(recon)
        cloud_tracks = PatchCloud.from_tracks(
            views,
            np.asarray(recon.positions_xyzw, np.float64),
            np.asarray(recon.track_point_indexes, np.uint32),
            np.asarray(recon.track_image_indexes, np.uint32),
            **kwargs,
        )

        # Localize a handful of points for speed; the same subset both ways.
        subset = [int(p) for p in np.asarray(cloud.point_indexes)[:8]]
        recon_out = cloud.localize_keypoints(recon, images, point_indexes=subset)

        view_sets = _view_sets_from_recon(recon)
        views_out = cloud_tracks.localize_keypoints(
            views, images, view_sets=view_sets, point_indexes=subset
        )

        assert len(recon_out) == len(views_out)
        for a, b in zip(recon_out, views_out):
            assert a["point_index"] == b["point_index"]
            np.testing.assert_array_equal(
                np.asarray(a["views"]), np.asarray(b["views"])
            )
            np.testing.assert_array_equal(
                np.asarray(a["keypoints"]), np.asarray(b["keypoints"])
            )
            np.testing.assert_array_equal(
                np.asarray(a["offsets_px"]), np.asarray(b["offsets_px"])
            )

    def test_select_views_candidate_override_in_recon_mode(self, seoul_bull_workspace):
        from sfmtool.xform._images import load_workspace_images

        recon = SfmrReconstruction.load(seoul_bull_workspace)
        images = load_workspace_images(recon)
        cloud = PatchCloud.from_reconstruction(
            recon, normal="mean_viewing", extent="pixel_radius", extent_value=4.0
        )

        # Pick a point and an image its track did NOT observe; overriding its
        # candidate list must make that image show up in the always-admitted set,
        # proving `candidate_views` took precedence over the track views.
        sets = _view_sets_from_recon(recon)
        pid = int(np.asarray(cloud.point_indexes)[0])
        n_images = recon.image_count
        track_imgs = set(sets[pid])
        extra = next(i for i in range(n_images) if i not in track_imgs)

        out = cloud.select_views(
            recon,
            images,
            candidate_views={pid: [extra]},
            point_indexes=[pid],
        )
        assert len(out) == 1
        admitted = set(int(i) for i in np.asarray(out[0]["admitted"]))
        assert extra in admitted
