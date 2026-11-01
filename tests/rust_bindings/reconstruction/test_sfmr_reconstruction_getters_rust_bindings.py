# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The array getters of ``SfmrReconstruction``: every array is read-only; a
column held as per-point or per-image records is gathered once and returned as
the same array on each read; a column held as one block is a view of it."""

import gc
import sys

import numpy as np
import pytest

from sfmtool import THUMBNAIL_SIZE
from sfmtool.fileio import read_sfmr
from sfmtool.reconstruction import SfmrReconstruction

# Gathered on the first read and kept: the same object on every read.
GATHERED = [
    "camera_indexes",
    "quaternions_wxyz",
    "translations",
    "focal_adjustments",
    "principal_point_adjustments_xy",
    "radial_bspline_adjustments",
    "rolling_shutter_rates_xyz",
    "positions",
    "positions_xyzw",
    "point_is_at_infinity",
    "colors",
    "rms_reprojection_errors",
    "normals",
    "track_image_indexes",
    "track_point_indexes",
    "depth_histogram_counts",
]
# Views of a block the reconstruction holds: a new view of the same memory on
# every read.
VIEWS = [
    "keypoints_xy",
    "track_feature_indexes",
    "observation_counts",
    "observation_confidence",
    "reference_observations",
    "normal_confidence",
    "point_constraints",
    "constraint_distances",
    "constraint_reference_images",
    "patch_bitmaps",
    "thumbnails_y_x_rgb",
]

_HELD, _RANGED, _NO_REFERENCE_IMAGE = 2, 1, 0xFFFFFFFF


def _expected(data: dict, extra: dict) -> dict:
    """What each getter holds, from the file's own columns as ``read_sfmr``
    reads them and the columns the test passed to ``clone_with_changes``."""
    xyzw = data["positions_xyzw"]
    expected = {
        "camera_indexes": data["camera_indexes"],
        "quaternions_wxyz": data["quaternions_wxyz"],
        "translations": data["translations_xyz"],
        "radial_bspline_adjustments": None,
        "rolling_shutter_rates_xyz": None,
        "positions": xyzw[:, :3],
        "positions_xyzw": xyzw,
        "point_is_at_infinity": xyzw[:, 3] == 0.0,
        "colors": data["colors_rgb"],
        "rms_reprojection_errors": data["reprojection_errors"],
        "normals": data["normals_xyz"],
        "track_image_indexes": data["image_indexes"],
        "track_point_indexes": data["point_indexes"],
        "depth_histogram_counts": data["observed_depth_histogram_counts"],
        "keypoints_xy": data["keypoints_xy"],
        "track_feature_indexes": data["feature_indexes"],
        "observation_counts": data["observation_counts"],
        "reference_observations": data["reference_observations"],
    }
    expected.update(extra)
    return expected


@pytest.fixture
def every_column(seoul_bull_ground_truth_sfmr):
    """The seoul_bull ground truth with every optional array column set, and
    the values each getter should return."""
    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    data = read_sfmr(seoul_bull_ground_truth_sfmr)
    n, k, n_img = recon.point_count, recon.observation_count, recon.image_count
    rng = np.random.default_rng(7)

    finite = np.flatnonzero(data["positions_xyzw"][:, 3] != 0.0)
    constraints = np.zeros(n, dtype=np.uint8)
    constraints[finite[0]] = _HELD
    constraints[finite[1]] = _RANGED
    distances = np.full(n, np.nan)
    distances[finite[1]] = 12.5
    references = np.full(n, _NO_REFERENCE_IMAGE, dtype=np.uint32)
    references[finite[1]] = 0

    extra = {
        "normal_confidence": rng.integers(0, 256, n, dtype=np.uint8),
        "observation_confidence": rng.integers(0, 256, k, dtype=np.uint8),
        "point_constraints": constraints,
        "constraint_distances": distances,
        "constraint_reference_images": references,
        "patch_bitmaps": rng.integers(0, 256, (n, 4, 4, 4), dtype=np.uint8),
        "thumbnails_y_x_rgb": rng.integers(
            0, 256, (n_img, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3), dtype=np.uint8
        ),
        "focal_adjustments": rng.uniform(-0.01, 0.01, n_img),
        "principal_point_adjustments_xy": rng.uniform(-0.01, 0.01, (n_img, 2)),
    }
    # A file stores the two adjustments only for a camera that lists them.
    per_image = [["focal", "principal_point"]] * recon.camera_count
    recon = recon.clone_with_changes(per_image_adjustments=per_image, **extra)
    return recon, _expected(data, extra)


def _assert_read_only(array):
    assert not array.flags.writeable
    # Nor can a caller turn writing back on: numpy refuses for an array whose
    # base is not a writeable buffer.
    with pytest.raises(ValueError, match="WRITEABLE"):
        array.setflags(write=True)
    if array.size:
        with pytest.raises(ValueError, match="read-only"):
            array[(0,) * array.ndim] = array[(0,) * array.ndim]


class TestEveryGetter:
    def test_the_lists_name_every_array_getter(self, every_column):
        recon, expected = every_column
        assert sorted(GATHERED + VIEWS) == sorted(expected)
        # `track_feature_indexes` is the one view this fixture has no column
        # for (embedded patches); the sift_files tests below read it.
        assert recon.track_feature_indexes is None

    @pytest.mark.parametrize("name", GATHERED + VIEWS)
    def test_values_dtype_and_shape(self, every_column, name):
        recon, expected = every_column
        value, want = getattr(recon, name), expected[name]
        if want is None:
            assert value is None
            return
        assert value.dtype == want.dtype
        assert value.shape == want.shape
        np.testing.assert_array_equal(value, want)

    @pytest.mark.parametrize("name", GATHERED + VIEWS)
    def test_read_only(self, every_column, name):
        recon, _ = every_column
        value = getattr(recon, name)
        if value is not None:
            _assert_read_only(value)

    @pytest.mark.parametrize("name", GATHERED)
    def test_a_gathered_column_is_the_same_array_each_read(self, every_column, name):
        recon, _ = every_column
        first = getattr(recon, name)
        assert getattr(recon, name) is first

    @pytest.mark.parametrize("name", VIEWS)
    def test_a_view_shares_the_objects_memory(self, every_column, name):
        recon, _ = every_column
        first = getattr(recon, name)
        if first is None:
            return
        second = getattr(recon, name)
        assert np.shares_memory(first, second)
        assert first.base is recon

    def test_a_view_keeps_its_reconstruction_alive(self, seoul_bull_ground_truth_sfmr):
        # Loaded here rather than taken from a fixture, which pytest would keep
        # alive itself.
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        expected = read_sfmr(seoul_bull_ground_truth_sfmr)["keypoints_xy"]
        view = recon.keypoints_xy
        del recon
        gc.collect()
        # The view's base is now the only reference to the reconstruction.
        assert sys.getrefcount(view.base) == 2
        # Allocate and free memory so that a freed block would be reused.
        junk = [np.ones(len(expected) * 2, dtype=np.float32) for _ in range(8)]
        del junk
        np.testing.assert_array_equal(view, expected)

    def test_a_copy_is_writeable_and_reaches_nothing(self, every_column):
        recon, expected = every_column
        positions = recon.positions_xyzw.copy()
        positions[0, 0] += 1.0
        np.testing.assert_array_equal(recon.positions_xyzw, expected["positions_xyzw"])
        # And it goes back in through clone_with_changes as before.
        moved = recon.clone_with_changes(positions=positions)
        assert moved.positions_xyzw[0, 0] == positions[0, 0]

    def test_a_clone_gathers_its_own_arrays(self, every_column):
        recon, _ = every_column
        positions = recon.positions_xyzw.copy()
        positions[0, :3] += 1.0
        moved = recon.clone_with_changes(positions=positions)
        assert moved.positions_xyzw is not recon.positions_xyzw
        np.testing.assert_array_equal(moved.positions_xyzw, positions)


class TestSiftFilesColumns:
    def test_feature_indexes_and_thumbnails(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        data = read_sfmr(seoul_bull_sfmr_only)
        for name, key in [
            ("track_feature_indexes", "feature_indexes"),
            ("thumbnails_y_x_rgb", "thumbnails_y_x_rgb"),
            ("keypoints_xy", "keypoints_xy"),
        ]:
            value = getattr(recon, name)
            assert value.dtype == data[key].dtype
            np.testing.assert_array_equal(value, data[key])
            _assert_read_only(value)
            assert value.base is recon


class TestEmpty:
    @pytest.fixture
    def no_points(self, seoul_bull_ground_truth_sfmr):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        empty = recon.filter_points_by_mask(np.zeros(recon.point_count, dtype=bool))
        assert empty.point_count == 0 and empty.observation_count == 0
        return empty

    @pytest.mark.parametrize(
        ("name", "dtype", "shape"),
        [
            ("positions", np.float64, (0, 3)),
            ("positions_xyzw", np.float64, (0, 4)),
            ("point_is_at_infinity", np.bool_, (0,)),
            ("colors", np.uint8, (0, 3)),
            ("rms_reprojection_errors", np.float32, (0,)),
            ("normals", np.float32, (0, 3)),
            ("observation_counts", np.uint32, (0,)),
            ("reference_observations", np.int32, (0,)),
            ("track_image_indexes", np.uint32, (0,)),
            ("track_point_indexes", np.uint32, (0,)),
            ("keypoints_xy", np.float32, (0, 2)),
        ],
    )
    def test_no_points(self, no_points, name, dtype, shape):
        value = getattr(no_points, name)
        assert value.dtype == dtype
        assert value.shape == shape
        assert not value.flags.writeable
        if name in GATHERED:
            assert getattr(no_points, name) is value

    @pytest.mark.parametrize(
        ("name", "dtype", "shape"),
        [
            ("camera_indexes", np.uint32, (0,)),
            ("quaternions_wxyz", np.float64, (0, 4)),
            ("translations", np.float64, (0, 3)),
        ],
    )
    def test_no_images(self, no_points, name, dtype, shape):
        empty = no_points.subset_by_image_indices(np.zeros(0, dtype=np.uint32))
        assert empty.image_count == 0
        value = getattr(empty, name)
        assert value.dtype == dtype
        assert value.shape == shape
        assert not value.flags.writeable
        assert getattr(empty, name) is value


class TestMutatingMethods:
    """``save`` and the two recompute methods are the only methods that change
    a reconstruction in place."""

    def test_recompute_point_errors_gathers_the_errors_again(
        self, seoul_bull_ground_truth_sfmr
    ):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        zeroed = recon.clone_with_changes(
            rms_reprojection_errors=np.zeros(recon.point_count, dtype=np.float32)
        )
        before = zeroed.rms_reprojection_errors
        positions = zeroed.positions_xyzw
        assert not before.any()

        zeroed.recompute_point_errors()
        after = zeroed.rms_reprojection_errors
        assert after is not before
        assert after.any()
        assert zeroed.rms_reprojection_errors is after
        _assert_read_only(after)
        # The array read before keeps the errors it was read with.
        assert not before.any()
        # Nothing else the call left alone is gathered again.
        assert zeroed.positions_xyzw is positions

    def test_recompute_infinity_point_errors_gathers_the_errors_again(
        self, seoul_bull_ground_truth_sfmr
    ):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        at_infinity = recon.point_is_at_infinity
        assert at_infinity.any() and not at_infinity.all()
        zeroed = recon.clone_with_changes(
            rms_reprojection_errors=np.zeros(recon.point_count, dtype=np.float32)
        )
        before = zeroed.rms_reprojection_errors

        zeroed.recompute_infinity_point_errors()
        after = zeroed.rms_reprojection_errors
        assert after is not before
        assert not before.any()
        assert after[at_infinity].any()
        assert not after[~at_infinity].any()

    def test_views_survive_every_mutating_method(self, every_column, tmp_path):
        # A view points into the reconstruction's own storage, which is sound
        # only while no method moves or rewrites that storage. Hold a view of
        # every block across every method that takes the object mutably.
        recon, expected = every_column
        views = {name: getattr(recon, name) for name in VIEWS}
        views = {name: v for name, v in views.items() if v is not None}
        gathered = {name: getattr(recon, name) for name in GATHERED}

        recon.save(tmp_path / "plain.sfmr")
        recon.save(tmp_path / "stamped.sfmr", operation="xform")
        recon.save(
            tmp_path / "minimal.sfmr",
            operation="xform",
            minimal=True,
            workspace_path=".",
        )
        recon.recompute_point_errors()
        recon.recompute_infinity_point_errors()

        for name, view in views.items():
            np.testing.assert_array_equal(view, expected[name])
            assert np.shares_memory(view, getattr(recon, name))
        for name, array in gathered.items():
            if name != "rms_reprojection_errors":
                assert getattr(recon, name) is array
