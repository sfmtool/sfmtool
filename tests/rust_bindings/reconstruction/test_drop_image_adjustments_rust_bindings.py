# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `SfmrReconstruction.drop_image_adjustments` on the seoul_bull
ground truth whose one `SFMTOOL_PINHOLE` camera varies all four per-image
parameters, and for `SfmrReconstruction.varied_image_adjustments` on it and on
the kerry_park ground truth, whose two cameras vary none."""

import shutil

import numpy as np
import pytest

from sfmtool.reconstruction import SfmrReconstruction
from tests.conftest import KERRY_PARK_GROUND_TRUTH, SEOUL_BULL_DIR

GROUND_TRUTH_V02 = SEOUL_BULL_DIR / "seoul_bull_sculpture_ground_truth_v02.sfmr"
ALL = ["focal", "principal_point", "radial_bspline", "rolling_shutter"]


@pytest.fixture(scope="module")
def recon():
    return SfmrReconstruction.load(GROUND_TRUTH_V02)


def parameters(camera):
    return camera.to_dict()["parameters"]


def test_the_fixture_varies_every_parameter(recon):
    assert recon.per_image_adjustments == [ALL]
    assert recon.rolling_shutter_scanlines == ["COLUMNS"]


def test_dropping_every_parameter_folds_the_mean_into_the_camera(recon, tmp_path):
    s = recon.focal_adjustments
    a = recon.principal_point_adjustments_xy
    before = parameters(recon.cameras[0])

    dropped = recon.drop_image_adjustments(0, ALL)

    assert dropped.per_image_adjustments == [[]]
    assert dropped.focal_adjustments is None
    assert dropped.principal_point_adjustments_xy is None
    assert dropped.radial_bspline_adjustments is None
    assert dropped.rolling_shutter_rates_xyz is None
    # The scanlines setting describes the sensor and is kept.
    assert dropped.rolling_shutter_scanlines == ["COLUMNS"]
    after = parameters(dropped.cameras[0])
    assert after["focal_length"] == pytest.approx(
        before["focal_length"] * np.exp(np.mean(np.log1p(s))), rel=1e-12
    )
    short_side = min(recon.cameras[0].width, recon.cameras[0].height)
    mean_a = a.mean(axis=0)
    assert after["principal_point_x"] == pytest.approx(
        before["principal_point_x"] + mean_a[0] * short_side, abs=1e-9
    )
    assert after["principal_point_y"] == pytest.approx(
        before["principal_point_y"] + mean_a[1] * short_side, abs=1e-9
    )
    # Every image is now read through the camera itself.
    for i in range(dropped.image_count):
        assert dropped.image_intrinsics(i) == dropped.cameras[0]
    np.testing.assert_array_equal(dropped.positions_xyzw, recon.positions_xyzw)
    np.testing.assert_array_equal(dropped.quaternions_wxyz, recon.quaternions_wxyz)
    # This object is not changed.
    assert recon.per_image_adjustments == [ALL]
    assert parameters(recon.cameras[0]) == before

    # The result is a file the format accepts. A save resolves the workspace
    # the file is in, so the copy gets a marker of its own.
    shutil.copy(SEOUL_BULL_DIR / ".sfm-workspace.json", tmp_path)
    path = tmp_path / "dropped.sfmr"
    dropped.save(path)
    reread = SfmrReconstruction.load(path)
    assert reread.per_image_adjustments == [[]]
    assert reread.rolling_shutter_scanlines == ["COLUMNS"]


def test_a_parameter_left_alone_is_kept(recon):
    dropped = recon.drop_image_adjustments(0, ["rolling_shutter"])
    assert dropped.per_image_adjustments == [ALL[:3]]
    np.testing.assert_array_equal(dropped.focal_adjustments, recon.focal_adjustments)
    assert dropped.rolling_shutter_rates_xyz is None
    # No rate is folded, so the camera is the one it was.
    assert dropped.cameras[0] == recon.cameras[0]
    # Dropping it again changes nothing.
    again = dropped.drop_image_adjustments(0, ["rolling_shutter"])
    assert again.per_image_adjustments == dropped.per_image_adjustments
    assert again.cameras[0] == dropped.cameras[0]


def test_an_image_at_the_mean_is_read_as_before(recon):
    # Put every image's focal scale and offset at one shared value: it is then
    # the mean, and the image's intrinsics survive the drop to rounding.
    n = recon.image_count
    held = recon.clone_with_changes(
        focal_adjustments=np.full(n, 0.004),
        principal_point_adjustments_xy=np.tile([0.002, -0.003], (n, 1)),
    )
    before = held.image_intrinsics(5).to_dict()["parameters"]
    dropped = held.drop_image_adjustments(0, ["focal", "principal_point"])
    after = dropped.image_intrinsics(5).to_dict()["parameters"]
    for key in ("focal_length", "principal_point_x", "principal_point_y"):
        assert after[key] == pytest.approx(before[key], rel=1e-14), key
    assert after["focal_length"] != parameters(held.cameras[0])["focal_length"]


def test_bad_arguments_raise_value_error(recon):
    with pytest.raises(ValueError, match="'distortion'|\"distortion\" is not one of"):
        recon.drop_image_adjustments(0, ["distortion"])
    with pytest.raises(ValueError, match="camera 3 does not exist"):
        recon.drop_image_adjustments(3, ["focal"])


def test_varied_image_adjustments_lists_every_flagged_parameter(recon):
    assert recon.varied_image_adjustments(0) == ALL


def test_a_flag_with_every_value_zero_still_varies(recon):
    n = recon.image_count
    zeroed = recon.clone_with_changes(
        focal_adjustments=np.zeros(n),
        principal_point_adjustments_xy=np.zeros((n, 2)),
        radial_bspline_adjustments=np.zeros((n, 2)),
        rolling_shutter_rates_xyz=np.zeros((n, 3)),
    )
    assert zeroed.per_image_adjustments == [ALL]
    assert zeroed.varied_image_adjustments(0) == ALL


@pytest.fixture(scope="module")
def kerry_park():
    return SfmrReconstruction.load(KERRY_PARK_GROUND_TRUTH)


def test_a_camera_with_no_flag_and_no_value_varies_nothing(kerry_park):
    assert kerry_park.per_image_adjustments == [[], []]
    assert kerry_park.varied_image_adjustments(0) == []
    assert kerry_park.varied_image_adjustments(1) == []


def test_a_non_zero_value_with_no_flag_varies(kerry_park):
    # One image of camera 1 holds a principal-point offset its camera does not
    # flag; camera 0 is unchanged.
    image = int(np.flatnonzero(np.asarray(kerry_park.camera_indexes) == 1)[0])
    offsets = np.zeros((kerry_park.image_count, 2))
    offsets[image] = [0.001, 0.0]
    stray = kerry_park.clone_with_changes(principal_point_adjustments_xy=offsets)
    assert stray.per_image_adjustments == [[], []]
    assert stray.varied_image_adjustments(0) == []
    assert stray.varied_image_adjustments(1) == ["principal_point"]


def test_varied_image_adjustments_refuses_an_unknown_camera(recon):
    with pytest.raises(ValueError, match="camera 1 does not exist"):
        recon.varied_image_adjustments(1)
