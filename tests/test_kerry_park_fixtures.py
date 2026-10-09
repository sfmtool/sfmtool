# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""What the kerry_park ground-truth fixtures promise.

``kerry_park_ground_truth_sfmr`` is a loadable copy of the checked-in ground
truth; ``kerry_park_workspace`` is a SIFT-backed workspace whose cameras, poses
and rig are that ground truth's for the 8-frame prefix, with the cluster tracks
triangulated at them.
"""

from pathlib import Path

import numpy as np

from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.sift.file import get_sift_path_for_image

from .conftest import (
    KERRY_PARK_GROUND_TRUTH,
    KERRY_PARK_PREFIX_FRAME_COUNT,
    kerry_park_missing_guarantee,
)

#: The deterministic build gives 516 points over 2069 observations; the floor
#: leaves room for a matcher or SIFT change without letting the cloud go thin.
MIN_POINT_COUNT = 400
#: The reprojection gate the fixture triangulates under.
BAR_PX = 2.0


def _prefix_indexes(recon) -> np.ndarray:
    """The ground truth's images in the 8-frame prefix, in its order."""
    return np.array(
        [
            i
            for i, name in enumerate(recon.image_names)
            if int(Path(name).stem.split("_")[-1]) <= KERRY_PARK_PREFIX_FRAME_COUNT
        ],
        dtype=np.int64,
    )


def test_ground_truth_fixture_loads(kerry_park_ground_truth_sfmr: Path):
    recon = SfmrReconstruction.load(kerry_park_ground_truth_sfmr)

    assert kerry_park_ground_truth_sfmr.parent != KERRY_PARK_GROUND_TRUTH.parent
    assert (kerry_park_ground_truth_sfmr.parent / ".sfm-workspace.json").is_file()
    assert recon.image_count == 48
    assert recon.camera_count == 2
    assert all(c.model == "SFMTOOL_FISHEYE" for c in recon.cameras)
    assert recon.rig_frame_data is not None
    assert recon.feature_source == "embedded_patches"
    assert recon.track_feature_indexes is None
    assert recon.world_space_unit == "m"


def test_workspace_fixture_matches_ground_truth(kerry_park_workspace: Path):
    recon = SfmrReconstruction.load(kerry_park_workspace)
    truth = SfmrReconstruction.load(KERRY_PARK_GROUND_TRUTH)
    prefix = _prefix_indexes(truth)

    assert recon.image_count == 2 * KERRY_PARK_PREFIX_FRAME_COUNT
    assert list(recon.image_names) == [truth.image_names[i] for i in prefix]
    assert recon.world_space_unit == "m"
    # The camera, pose and rig columns are the ground truth's, verbatim.
    assert recon.cameras == truth.cameras
    np.testing.assert_array_equal(
        recon.camera_indexes, np.asarray(truth.camera_indexes)[prefix]
    )
    np.testing.assert_array_equal(
        recon.quaternions_wxyz, np.asarray(truth.quaternions_wxyz)[prefix]
    )
    np.testing.assert_array_equal(
        recon.translations, np.asarray(truth.translations)[prefix]
    )
    rig, truth_rig = recon.rig_frame_data, truth.rig_frame_data
    for key in ("sensor_quaternions_wxyz", "sensor_translations_xyz"):
        np.testing.assert_array_equal(rig[key], truth_rig[key])
    np.testing.assert_array_equal(
        rig["image_sensor_indexes"],
        np.asarray(truth_rig["image_sensor_indexes"])[prefix],
    )


def test_workspace_fixture_is_sift_backed(kerry_park_workspace: Path):
    recon = SfmrReconstruction.load(kerry_park_workspace)
    workspace_dir = kerry_park_workspace.parent

    assert recon.feature_source == "sift_files"
    assert recon.track_feature_indexes is not None
    for name in recon.image_names:
        image_path = workspace_dir / name
        assert image_path.is_file()
        assert get_sift_path_for_image(image_path).is_file()


def test_workspace_fixture_points_are_well_placed(kerry_park_workspace: Path):
    recon = SfmrReconstruction.load(kerry_park_workspace)

    assert recon.point_count >= MIN_POINT_COUNT
    assert recon.infinity_point_count == 0
    assert (np.asarray(recon.observation_counts) >= 3).all()
    assert kerry_park_missing_guarantee(recon) is None

    # Per-observation errors, read against the .sift keypoints.
    errors = np.concatenate(
        [
            np.asarray(recon.compute_observation_reprojection_errors(i))[:, 1]
            for i in range(recon.image_count)
        ]
    )
    assert len(errors) == recon.observation_count
    assert np.isfinite(errors).all()
    assert errors.max() <= BAR_PX + 1e-3
    assert np.median(errors) < 0.5
