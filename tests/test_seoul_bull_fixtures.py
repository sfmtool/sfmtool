# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""What the seoul_bull ground-truth fixtures promise.

``seoul_bull_ground_truth_sfmr`` is a loadable copy of the checked-in ground
truth; ``seoul_bull_workspace`` is a SIFT-backed workspace whose cameras and
poses are that ground truth's, with the cluster tracks triangulated at them.
"""

from pathlib import Path

import numpy as np

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool.sift.file import get_sift_path_for_image

from .conftest import SEOUL_BULL_GROUND_TRUTH

#: The deterministic build gives 2486 points over 6132 observations; the floor
#: leaves room for a matcher or SIFT change without letting the cloud go thin.
MIN_POINT_COUNT = 2000
#: The reprojection gate the fixture triangulates under.
BAR_PX = 2.0


def test_ground_truth_fixture_loads(seoul_bull_ground_truth_sfmr: Path):
    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)

    assert seoul_bull_ground_truth_sfmr.parent != SEOUL_BULL_GROUND_TRUTH.parent
    assert (seoul_bull_ground_truth_sfmr.parent / ".sfm-workspace.json").is_file()
    assert recon.image_count == 17
    assert recon.feature_source == "embedded_patches"
    assert recon.track_feature_indexes is None
    assert recon.world_space_unit == "m"


def test_workspace_fixture_matches_ground_truth(seoul_bull_workspace: Path):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    truth = SfmrReconstruction.load(SEOUL_BULL_GROUND_TRUTH)

    assert recon.image_count == 17
    assert [Path(n).name for n in recon.image_names] == list(truth.image_names)
    assert recon.world_space_unit == "m"
    # The camera and pose columns are the ground truth's, verbatim.
    assert recon.cameras == truth.cameras
    np.testing.assert_array_equal(recon.camera_indexes, truth.camera_indexes)
    np.testing.assert_array_equal(recon.quaternions_wxyz, truth.quaternions_wxyz)
    np.testing.assert_array_equal(recon.translations, truth.translations)


def test_workspace_fixture_is_sift_backed(seoul_bull_workspace: Path):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    workspace_dir = seoul_bull_workspace.parent

    assert recon.feature_source == "sift_files"
    assert recon.track_feature_indexes is not None
    for name in recon.image_names:
        image_path = workspace_dir / name
        assert image_path.is_file()
        assert get_sift_path_for_image(image_path).is_file()


def test_workspace_fixture_points_are_well_placed(seoul_bull_workspace: Path):
    recon = SfmrReconstruction.load(seoul_bull_workspace)

    assert recon.point_count >= MIN_POINT_COUNT
    assert recon.infinity_point_count == 0
    assert (np.asarray(recon.observation_counts) >= 2).all()

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
