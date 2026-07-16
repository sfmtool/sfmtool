# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.analysis`."""

import sfmtool._sfmtool.analysis as analysis

_EXPECTED_FUNCTIONS = (
    "apply_se3_to_camera_poses_py",
    "compute_narrow_track_mask",
    "estimate_alignment_rs",
    "ransac_alignment_rs",
    "find_point_correspondences_py",
    "merge_points_and_tracks_py",
    "filter_tracks_by_point_mask_py",
    "triangulate_batch",
    "epipolar_curves",
    "build_covisibility_pairs_py",
    "build_frustum_intersection_pairs_py",
)


def test_all_analysis_bindings_registered():
    """Every expected function is present on `_sfmtool.analysis`."""
    missing = [name for name in _EXPECTED_FUNCTIONS if not hasattr(analysis, name)]
    assert not missing, f"missing analysis bindings: {missing}"
    for name in _EXPECTED_FUNCTIONS:
        assert callable(getattr(analysis, name)), f"{name} is not callable"


def test_analysis_submodule_public_name():
    """The submodule reports its public `__name__` so binding objects'
    `__module__` reads `sfmtool.analysis`."""
    assert analysis.__name__ == "sfmtool.analysis"
    for name in _EXPECTED_FUNCTIONS:
        assert getattr(analysis, name).__module__ == "sfmtool.analysis"
