# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Malformed input to the ``sfmtool.analysis`` alignment, correspondence,
merge and image-pair-graph bindings raises ``ValueError`` naming what is
wrong, where each once raised ``PanicException``."""

import numpy as np
import pytest

from sfmtool.analysis import (
    build_covisibility_pairs,
    build_frustum_intersection_pairs,
    compute_narrow_track_mask,
    find_point_correspondences,
    merge_points_and_tracks,
    ransac_alignment,
)
from sfmtool.reconstruction import SfmrReconstruction
from tests.conftest import SEOUL_BULL_GROUND_TRUTH

TWO_POSES = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])


def _u32(values):
    return np.array(values, dtype=np.uint32)


@pytest.mark.parametrize(
    ("translations", "positions", "match"),
    [
        (np.zeros((1, 3)), np.zeros((1, 4)), "translations has 3 values but 2 rows"),
        (
            np.zeros((2, 3)),
            np.zeros((1, 3)),
            "positions_xyzw has 3 values but 1 rows of 4",
        ),
    ],
)
def test_compute_narrow_track_mask_shapes(translations, positions, match):
    with pytest.raises(ValueError, match=match):
        compute_narrow_track_mask(
            TWO_POSES, translations, positions, _u32([0, 0]), _u32([0, 1]), 0.1
        )


@pytest.mark.parametrize(
    ("source", "target", "match"),
    [
        (np.zeros((3, 2)), np.zeros((3, 2)), "source_points has 6 values"),
        (np.zeros((5, 3)), np.zeros((4, 3)), "target_points has 12 values"),
    ],
)
def test_ransac_alignment_shapes(source, target, match):
    with pytest.raises(ValueError, match=match):
        ransac_alignment(source, target)


def test_find_point_correspondences_track_column_lengths():
    with pytest.raises(ValueError, match="the source track columns differ in length"):
        find_point_correspondences(
            _u32([0, 0]),
            _u32([0]),
            _u32([0, 1]),
            _u32([0]),
            _u32([0]),
            _u32([0]),
            _u32([0]),
            _u32([0]),
        )


def test_build_covisibility_pairs_image_index_out_of_range():
    with pytest.raises(ValueError, match="holds image 5, out of range for 1 images"):
        build_covisibility_pairs(TWO_POSES[:1], _u32([0, 0]), _u32([0, 5]))


def test_build_frustum_intersection_pairs_short_intrinsics():
    one = np.ones(1)
    with pytest.raises(ValueError, match="fx has 1 values but 2 images"):
        build_frustum_intersection_pairs(
            TWO_POSES,
            np.zeros((2, 3)),
            one,
            one,
            one,
            one,
            _u32([10]),
            _u32([10]),
            np.ones((2, 4), dtype=np.uint32),
            np.ones(2),
            np.full(2, 2.0),
        )


def test_merge_points_and_tracks_image_mapping_out_of_range():
    recon = SfmrReconstruction.load(str(SEOUL_BULL_GROUND_TRUTH))
    with pytest.raises(
        ValueError, match="image_mapping names reconstruction 5, out of range"
    ):
        merge_points_and_tracks([recon], [], {"a": [(5, 0)]})
