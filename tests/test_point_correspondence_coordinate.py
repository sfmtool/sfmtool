# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for coordinate-based 3D point correspondence.

These exercise the pure voting core (``_vote_point_correspondences``), which
matches observations by 2D keypoint coordinate and is independent of the SIFT
files — so it can be tested with synthetic observation dictionaries.
"""

import numpy as np
import pytest

from sfmtool._point_correspondence import _vote_point_correspondences


def _obs(xy, point_ids):
    return (np.asarray(xy, dtype=np.float64), np.asarray(point_ids, dtype=np.int64))


def test_matches_coincident_keypoints_across_backends():
    # Source and target use different feature indices/orderings, but the same
    # scene keypoints land at (nearly) the same pixels. Point 10<->20 appears in
    # both shared images; 11<->21 only in the first.
    source = {
        0: _obs([[10.0, 10.0], [50.0, 50.0]], [10, 11]),
        1: _obs([[12.0, 12.0]], [10]),
    }
    target = {
        0: _obs([[50.3, 49.8], [10.2, 9.9]], [21, 20]),  # reordered
        1: _obs([[11.8, 12.1]], [20]),
    }
    shared = [(0, 0), (1, 1)]

    corr = _vote_point_correspondences(
        source, target, shared, pixel_threshold=2.0, min_votes=2
    )
    # 10<->20 is supported by both images (2 votes); 11<->21 by only one.
    assert corr == {10: 20}


def test_min_votes_one_keeps_single_image_pairs():
    source = {0: _obs([[10.0, 10.0], [50.0, 50.0]], [10, 11])}
    target = {0: _obs([[10.1, 9.9], [49.9, 50.2]], [20, 21])}

    corr = _vote_point_correspondences(
        source, target, [(0, 0)], pixel_threshold=2.0, min_votes=1
    )
    assert corr == {10: 20, 11: 21}


def test_threshold_rejects_distant_keypoints():
    source = {0: _obs([[10.0, 10.0]], [10])}
    target = {0: _obs([[20.0, 20.0]], [20])}  # ~14px away

    assert (
        _vote_point_correspondences(
            source, target, [(0, 0)], pixel_threshold=2.0, min_votes=1
        )
        == {}
    )


def test_mutual_nearest_neighbor_only():
    # Two source keypoints both nearest to the same single target keypoint; only
    # the mutually-nearest pair should match.
    source = {0: _obs([[10.0, 10.0], [10.8, 10.0]], [10, 11])}
    target = {0: _obs([[10.1, 10.0]], [20])}

    corr = _vote_point_correspondences(
        source, target, [(0, 0)], pixel_threshold=2.0, min_votes=1
    )
    assert corr == {10: 20}


def test_one_to_one_resolution_prefers_strongest_support():
    # 10 wants 20 with 2 votes; 11 wants 20 with 1 vote. 20 can only be claimed
    # once, so the stronger pair (10<->20) wins and 11 is dropped.
    source = {
        0: _obs([[10.0, 10.0], [10.05, 10.0]], [10, 11]),
        1: _obs([[10.0, 10.0]], [10]),
    }
    target = {
        0: _obs([[10.0, 10.0]], [20]),
        1: _obs([[10.0, 10.0]], [20]),
    }
    # Image 0 mutual-NN: target's single kp matches whichever source kp is
    # closest (10), giving 10<->20. Image 1 gives 10<->20 again -> 2 votes.
    corr = _vote_point_correspondences(
        source, target, [(0, 0), (1, 1)], pixel_threshold=2.0, min_votes=1
    )
    assert corr == {10: 20}


def test_skips_images_missing_from_one_side():
    source = {0: _obs([[10.0, 10.0]], [10])}
    target = {5: _obs([[10.0, 10.0]], [20])}
    # Shared pair references images that have no observations on one side.
    assert (
        _vote_point_correspondences(
            source, target, [(0, 1)], pixel_threshold=2.0, min_votes=1
        )
        == {}
    )


@pytest.mark.parametrize("min_votes", [2, 3])
def test_requires_min_votes(min_votes):
    source = {0: _obs([[10.0, 10.0]], [10])}
    target = {0: _obs([[10.0, 10.0]], [20])}
    # Only one supporting image -> needs min_votes <= 1 to keep.
    assert (
        _vote_point_correspondences(
            source, target, [(0, 0)], pixel_threshold=2.0, min_votes=min_votes
        )
        == {}
    )
