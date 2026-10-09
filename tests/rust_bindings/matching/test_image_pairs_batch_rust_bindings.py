# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that match_image_pairs_batch checks its indexes before matching."""

import numpy as np
import pytest

from sfmtool.matching import match_image_pairs_batch


def _batch_args(n_images=2, n_cameras=1):
    """Arguments for a small batch with no features, as a dict of keywords."""
    k = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    return {
        "pairs": [(0, 1)],
        "intrinsics": [k] * n_cameras,
        "rotations": [np.eye(3)] * n_images,
        "translations": [np.array([float(i), 0.0, 0.0]) for i in range(n_images)],
        "camera_indices": np.zeros(n_images, dtype=np.int64),
        "positions_list": [np.zeros((0, 2))] * n_images,
        "descriptors_list": [np.zeros((0, 128), dtype=np.uint8)] * n_images,
        "widths": [100] * n_cameras,
        "heights": [100] * n_cameras,
        "window_size": 30,
        "threshold": None,
        "rectification_margin": 50,
    }


def test_valid_batch_returns_one_result_per_pair():
    assert match_image_pairs_batch(**_batch_args()) == [[]]


@pytest.mark.parametrize("bad_camera", [-1, 1, 7])
def test_bad_camera_index_raises_value_error(bad_camera):
    args = _batch_args()
    args["camera_indices"] = np.array([0, bad_camera], dtype=np.int64)
    with pytest.raises(ValueError, match="camera index"):
        match_image_pairs_batch(**args)


@pytest.mark.parametrize("pair", [(0, 2), (5, 1)])
def test_bad_pair_image_index_raises_value_error(pair):
    args = _batch_args()
    args["pairs"] = [pair]
    with pytest.raises(ValueError, match="out of range for 2 images"):
        match_image_pairs_batch(**args)


def test_per_image_length_mismatch_raises_value_error():
    args = _batch_args()
    args["rotations"] = [np.eye(3)]
    with pytest.raises(ValueError, match="rotations has 1 entries"):
        match_image_pairs_batch(**args)


def test_affines_length_mismatch_raises_value_error():
    args = _batch_args()
    args["affines_list"] = [np.zeros((0, 4))]
    with pytest.raises(ValueError, match="affines_list has 1 entries"):
        match_image_pairs_batch(**args)


def test_per_camera_length_mismatch_raises_value_error():
    args = _batch_args()
    args["widths"] = [100, 100]
    with pytest.raises(ValueError, match="widths has 2 entries"):
        match_image_pairs_batch(**args)
