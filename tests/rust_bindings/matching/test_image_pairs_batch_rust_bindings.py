# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that match_image_pair and match_image_pairs_batch check their inputs
before matching, raising ValueError rather than panicking."""

import numpy as np
import pytest

from sfmtool.matching import match_image_pair, match_image_pairs_batch

_K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])


def _features():
    """Four features with distinct descriptors and identity affine shapes."""
    positions = np.array([[100.0, 80.0], [500.0, 80.0], [100.0, 400.0], [500.0, 400.0]])
    descriptors = np.zeros((4, 128), dtype=np.uint8)
    descriptors[:, 0] = [0, 50, 100, 150]
    affines = np.tile([1.0, 0.0, 0.0, 1.0], (4, 1))
    return positions, descriptors, affines


def _pair_args():
    """Arguments for match_image_pair between two copies of _features()
    with sideways motion, as a dict of keywords."""
    positions, descriptors, _ = _features()
    return {
        "k1": _K,
        "k2": _K,
        "r1": np.eye(3),
        "r2": np.eye(3),
        "t1": np.zeros(3),
        "t2": np.array([1.0, 0.0, 0.0]),
        "width1": 640,
        "height1": 480,
        "width2": 640,
        "height2": 480,
        "positions1": positions,
        "descriptors1": descriptors,
        "positions2": positions,
        "descriptors2": descriptors,
        "window_size": 5,
    }


def _with_filter(args, affines1, affines2):
    """`args` with the given affine shapes and the default filter settings."""
    return {
        **args,
        "affines1": affines1,
        "affines2": affines2,
        "max_angle_difference": 15.0,
        "min_triangulation_angle": 5.0,
        "geometric_size_ratio_min": 0.8,
        "geometric_size_ratio_max": 1.25,
    }


def _batch_args(n_images=2, n_cameras=1):
    """Arguments for a batch over images of _features(), as a dict of
    keywords."""
    positions, descriptors, _ = _features()
    return {
        "pairs": [(0, 1)],
        "intrinsics": [_K] * n_cameras,
        "rotations": [np.eye(3)] * n_images,
        "translations": [np.array([float(i), 0.0, 0.0]) for i in range(n_images)],
        "camera_indices": np.zeros(n_images, dtype=np.int64),
        "positions_list": [positions] * n_images,
        "descriptors_list": [descriptors] * n_images,
        "widths": [640] * n_cameras,
        "heights": [480] * n_cameras,
        "window_size": 5,
        "threshold": None,
        "rectification_margin": 50,
    }


def _batch_with_filter(args, affines_list):
    return {
        **args,
        "affines_list": affines_list,
        "max_angle_difference": 15.0,
        "min_triangulation_angle": 5.0,
        "geometric_size_ratio_min": 0.8,
        "geometric_size_ratio_max": 1.25,
    }


# match_image_pair


def test_valid_pair_matches_every_feature_to_itself():
    matches = match_image_pair(**_pair_args())
    assert [(i, j) for i, j, _ in matches] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    _, _, affines = _features()
    filtered = match_image_pair(**_with_filter(_pair_args(), affines, affines))
    assert filtered == matches


@pytest.mark.parametrize("name", ["k1", "k2", "r1", "r2"])
@pytest.mark.parametrize("shape", [(2, 2), (1, 9), (3, 4)])
def test_pair_matrix_not_3x3_raises_value_error(name, shape):
    args = _pair_args()
    args[name] = np.ones(shape)
    with pytest.raises(
        ValueError, match=rf"{name} has shape .* must have shape \(3, 3\)"
    ):
        match_image_pair(**args)


@pytest.mark.parametrize("name", ["t1", "t2"])
@pytest.mark.parametrize("length", [2, 4])
def test_pair_translation_not_length_3_raises_value_error(name, length):
    args = _pair_args()
    args[name] = np.zeros(length)
    with pytest.raises(ValueError, match=rf"{name} has shape \({length},\)"):
        match_image_pair(**args)


@pytest.mark.parametrize("name", ["positions1", "positions2"])
def test_pair_positions_not_two_columns_raises_value_error(name):
    args = _pair_args()
    args[name] = np.zeros((4, 3))
    with pytest.raises(ValueError, match=rf"{name} has shape \(4, 3\)"):
        match_image_pair(**args)


@pytest.mark.parametrize("name", ["descriptors1", "descriptors2"])
def test_pair_descriptors_with_too_few_rows_raise_value_error(name):
    args = _pair_args()
    args[name] = args[name][:3]
    with pytest.raises(ValueError, match=rf"{name} has shape \(3, 128\)"):
        match_image_pair(**args)


def test_pair_descriptors_of_different_widths_raise_value_error():
    args = _pair_args()
    args["descriptors2"] = np.zeros((4, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"descriptors2 .* must have shape \(4, 128\)"):
        match_image_pair(**args)


@pytest.mark.parametrize("which", [1, 2])
@pytest.mark.parametrize("shape", [(3, 4), (4, 3)])
def test_pair_bad_affine_shape_raises_value_error(which, shape):
    _, _, affines = _features()
    bad = np.zeros(shape)
    args = _with_filter(
        _pair_args(), bad if which == 1 else affines, bad if which == 2 else affines
    )
    with pytest.raises(ValueError, match=rf"affines{which} has shape"):
        match_image_pair(**args)


# match_image_pairs_batch


def test_valid_batch_returns_one_result_per_pair():
    args = _batch_args()
    result = match_image_pairs_batch(**args)
    assert len(result) == 1
    assert [(i, j) for i, j, _ in result[0]] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert result[0] == match_image_pair(**_pair_args())
    _, _, affines = _features()
    assert match_image_pairs_batch(**_batch_with_filter(args, [affines] * 2)) == result


def test_batch_with_no_features_returns_empty_matches():
    args = _batch_args()
    args["positions_list"] = [np.zeros((0, 2))] * 2
    args["descriptors_list"] = [np.zeros((0, 128), dtype=np.uint8)] * 2
    assert match_image_pairs_batch(**args) == [[]]


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
    args["affines_list"] = [np.zeros((4, 4))]
    with pytest.raises(ValueError, match="affines_list has 1 entries"):
        match_image_pairs_batch(**args)


def test_per_camera_length_mismatch_raises_value_error():
    args = _batch_args()
    args["widths"] = [640, 640]
    with pytest.raises(ValueError, match="widths has 2 entries"):
        match_image_pairs_batch(**args)


@pytest.mark.parametrize("shape", [(2, 2), (1, 9)])
def test_batch_intrinsics_not_3x3_raises_value_error(shape):
    args = _batch_args()
    args["intrinsics"] = [np.ones(shape)]
    with pytest.raises(ValueError, match=r"intrinsics\[0\] has shape"):
        match_image_pairs_batch(**args)


@pytest.mark.parametrize("shape", [(2, 2), (1, 9)])
def test_batch_rotation_not_3x3_raises_value_error(shape):
    args = _batch_args()
    args["rotations"] = [np.eye(3), np.ones(shape)]
    with pytest.raises(ValueError, match=r"rotations\[1\] has shape"):
        match_image_pairs_batch(**args)


def test_batch_translation_not_length_3_raises_value_error():
    args = _batch_args()
    args["translations"] = [np.zeros(3), np.zeros(2)]
    with pytest.raises(ValueError, match=r"translations\[1\] has shape \(2,\)"):
        match_image_pairs_batch(**args)


def test_batch_positions_not_two_columns_raises_value_error():
    args = _batch_args()
    args["positions_list"] = [args["positions_list"][0], np.zeros((4, 3))]
    with pytest.raises(ValueError, match=r"positions_list\[1\] has shape \(4, 3\)"):
        match_image_pairs_batch(**args)


def test_batch_descriptors_with_too_few_rows_raise_value_error():
    args = _batch_args()
    descriptors = args["descriptors_list"][0]
    args["descriptors_list"] = [descriptors, descriptors[:3]]
    with pytest.raises(ValueError, match=r"descriptors_list\[1\] has shape \(3, 128\)"):
        match_image_pairs_batch(**args)


def test_batch_descriptors_of_different_widths_raise_value_error():
    args = _batch_args()
    args["descriptors_list"] = [
        args["descriptors_list"][0],
        np.zeros((4, 64), dtype=np.uint8),
    ]
    with pytest.raises(ValueError, match=r"descriptors_list\[1\] .* \(4, 128\)"):
        match_image_pairs_batch(**args)


def test_batch_affines_with_too_few_rows_raise_value_error():
    _, _, affines = _features()
    args = _batch_with_filter(_batch_args(), [affines, affines[:3]])
    with pytest.raises(ValueError, match=r"affines_list\[1\] has shape \(3, 4\)"):
        match_image_pairs_batch(**args)


def test_batch_checks_an_image_no_pair_names():
    args = _batch_args(n_images=3)
    args["descriptors_list"] = args["descriptors_list"][:2] + [
        np.zeros((2, 128), dtype=np.uint8)
    ]
    with pytest.raises(ValueError, match=r"descriptors_list\[2\]"):
        match_image_pairs_batch(**args)
