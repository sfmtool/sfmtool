# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sfmtool SIFT extraction bindings (`extract_sift`, `detect_sift_keypoints`)."""

import numpy as np
import pytest

from sfmtool._sfmtool.sift import (
    affine_shapes_from_similarity,
    describe_keypoints,
    detect_sift_keypoints,
    extract_sift,
)


def _noise_image(n: int = 224, seed: int = 0) -> np.ndarray:
    """A random RGB image — high-frequency content yields plenty of keypoints."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.integers(0, 256, (n, n, 3), dtype=np.uint8))


class TestParamValidation:
    """Degenerate params are rejected with ValueError instead of panicking."""

    @pytest.mark.parametrize(
        "params",
        [
            {"orientation_bins": 0},  # would hit rem_euclid(0)
            {"octave_layers": 0},  # would make k = 2^inf
            {"sigma": 0.0},
            {"sigma": float("inf")},
            {"blur_radius_factor": 0.0},
            {"input_sigma": float("nan")},
            {"descriptor_magnification": 0.0},
            {"descriptor_clamp": 0.0},
            {"descriptor_width": 5},  # fixed at 4
            {"descriptor_bins": 7},  # fixed at 8
            {"unknown_key": 1},
        ],
    )
    def test_bad_params_raise(self, params):
        with pytest.raises(ValueError):
            extract_sift(_noise_image(64), params)

    def test_defaults_and_valid_overrides_work(self):
        img = _noise_image(96)
        assert extract_sift(img)[0].shape[1] == 2
        # A valid magnification/clamp override is accepted.
        extract_sift(img, {"descriptor_magnification": 4.0, "descriptor_clamp": 0.3})


class TestMaxDescribed:
    """`max_described` describes only the top-k prefix; detection returns all."""

    def test_partial_describe_shapes(self):
        img = _noise_image(224)
        pos, _aff, desc = extract_sift(img)
        n = len(pos)
        assert n > 10, f"need enough keypoints for the test (got {n})"
        assert desc.shape == (n, 128)

        p2, _a2, d2 = extract_sift(img, None, 10)
        assert len(p2) == n, "detection must return every keypoint"
        assert d2.shape == (10, 128), "only the prefix is described"
        assert np.array_equal(d2, desc[:10]), "prefix must match the full extract"

        # A cap >= the keypoint count (or None) describes everything.
        assert extract_sift(img, None, n + 100)[2].shape[0] == n
        assert extract_sift(img, None, None)[2].shape[0] == n

    def test_detect_only_matches_extract_keypoints(self):
        img = _noise_image(160)
        det_pos, det_aff, responses = detect_sift_keypoints(img)
        ext_pos, _ext_aff, _desc = extract_sift(img)
        assert det_pos.shape == ext_pos.shape
        assert len(responses) == len(det_pos)
        assert np.array_equal(det_pos, ext_pos)


class TestDeterminismAndOrder:
    def test_repeatable_output(self):
        img = _noise_image(224, seed=3)
        pos_a, aff_a, desc_a = extract_sift(img)
        pos_b, aff_b, desc_b = extract_sift(img)
        assert np.array_equal(pos_a, pos_b)
        assert np.array_equal(aff_a, aff_b)
        assert np.array_equal(desc_a, desc_b)

    def test_sorted_by_descending_size(self):
        # The .sift format / cap reproducibility relies on a stable descending-size
        # order; verify the invariant holds on a keypoint-rich image.
        _pos, aff, _desc = extract_sift(_noise_image(224, seed=3))
        sizes = 0.5 * (
            np.linalg.norm(aff[:, :, 0], axis=1) + np.linalg.norm(aff[:, :, 1], axis=1)
        )
        assert len(sizes) > 50
        assert np.all(np.diff(sizes) <= 1e-4), "keypoints not sorted by descending size"


class TestDescribeKeypoints:
    """`describe_keypoints` describes keypoints the detector did not find."""

    def test_describing_the_detections_reproduces_their_descriptors(self):
        """Both query forms reproduce the extractor's bytes exactly.

        A descriptor is a pure function of the scale space and a keypoint, so
        asking for one at a detection's own position, shape and size must return
        the detection's own descriptor -- the octave and pyramid level are
        recovered from the size rather than remembered.
        """
        img = _noise_image(224, seed=5)
        pos, aff, desc = extract_sift(img)
        n = len(pos)
        assert n > 50, f"need a non-trivial keypoint count, got {n}"

        from_shape = describe_keypoints(img, pos, aff)
        assert from_shape.shape == (n, 128)
        assert np.array_equal(from_shape, desc)

        # The same keypoints stated as (size, orientation).
        scales = 0.5 * (
            np.linalg.norm(aff[:, :, 0], axis=1) + np.linalg.norm(aff[:, :, 1], axis=1)
        )
        orientations = np.arctan2(aff[:, 1, 0], aff[:, 0, 0])
        shapes = affine_shapes_from_similarity(
            scales.astype(np.float32), orientations.astype(np.float32)
        )
        assert shapes.shape == (n, 2, 2)
        assert np.array_equal(describe_keypoints(img, pos, shapes), desc)

    def test_a_keypoint_of_its_own_beside_the_detections(self):
        """A pixel nothing detected describes fine, and differently."""
        img = _noise_image(128, seed=7)
        pos, aff, _desc = extract_sift(img)
        query_pos = np.array([[64.5, 64.5]], dtype=np.float32)
        query_aff = affine_shapes_from_similarity(
            np.array([5.0], dtype=np.float32), np.array([0.4], dtype=np.float32)
        )
        row = describe_keypoints(img, query_pos, query_aff)
        assert row.shape == (1, 128)
        assert row.any(), "a textured patch must describe to something"
        # Not a copy of some detection's row.
        described = describe_keypoints(img, pos, aff)
        assert not (described == row).all(axis=1).any()

    @pytest.mark.parametrize(
        "position",
        [
            [-1.0, 10.0],  # left of the image
            [10.0, 128.0],  # one row past the bottom
            [500.0, 10.0],  # far right
            [float("nan"), 10.0],
        ],
    )
    def test_a_keypoint_outside_the_image_is_refused(self, position):
        img = _noise_image(128)
        pos = np.array([[20.0, 20.0], position], dtype=np.float32)
        aff = affine_shapes_from_similarity(
            np.array([4.0, 4.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        )
        with pytest.raises(ValueError, match="outside"):
            describe_keypoints(img, pos, aff)

    def test_a_keypoint_with_no_size_is_refused(self):
        img = _noise_image(128)
        pos = np.array([[20.0, 20.0], [30.0, 30.0]], dtype=np.float32)
        aff = np.zeros((2, 2, 2), dtype=np.float32)
        aff[0] = [[4.0, 0.0], [0.0, 4.0]]
        with pytest.raises(ValueError, match="positive size"):
            describe_keypoints(img, pos, aff)

    def test_mismatched_arrays_are_refused(self):
        img = _noise_image(64)
        pos = np.zeros((3, 2), dtype=np.float32) + 20.0
        with pytest.raises(ValueError, match="affine_shapes"):
            describe_keypoints(img, pos, np.zeros((2, 2, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="positions"):
            describe_keypoints(
                img,
                np.zeros((3, 3), dtype=np.float32),
                np.zeros((3, 2, 2), dtype=np.float32),
            )

    def test_no_keypoints_describes_nothing(self):
        out = describe_keypoints(
            _noise_image(64),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2, 2), dtype=np.float32),
        )
        assert out.shape == (0, 128)
