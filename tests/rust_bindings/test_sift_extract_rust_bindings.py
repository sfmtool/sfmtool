# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the sfmtool SIFT extraction bindings (`extract_sift`, `detect_sift_keypoints`)."""

import numpy as np
import pytest

from sfmtool._sfmtool import detect_sift_keypoints, extract_sift


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
