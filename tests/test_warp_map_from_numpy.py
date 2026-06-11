# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :py:meth:`WarpMap.from_numpy`, the inverse of ``to_numpy``."""

from __future__ import annotations

import numpy as np
import pytest

from sfmtool._sfmtool import ImagePyramid, WarpMap


def test_from_numpy_roundtrips_through_to_numpy():
    """Coordinates put in via from_numpy come back unchanged from to_numpy."""
    h, w = 5, 7
    rng = np.random.default_rng(0)
    map_x = rng.uniform(0, w, size=(h, w)).astype(np.float32)
    map_y = rng.uniform(0, h, size=(h, w)).astype(np.float32)

    wm = WarpMap.from_numpy(map_x, map_y)
    assert (wm.width, wm.height) == (w, h)

    out_x, out_y = wm.to_numpy()
    np.testing.assert_array_equal(out_x, map_x)
    np.testing.assert_array_equal(out_y, map_y)


def test_from_numpy_identity_map_remaps_image_to_itself():
    """An identity coordinate map resamples an image to itself.

    Pixel centers sit at half-integer coordinates (pixel ``(c, r)`` is sampled
    at source ``(c + 0.5, r + 0.5)``), matching the keypoint-position convention.
    """
    h, w = 4, 6
    image = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))

    wm = WarpMap.from_numpy(
        (cols + 0.5).astype(np.float32), (rows + 0.5).astype(np.float32)
    )
    out = np.asarray(wm.remap_bilinear(image))

    np.testing.assert_array_equal(out, image)


def test_from_numpy_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        WarpMap.from_numpy(np.zeros((3, 4), np.float32), np.zeros((4, 3), np.float32))


def test_image_pyramid_remap_aniso_matches_one_shot():
    """ImagePyramid.remap_aniso reuses a prebuilt pyramid but must match the
    one-shot WarpMap.remap_aniso pixel for pixel."""
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(40, 48), dtype=np.uint8)

    # A 2x-downscaling (compressive) map so the anisotropic path engages.
    h, w = 20, 24
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (cols * 2 + 0.5).astype(np.float32)
    map_y = (rows * 2 + 0.5).astype(np.float32)

    one_shot = np.asarray(WarpMap.from_numpy(map_x, map_y).remap_aniso(image))

    pyr = ImagePyramid(image)
    reused = np.asarray(pyr.remap_aniso(WarpMap.from_numpy(map_x, map_y)))

    assert pyr.num_levels >= 2
    np.testing.assert_array_equal(reused, one_shot)
