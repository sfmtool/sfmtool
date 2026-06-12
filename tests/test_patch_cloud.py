# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for PatchCloud.from_reconstruction against a real reconstruction.

These exercise the FeatureSize extent (the default), which sizes each patch
from the observing keypoints' scales read from the workspace ``.sift`` files —
a path the Rust unit tests cannot cover without on-disk SIFT fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfmtool._sfmtool import PatchCloud, SfmrReconstruction


def test_feature_size_success_path(sfmrfile_reconstruction_with_17_images: Path):
    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    cloud = PatchCloud.from_reconstruction(
        recon, extent="feature_size", extent_value=5.0
    )

    assert len(cloud) > 0
    halves = np.array([cloud[i].half_extent for i in range(len(cloud))])
    # Every patch got a real, positive world size from its keypoint scales.
    assert np.all(np.isfinite(halves))
    assert np.all(halves > 0.0)
    # The two axes are square (isotropic feature size).
    np.testing.assert_allclose(halves[:, 0], halves[:, 1])
    # Real scenes have a spread of feature scales, so sizes vary across patches
    # (i.e. this is genuinely per-patch, not a constant fallback).
    assert halves[:, 0].std() > 0.0


def test_feature_size_scales_linearly_with_factor(
    sfmrfile_reconstruction_with_17_images: Path,
):
    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    cloud5 = PatchCloud.from_reconstruction(
        recon, extent="feature_size", extent_value=5.0
    )
    cloud10 = PatchCloud.from_reconstruction(
        recon, extent="feature_size", extent_value=10.0
    )

    assert list(cloud5.point_ids) == list(cloud10.point_ids)
    h5 = np.array([cloud5[i].half_extent[0] for i in range(len(cloud5))])
    h10 = np.array([cloud10[i].half_extent[0] for i in range(len(cloud10))])
    # half = factor * reduce(per-view feature size): doubling the factor doubles
    # every patch's half-size (pins the `factor *` and a deterministic reduce).
    np.testing.assert_allclose(h10, 2.0 * h5, rtol=1e-12)
