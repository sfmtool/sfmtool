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

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.patches import PatchCloud


def test_feature_size_success_path(seoul_bull_workspace: Path):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
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
    seoul_bull_workspace: Path,
):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    cloud5 = PatchCloud.from_reconstruction(
        recon, extent="feature_size", extent_value=5.0
    )
    cloud10 = PatchCloud.from_reconstruction(
        recon, extent="feature_size", extent_value=10.0
    )

    assert list(cloud5.point_indexes) == list(cloud10.point_indexes)
    h5 = np.array([cloud5[i].half_extent[0] for i in range(len(cloud5))])
    h10 = np.array([cloud10[i].half_extent[0] for i in range(len(cloud10))])
    # half = factor * reduce(per-view feature size): doubling the factor doubles
    # every patch's half-size (pins the `factor *` and a deterministic reduce).
    np.testing.assert_allclose(h10, 2.0 * h5, rtol=1e-12)


def test_from_reconstruction_excludes_or_includes_points_at_infinity(
    seoul_bull_workspace: Path,
):
    """exclude_points_at_infinity gates whether infinity points get a frame."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    pos = np.asarray(recon.positions_xyzw, dtype=np.float64)
    counts = np.bincount(
        np.asarray(recon.track_point_indexes), minlength=recon.point_count
    )
    pi = int(np.argmax(counts))
    xyz = pos[pi, :3]
    pos[pi] = np.append(xyz / np.linalg.norm(xyz), 0.0)
    recon = recon.clone_with_changes(positions=pos)

    # Default (exclude=False): the infinity point gets a tangent-sphere frame.
    default_cloud = PatchCloud.from_reconstruction(
        recon, extent="fixed", extent_value=1.0
    )
    ids = list(default_cloud.point_indexes)
    assert pi in set(ids)
    patch = default_cloud[ids.index(pi)]
    assert patch.w == 0.0
    d = xyz / np.linalg.norm(xyz)
    np.testing.assert_allclose(np.asarray(patch.normal), -d, atol=1e-6)

    # exclude=True: finite only — the infinity point gets no patch.
    finite_only = PatchCloud.from_reconstruction(
        recon, extent="fixed", extent_value=1.0, exclude_points_at_infinity=True
    )
    assert pi not in set(finite_only.point_indexes)
    assert len(default_cloud) == len(finite_only) + 1


def test_feature_size_handles_fisheye(kerry_park_workspace: Path):
    """FeatureSize sizes every point of a fisheye (FoV > 180°) rig.

    A fisheye sees points past 90° off axis at camera-frame z <= 0; the world
    size is back-projected from the ray distance d = ‖X − C‖ (always positive),
    not the optical-axis depth z, so such peripheral points size fine. The old
    pinhole `σ·z/f` (gated on z > 0) raised ``MissingFeatureScale`` for a point
    whose every observation was peripheral. See ``specs/core/patch-cloud.md``.
    """
    recon = SfmrReconstruction.load(kerry_park_workspace)
    cloud = PatchCloud.from_reconstruction(
        recon, extent="feature_size", extent_value=5.0
    )

    assert len(cloud) > 0
    halves = np.array([cloud[i].half_extent for i in range(len(cloud))])
    assert np.all(np.isfinite(halves))
    assert np.all(halves > 0.0)
