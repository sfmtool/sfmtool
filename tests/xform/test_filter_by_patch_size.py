# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FilterByPatchSizeTransform (coarse world-space patch cull)."""

import numpy as np
import pytest

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import FilterByPatchSizeTransform


@pytest.fixture(scope="module")
def embedded_recon(seoul_bull_workspace_once) -> SfmrReconstruction:
    """An ``embedded_patches`` recon carrying per-point patch frames."""
    return SfmrReconstruction.load(seoul_bull_workspace_once).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )


def _patch_sizes(recon: SfmrReconstruction) -> np.ndarray:
    """World-space patch size per point, same measure the filter culls on."""
    cloud = recon.patches
    return np.array(
        [
            np.sqrt(abs(cloud[i].half_extent[0]) * abs(cloud[i].half_extent[1]))
            for i in range(len(cloud))
        ]
    )


def test_invalid_multiplier():
    with pytest.raises(ValueError, match="Multiplier must be positive"):
        FilterByPatchSizeTransform(multiplier=0.0)
    with pytest.raises(ValueError, match="Multiplier must be positive"):
        FilterByPatchSizeTransform(multiplier=-1.0)


def test_lenient_multiplier_keeps_all(embedded_recon):
    # A huge multiplier puts the threshold above every patch, removing nothing.
    out = FilterByPatchSizeTransform(multiplier=1e9).apply(embedded_recon)
    assert out.point_count == embedded_recon.point_count


def test_removes_large_patches_keeps_small(embedded_recon):
    recon = embedded_recon
    sizes = _patch_sizes(recon)
    multiplier = 3.0
    threshold = multiplier * float(np.median(sizes))
    expected_keep = int(np.sum(sizes <= threshold))
    expected_removed = recon.point_count - expected_keep
    if expected_removed == 0:
        pytest.skip("no patches above 3x median on this fixture")

    out = FilterByPatchSizeTransform(multiplier=multiplier).apply(recon)
    # Coarse (large) patches are removed; the survivor count matches the mask.
    assert out.point_count == expected_keep
    # Every survivor is at or below the data-derived threshold (points under the
    # threshold are kept); every removed patch was above it.
    out_sizes = _patch_sizes(out)
    assert np.all(out_sizes <= threshold + 1e-12)
    assert np.max(sizes) > threshold


def test_description():
    desc = FilterByPatchSizeTransform(multiplier=3.0).description()
    assert "patch size" in desc
    assert "3.00" in desc


def test_requires_embedded_patches(seoul_bull_workspace_once):
    """Filtering a sift_files recon (no patch frames) is a clear, actionable error."""
    recon = SfmrReconstruction.load(seoul_bull_workspace_once)
    assert recon.patches is None
    with pytest.raises(ValueError, match="embedded_patches"):
        FilterByPatchSizeTransform(multiplier=3.0).apply(recon)
