# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for FilterByLocalizabilityTransform (keypoint uncertainty cull)."""

import numpy as np
import pytest

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import (
    FilterByLocalizabilityTransform,
    RefineKeypointsTransform,
)

R = 12


@pytest.fixture(scope="module")
def embedded_with_bitmaps(seoul_bull_workspace_once) -> SfmrReconstruction:
    """An ``embedded_patches`` recon carrying per-point consensus bitmaps."""
    recon = SfmrReconstruction.load(seoul_bull_workspace_once).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )
    return RefineKeypointsTransform(bitmaps=True, resolution=R, max_gn_steps=3).apply(
        recon
    )


def _sigma_pos(recon: SfmrReconstruction) -> np.ndarray:
    # The filter culls on the intrinsic patch-grid-px score; the tests derive
    # their thresholds from the same quantity so they stay unit-consistent.
    result = recon.patches.score_localizability(recon, recon.patch_bitmaps)
    return np.asarray(result["sigma_pos_grid"], dtype=float)


def test_invalid_threshold():
    with pytest.raises(ValueError, match="Threshold must be positive"):
        FilterByLocalizabilityTransform(threshold=0.0)
    with pytest.raises(ValueError, match="Threshold must be positive"):
        FilterByLocalizabilityTransform(threshold=-1.0)
    with pytest.raises(ValueError, match="sigma_noise must be positive"):
        FilterByLocalizabilityTransform(threshold=1.0, sigma_noise=0.0)


def test_lenient_threshold_keeps_all(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    out = FilterByLocalizabilityTransform(threshold=1e9).apply(recon)
    # A huge τ removes nothing (unscorable points are kept too).
    assert out.point_count == recon.point_count


def test_removes_high_uncertainty_points(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    sigma = _sigma_pos(recon)
    finite = sigma[np.isfinite(sigma)]
    assert finite.size > 0
    # Pick a threshold in the middle of the scored distribution so some points are
    # above it and get culled, and the survivor count matches the keep-mask.
    tau = float(np.median(finite))
    expected_removed = int(np.sum(sigma > tau))
    if expected_removed == 0:
        pytest.skip("no points above the median σ_pos on this fixture")

    out = FilterByLocalizabilityTransform(threshold=tau).apply(recon)
    assert out.point_count == recon.point_count - expected_removed
    # Every survivor is at or below τ (NaN-scored survivors are allowed through).
    out_sigma = _sigma_pos(out)
    kept = out_sigma[np.isfinite(out_sigma)]
    assert np.all(kept <= tau + 1e-9)


def test_keeps_unscorable_points(embedded_with_bitmaps):
    """A point whose σ_pos is NaN (empty consensus / no finite-depth view) is kept
    — the filter only drops points it has positive evidence are poorly localized."""
    recon = embedded_with_bitmaps
    sigma = _sigma_pos(recon)
    n_nan = int(np.sum(~np.isfinite(sigma)))
    # Cull everything scorable with a tiny τ; the NaN points must survive.
    tiny = float(np.nanmin(sigma[np.isfinite(sigma)])) * 0.5
    if tiny <= 0 or n_nan == 0:
        pytest.skip("fixture has no unscorable points to check")
    out = FilterByLocalizabilityTransform(threshold=tiny).apply(recon)
    assert out.point_count >= n_nan


def test_requires_patch_bitmaps(seoul_bull_workspace_once):
    """Filtering an embedded recon with no bitmaps is a clear error."""
    recon = SfmrReconstruction.load(seoul_bull_workspace_once).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )
    assert recon.patch_bitmaps is None
    with pytest.raises(ValueError, match="patch bitmaps"):
        FilterByLocalizabilityTransform(threshold=1.0).apply(recon)


def test_description():
    desc = FilterByLocalizabilityTransform(threshold=1.5).description()
    assert "keypoint uncertainty" in desc
    assert "1.50" in desc


def test_no_points_remain_raises(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    sigma = _sigma_pos(recon)
    finite = sigma[np.isfinite(sigma)]
    # If every point is either unscorable (kept) or above a floor, a threshold
    # below the minimum finite σ_pos still keeps the NaN points — so only assert
    # the "no points remain" error when there are no unscorable survivors.
    if np.any(~np.isfinite(sigma)):
        pytest.skip("unscorable points would survive any threshold")
    tau = float(finite.min()) * 0.5
    with pytest.raises(ValueError, match="No points remain"):
        FilterByLocalizabilityTransform(threshold=tau).apply(recon)
