# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the batch triangulation PyO3 bindings.

Covers the free ``triangulate_batch`` (rays in, point + diagnostics out) and the
``SfmrReconstruction.triangulation_diagnostics`` convenience over a loaded
reconstruction's stored points. See specs/core/batch-triangulation-api.md.
"""

import numpy as np
import pytest

from sfmtool._sfmtool import SfmrReconstruction, triangulate_batch


def _rays_to(target, centers):
    dirs = np.asarray(target, float) - np.asarray(centers, float)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def test_triangulate_batch_recovers_finite_point():
    target = np.array([0.0, 0.0, 5.0])
    centers = np.array([[-2.0, 0, 0], [0, 0, 0], [2, 1, 0]])
    dirs = _rays_to(target, centers)
    offsets = np.array([0, 3], dtype=np.int64)

    out = triangulate_batch(dirs, centers, offsets)

    assert set(out) == {
        "points",
        "eigenvalues",
        "condition_number",
        "in_front_of_all_cameras",
    }
    assert out["points"].shape == (1, 3)
    assert out["eigenvalues"].shape == (1, 3)
    np.testing.assert_allclose(out["points"][0], target, atol=1e-9)
    assert np.isfinite(out["condition_number"][0])
    assert bool(out["in_front_of_all_cameras"][0])
    # Eigenvalues are ascending and sum to 2K.
    eig = out["eigenvalues"][0]
    assert eig[0] <= eig[1] <= eig[2]
    np.testing.assert_allclose(eig.sum(), 6.0, atol=1e-9)


def test_triangulate_batch_parallel_rays_degenerate():
    # Two tracks: finite, then parallel (point at infinity).
    target = np.array([0.0, 0.0, 5.0])
    c0 = np.array([[-1.0, 0, 0], [1, 0, 0]])
    d0 = _rays_to(target, c0)
    c1 = np.array([[0.0, 0, 0], [0, 1, 0]])
    d1 = np.array([[1.0, 0, 0], [1, 0, 0]])

    dirs = np.vstack([d0, d1])
    centers = np.vstack([c0, c1])
    offsets = np.array([0, 2, 4], dtype=np.int64)

    out = triangulate_batch(dirs, centers, offsets)
    assert out["condition_number"].shape == (2,)
    assert np.isfinite(out["condition_number"][0])
    assert np.isinf(out["condition_number"][1])
    assert bool(out["in_front_of_all_cameras"][0])
    assert not bool(out["in_front_of_all_cameras"][1])


def test_triangulate_batch_rejects_bad_offsets():
    """Malformed offsets raise ValueError instead of panicking on an OOB index."""
    dirs = np.array([[0.0, 0, 1], [0, 0, 1]])
    centers = np.array([[0.0, 0, 0], [1, 0, 0]])

    # Out of range: last offset exceeds the ray count.
    with pytest.raises(ValueError):
        triangulate_batch(dirs, centers, np.array([0, 5], dtype=np.int64))

    # Negative offset.
    with pytest.raises(ValueError):
        triangulate_batch(dirs, centers, np.array([-1, 2], dtype=np.int64))

    # Non-monotonic offsets.
    with pytest.raises(ValueError):
        triangulate_batch(dirs, centers, np.array([0, 2, 1], dtype=np.int64))


def test_triangulation_diagnostics_shapes_and_nan(
    seoul_bull_sfmr_only,
):
    recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    diag = recon.triangulation_diagnostics(noise_px=1.0)

    assert set(diag) == {"condition_number", "depth_sigma", "inverse_depth_z"}
    m = recon.point_count
    for arr in diag.values():
        assert arr.shape == (m,)

    at_infinity = np.asarray(recon.point_is_at_infinity)
    # Points at infinity have no finite depth → NaN diagnostics across the board.
    assert np.isnan(diag["inverse_depth_z"][at_infinity]).all()
    assert np.isnan(diag["condition_number"][at_infinity]).all()
    assert np.isnan(diag["depth_sigma"][at_infinity]).all()

    # The 17-image seoul_bull fixture is an ordinary forward-facing capture.
    # COLMAP's solve jitters in *content* run to run (the point count swings
    # ~10%), but its *character* is stable, so the assertions below pin that
    # character with generous margins rather than exact values.
    finite = ~at_infinity
    z = diag["inverse_depth_z"][finite]
    sigma = diag["depth_sigma"][finite]
    cond = diag["condition_number"][finite]
    diagnosable = np.isfinite(z)
    n = int(diagnosable.sum())
    assert n > 0

    # depth_sigma is a real 1σ uncertainty: finite and strictly positive
    # wherever the depth is diagnosable.
    assert np.isfinite(sigma[diagnosable]).all()
    assert (sigma[diagnosable] > 0.0).all()
    # condition_number is finite exactly where the z-score is.
    assert np.isfinite(cond[diagnosable]).all()

    zf = z[diagnosable]
    # A forward-facing capture triangulates well: the bulk of points have a
    # large z-score (observed medians 46–104 across solves; >20 is a safe floor)
    # and only a small fraction sit near infinity (observed 0–4%; <15% is safe).
    assert np.median(zf) > 20.0
    near_infinity_fraction = float((zf < 4.0).mean())
    assert near_infinity_fraction < 0.15
    # Depth is measured from the camera-cloud centroid, so an in-front point can
    # in principle read negative; for a forward capture that is rare, so assert
    # the overwhelming majority are positive rather than the brittle "all >= 0".
    assert float((zf > 0.0).mean()) >= 0.95

    # The triangulation is well-conditioned: the median condition number sits
    # far below the 1e4 classifier pre-filter (observed medians 32–105).
    assert np.median(cond[diagnosable]) < 1000.0


def test_triangulation_diagnostics_flags_distant_as_low_z(
    seoul_bull_sfmr_only,
):
    """Points the classifier moves to infinity carry the lowest z-scores."""
    recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    diag = recon.triangulation_diagnostics(noise_px=1.0)
    z = diag["inverse_depth_z"]

    # Reclassify with a high noise floor to surface near-infinity points, then
    # check those points sat at low z in the original diagnostics.
    classified = recon.classify_points_at_infinity(50.0)
    newly_infinite = np.asarray(classified.point_is_at_infinity) & ~np.asarray(
        recon.point_is_at_infinity
    )
    if newly_infinite.any():
        # Their z-scores are below the well-resolved median.
        median_z = np.nanmedian(z[np.isfinite(z)])
        assert np.nanmedian(z[newly_infinite]) < median_z
