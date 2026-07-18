# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``PatchCloud.score_localizability`` PyO3 batch scorer.

Cross-checks the binding's per-point keypoint uncertainty ``σ_pos`` against an
independent NumPy reimplementation of the documented math
(``specs/core/patch-localizability.md``): the noise-normalized structure tensor
on each consensus patch, mapped to source px by the recon geometry.
"""

import numpy as np
import pytest

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.patches import PatchCloud
from sfmtool._sfmtool.geometry import RotQuaternion
from sfmtool.xform import RefineKeypointsTransform

R = 12  # small consensus grid — correctness, not quality
SIGMA_NOISE = 3.0


@pytest.fixture(scope="module")
def embedded_with_bitmaps(seoul_bull_workspace_once) -> SfmrReconstruction:
    """An ``embedded_patches`` recon carrying per-point consensus ``patch_bitmaps``
    (rendered by a cheap sub-pixel refine pass)."""
    recon = SfmrReconstruction.load(seoul_bull_workspace_once).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )
    out = RefineKeypointsTransform(bitmaps=True, resolution=R, max_gn_steps=3).apply(
        recon
    )
    assert out.patch_bitmaps is not None
    return out


def _reference_sigma_pos(recon: SfmrReconstruction, sigma_noise: float = SIGMA_NOISE):
    """Independent per-point ``σ_pos`` (source px) reference, mirroring the binding
    (luminance structure tensor, summed λ₂, median-over-views grid→px map with the
    canonical ``depth = -(R·X + t)_z`` and focal ``0.5·(fx + fy)``)."""
    bitmaps = np.asarray(recon.patch_bitmaps)
    p, r = bitmaps.shape[0], bitmaps.shape[1]
    # Score against the scorer's own window kernel (via the binding) rather than
    # a reimplemented gaussian disk.
    w = np.asarray(PatchCloud.window_weights(r, window="gaussian_disk"))
    gray = (
        0.299 * bitmaps[..., 0] + 0.587 * bitmaps[..., 1] + 0.114 * bitmaps[..., 2]
    ).astype(np.float64)
    valid = bitmaps.reshape(p, -1).any(axis=1)

    lam1 = np.full(p, np.nan)
    lam2 = np.full(p, np.nan)
    theta = np.full(p, np.nan)
    for i in np.nonzero(valid)[0]:
        gy, gx = np.gradient(gray[i])
        sxx = float((w * gx * gx).sum())
        syy = float((w * gy * gy).sum())
        sxy = float((w * gx * gy).sum())
        tr = sxx + syy
        disc = max(tr * tr - 4 * (sxx * syy - sxy * sxy), 0.0) ** 0.5
        lam1[i] = 0.5 * (tr + disc)
        lam2[i] = 0.5 * (tr - disc)
        if abs(sxy) > 1e-12:
            theta[i] = np.arctan2(lam2[i] - sxx, sxy)
        else:
            theta[i] = 0.0 if sxx <= syy else np.pi / 2
    sigma_grid = sigma_noise / np.sqrt(np.maximum(lam2, 1e-12))

    # grid → source-px scale (median over the point's observing views).
    positions = np.asarray(recon.positions, np.float64)
    # Homogeneous w (0 for a point at infinity, whose position is a direction):
    # the depth rotates the point but only translates it by w, matching the
    # binding's transform_point_homogeneous.
    hw = np.asarray(recon.positions_xyzw, np.float64)[:, 3]
    quats = np.asarray(recon.quaternions_wxyz, np.float64)
    trans = np.asarray(recon.translations, np.float64)
    rots = [RotQuaternion.from_wxyz_array(q).to_rotation_matrix() for q in quats]
    cam_idx = np.asarray(recon.camera_indexes)
    cams = recon.cameras
    focal = np.array(
        [0.5 * sum(cams[int(c)].focal_lengths) for c in cam_idx], np.float64
    )
    cloud = recon.patches
    half = np.full(p, np.nan)
    for i, pid in enumerate(np.asarray(cloud.point_indexes)):
        half[int(pid)] = float(cloud[i].half_extent[0])

    tpid = np.asarray(recon.track_point_indexes)
    timg = np.asarray(recon.track_image_indexes)
    by_pt: dict[int, list[float]] = {}
    for k in range(len(tpid)):
        pid, im = int(tpid[k]), int(timg[k])
        if not np.isfinite(half[pid]):
            continue
        depth = -(rots[im] @ positions[pid] + trans[im] * hw[pid])[2]
        if depth > 1e-6:
            by_pt.setdefault(pid, []).append(
                (half[pid] / (r / 2.0)) * focal[im] / depth
            )
    scale = np.full(p, np.nan)
    for pid, vals in by_pt.items():
        scale[pid] = float(np.median(vals))

    return sigma_grid * scale, lam1, lam2, theta, sigma_grid


def test_score_localizability_shapes_and_keys(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    cloud = recon.patches
    result = cloud.score_localizability(
        recon, recon.patch_bitmaps, sigma_noise=SIGMA_NOISE
    )
    assert set(result) == {"sigma_pos_px", "sigma_pos_grid", "lam1", "lam2", "theta"}
    for key, arr in result.items():
        arr = np.asarray(arr)
        assert arr.shape == (recon.point_count,), key
        assert arr.dtype == np.float64, key


def test_score_localizability_matches_numpy_reference(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    cloud = recon.patches
    result = cloud.score_localizability(
        recon, recon.patch_bitmaps, sigma_noise=SIGMA_NOISE
    )
    ref_sigma, ref_lam1, ref_lam2, ref_theta, ref_grid = _reference_sigma_pos(recon)

    got_sigma = np.asarray(result["sigma_pos_px"])
    got_lam1 = np.asarray(result["lam1"])
    got_lam2 = np.asarray(result["lam2"])
    got_grid = np.asarray(result["sigma_pos_grid"])

    # Empty (unscorable) patches are NaN in both.
    np.testing.assert_array_equal(np.isnan(got_grid), np.isnan(ref_grid))

    scored = np.isfinite(ref_lam2)
    assert scored.sum() > 0, "fixture produced no scorable consensus patches"
    np.testing.assert_allclose(got_lam1[scored], ref_lam1[scored], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_lam2[scored], ref_lam2[scored], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_grid[scored], ref_grid[scored], rtol=1e-6)

    # σ_pos in source px agrees wherever both map to a finite depth.
    both = np.isfinite(got_sigma) & np.isfinite(ref_sigma)
    assert both.sum() > 0
    np.testing.assert_allclose(got_sigma[both], ref_sigma[both], rtol=1e-5)


def test_score_localizability_theta_matches(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    cloud = recon.patches
    result = cloud.score_localizability(recon, recon.patch_bitmaps)
    _, _, _, ref_theta, ref_grid = _reference_sigma_pos(recon)
    got = np.asarray(result["theta"])
    scored = np.isfinite(ref_theta)
    # theta is a line direction (mod π); compare via sin(2θ)/cos(2θ).
    np.testing.assert_allclose(
        np.sin(2 * got[scored]), np.sin(2 * ref_theta[scored]), atol=1e-6
    )
    np.testing.assert_allclose(
        np.cos(2 * got[scored]), np.cos(2 * ref_theta[scored]), atol=1e-6
    )


def test_score_localizability_sigma_is_nonnegative(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    result = recon.patches.score_localizability(recon, recon.patch_bitmaps)
    sigma = np.asarray(result["sigma_pos_px"])
    assert np.all(sigma[np.isfinite(sigma)] >= 0.0)
    assert np.all(np.asarray(result["lam2"])[np.isfinite(result["lam2"])] >= 0.0)


def test_window_weights_kernel():
    """The exposed scorer window is a gaussian disk: (R, R), peaked at the centre,
    zero outside the unit disk (corners), and rejects an unknown window name."""
    w = np.asarray(PatchCloud.window_weights(12, window="gaussian_disk"))
    assert w.shape == (12, 12)
    assert w[0, 0] == 0.0 and w[-1, -1] == 0.0  # corners fall outside the disk
    assert w[6, 6] > 0.0 and w.max() == w[5:7, 5:7].max()  # peak at the centre
    with pytest.raises(ValueError, match="unknown window"):
        PatchCloud.window_weights(12, window="triangle")


def test_score_localizability_rejects_wrong_bitmap_rows(embedded_with_bitmaps):
    recon = embedded_with_bitmaps
    cloud = recon.patches
    bad = np.asarray(recon.patch_bitmaps)[:-1]  # one row short
    with pytest.raises(ValueError, match="rows"):
        cloud.score_localizability(recon, bad)
