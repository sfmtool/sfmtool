# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the point-or-bearing PyO3 bindings.

Covers the free ``bearing_score_batch``, ``fit_point_and_bearing_batch`` and
``observed_rays`` with their constants, and the ``SfmrReconstruction``
conveniences ``reprojection_noise_px``, ``reprojection_noise`` and
``point_or_bearing_scores``. See
specs/core/reconstruction/batch-triangulation-api.md § "Point or bearing".
"""

import numpy as np
import pytest

from sfmtool._sfmtool.analysis import (
    DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD,
    DEFAULT_POINT_FIT_MAX_ITERATIONS,
    DEFAULT_SOFT_L1_SCALE,
    bearing_score_batch,
    fit_point_and_bearing_batch,
    observed_rays,
)
from sfmtool._sfmtool.geometry import CameraIntrinsics
from sfmtool._sfmtool.reconstruction import SfmrReconstruction

SCORE_KEYS = {
    "scored",
    "bearing",
    "bearing_cost",
    "depth_score",
    "midpoint_bound",
    "bearing_in_front_of_all_cameras",
    "num_views",
    "is_finite",
}

FIT_KEYS = {
    "fitted",
    "bearing",
    "bearing_cost",
    "anchor",
    "direction",
    "inverse_depth",
    "point",
    "point_cost",
    "depth_likelihood_ratio",
    "in_front_of_all_cameras",
    "num_views",
}


def _isotropic_weights(dirs, sigma_rad):
    """(1/σ) Bᵀ per ray, B an orthonormal basis perpendicular to the ray."""
    weights = np.empty((len(dirs), 2, 3))
    for i, d in enumerate(dirs):
        helper = np.eye(3)[np.argmin(np.abs(d))]
        b1 = np.cross(d, helper)
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(d, b1)
        weights[i] = np.stack([b1, b2]) / sigma_rad
    return weights


def _two_tracks():
    """A near point seen from three cameras, then two parallel rays."""
    target = np.array([0.0, 0.0, 5.0])
    c0 = np.array([[-2.0, 0, 0], [0, 0, 0], [2, 1, 0]])
    d0 = target - c0
    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
    c1 = np.array([[0.0, 0, 0], [0.01, 0, 0]])
    d1 = np.array([[0.0, 0, 1], [0, 0, 1]])
    dirs = np.vstack([d0, d1])
    centers = np.vstack([c0, c1])
    offsets = np.array([0, 3, 5], dtype=np.int64)
    weights = _isotropic_weights(dirs, 1e-3)
    return target, dirs, centers, offsets, weights


def test_constants():
    assert DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD == 25.0
    assert DEFAULT_SOFT_L1_SCALE == 3.0
    assert DEFAULT_POINT_FIT_MAX_ITERATIONS == 20


def test_bearing_score_batch_shapes_and_verdicts():
    _, dirs, centers, offsets, weights = _two_tracks()
    out = bearing_score_batch(dirs, centers, offsets, weights)
    assert set(out) == SCORE_KEYS
    assert out["bearing"].shape == (2, 3)
    for key in ("bearing_cost", "depth_score", "midpoint_bound"):
        assert out[key].shape == (2,)
        assert out[key].dtype == np.float64
    for key in ("scored", "bearing_in_front_of_all_cameras", "is_finite"):
        assert out[key].dtype == np.bool_
    assert out["num_views"].dtype == np.int64
    assert out["scored"].tolist() == [True, True]
    assert out["num_views"].tolist() == [3, 2]
    # The near point asks for a depth; the parallel rays fit a bearing exactly.
    assert out["is_finite"].tolist() == [True, False]
    np.testing.assert_allclose(out["bearing"][1], [0, 0, 1], atol=1e-12)
    assert out["bearing_cost"][1] < 1e-12

    # A threshold above every score calls nothing finite.
    high = bearing_score_batch(dirs, centers, offsets, weights, threshold=1e30)
    assert not high["is_finite"].any()


def test_bearing_score_batch_unscored_track():
    _, dirs, centers, _, weights = _two_tracks()
    out = bearing_score_batch(
        dirs, centers, np.array([0, 1, 5], dtype=np.int64), weights
    )
    assert out["scored"].tolist() == [False, True]
    assert np.isnan(out["bearing"][0]).all()
    assert np.isnan(out["depth_score"][0])
    assert out["num_views"][0] == 0
    assert not out["is_finite"][0]


def test_bearing_score_batch_rejects_bad_input():
    _, dirs, centers, offsets, weights = _two_tracks()
    with pytest.raises(ValueError):
        bearing_score_batch(dirs, centers, np.array([0, 9], dtype=np.int64), weights)
    with pytest.raises(ValueError):
        bearing_score_batch(dirs, centers, offsets, weights[:, :, :2].copy())
    with pytest.raises(ValueError):
        bearing_score_batch(dirs, centers, offsets, weights[:4].copy())
    with pytest.raises(ValueError):
        bearing_score_batch(dirs, centers[:4], offsets, weights)
    with pytest.raises(ValueError):
        bearing_score_batch(dirs[:, :2].copy(), centers, offsets, weights)


def test_fit_point_and_bearing_batch():
    target, dirs, centers, offsets, weights = _two_tracks()
    out = fit_point_and_bearing_batch(dirs, centers, offsets, weights)
    assert set(out) == FIT_KEYS
    for key in ("bearing", "anchor", "direction", "point"):
        assert out[key].shape == (2, 3)
    assert out["fitted"].tolist() == [True, True]
    np.testing.assert_allclose(out["point"][0], target, atol=1e-6)
    assert out["in_front_of_all_cameras"][0]
    # The parallel rays' best point is the bearing: no finite position.
    assert out["inverse_depth"][1] == 0.0
    assert np.isnan(out["point"][1]).all()

    # Plain least squares, a start for one track only and an anchor.
    starts = np.array([[0.1, 0.0, 5.2], [np.nan, np.nan, np.nan]])
    anchors = np.array([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
    plain = fit_point_and_bearing_batch(
        dirs,
        centers,
        offsets,
        weights,
        starts=starts,
        anchors=anchors,
        soft_l1_scale=None,
        max_iterations=50,
    )
    np.testing.assert_allclose(plain["point"][0], target, atol=1e-6)
    np.testing.assert_allclose(plain["anchor"][0], [0, 0, 0])
    score = bearing_score_batch(dirs, centers, offsets, weights)
    # Plain least squares: Λ never falls below the midpoint bound.
    assert plain["depth_likelihood_ratio"][0] >= score["midpoint_bound"][0] * (1 - 1e-9)


def test_fit_point_and_bearing_batch_rejects_bad_input():
    _, dirs, centers, offsets, weights = _two_tracks()
    with pytest.raises(ValueError):
        fit_point_and_bearing_batch(
            dirs, centers, offsets, weights, starts=np.zeros((3, 3))
        )
    with pytest.raises(ValueError):
        fit_point_and_bearing_batch(
            dirs, centers, offsets, weights, anchors=np.zeros((2, 2))
        )
    with pytest.raises(ValueError):
        fit_point_and_bearing_batch(dirs, centers, offsets, weights, soft_l1_scale=-1)


def _pinhole():
    return CameraIntrinsics(
        "PINHOLE",
        640,
        480,
        {
            "focal_length_x": 500.0,
            "focal_length_y": 500.0,
            "principal_point_x": 320.0,
            "principal_point_y": 240.0,
        },
    )


def test_observed_rays():
    camera = _pinhole()
    identity = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    pixels = np.array([[320.0, 240.0], [400.0, 100.0]])
    out = observed_rays(camera, identity, pixels, 0.5)
    assert set(out) == {"valid", "dirs", "weights"}
    assert out["valid"].tolist() == [True, True]
    assert out["dirs"].shape == (2, 3)
    assert out["weights"].shape == (2, 2, 3)
    # The principal point looks down the camera's -Z axis.
    np.testing.assert_allclose(out["dirs"][0], [0, 0, -1], atol=1e-12)
    for d, w in zip(out["dirs"], out["weights"]):
        np.testing.assert_allclose(np.linalg.norm(d), 1.0)
        # Moving along the ray does not move its pixel.
        np.testing.assert_allclose(w @ d, 0.0, atol=1e-9)
    # The weight is over the noise.
    half = observed_rays(camera, identity, pixels, 1.0)
    np.testing.assert_allclose(half["weights"], out["weights"] / 2)

    with pytest.raises(ValueError):
        observed_rays(camera, identity, pixels, 0.0)
    with pytest.raises(ValueError):
        observed_rays(camera, identity[:1], pixels, 0.5)
    with pytest.raises(ValueError):
        observed_rays(camera, identity, pixels[:, :1].copy(), 0.5)


def test_reprojection_noise(seoul_bull_ground_truth_sfmr):
    recon = SfmrReconstruction.load(str(seoul_bull_ground_truth_sfmr))
    sigma = recon.reprojection_noise_px()
    assert isinstance(sigma, float)
    # The in-repo ground truth measures 0.4677 px over 1,229 observations; four
    # mismatched keypoints, 7 to 16 px off, are left out as outliers (with them
    # the RMS is 0.6460 px).
    assert sigma == pytest.approx(0.46769, abs=5e-5)

    noise = recon.reprojection_noise()
    assert noise["sigma_px"] == sigma
    finite = ~recon.point_is_at_infinity
    finite_obs = finite[recon.track_point_indexes]
    assert noise["observation_count"] <= int(finite_obs.sum())
    assert noise["per_camera_sigma_px"].shape == (recon.camera_count,)
    assert noise["per_camera_observation_count"].sum() == noise["observation_count"]
    assert noise["observation_count"] == 1229
    assert noise["outlier_count"] == 4


def _rays_by_hand(recon, sigma_px):
    """Each point's rays, centres and weights through ``observed_rays``."""
    quats = recon.quaternions_wxyz
    trans = recon.translations
    rot = [_rotation(q) for q in quats]
    centers_by_image = np.array([-(r.T @ t) for r, t in zip(rot, trans)])
    cameras = recon.cameras
    cam_idx = recon.camera_indexes
    images = recon.track_image_indexes.astype(np.int64)
    pixels = recon.keypoints_xy.astype(np.float64)
    dirs = np.full((len(images), 3), np.nan)
    weights = np.full((len(images), 2, 3), np.nan)
    valid = np.zeros(len(images), dtype=bool)
    for c, camera in enumerate(cameras):
        rows = np.flatnonzero(cam_idx[images] == c)
        out = observed_rays(camera, quats[images[rows]], pixels[rows], sigma_px)
        dirs[rows] = out["dirs"]
        weights[rows] = out["weights"]
        valid[rows] = out["valid"]
    counts = recon.observation_counts.astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)])
    keep_dirs, keep_centers, keep_weights, offsets = [], [], [], [0]
    for p in range(len(counts)):
        rows = np.arange(starts[p], starts[p + 1])
        rows = rows[valid[rows]]
        keep_dirs.append(dirs[rows])
        keep_centers.append(centers_by_image[images[rows]])
        keep_weights.append(weights[rows])
        offsets.append(offsets[-1] + len(rows))
    return (
        np.concatenate(keep_dirs),
        np.concatenate(keep_centers),
        np.array(offsets, dtype=np.int64),
        np.concatenate(keep_weights),
    )


def _rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_point_or_bearing_scores_agrees_with_the_batch_bindings(
    seoul_bull_ground_truth_sfmr,
):
    recon = SfmrReconstruction.load(str(seoul_bull_ground_truth_sfmr))
    out = recon.point_or_bearing_scores(fit=True)
    m = recon.point_count
    assert set(out) == SCORE_KEYS | {"sigma_px", "point_indexes", "fit"}
    assert set(out["fit"]) == FIT_KEYS
    assert out["sigma_px"] == recon.reprojection_noise_px()
    np.testing.assert_array_equal(out["point_indexes"], np.arange(m))
    assert out["bearing"].shape == (m, 3)
    assert out["fit"]["point"].shape == (m, 3)
    assert out["scored"].all()

    dirs, centers, offsets, weights = _rays_by_hand(recon, out["sigma_px"])
    by_hand = bearing_score_batch(dirs, centers, offsets, weights)
    for key in ("bearing", "bearing_cost", "depth_score", "midpoint_bound"):
        np.testing.assert_allclose(out[key], by_hand[key], rtol=1e-9, atol=1e-9)
    for key in ("scored", "is_finite", "num_views", "bearing_in_front_of_all_cameras"):
        np.testing.assert_array_equal(out[key], by_hand[key])

    positions = recon.positions_xyzw
    starts = np.where(positions[:, 3:] != 0, positions[:, :3], np.nan)
    fit = fit_point_and_bearing_batch(
        dirs, centers, offsets, weights, starts=starts, soft_l1_scale=None
    )
    np.testing.assert_allclose(
        out["fit"]["depth_likelihood_ratio"],
        fit["depth_likelihood_ratio"],
        rtol=1e-6,
        atol=1e-6,
    )
    # The camera centres here are computed in numpy, so the iterative fit can
    # end a few parts in 10^8 away from the convenience's.
    np.testing.assert_allclose(out["fit"]["point"], fit["point"], rtol=1e-6)

    # The stored finite points of a ground truth are finite here too.
    finite = ~recon.point_is_at_infinity
    assert out["is_finite"][finite].mean() > 0.9


def test_point_or_bearing_scores_subset_and_options(seoul_bull_ground_truth_sfmr):
    recon = SfmrReconstruction.load(str(seoul_bull_ground_truth_sfmr))
    full = recon.point_or_bearing_scores(sigma_px=0.5)
    idx = np.array([5, 0, 5, recon.point_count - 1], dtype=np.int64)
    sub = recon.point_or_bearing_scores(point_indexes=idx, sigma_px=0.5)
    np.testing.assert_array_equal(sub["point_indexes"], idx)
    np.testing.assert_array_equal(sub["depth_score"], full["depth_score"][idx])
    assert "fit" not in sub
    # A larger noise level lowers every score.
    noisy = recon.point_or_bearing_scores(point_indexes=idx, sigma_px=1.0)
    np.testing.assert_allclose(noisy["bearing_cost"], sub["bearing_cost"] / 4)

    robust = recon.point_or_bearing_scores(
        point_indexes=idx, sigma_px=0.5, fit=True, soft_l1_scale=3.0
    )
    assert robust["fit"]["fitted"].all()

    with pytest.raises(IndexError):
        recon.point_or_bearing_scores(point_indexes=np.array([recon.point_count]))
    with pytest.raises(IndexError):
        recon.point_or_bearing_scores(point_indexes=np.array([-1]))
    with pytest.raises(ValueError):
        recon.point_or_bearing_scores(sigma_px=0.0)
    with pytest.raises(ValueError):
        recon.point_or_bearing_scores(fit=True, soft_l1_scale=-2.0)


def test_point_or_bearing_scores_index_types(seoul_bull_ground_truth_sfmr):
    recon = SfmrReconstruction.load(str(seoul_bull_ground_truth_sfmr))
    expected = recon.point_or_bearing_scores(
        point_indexes=np.array([4, 1], dtype=np.int64), sigma_px=0.5
    )["depth_score"]
    for idx in (
        [4, 1],
        (4, 1),
        np.array([4, 1], dtype=np.int32),
        np.array([4, 1], dtype=np.uint16),
        np.array([4, 1], dtype=np.uint64),
    ):
        out = recon.point_or_bearing_scores(point_indexes=idx, sigma_px=0.5)
        np.testing.assert_array_equal(out["depth_score"], expected)
        assert out["point_indexes"].dtype == np.int64
    assert recon.point_or_bearing_scores(point_indexes=[], sigma_px=0.5)[
        "scored"
    ].shape == (0,)
    with pytest.raises(IndexError):
        recon.point_or_bearing_scores(point_indexes=[-1])
    with pytest.raises(TypeError):
        recon.point_or_bearing_scores(point_indexes=np.array([1.0, 2.0]))


def test_threshold_is_validated(seoul_bull_ground_truth_sfmr):
    _, dirs, centers, offsets, weights = _two_tracks()
    recon = SfmrReconstruction.load(str(seoul_bull_ground_truth_sfmr))
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            bearing_score_batch(dirs, centers, offsets, weights, threshold=bad)
        with pytest.raises(ValueError):
            recon.point_or_bearing_scores(threshold=bad)
    zero = bearing_score_batch(dirs, centers, offsets, weights, threshold=0.0)
    assert zero["is_finite"].tolist() == [True, True]
