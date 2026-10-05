# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the observation adjacency PyO3 binding.

``build_observation_adjacency`` turns per-observation keypoints plus per-point
radii into a symmetric CSR graph of points that are next to each other on the
imaged surface. See specs/core/analysis/observation-adjacency-graph.md.
"""

import numpy as np
import pytest

from sfmtool._sfmtool.analysis import build_observation_adjacency

_KEYS = {
    "offsets",
    "neighbours",
    "separation_med",
    "separation_min",
    "separation_max",
    "shared_images",
    "annulus_hits",
    "range_mismatch",
}


def _build(
    observations,
    positions,
    *,
    n_images=2,
    radii=None,
    infinity=None,
    quaternions=None,
    translations=None,
    **params,
):
    """Run the binding over ``(point, image, x, y)`` observation tuples.

    Poses default to the identity, i.e. every camera centered at the origin.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_points = len(positions)
    if quaternions is None:
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_images, 1))
    if translations is None:
        translations = np.zeros((n_images, 3))
    params.setdefault("b_max", 10.0)
    return build_observation_adjacency(
        np.array([[o[2], o[3]] for o in observations], dtype=np.float64).reshape(-1, 2),
        np.array([o[0] for o in observations], dtype=np.uint32),
        np.array([o[1] for o in observations], dtype=np.uint32),
        np.ones(n_points, dtype=np.float32)
        if radii is None
        else np.asarray(radii, dtype=np.float32),
        np.zeros(n_points, dtype=bool)
        if infinity is None
        else np.asarray(infinity, dtype=bool),
        positions,
        np.asarray(quaternions, dtype=np.float64),
        np.asarray(translations, dtype=np.float64),
        **params,
    )


def _neighbours(out, p):
    return out["neighbours"][out["offsets"][p] : out["offsets"][p + 1]]


def _slice(out, key, p):
    return out[key][out["offsets"][p] : out["offsets"][p + 1]]


def test_surface_pair_becomes_an_edge():
    # Two points side by side on a surface, 2 px apart in both images.
    obs = [(0, 0, 0.0, 0.0), (1, 0, 2.0, 0.0), (0, 1, 5.0, 5.0), (1, 1, 7.0, 5.0)]
    out = _build(obs, [[0.0, 0.0, 10.0], [0.2, 0.0, 10.0]])

    assert set(out) == _KEYS
    assert out["offsets"].dtype == np.uint32
    assert out["neighbours"].dtype == np.uint32
    assert out["separation_med"].dtype == np.float32
    assert out["shared_images"].dtype == np.uint32

    np.testing.assert_array_equal(out["offsets"], [0, 1, 2])
    np.testing.assert_array_equal(out["neighbours"], [1, 0])
    np.testing.assert_allclose(out["separation_med"], [2.0, 2.0])
    np.testing.assert_allclose(out["separation_min"], [2.0, 2.0])
    np.testing.assert_allclose(out["separation_max"], [2.0, 2.0])
    np.testing.assert_array_equal(out["shared_images"], [2, 2])
    np.testing.assert_array_equal(out["annulus_hits"], [2, 2])
    assert out["range_mismatch"][0] < 0.05


def test_range_vet_rejects_one_behind_the_other():
    # Same image geometry, but the second point sits 5 units further along the
    # viewing ray.
    obs = [(0, 0, 0.0, 0.0), (1, 0, 2.0, 0.0), (0, 1, 5.0, 5.0), (1, 1, 7.0, 5.0)]
    stacked = [[0.0, 0.0, 10.0], [0.0, 0.0, 15.0]]
    out = _build(obs, stacked)
    assert out["neighbours"].size == 0
    np.testing.assert_array_equal(out["offsets"], [0, 0, 0])

    # An infinite tolerance disables the vet, so the pair comes back.
    out = _build(obs, stacked, range_tol=float("inf"))
    np.testing.assert_array_equal(out["neighbours"], [1, 0])
    assert out["range_mismatch"][0] == pytest.approx(0.4, rel=1e-6)


def test_a_lo_zero_admits_coincident_observations():
    # Two points whose observations coincide exactly: the duplicate regime.
    obs = [(0, 0, 0.0, 0.0), (1, 0, 0.0, 0.0), (0, 1, 3.0, 3.0), (1, 1, 3.0, 3.0)]
    positions = [[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]

    out = _build(obs, positions)
    assert out["neighbours"].size == 0

    out = _build(obs, positions, a_lo=0.0)
    np.testing.assert_array_equal(out["neighbours"], [1, 0])
    np.testing.assert_allclose(out["separation_med"], [0.0, 0.0])
    np.testing.assert_array_equal(out["annulus_hits"], [2, 2])


def test_csr_is_symmetric_and_ordered_by_separation_then_index():
    # A hub with three neighbours at 4, 2 and 2 pair radii.
    layout = [(0, 0.0, 0.0), (1, 4.0, 0.0), (2, 0.0, 2.0), (3, -2.0, 0.0)]
    obs = [(p, image, x, y) for image in (0, 1) for (p, x, y) in layout]
    out = _build(obs, [[0.0, 0.0, 10.0]] * 4)

    n_points = len(out["offsets"]) - 1
    assert n_points == 4
    assert out["offsets"][0] == 0
    assert out["offsets"][-1] == len(out["neighbours"])

    # Ties on the median separation break on the neighbour index.
    np.testing.assert_array_equal(_neighbours(out, 0), [2, 3, 1])
    np.testing.assert_allclose(_slice(out, "separation_med", 0), [2.0, 2.0, 4.0])

    for p in range(n_points):
        seps = _slice(out, "separation_med", p)
        assert np.all(np.diff(seps) >= 0), f"row {p} is ordered by separation"
        for q in _neighbours(out, p):
            assert p in _neighbours(out, q), f"edge {p}-{q} is not symmetric"


def test_pair_radius_is_the_smaller_of_the_two():
    obs = [(0, 0, 0.0, 0.0), (1, 0, 4.0, 0.0), (0, 1, 0.0, 0.0), (1, 1, 4.0, 0.0)]
    positions = [[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]
    radii = [1.0, 4.0]

    # r_pair = min(1, 4) = 1, so 4 px is 4 pair radii.
    out = _build(obs, positions, radii=radii, b_max=3.0)
    assert out["neighbours"].size == 0
    out = _build(obs, positions, radii=radii, b_max=5.0)
    np.testing.assert_allclose(out["separation_med"], [4.0, 4.0])


def test_infinity_and_non_positive_radius_exclude_points():
    obs = [(0, 0, 0.0, 0.0), (1, 0, 2.0, 0.0), (0, 1, 0.0, 0.0), (1, 1, 2.0, 0.0)]
    positions = [[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]

    assert _build(obs, positions)["neighbours"].size == 2
    assert _build(obs, positions, infinity=[False, True])["neighbours"].size == 0
    assert _build(obs, positions, radii=[1.0, 0.0])["neighbours"].size == 0


def test_camera_centers_are_derived_from_the_poses():
    # 90 degrees about +Z with t = -R C places the camera at (10, 0, 0). Both
    # points are 10 units from that center — and different distances from any
    # other reading of the pose — so the range vet passes only if the binding
    # derived the center as -R^T t.
    sqrt_half = np.sqrt(0.5)
    quaternions = np.tile(np.array([sqrt_half, 0.0, 0.0, sqrt_half]), (2, 1))
    translations = np.tile(np.array([0.0, -10.0, 0.0]), (2, 1))
    obs = [(0, 0, 0.0, 0.0), (1, 0, 2.0, 0.0), (0, 1, 0.0, 0.0), (1, 1, 2.0, 0.0)]
    positions = [[10.0, 0.0, 10.0], [20.0, 0.0, 0.0]]

    out = _build(obs, positions, quaternions=quaternions, translations=translations)
    np.testing.assert_array_equal(out["neighbours"], [1, 0])
    np.testing.assert_allclose(out["range_mismatch"], [0.0, 0.0], atol=1e-6)


def test_empty_inputs():
    out = _build([], np.zeros((0, 3)), n_images=0)
    np.testing.assert_array_equal(out["offsets"], [0])
    assert out["neighbours"].size == 0

    # Points, but each observed in its own image: nothing is shared.
    out = _build([(0, 0, 0.0, 0.0), (1, 1, 2.0, 0.0)], [[0.0, 0.0, 10.0]] * 2)
    assert out["neighbours"].size == 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"a_lo": 2.0, "b_max": 1.0}, "a_lo"),
        ({"b_max": -1.0}, "a_lo"),
    ],
)
def test_invalid_annulus_raises(kwargs, message):
    obs = [(0, 0, 0.0, 0.0), (1, 0, 2.0, 0.0)]
    with pytest.raises(ValueError, match=message):
        _build(obs, [[0.0, 0.0, 10.0]] * 2, **kwargs)


def test_mismatched_lengths_raise():
    keypoints = np.zeros((3, 2))
    with pytest.raises(ValueError, match="track_point_indexes"):
        build_observation_adjacency(
            keypoints,
            np.zeros(2, dtype=np.uint32),
            np.zeros(3, dtype=np.uint32),
            np.ones(2, dtype=np.float32),
            np.zeros(2, dtype=bool),
            np.zeros((2, 3)),
            np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (1, 1)),
            np.zeros((1, 3)),
            b_max=10.0,
        )


def test_out_of_range_index_raises():
    obs = [(0, 0, 0.0, 0.0), (5, 0, 2.0, 0.0)]
    with pytest.raises(ValueError, match="out of range"):
        _build(obs, [[0.0, 0.0, 10.0]] * 2)
