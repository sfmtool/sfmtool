# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adjacency surfel normals PyO3 binding.

``estimate_adjacency_surfel_normals`` fits a plane through each selected point
over the directions to its image-space neighbours, and reports how well
determined the result is. See specs/core/analysis/adjacency-surfel-normals.md.
"""

import numpy as np
import pytest

from sfmtool._sfmtool.analysis import estimate_adjacency_surfel_normals

_KEYS = {
    "normals",
    "n_eff",
    "anisotropy",
    "sectors",
    "sigma_deg",
    "resid_deg",
    "n_support",
    "determined",
}

_UP = np.array([0.0, 0.0, 1.0])


def _ring(angles_deg):
    """Unit-radius positions at the given in-plane angles, on ``z = 0``."""
    a = np.radians(np.asarray(angles_deg, dtype=np.float64))
    return np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)


# Eight evenly spread directions, each in the middle of its own default sector,
# so no libm's last-ulp disagreement can push a row across a sector boundary.
_FULL_RING = _ring([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])
# Five directions inside a single 80-degree wedge: real support, no coverage.
_WEDGE = _ring([5, 25, 50, 70, 85])


def _hub(neighbours, *, view=(0.0, 0.0, 1.0), selected=None, extras=None, **params):
    """Fit point 0, sitting at the origin with ``neighbours`` hung off it.

    The neighbours become points ``1..k``, each with an empty CSR row, so only
    point 0 has anything to fit.
    """
    neighbours = np.asarray(neighbours, dtype=np.float64).reshape(-1, 3)
    k = len(neighbours)
    positions = np.vstack([np.zeros((1, 3)), neighbours])
    n = len(positions)

    offsets = np.full(n + 1, k, dtype=np.uint32)
    offsets[0] = 0
    view_dirs = np.tile(_UP, (n, 1))
    view_dirs[0] = view
    if selected is None:
        selected = np.zeros(n, dtype=bool)
        selected[0] = True
    return estimate_adjacency_surfel_normals(
        positions,
        offsets,
        np.arange(1, k + 1, dtype=np.uint32),
        view_dirs,
        np.asarray(selected, dtype=bool),
        extras=extras,
        **params,
    )


def _angle_deg(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _assert_unfitted(out, p):
    assert np.all(np.isnan(out["normals"][p]))
    for key in _KEYS - {"normals", "determined"}:
        assert np.isnan(out[key][p]), f"{key}[{p}] = {out[key][p]}, expected NaN"
    assert not out["determined"][p]


def test_exact_plane_recovers_its_normal():
    out = _hub(_FULL_RING)

    assert set(out) == _KEYS
    assert out["normals"].shape == (9, 3)
    assert out["normals"].dtype == np.float64
    assert out["n_eff"].dtype == np.float64
    assert out["determined"].dtype == np.bool_

    assert _angle_deg(out["normals"][0], _UP) < 1e-9
    assert out["n_support"][0] == 8.0
    assert out["n_eff"][0] == pytest.approx(8.0)
    assert out["anisotropy"][0] == pytest.approx(1.0)
    assert out["sectors"][0] == 8.0
    # Every residual is zero, so the robust scale sits on its floor.
    assert out["sigma_deg"][0] == pytest.approx(2.0)
    assert out["resid_deg"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["determined"][0]


def test_the_normal_takes_the_side_the_view_direction_is_on():
    # The geometry cannot tell the two sides of a plane apart; the caller's
    # reference direction is what picks one.
    up = _hub(_FULL_RING, view=(0.0, 0.0, 1.0))
    down = _hub(_FULL_RING, view=(0.0, 0.0, -1.0))

    assert _angle_deg(up["normals"][0], _UP) < 1e-9
    assert _angle_deg(down["normals"][0], -_UP) < 1e-9
    np.testing.assert_allclose(up["normals"][0], -down["normals"][0], atol=1e-12)


def test_gross_off_surface_neighbours_are_redescended_away():
    # Eight neighbours on the surface plus two lying almost along the viewing
    # direction — points on another surface, seen next to this one.
    neighbours = np.vstack([_FULL_RING, [[0.15, 0.0, 1.0], [0.10, 0.12, 1.0]]])

    raw = _hub(neighbours, irls_iters=0)
    assert _angle_deg(raw["normals"][0], _UP) > 5.0
    assert np.isnan(raw["sigma_deg"][0]), "no pass ran, so there is no scale"

    out = _hub(neighbours)
    assert _angle_deg(out["normals"][0], _UP) < 1e-9
    assert out["n_support"][0] == 10.0
    assert out["n_eff"][0] == pytest.approx(8.0), "the two outliers weigh nothing"
    assert out["resid_deg"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["determined"][0]


def test_a_collinear_line_fails_the_anisotropy_gate():
    line = np.array([[x, 0.0, 0.0] for x in (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)])
    out = _hub(line)

    assert out["n_support"][0] == 6.0
    assert out["n_eff"][0] == pytest.approx(6.0), "support is not the problem"
    assert out["anisotropy"][0] < 1e-12
    assert not out["determined"][0]


def test_a_one_sided_neighbourhood_fails_the_sector_gate():
    out = _hub(_WEDGE)

    assert out["sectors"][0] == 2.0
    assert out["n_eff"][0] == pytest.approx(5.0), "support is not the problem"
    assert out["anisotropy"][0] > 0.10, "anisotropy is not the problem"
    assert not out["determined"][0]

    # The same number of rows spread over both sides occupies twice the
    # sectors, and that alone flips the verdict.
    spread = np.vstack([_WEDGE, _ring([185, 205, 230, 250, 265])])
    out = _hub(spread)
    assert out["sectors"][0] == 4.0
    assert out["determined"][0]


def test_extras_fill_the_empty_sectors_of_an_under_determined_point():
    assert not _hub(_WEDGE)["determined"][0]

    out = _hub(_WEDGE, extras={0: _ring([140, 230, 320])})
    assert out["n_support"][0] == 8.0, "extras count as support"
    assert out["n_eff"][0] == pytest.approx(8.0)
    assert out["sectors"][0] == 5.0
    assert _angle_deg(out["normals"][0], _UP) < 1e-9
    assert out["determined"][0]


def test_extras_alone_can_carry_a_point_with_no_graph_neighbours():
    out = _hub(np.zeros((0, 3)), extras={0: _ring([0, 120, 240])})
    assert out["n_support"][0] == 3.0
    assert out["sectors"][0] == 3.0
    assert _angle_deg(out["normals"][0], _UP) < 1e-9


def test_extras_for_unselected_points_are_ignored():
    plain = _hub(_FULL_RING)
    out = _hub(_FULL_RING, extras={3: np.array([[0.0, 0.0, 5.0], [0.0, 5.0, 0.0]])})

    np.testing.assert_array_equal(out["normals"][0], plain["normals"][0])
    assert out["n_support"][0] == plain["n_support"][0]
    _assert_unfitted(out, 3)


def test_fewer_than_two_neighbours_yields_nan():
    _assert_unfitted(_hub(np.zeros((0, 3))), 0)
    _assert_unfitted(_hub([[1.0, 0.0, 0.0]]), 0)
    # A neighbour coincident with the point carries no direction and is dropped
    # before it can count as support.
    _assert_unfitted(_hub([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), 0)


def test_unselected_points_stay_nan():
    out = _hub(_FULL_RING)
    for p in range(1, len(out["normals"])):
        _assert_unfitted(out, p)

    none = _hub(_FULL_RING, selected=np.zeros(9, dtype=bool))
    for p in range(9):
        _assert_unfitted(none, p)


def test_parameters_move_the_verdict():
    assert not _hub(_WEDGE)["determined"][0]
    assert _hub(_WEDGE, det_sectors=2)["determined"][0]
    assert not _hub(_WEDGE, det_n_eff=6.0, det_sectors=2)["determined"][0]

    # Fewer sectors coarsen the coverage measure.
    assert _hub(_FULL_RING, n_sectors=4)["sectors"][0] == 4.0

    # A larger floor is reported as the scale it imposed.
    assert _hub(_FULL_RING, sigma_floor_deg=5.0)["sigma_deg"][0] == pytest.approx(5.0)


def test_empty_cloud():
    out = estimate_adjacency_surfel_normals(
        np.zeros((0, 3)),
        np.zeros(1, dtype=np.uint32),
        np.zeros(0, dtype=np.uint32),
        np.zeros((0, 3)),
        np.zeros(0, dtype=bool),
    )
    assert out["normals"].shape == (0, 3)
    assert out["determined"].size == 0


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda a: a.__setitem__("view_dirs", np.zeros((3, 3))), "view_dirs"),
        (lambda a: a.__setitem__("selected", np.zeros(3, dtype=bool)), "selected"),
        (lambda a: a.__setitem__("offsets", np.zeros(4, dtype=np.uint32)), "offsets"),
        (
            lambda a: a.__setitem__("offsets", np.array([0, 2, 1], dtype=np.uint32)),
            "non-decreasing",
        ),
        (
            lambda a: a.__setitem__("neighbours", np.array([7], dtype=np.uint32)),
            "out of range",
        ),
    ],
)
def test_malformed_inputs_raise(mutate, message):
    args = {
        "positions": np.zeros((2, 3)),
        "offsets": np.array([0, 1, 1], dtype=np.uint32),
        "neighbours": np.array([1], dtype=np.uint32),
        "view_dirs": np.tile(_UP, (2, 1)),
        "selected": np.ones(2, dtype=bool),
    }
    mutate(args)
    with pytest.raises(ValueError, match=message):
        estimate_adjacency_surfel_normals(**args)


def test_offsets_must_match_the_neighbour_count():
    with pytest.raises(ValueError, match="neighbours has"):
        estimate_adjacency_surfel_normals(
            np.zeros((2, 3)),
            np.array([0, 3, 3], dtype=np.uint32),
            np.array([1], dtype=np.uint32),
            np.tile(_UP, (2, 1)),
            np.ones(2, dtype=bool),
        )


@pytest.mark.parametrize(
    "extras, message",
    [
        ({50: np.zeros((1, 3))}, "out of range"),
        ({0: np.zeros((1, 2))}, r"shape \(k, 3\)"),
        ({0: "not an array"}, "float64 array"),
    ],
)
def test_malformed_extras_raise(extras, message):
    with pytest.raises(ValueError, match=message):
        _hub(_FULL_RING, extras=extras)
