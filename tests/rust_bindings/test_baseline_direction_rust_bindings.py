# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the baseline-direction binding
(``sfmtool._sfmtool.geometry.baseline_directions``).

With both frames' rotations held, the direction between their centres is the
null space of the rows ``u_i x u_j``, one row per shared point. Rows whose
parallax is inside the bound carry no baseline and are dropped; the sign is
fixed by cheirality. Edges are flattened CSR-style, so a whole graph is one
call. Rotations are built with numpy (the test env has no scipy).
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.geometry import baseline_directions

ROUNDS, KEEP = 5, 0.6
TOL = np.radians(0.05)


def _edge(centre_i, centre_j, world):
    """The two frames' unit world rays over a shared cloud."""
    a = np.asarray(world, float) - np.asarray(centre_i, float)
    b = np.asarray(world, float) - np.asarray(centre_j, float)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a, b


def _cloud(n, seed=7, spread=3.0):
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(-spread, spread, n),
            rng.uniform(-spread, spread, n),
            -rng.uniform(4.0, 30.0, n),
        ],
        axis=1,
    )


def _call(pairs, tol=TOL, rounds=ROUNDS, keep=KEEP):
    counts = [len(a) for a, _b in pairs]
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    empty = np.zeros((0, 3))
    ri = np.concatenate([a for a, _b in pairs]) if pairs else empty
    rj = np.concatenate([b for _a, b in pairs]) if pairs else empty
    return baseline_directions(
        np.ascontiguousarray(ri),
        np.ascontiguousarray(rj),
        offsets,
        float(tol),
        int(rounds),
        float(keep),
    )


def test_the_direction_is_the_baseline_the_centres_have():
    c_i, c_j = np.zeros(3), np.array([1.0, 0.0, 0.0])
    out = _call([_edge(c_i, c_j, _cloud(60))])
    assert bool(out["stated"][0])
    npt.assert_allclose(out["direction"][0], [1.0, 0.0, 0.0], atol=1e-9)
    assert out["n_rows"][0] == 60
    assert out["n_used"][0] == 60
    assert out["cheiral_fraction"][0] == 1.0
    assert out["residual_median_rad"][0] < 1e-12
    assert out["parallax_max_deg"][0] >= out["parallax_median_deg"][0]


def test_the_sign_is_the_cheiral_one():
    world = _cloud(60)
    a, b = _edge(np.zeros(3), np.array([1.0, 0.0, 0.0]), world)
    forward = _call([(a, b)])
    backward = _call([(b, a)])
    npt.assert_allclose(forward["direction"][0], [1.0, 0.0, 0.0], atol=1e-9)
    npt.assert_allclose(backward["direction"][0], [-1.0, 0.0, 0.0], atol=1e-9)


def test_an_edge_with_no_parallax_states_nothing():
    world = _cloud(60)
    out = _call([_edge(np.zeros(3), np.zeros(3), world)])
    assert not bool(out["stated"][0])
    assert np.isnan(out["direction"][0]).all()
    assert out["n_used"][0] == 0


def test_rows_inside_the_bound_are_dropped():
    near = _cloud(60)
    far = np.stack(
        [np.linspace(-1.0, 1.0, 20), np.zeros(20), np.full(20, -1.0e7)], axis=1
    )
    world = np.concatenate([near, far])
    out = _call([_edge(np.zeros(3), np.array([1.0, 0.0, 0.0]), world)])
    assert out["n_rows"][0] == 80
    assert out["n_used"][0] < 80
    npt.assert_allclose(out["direction"][0], [1.0, 0.0, 0.0], atol=1e-9)


def test_a_whole_graph_is_one_call():
    world = _cloud(50)
    pairs = [
        _edge(np.zeros(3), np.array([1.0, 0.0, 0.0]), world),
        _edge(np.zeros(3), np.array([0.0, 1.0, 0.0]), world),
        _edge(np.zeros(3), np.zeros(3), world),
    ]
    out = _call(pairs)
    npt.assert_array_equal(out["stated"], [True, True, False])
    npt.assert_allclose(out["direction"][0], [1.0, 0.0, 0.0], atol=1e-9)
    npt.assert_allclose(out["direction"][1], [0.0, 1.0, 0.0], atol=1e-9)
    assert out["direction"].shape == (3, 3)


def test_the_call_repeats_itself_bit_for_bit():
    world = _cloud(80, seed=11)
    pairs = [_edge(np.zeros(3), np.array([1.0, 0.3, -0.2]), world)]
    a = _call(pairs)
    b = _call(pairs)
    for key in ("direction", "n_used", "condition", "residual_median_rad"):
        npt.assert_array_equal(a[key], b[key])


def test_the_inputs_are_checked():
    world = _cloud(10)
    a, b = _edge(np.zeros(3), np.array([1.0, 0.0, 0.0]), world)
    offsets = np.array([0, 10], np.int64)
    with pytest.raises(ValueError, match="shape"):
        baseline_directions(a[:, :2], b[:, :2], offsets, TOL, ROUNDS, KEEP)
    with pytest.raises(ValueError, match="same length"):
        baseline_directions(a, b[:-1], offsets, TOL, ROUNDS, KEEP)
    with pytest.raises(ValueError, match="non-decreasing"):
        baseline_directions(a, b, np.array([0, 8, 4], np.int64), TOL, ROUNDS, KEEP)
    with pytest.raises(ValueError, match="exceeds"):
        baseline_directions(a, b, np.array([0, 99], np.int64), TOL, ROUNDS, KEEP)
