# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The baseline direction read off a pair's refused rows."""

import numpy as np

from seed_relax.graph import baseline_direction, largest_component

BASELINE = np.array([1.0, 0.0, 0.0])
TOL = np.radians(0.5)


def _rays(points, c_i, c_j):
    """Unit world rays of ``points`` from two camera centres."""
    a = points - c_i
    b = points - c_j
    return (
        a / np.linalg.norm(a, axis=1, keepdims=True),
        b / np.linalg.norm(b, axis=1, keepdims=True),
    )


def _near_and_far(n_near=12, n_far=12):
    """Points at a parallax well past the bound, and points well inside it."""
    rng_near = np.linspace(-1.0, 1.0, n_near)
    near = np.stack(
        [rng_near, np.linspace(-0.5, 0.5, n_near), np.full(n_near, -4.0)], axis=1
    )
    rng_far = np.linspace(-1.0, 1.0, n_far)
    far = np.stack(
        [
            rng_far * 4.0e4,
            np.linspace(-0.5, 0.5, n_far) * 4.0e4,
            np.full(n_far, -2.0e5),
        ],
        axis=1,
    )
    return near, far


def test_only_the_rows_past_the_bound_carry_the_direction():
    near, far = _near_and_far()
    c_i, c_j = np.zeros(3), BASELINE
    u_i, u_j = _rays(np.concatenate([near, far]), c_i, c_j)
    got = baseline_direction(u_i, u_j, TOL)
    assert got is not None
    # The far rows sit inside the bound and are dropped, not down-weighted.
    assert got["n_rows"] == len(near) + len(far)
    assert got["n_used"] == len(near)
    assert abs(abs(float(got["d"] @ BASELINE)) - 1.0) < 1e-9


def test_the_sign_is_the_cheiral_one():
    near, _far = _near_and_far()
    u_i, u_j = _rays(near, np.zeros(3), BASELINE)
    got = baseline_direction(u_i, u_j, TOL)
    assert float(got["d"] @ BASELINE) > 0
    assert got["cheir_frac"] == 1.0
    # Swapping the two frames reverses the baseline, and the cheirality vote
    # follows it rather than the row order.
    back = baseline_direction(u_j, u_i, TOL)
    assert float(back["d"] @ BASELINE) < 0


def test_a_pair_with_fewer_than_three_parallax_rows_refuses():
    near, far = _near_and_far(n_near=2, n_far=12)
    u_i, u_j = _rays(np.concatenate([near, far]), np.zeros(3), BASELINE)
    assert baseline_direction(u_i, u_j, TOL) is None
    # And so does a pair whose rows are all inside the bound.
    u_i, u_j = _rays(far, np.zeros(3), BASELINE)
    assert baseline_direction(u_i, u_j, TOL) is None


def test_largest_component_takes_the_biggest_connected_piece():
    dirs = {(0, 1): None, (1, 2): None, (2, 3): None, (5, 6): None}
    assert largest_component([0, 1, 2, 3, 4, 5, 6], dirs) == [0, 1, 2, 3]
    assert largest_component([0, 1, 2, 3, 4, 5, 6], {}) == []
