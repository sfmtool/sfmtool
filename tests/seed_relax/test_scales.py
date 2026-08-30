# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Relative baseline lengths, from the depths each pair's own solve implies."""

import numpy as np

from seed_relax.scales import relative_lengths, two_view_depths

#: Four frames on a line at deliberately UNEVEN spacing, so the lengths carry
#: something the directions cannot: every baseline of them points the same way.
CENTRES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.0, 0.0, 0.0]])
CLOUD = np.stack(
    [
        np.linspace(-6.0, 6.0, 40),
        np.linspace(-4.0, 4.0, 40),
        -np.linspace(9.0, 30.0, 40),
    ],
    axis=1,
)


def _rays(centre):
    v = CLOUD - centre
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _graph(centres=CENTRES, clusters=None):
    """``(keys, depths)`` over every pair, read the way an edge reads them."""
    n = len(centres)
    keys = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cl = np.arange(len(CLOUD)) if clusters is None else clusters
    depths = {}
    for i, j in keys:
        b = centres[j] - centres[i]
        d = b / np.linalg.norm(b)
        z_i, z_j, _mid = two_view_depths(_rays(centres[i]), _rays(centres[j]), d)
        depths[(i, j)] = (
            np.concatenate([np.full(len(cl), i), np.full(len(cl), j)]),
            np.concatenate([cl, cl]),
            np.concatenate([z_i, z_j]),
        )
    return keys, depths


def test_the_depths_state_the_ratio_of_the_baselines():
    keys, depths = _graph()
    ell, spread, tied = relative_lengths(keys, depths)
    truth = np.array([np.linalg.norm(CENTRES[j] - CENTRES[i]) for i, j in keys], float)
    truth = truth / np.median(truth)
    assert np.isfinite(ell).all()
    assert np.allclose(ell / np.median(ell), truth, rtol=1e-9)
    # The fit explains its own rows, so the scatter it leaves is nothing.
    assert float(np.nanmax(spread)) < 1e-9
    assert int(tied.min()) > 0


def test_an_edge_that_shares_no_cluster_states_no_length():
    keys, depths = _graph()
    # One edge is given clusters nobody else saw, so nothing ties its depths to
    # another baseline: a length for it would be invented rather than read.
    lone = keys[0]
    frames, clusters, z = depths[lone]
    depths[lone] = (frames, clusters + 10_000, z)
    ell, _spread, tied = relative_lengths(keys, depths)
    assert not np.isfinite(ell[keys.index(lone)])
    assert tied[keys.index(lone)] == 0
    assert np.isfinite(ell[1:]).all()


def test_a_wild_depth_moves_nothing():
    keys, depths = _graph()
    clean, _s, _t = relative_lengths(keys, depths)
    frames, clusters, z = depths[keys[2]]
    z = z.copy()
    z[: len(z) // 6] *= 1000.0
    depths[keys[2]] = (frames, clusters, z)
    ell, spread, _tied = relative_lengths(keys, depths)
    assert np.allclose(ell, clean, rtol=1e-3)
    # The edge that carries the wild rows is the one whose scatter shows it.
    assert np.nanargmax(spread) == 2
