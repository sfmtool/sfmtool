# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The radius bands, and the population each of them admits."""

import types

import numpy as np
import pytest

from seed_relax.fleet_constants import RING_RATIO_P1
from seed_relax.rings import assign_rings, band_order, octave_edges, ring_cap


def test_the_fleet_constant_gives_five_octaves():
    assert octave_edges(RING_RATIO_P1) == [
        float("inf"),
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
    ]


@pytest.mark.parametrize(
    "p1,want",
    [
        (0.6, [float("inf"), 1.0, 0.5]),
        (0.5, [float("inf"), 1.0, 0.5]),
        (0.4, [float("inf"), 1.0, 0.5, 0.25]),
        (0.03, [float("inf"), 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]),
    ],
)
def test_the_grid_runs_down_in_octaves_to_the_pooled_percentile(p1, want):
    assert octave_edges(p1) == want


@pytest.mark.parametrize("bad", [0.0, 1.0, 2.0, -0.1])
def test_a_ratio_outside_the_unit_interval_is_refused(bad):
    with pytest.raises(ValueError):
        octave_edges(bad)


def test_the_bands_are_half_open_and_the_last_edge_cuts():
    edges = octave_edges(RING_RATIO_P1)
    floor = 20.0
    # One radius exactly on each edge, plus one above the floor and one below
    # the last edge.  The top ring is open, so a cluster coarser than the
    # member's own floor lands in it rather than outside the grid.
    radii = floor * np.array([4.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.05])
    got = assign_rings(radii, floor, edges)
    assert got.tolist() == [0, 0, 1, 2, 3, 4, -1]


def test_a_ring_admits_its_whole_band_by_default():
    # No count at all, which is what admitting the whole band means: the
    # slice a caller takes with it is the identity.
    assert ring_cap(types.SimpleNamespace(ring_cap=0)) is None
    assert ring_cap(types.SimpleNamespace()) is None
    assert np.arange(7)[: ring_cap(types.SimpleNamespace(ring_cap=0))].tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ]


def test_an_absolute_count_is_honoured():
    assert ring_cap(types.SimpleNamespace(ring_cap=50)) == 50
    # A function of its inputs alone: nothing is carried between calls.
    assert ring_cap(types.SimpleNamespace(ring_cap=50)) == 50
    assert ring_cap(types.SimpleNamespace(ring_cap=-3)) is None


def test_a_band_is_admitted_coarsest_first_with_the_cluster_id_on_a_tie():
    cand = np.array([11, 4, 7, 2, 9], np.int64)
    radius = np.array([3.0, 5.0, 3.0, 1.0, 5.0])
    order = band_order(cand, radius)
    assert cand[order].tolist() == [4, 9, 7, 11, 2]
    # And that is the order an absolute count cuts.
    assert cand[order[: ring_cap(types.SimpleNamespace(ring_cap=3))]].tolist() == [
        4,
        9,
        7,
    ]


def test_the_ring_a_cluster_lands_in_is_a_function_of_the_file():
    edges = octave_edges(RING_RATIO_P1)
    radii = np.array([31.0, 19.0, 9.0, 4.0, 1.5, 0.4])
    a = assign_rings(radii, 20.0, edges)
    b = assign_rings(radii, 20.0, edges)
    assert a.tobytes() == b.tobytes()
    # Halving the floor moves every cluster one ring coarser.
    coarser = assign_rings(radii, 10.0, edges)
    for x, y in zip(a.tolist(), coarser.tolist()):
        if x > 0:
            assert y == x - 1
