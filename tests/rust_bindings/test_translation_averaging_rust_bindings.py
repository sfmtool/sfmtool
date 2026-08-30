# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the translation-averaging bindings
(``sfmtool._sfmtool.geometry.average_translations`` and its three siblings).

With every rotation held, the camera centres of a graph of pairwise baseline
directions are one linear problem: the true centres are what the objective
sends to zero, so the constellation is the form's own null space. The lengths
the two-view depths state close the freedoms a colinear path leaves, and
cheirality settles the one bit the directions cannot, which of the two
mirror-image constellations has the structure in front of the cameras.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.geometry import (
    average_translations,
    direction_reading,
    orientation_reading,
    relative_lengths,
)

TRUE_CENTRES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.1, -0.2],
        [2.1, 0.0, 0.3],
        [3.0, -0.4, 0.1],
        [3.8, 0.5, -0.5],
        [4.9, 0.2, 0.6],
        [5.7, -0.3, 0.0],
        [6.6, 0.1, 0.4],
    ]
)
#: Six frames on one straight line, evenly spaced. Every baseline of it carries
#: the same direction, which is what a camera walking a line produces.
COLINEAR = np.stack([np.array([0.4 * f, 0.0, 0.0]) for f in range(6)])


def _all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _graph(centres, pairs, weight=10.0):
    """``(edges, directions, weights)`` for the exact constellation."""
    edges = np.asarray(pairs, np.int64)
    b = centres[edges[:, 1]] - centres[edges[:, 0]]
    d = b / np.linalg.norm(b, axis=1, keepdims=True)
    return edges, np.ascontiguousarray(d), np.full(len(pairs), float(weight))


def _lengths(centres, pairs):
    """The true relative baseline lengths, on one arbitrary common scale."""
    ell = np.array(
        [float(np.linalg.norm(centres[j] - centres[i])) for i, j in pairs], float
    )
    return ell / np.median(ell)


def _shape(centres):
    """The constellation modulo scale and shift: unit pairwise vectors."""
    n = len(centres)
    v = np.stack([centres[j] - centres[i] for i in range(n) for j in range(i + 1, n)])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _spacing(cen):
    """Consecutive gaps, over their own largest."""
    gap = np.linalg.norm(cen[1:] - cen[:-1], axis=1)
    return gap / gap.max()


def test_the_constellation_is_read_off_the_forms_null_space():
    pairs = _all_pairs(len(TRUE_CENTRES))
    edges, d, w = _graph(TRUE_CENTRES, pairs)
    cen, lam, res, census = average_translations(
        edges, d, w, n_frames=len(TRUE_CENTRES)
    )
    assert census["solved"]
    assert census["read_off_null"]
    assert census["n_null"] == 1
    assert census["n_free"] == 0
    assert census["n_lengths"] == 0
    assert cen.shape == (len(TRUE_CENTRES), 3)
    assert float(np.abs(_shape(cen) - _shape(TRUE_CENTRES)).max()) < 1e-9
    # Every edge points forward along its own measured direction.
    assert lam.min() > 0
    assert res.max() < 1e-9
    # The shift gauge: the centres are their own mean.
    npt.assert_allclose(cen.mean(axis=0), 0.0, atol=1e-9)


def test_a_colinear_path_does_not_state_its_own_spacing():
    pairs = _all_pairs(len(COLINEAR))
    edges, d, w = _graph(COLINEAR, pairs)
    cen, _lam, _res, census = average_translations(edges, d, w, n_frames=len(COLINEAR))
    # One null direction per free spacing: six frames on a line have five gaps,
    # one of which the scale gauge fixes, so four are free on top of the
    # constellation's own direction.
    assert census["n_free"] == len(COLINEAR) - 2
    assert not census["read_off_null"]
    assert not np.allclose(_spacing(cen), 1.0, atol=1e-3)

    # The pairs' own lengths close every one of those freedoms.
    ell = _lengths(COLINEAR, pairs)
    cen, _lam, _res, census = average_translations(
        edges, d, w, ell, w, n_frames=len(COLINEAR)
    )
    assert census["n_free"] == 0
    assert census["n_lengths"] == len(pairs)
    npt.assert_allclose(_spacing(cen), 1.0, atol=1e-6)


def test_the_directions_own_reading_ignores_the_lengths():
    pairs = _all_pairs(len(COLINEAR))
    edges, d, w = _graph(COLINEAR, pairs)
    ell = _lengths(COLINEAR, pairs)
    alone = direction_reading(edges, d, w, len(COLINEAR))
    _c, _l, _r, with_len = average_translations(
        edges, d, w, ell, w, n_frames=len(COLINEAR)
    )
    assert alone["n_free"] == len(COLINEAR) - 2
    assert with_len["n_free"] == 0
    assert alone["lam2_rel"] < with_len["lam2_rel"]
    assert alone["n_lengths"] == 0


def test_a_frame_on_one_lengthless_edge_is_loose():
    # Frame 8 hangs off frame 0 by a single edge and states no length for it,
    # so nothing says how far along that edge it sits.
    centres = np.vstack([TRUE_CENTRES, TRUE_CENTRES[0] + np.array([0.0, 2.0, 0.0])])
    pairs = _all_pairs(len(TRUE_CENTRES)) + [(0, 8)]
    edges, d, w = _graph(centres, pairs)
    ell = _lengths(centres, pairs)
    ell[-1] = np.nan
    cen, _lam, _res, census = average_translations(
        edges, d, w, ell, w, n_frames=len(centres)
    )
    assert census["n_loose"] == 1
    assert not census["read_off_null"]
    assert census["n_lengths"] == len(pairs) - 1
    # The eight frames the graph does determine are still where they belong.
    assert float(np.abs(_shape(cen[:8]) - _shape(TRUE_CENTRES)).max()) < 1e-3


def test_a_contradicted_edge_is_reweighted_out_of_the_solve():
    pairs = _all_pairs(len(TRUE_CENTRES))
    edges, d, w = _graph(TRUE_CENTRES, pairs)
    bad = pairs.index((2, 5))
    tilted = -d[bad] + np.array([0.0, 0.9, 0.0])
    d = d.copy()
    d[bad] = tilted / np.linalg.norm(tilted)
    cen, _lam, res, _census = average_translations(
        edges, d, w, n_frames=len(TRUE_CENTRES)
    )
    assert int(np.argmax(res)) == bad
    assert float(np.abs(_shape(cen) - _shape(TRUE_CENTRES)).max()) < 5e-2


def test_a_graph_stating_no_baseline_solves_nothing():
    edges = np.array([[0, 1], [1, 2]], np.int64)
    d = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cen, lam, res, census = average_translations(edges, d, np.zeros(2), n_frames=3)
    assert not census["solved"]
    assert cen.shape == (0, 3)
    assert lam.size == 0 and res.size == 0


def test_the_call_repeats_itself_bit_for_bit():
    pairs = _all_pairs(len(TRUE_CENTRES))
    edges, d, w = _graph(TRUE_CENTRES, pairs)
    a = average_translations(edges, d, w, n_frames=len(TRUE_CENTRES))
    b = average_translations(edges, d, w, n_frames=len(TRUE_CENTRES))
    for x, y in zip(a[:3], b[:3]):
        assert x.tobytes() == y.tobytes()
    assert a[3] == b[3]


def test_fortran_order_inputs_are_accepted():
    pairs = _all_pairs(len(TRUE_CENTRES))
    edges, d, w = _graph(TRUE_CENTRES, pairs)
    c_order = average_translations(edges, d, w, n_frames=len(TRUE_CENTRES))
    f_order = average_translations(
        np.asfortranarray(edges),
        np.asfortranarray(d),
        w,
        n_frames=len(TRUE_CENTRES),
    )
    npt.assert_array_equal(c_order[0], f_order[0])
    assert f_order[0].flags["C_CONTIGUOUS"]


# ── Relative lengths ──────────────────────────────────────────────────────

#: Four frames on a line at deliberately UNEVEN spacing, so the lengths carry
#: something the directions cannot: every baseline of them points the same way.
SPACED = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [4.0, 0.0, 0.0]])
CLOUD = np.stack(
    [
        np.linspace(-6.0, 6.0, 40),
        np.linspace(-4.0, 4.0, 40),
        -np.linspace(9.0, 30.0, 40),
    ],
    axis=1,
)


def _depth_rows(centres=SPACED, point_shift=0):
    """One row per (edge, frame, point), depths in units of the edge's own
    baseline. ``point_shift`` moves edge zero's points out of everyone else's
    numbering, which is how a lone edge is built."""
    pairs = _all_pairs(len(centres))
    ee, ff, pp, zz = [], [], [], []
    for e, (i, j) in enumerate(pairs):
        base = float(np.linalg.norm(centres[j] - centres[i]))
        shift = point_shift if e == 0 else 0
        for k, (frame, centre) in enumerate(((i, centres[i]), (j, centres[j]))):
            del k
            z = np.linalg.norm(CLOUD - centre, axis=1) / base
            ee.append(np.full(len(CLOUD), e, np.int64))
            ff.append(np.full(len(CLOUD), frame, np.int64))
            pp.append(np.arange(len(CLOUD), dtype=np.int64) + shift)
            zz.append(z)
    return (
        np.concatenate(ee),
        np.concatenate(ff),
        np.concatenate(pp),
        np.concatenate(zz),
    )


def test_the_depths_state_the_ratio_of_the_baselines():
    pairs = _all_pairs(len(SPACED))
    ell, spread, tied = relative_lengths(*_depth_rows(), len(pairs))
    truth = _lengths(SPACED, pairs)
    assert np.isfinite(ell).all()
    npt.assert_allclose(ell / np.median(ell), truth, rtol=1e-9)
    # The fit explains its own rows, so the scatter it leaves is nothing.
    assert float(np.nanmax(spread)) < 1e-9
    assert int(tied.min()) > 0


def test_an_edge_that_shares_no_point_states_no_length():
    pairs = _all_pairs(len(SPACED))
    ell, _spread, tied = relative_lengths(*_depth_rows(point_shift=10_000), len(pairs))
    assert not np.isfinite(ell[0])
    assert tied[0] == 0
    assert np.isfinite(ell[1:]).all()


def test_a_graph_with_no_tied_row_states_nothing():
    ee = np.array([0, 0, 0, 1, 1, 1], np.int64)
    ff = np.array([0, 0, 0, 1, 1, 1], np.int64)
    pp = np.array([0, 1, 2, 3, 4, 5], np.int64)
    zz = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ell, spread, tied = relative_lengths(ee, ff, pp, zz, 2)
    assert np.isnan(ell).all()
    assert np.isnan(spread).all()
    npt.assert_array_equal(tied, [0, 0])


def test_an_edge_below_the_tie_floor_states_no_length():
    # Two edges share exactly two rows, which is under the floor of three.
    ee = np.array([0, 0, 0, 1, 1, 1], np.int64)
    ff = np.array([0, 0, 0, 0, 0, 0], np.int64)
    pp = np.array([0, 1, 7, 0, 1, 8], np.int64)
    zz = np.array([1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
    ell, _spread, tied = relative_lengths(ee, ff, pp, zz, 2)
    npt.assert_array_equal(tied, [2, 2])
    assert np.isnan(ell).all()
    # At a floor of two the same graph does state its ratio.  Edge one reads
    # the shared points at twice edge zero's depth, and a depth is in units of
    # its own baseline, so edge one's baseline is half as long.
    ell, _spread, _tied = relative_lengths(ee, ff, pp, zz, 2, min_tied=2)
    assert np.isfinite(ell).all()
    npt.assert_allclose(ell[1] / ell[0], 0.5, rtol=1e-9)


# ── Orientation ───────────────────────────────────────────────────────────

N_FRAMES = 5
TOL = np.radians(0.5)


def _constellation():
    """Frames along a short arc, points in front of every one of them."""
    centres = np.stack([np.array([0.6 * f, 0.0, 0.0]) for f in range(N_FRAMES)])
    pts = np.stack(
        [np.linspace(-2.0, 2.0, 9), np.linspace(-1.0, 1.0, 9), np.full(9, -6.0)],
        axis=1,
    )
    rays, pof, fof = [], [], []
    for f in range(N_FRAMES):
        d = pts - centres[f]
        rays.append(d / np.linalg.norm(d, axis=1, keepdims=True))
        pof.append(np.arange(len(pts), dtype=np.int64))
        fof.append(np.full(len(pts), f, np.int64))
    return (
        centres,
        np.concatenate(rays),
        np.concatenate(pof),
        np.concatenate(fof),
    )


def test_the_right_way_up_reads_positive():
    centres, rays, pof, fof = _constellation()
    got = orientation_reading(centres, rays, pof, fof, TOL)
    assert got["angw"] > 0
    assert got["obs_frac"] == 1.0
    assert got["margin_frac"] == 1.0
    assert got["pts"] == 9
    assert got["behind"] == 0
    assert got["thin"] == 0


def test_the_reading_is_exactly_antisymmetric_under_reflection():
    centres, rays, pof, fof = _constellation()
    up = orientation_reading(centres, rays, pof, fof, TOL)
    down = orientation_reading(-centres, rays, pof, fof, TOL)
    assert down["angw"] == -up["angw"]
    assert down["obs_front"] == up["obs_total"] - up["obs_front"]
    assert down["margin_frac"] == up["margin_frac"]


# ── Input validation ──────────────────────────────────────────────────────


def test_the_averaging_inputs_are_checked():
    pairs = _all_pairs(len(TRUE_CENTRES))
    edges, d, w = _graph(TRUE_CENTRES, pairs)
    n = len(TRUE_CENTRES)
    with pytest.raises(ValueError, match="shape"):
        average_translations(edges[:, :1].copy(), d, w, n_frames=n)
    with pytest.raises(ValueError, match="outside n_frames"):
        average_translations(edges, d, w, n_frames=3)
    with pytest.raises(ValueError, match="not unit"):
        average_translations(edges, 2.0 * d, w, n_frames=n)
    with pytest.raises(ValueError, match="one entry per edge"):
        average_translations(edges, d, w[:-1].copy(), n_frames=n)
    with pytest.raises(ValueError, match="itself"):
        average_translations(np.array([[0, 0]], np.int64), d[:1], w[:1], n_frames=n)
    with pytest.raises(ValueError, match="at least one"):
        average_translations(edges, d, w, n_frames=n, rounds=0)


def test_the_length_inputs_are_checked():
    ee = np.array([0, 0, 1, 1], np.int64)
    ff = np.zeros(4, np.int64)
    pp = np.array([0, 1, 0, 1], np.int64)
    zz = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="outside"):
        relative_lengths(ee, ff, pp, zz, 1)
    with pytest.raises(ValueError, match="not positive"):
        relative_lengths(ee, ff, pp, np.array([1.0, 0.0, 3.0, 4.0]), 2)
    with pytest.raises(ValueError, match="one entry per row"):
        relative_lengths(ee, ff[:-1].copy(), pp, zz, 2)


def test_the_orientation_inputs_are_checked():
    centres, rays, pof, fof = _constellation()
    with pytest.raises(ValueError, match="n_frame, 3"):
        orientation_reading(centres[:, :2].copy(), rays, pof, fof, TOL)
    with pytest.raises(ValueError, match="n_ray, 3"):
        orientation_reading(centres, rays[:, :2].copy(), pof, fof, TOL)
    with pytest.raises(ValueError, match="one entry per ray"):
        orientation_reading(centres, rays, pof[:-1].copy(), fof, TOL)
    with pytest.raises(ValueError, match="outside"):
        orientation_reading(centres[:2].copy(), rays, pof, fof, TOL)
