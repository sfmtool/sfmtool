# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Camera centres from pairwise baselines."""

import numpy as np

from seed_relax.averaging import centres_by_averaging, direction_reading

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
#: The token disagreement every measured graph carries, so the recovery below
#: is read at a perturbation a real reading is always past and the error it
#: produces is proportional to it.
JITTER = 1e-6
#: Six frames on one straight line, evenly spaced. Every baseline of it carries
#: the same direction, which is what a camera walking a line produces.
COLINEAR = np.stack([np.array([0.4 * f, 0.0, 0.0]) for f in range(6)])


def _all_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _edges(centres, pairs, jitter=JITTER):
    dirs, weights = {}, {}
    for k, (i, j) in enumerate(pairs):
        b = centres[j] - centres[i]
        d = b / np.linalg.norm(b)
        off = np.array([np.sin(k * 1.7), np.cos(k * 2.3), np.sin(k * 0.9)])
        off = off - (off @ d) * d
        d = d + jitter * off / np.linalg.norm(off)
        dirs[(i, j)] = d / np.linalg.norm(d)
        weights[(i, j)] = 10.0
    return dirs, weights


def _shape(centres):
    """The constellation modulo scale and shift: unit pairwise vectors."""
    n = len(centres)
    v = np.stack([centres[j] - centres[i] for i in range(n) for j in range(i + 1, n)])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_recovers_the_constellation_up_to_scale_and_shift():
    frames = list(range(len(TRUE_CENTRES)))
    dirs, weights = _edges(TRUE_CENTRES, _all_pairs(len(TRUE_CENTRES)))
    cen, lam, res, _read = centres_by_averaging(frames, dirs, weights)
    assert cen is not None
    assert float(np.abs(_shape(cen) - _shape(TRUE_CENTRES)).max()) < 10 * JITTER
    # Every edge points forward along its own measured direction.
    assert min(lam.values()) > 0
    assert max(res.values()) < 1e-4


def test_one_contradicted_edge_does_not_carry_the_solve():
    frames = list(range(len(TRUE_CENTRES)))
    dirs, weights = _edges(TRUE_CENTRES, _all_pairs(len(TRUE_CENTRES)))
    # An edge whose direction the rest of the graph contradicts: reversed and
    # tilted, at the same weight as every other edge.
    bad = (2, 5)
    d = -dirs[bad] + np.array([0.0, 0.9, 0.0])
    dirs[bad] = d / np.linalg.norm(d)
    cen, _lam, res, _read = centres_by_averaging(frames, dirs, weights)
    assert cen is not None
    assert float(np.abs(_shape(cen) - _shape(TRUE_CENTRES)).max()) < 5e-2
    # The contradicted edge is the one the graph disagrees with most.
    assert max(res, key=res.get) == bad


def test_two_calls_agree_to_the_bit():
    frames = list(range(len(TRUE_CENTRES)))
    dirs, weights = _edges(TRUE_CENTRES, _all_pairs(len(TRUE_CENTRES)))
    a, _l, _r, _rd = centres_by_averaging(frames, dirs, weights)
    b, _l2, _r2, _rd2 = centres_by_averaging(frames, dirs, weights)
    assert a.tobytes() == b.tobytes()


def test_directions_that_agree_exactly_are_read_exactly():
    # The true centres are what the objective sends to zero, so a graph with no
    # disagreement in it is read off the form's null space rather than solved
    # for against the form.
    frames = list(range(len(TRUE_CENTRES)))
    dirs, weights = _edges(TRUE_CENTRES, _all_pairs(len(TRUE_CENTRES)), jitter=0.0)
    cen, _lam, _res, read = centres_by_averaging(frames, dirs, weights)
    assert float(np.abs(_shape(cen) - _shape(TRUE_CENTRES)).max()) < 1e-9
    # One null direction, which is the constellation itself.
    assert read["n_null"] == 1
    assert read["n_free"] == 0


def _lengths(centres, pairs):
    """The true relative baseline lengths, on one arbitrary common scale."""
    ell = {k: float(np.linalg.norm(centres[k[1]] - centres[k[0]])) for k in pairs}
    med = float(np.median(list(ell.values())))
    return {k: v / med for k, v in ell.items()}


def _spacing(cen):
    """Consecutive gaps, over their own largest."""
    gap = np.linalg.norm(cen[1:] - cen[:-1], axis=1)
    return gap / gap.max()


def test_a_colinear_graph_does_not_state_its_own_spacing():
    frames = list(range(len(COLINEAR)))
    pairs = _all_pairs(len(COLINEAR))
    dirs, weights = _edges(COLINEAR, pairs, jitter=0.0)
    cen, _lam, _res, read = centres_by_averaging(frames, dirs, weights)
    # One null direction per free spacing: six frames on a line have five gaps,
    # one of which the scale gauge fixes, so four are free on top of the
    # constellation's own direction.
    assert read["n_free"] == len(COLINEAR) - 2
    assert read["n_lengths"] == 0
    # And what comes back is not the spacing the frames have.
    assert not np.allclose(_spacing(cen), 1.0, atol=1e-3)


def test_the_pairs_own_lengths_resolve_the_colinear_spacing():
    frames = list(range(len(COLINEAR)))
    pairs = _all_pairs(len(COLINEAR))
    dirs, weights = _edges(COLINEAR, pairs, jitter=0.0)
    ell = _lengths(COLINEAR, pairs)
    cen, _lam, _res, read = centres_by_averaging(frames, dirs, weights, ell, weights)
    assert read["n_free"] == 0
    assert read["n_lengths"] == len(pairs)
    assert np.allclose(_spacing(cen), 1.0, atol=1e-6)


def test_a_frame_on_one_lengthless_edge_stays_where_the_graph_left_it():
    # Frame 8 hangs off frame 0 by a single edge and states no length for it,
    # so nothing says how far along that edge it sits.  Its own freedom sends
    # the objective to zero at any distance, and reading the centres off that
    # null direction would send it to one nobody measured.
    cen8 = np.vstack([TRUE_CENTRES, TRUE_CENTRES[0] + np.array([0.0, 2.0, 0.0])])
    pairs = _all_pairs(len(TRUE_CENTRES)) + [(0, 8)]
    dirs, weights = _edges(cen8, pairs, jitter=0.0)
    ell = _lengths(cen8, pairs)
    ell.pop((0, 8))
    length_w = {k: weights[k] for k in ell}
    cen, _l, _r, read = centres_by_averaging(
        list(range(9)), dirs, weights, ell, length_w
    )
    assert read["n_loose"] == 1
    assert not read["read_off_null"]
    reach = np.linalg.norm(cen, axis=1)
    assert float(reach.max()) < 10.0 * float(np.median(reach))
    # The eight frames the graph does determine are still where they belong.
    assert float(np.abs(_shape(cen[:8]) - _shape(TRUE_CENTRES)).max()) < 1e-3


def test_the_directions_own_reading_is_recorded_before_the_lengths():
    frames = list(range(len(COLINEAR)))
    pairs = _all_pairs(len(COLINEAR))
    dirs, weights = _edges(COLINEAR, pairs, jitter=0.0)
    ell = _lengths(COLINEAR, pairs)
    # What the lengths closed is only visible against what the directions left.
    alone = direction_reading(frames, dirs, weights)
    _c, _l, _r, with_len = centres_by_averaging(frames, dirs, weights, ell, weights)
    assert alone["n_free"] == len(COLINEAR) - 2
    assert with_len["n_free"] == 0
    assert alone["lam2_rel"] < with_len["lam2_rel"]


def test_one_frame_off_the_line_leaves_the_rest_nearly_degenerate():
    # A single frame lifted off the line makes the form non-singular, but the
    # spacing of the five that stayed on it is still carried by almost nothing:
    # the form's second eigenvalue is orders below a general graph's.
    off = COLINEAR.copy()
    off[3] = off[3] + np.array([0.0, 0.02, 0.0])
    frames = list(range(len(off)))
    pairs = _all_pairs(len(off))
    dirs, weights = _edges(off, pairs, jitter=0.0)
    _cen, _lam, _res, read = centres_by_averaging(frames, dirs, weights)
    general = centres_by_averaging(
        list(range(len(TRUE_CENTRES))),
        *_edges(TRUE_CENTRES, _all_pairs(len(TRUE_CENTRES)), jitter=0.0),
    )[3]
    assert read["lam2_rel"] < 0.001 * general["lam2_rel"]
    # The lengths the pairs carry put it back where the general graph sits.
    ell = _lengths(off, pairs)
    cen, _l, _r, with_len = centres_by_averaging(frames, dirs, weights, ell, weights)
    assert with_len["lam2_rel"] > read["lam2_rel"]
    assert float(np.abs(_shape(cen) - _shape(off)).max()) < 1e-6
