# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the keypoint reach binding
(``sfmtool._sfmtool.analysis.keypoint_pairs_within_reach``).

The enumeration answers, per image of a track set, which other keypoints lie
inside this keypoint's own disk.  It is directed (the disk is the asking row's),
a row is never its own candidate, and a row whose reach is not finite asks
nothing while still answering for others.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.analysis import keypoint_pairs_within_reach


def _call(image, xy, reach):
    return keypoint_pairs_within_reach(
        np.asarray(image, np.int64),
        np.asarray(xy, float).reshape(-1, 2),
        np.asarray(reach, float),
    )


def _pairs(image, xy, reach):
    i, j, _d = _call(image, xy, reach)
    return list(zip(i.tolist(), j.tolist()))


def _brute(image, xy, reach):
    """The same relation by a double loop, in the same order."""
    image = np.asarray(image, np.int64)
    xy = np.asarray(xy, float).reshape(-1, 2)
    reach = np.asarray(reach, float)
    out = []
    for img in np.unique(image):
        rows = np.nonzero(image == img)[0]
        run = rows[np.argsort(xy[rows, 0], kind="stable")]
        for i in rows:
            if not np.isfinite(reach[i]):
                continue
            for j in run:
                if j == i:
                    continue
                dx, dy = xy[j] - xy[i]
                d = float(np.sqrt(dx * dx + dy * dy))
                if d <= reach[i]:
                    out.append((int(i), int(j), d))
    return out


# ------------------------------------------------------------------ exactness


def test_the_binding_reproduces_a_brute_force_double_loop():
    rng = np.random.default_rng(20260830)
    image = rng.integers(0, 4, 300).astype(np.int64)
    xy = rng.uniform(0.0, 40.0, (300, 2))
    reach = rng.uniform(0.5, 6.0, 300)
    reach[::13] = np.nan
    i, j, d = _call(image, xy, reach)
    want = _brute(image, xy, reach)
    assert list(zip(i.tolist(), j.tolist())) == [(a, b) for a, b, _ in want]
    npt.assert_array_equal(d, np.array([c for _a, _b, c in want]))


def test_no_rows_is_three_empty_arrays():
    i, j, d = _call(np.zeros(0, np.int64), np.zeros((0, 2)), np.zeros(0))
    assert (len(i), len(j), len(d)) == (0, 0, 0)
    assert (i.dtype, j.dtype, d.dtype) == (np.int64, np.int64, np.float64)


# ---------------------------------------------------------------- directedness


def test_only_the_reach_that_spans_the_separation_pairs():
    got = _pairs([0, 0], [[0.0, 0.0], [3.0, 0.0]], [5.0, 1.0])
    assert (0, 1) in got
    assert (1, 0) not in got


# ------------------------------------------------------------ self pair, NaN


def test_no_row_is_its_own_candidate():
    i, j, _d = _call([0, 0, 0], [[0.0, 0.0], [9.0, 9.0], [1.0, 0.0]], [1.0, 0.0, 1.0])
    assert len(i) > 0
    assert np.all(i != j)


def test_a_nan_reach_asks_nothing_and_still_answers_for_others():
    got = _pairs([0, 0], [[0.0, 0.0], [1.0, 0.0]], [np.nan, 4.0])
    assert got == [(1, 0)]


def test_a_nan_reach_is_not_an_error():
    i, _j, _d = _call([0], [[0.0, 0.0]], [np.nan])
    assert len(i) == 0


# ------------------------------------------------------------ image isolation


def test_identical_positions_in_different_images_never_pair():
    got = _pairs([0, 1, 2], [[4.0, 4.0]] * 3, [100.0] * 3)
    assert got == []


def test_images_come_out_in_ascending_index():
    got = _pairs(
        [7, 2, 7, 2],
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        [3.0] * 4,
    )
    assert got == [(1, 3), (3, 1), (0, 2), (2, 0)]


# ------------------------------------------------------------- memory order


def test_fortran_ordered_positions_give_the_same_answer():
    rng = np.random.default_rng(7)
    image = np.zeros(50, np.int64)
    xy = rng.uniform(0.0, 20.0, (50, 2))
    reach = rng.uniform(1.0, 5.0, 50)
    c_i, c_j, c_d = keypoint_pairs_within_reach(image, np.ascontiguousarray(xy), reach)
    f_i, f_j, f_d = keypoint_pairs_within_reach(
        np.asfortranarray(image),
        np.asfortranarray(xy),
        np.asfortranarray(reach),
    )
    npt.assert_array_equal(c_i, f_i)
    npt.assert_array_equal(c_j, f_j)
    npt.assert_array_equal(c_d, f_d)


def test_the_returned_arrays_are_c_contiguous():
    i, j, d = _call([0, 0], [[0.0, 0.0], [1.0, 0.0]], [4.0, 4.0])
    for arr in (i, j, d):
        assert arr.flags["C_CONTIGUOUS"]


# ------------------------------------------------------------------- refusals


def test_a_short_reach_array_is_refused():
    with pytest.raises(ValueError, match="reach_px"):
        _call([0, 0], [[0.0, 0.0], [1.0, 0.0]], [1.0])


def test_a_position_array_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match=r"xy_px must have shape"):
        keypoint_pairs_within_reach(
            np.zeros(3, np.int64), np.zeros((2, 2)), np.zeros(3)
        )


def test_a_negative_reach_is_refused_and_names_its_row():
    with pytest.raises(ValueError, match="row 1"):
        _call([0, 0], [[0.0, 0.0], [1.0, 0.0]], [1.0, -0.5])


# ---------------------------------------------------------------- determinism


def test_two_calls_agree_bit_for_bit():
    rng = np.random.default_rng(99)
    image = rng.integers(0, 6, 2000).astype(np.int64)
    xy = rng.uniform(0.0, 100.0, (2000, 2))
    reach = rng.uniform(0.5, 8.0, 2000)
    a = _call(image, xy, reach)
    b = _call(image, xy, reach)
    for x, y in zip(a, b):
        assert x.tobytes() == y.tobytes()
