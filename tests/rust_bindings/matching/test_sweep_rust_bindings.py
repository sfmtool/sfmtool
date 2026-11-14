# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that the sweep-matching bindings check their inputs before matching,
raising ValueError rather than panicking."""

import numpy as np
import pytest

from sfmtool.matching import (
    match_one_way_sweep,
    match_one_way_sweep_geometric,
    mutual_best_match_sweep,
    mutual_best_match_sweep_geometric,
    polar_mutual_best_match,
    polar_mutual_best_match_geometric,
)

_K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
# The fundamental matrix of pure forward motion, whose epipoles are finite.
_F = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def _features():
    """Five features with distinct descriptors and identity affine shapes."""
    keypoints = np.array(
        [[100.0, 80.0], [500.0, 90.0], [100.0, 400.0], [500.0, 410.0], [300.0, 240.0]]
    )
    descriptors = np.zeros((5, 128), dtype=np.uint8)
    descriptors[:, 0] = [0, 50, 100, 150, 200]
    affines = np.tile([1.0, 0.0, 0.0, 1.0], (5, 1))
    return keypoints, descriptors, affines


def _cameras():
    """k1, k2, r1, r2, t1, t2 for sideways motion."""
    return [_K, _K, np.eye(3), np.eye(3), np.zeros(3), np.array([1.0, 0.0, 0.0])]


def _call(name, **changes):
    """Call the binding `name` on two copies of _features(), with the
    positional arguments named in `changes` replaced."""
    kp, d, a = _features()
    k1, k2, r1, r2, t1, t2 = _cameras()
    args = {
        "match_one_way_sweep": dict(kpts1=kp, descs1=d, kpts2=kp, descs2=d),
        "mutual_best_match_sweep": dict(kpts1=kp, descs1=d, kpts2=kp, descs2=d),
        "polar_mutual_best_match": dict(kpts1=kp, descs1=d, kpts2=kp, descs2=d, f=_F),
        "match_one_way_sweep_geometric": dict(
            kpts1=kp,
            descs1=d,
            kpts2=kp,
            descs2=d,
            aff1=a,
            aff2=a,
            k1=k1,
            k2=k2,
            r1=r1,
            r2=r2,
            t1=t1,
            t2=t2,
        ),
        "mutual_best_match_sweep_geometric": dict(
            kpts1=kp,
            descs1=d,
            kpts2=kp,
            descs2=d,
            aff1=a,
            aff2=a,
            k1=k1,
            k2=k2,
            r1=r1,
            r2=r2,
            t1=t1,
            t2=t2,
        ),
        "polar_mutual_best_match_geometric": dict(
            kpts1=kp,
            descs1=d,
            kpts2=kp,
            descs2=d,
            aff1=a,
            aff2=a,
            f=_F,
            k1=k1,
            k2=k2,
            r1=r1,
            r2=r2,
            t1=t1,
            t2=t2,
        ),
    }[name]
    args.update(changes)
    fn = {
        "match_one_way_sweep": match_one_way_sweep,
        "match_one_way_sweep_geometric": match_one_way_sweep_geometric,
        "mutual_best_match_sweep": mutual_best_match_sweep,
        "mutual_best_match_sweep_geometric": mutual_best_match_sweep_geometric,
        "polar_mutual_best_match": polar_mutual_best_match,
        "polar_mutual_best_match_geometric": polar_mutual_best_match_geometric,
    }[name]
    return fn(*args.values(), 10)


_ALL = [
    "match_one_way_sweep",
    "match_one_way_sweep_geometric",
    "mutual_best_match_sweep",
    "mutual_best_match_sweep_geometric",
    "polar_mutual_best_match",
    "polar_mutual_best_match_geometric",
]
_GEOMETRIC = [n for n in _ALL if n.endswith("_geometric")]
_POLAR = [n for n in _ALL if n.startswith("polar")]


@pytest.mark.parametrize("name", _ALL)
def test_well_formed_input_matches(name):
    matches = _call(name)
    assert matches is not None
    assert len(matches) == 5


@pytest.mark.parametrize("name", _ALL)
def test_keypoints_with_one_column_raise(name):
    kp, _, _ = _features()
    with pytest.raises(
        ValueError, match=r"has shape \(5, 1\) but must have shape \(5, 2\)"
    ):
        _call(name, kpts1=np.ascontiguousarray(kp[:, :1]))


@pytest.mark.parametrize("name", _ALL)
def test_fewer_descriptor_rows_than_keypoints_raise(name):
    _, d, _ = _features()
    with pytest.raises(
        ValueError, match=r"has shape \(3, 128\) but must have shape \(5, 128\)"
    ):
        _call(name, descs2=d[:3])


@pytest.mark.parametrize("name", _ALL)
def test_descriptors_of_different_widths_raise(name):
    _, d, _ = _features()
    with pytest.raises(
        ValueError, match=r"has shape \(5, 64\) but must have shape \(5, 128\)"
    ):
        _call(name, descs2=np.ascontiguousarray(d[:, :64]))


@pytest.mark.parametrize("name", _GEOMETRIC)
@pytest.mark.parametrize(
    ("arg", "value", "expected"),
    [
        ("k1", np.eye(2), r"k1 has shape \(2, 2\) but must have shape \(3, 3\)"),
        ("r2", np.eye(4), r"r2 has shape \(4, 4\) but must have shape \(3, 3\)"),
        ("t1", np.zeros(2), r"t1 has shape \(2,\) but must have shape \(3,\)"),
        (
            "aff1",
            np.tile([1.0, 0.0, 0.0, 1.0], (2, 1)),
            r"has shape \(2, 4\) but must have shape \(5, 4\)",
        ),
    ],
)
def test_geometric_arguments_of_the_wrong_shape_raise(name, arg, value, expected):
    with pytest.raises(ValueError, match=expected):
        _call(name, **{arg: value})


@pytest.mark.parametrize("name", _POLAR)
def test_a_fundamental_matrix_of_the_wrong_shape_raises(name):
    with pytest.raises(
        ValueError, match=r"f_matrix has shape \(1, 9\) but must have shape \(3, 3\)"
    ):
        _call(name, f=_F.reshape(1, 9))
