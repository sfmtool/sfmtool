# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The global orientation bit, and the antisymmetry it rests on."""

import types

import numpy as np

from seed_relax.orientation import angw_bit

N_FRAMES = 5
TOL = np.radians(0.5)


def _constellation():
    """Frames along a short arc, points in front of every one of them."""
    centres = {f: np.array([0.6 * f, 0.0, 0.0]) for f in range(N_FRAMES)}
    pts = np.stack(
        [
            np.linspace(-2.0, 2.0, 9),
            np.linspace(-1.0, 1.0, 9),
            np.full(9, -6.0),
        ],
        axis=1,
    )
    rot = np.stack([np.eye(3)] * N_FRAMES)
    m = types.SimpleNamespace(rot=rot)
    per_frame = {}
    for f in range(N_FRAMES):
        d = pts - centres[f]
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        per_frame[f] = (
            np.arange(len(pts), dtype=np.int64),
            d,
            np.arange(len(pts), dtype=np.int64),
        )
    return m, per_frame, centres


def test_the_right_way_up_reads_positive():
    m, per_frame, placed = _constellation()
    got = angw_bit(m, per_frame, placed, TOL)
    assert got["angw"] > 0
    assert got["obs_frac"] == 1.0
    assert got["margin_frac"] == 1.0
    assert got["pts"] == 9
    assert got["behind"] == 0


def test_the_reading_is_exactly_antisymmetric_under_reflection():
    m, per_frame, placed = _constellation()
    up = angw_bit(m, per_frame, placed, TOL)
    down = angw_bit(m, per_frame, {f: -c for f, c in placed.items()}, TOL)
    assert down["angw"] == -up["angw"]
    assert down["obs_front"] == up["obs_total"] - up["obs_front"]
    assert down["margin_frac"] == up["margin_frac"]
