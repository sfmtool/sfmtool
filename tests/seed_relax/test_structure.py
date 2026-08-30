# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Point estimation at a fixed geometry, and the settled schedule."""

import numpy as np
import pytest

from seed_relax.structure import estimate_points, later_schedule

FOCAL = 500.0


@pytest.fixture(name="cam")
def _cam():
    from sfmtool._sfmtool.geometry import CameraIntrinsics

    return CameraIntrinsics.from_dict(
        {
            "model": "SIMPLE_PINHOLE",
            "width": 640,
            "height": 480,
            "parameters": {
                "focal_length": FOCAL,
                "principal_point_x": 320.0,
                "principal_point_y": 240.0,
            },
        }
    )


def _observations(cam, centres, points, pairs):
    """``(uv, slot_i, slot_c)`` for the listed ``(frame, point)`` pairs."""
    rot = np.stack([np.eye(3)] * len(centres))
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(centres), 1))
    slot_i = np.array([p[0] for p in pairs], np.int64)
    slot_c = np.array([p[1] for p in pairs], np.int64)
    xc = np.einsum("nij,nj->ni", rot[slot_i], points[slot_c] - centres[slot_i])
    uv = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(xc)), float)
    return quats, uv, slot_i, slot_c


def test_a_finite_point_comes_back_where_it_was(cam):
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points = np.array([[0.3, -0.2, -8.0]])
    q, uv, si, sc = _observations(cam, centres, points, [(0, 0), (1, 0), (2, 0)])
    pts, at_inf, census = estimate_points(
        cam, q, -centres, uv, si, sc, 1, np.radians(0.02)
    )
    assert not at_inf[0]
    assert np.allclose(pts[0], points[0], atol=1e-9)
    assert census["n_finite"] == 1
    assert census["n_seen"] == 1


def test_a_single_view_point_is_a_bearing(cam):
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points = np.array([[0.3, -0.2, -8.0]])
    q, uv, si, sc = _observations(cam, centres, points, [(0, 0)])
    pts, at_inf, census = estimate_points(
        cam, q, -centres, uv, si, sc, 1, np.radians(0.02)
    )
    assert at_inf[0]
    assert census["n_single"] == 1
    assert abs(float(np.linalg.norm(pts[0])) - 1.0) < 1e-12


def test_a_pair_inside_the_floor_is_a_bearing(cam):
    centres = np.array([[0.0, 0.0, 0.0], [1.0e-3, 0.0, 0.0]])
    points = np.array([[0.0, 0.0, -400.0]])
    q, uv, si, sc = _observations(cam, centres, points, [(0, 0), (1, 0)])
    pts, at_inf, census = estimate_points(
        cam, q, -centres, uv, si, sc, 1, np.radians(0.5)
    )
    assert at_inf[0]
    assert census["n_thin"] == 1
    assert abs(float(np.linalg.norm(pts[0])) - 1.0) < 1e-12


def test_a_point_behind_a_camera_is_a_bearing(cam):
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points = np.array([[0.3, -0.2, -8.0]])
    q, uv, si, sc = _observations(cam, centres, points, [(0, 0), (1, 0), (2, 0)])
    # Move the third camera past the point along its own viewing axis, so the
    # rays still cross but the crossing is behind that camera.
    moved = centres.copy()
    moved[2] = np.array([0.3, -0.2, -20.0])
    pts, at_inf, census = estimate_points(
        cam, q, -moved, uv, si, sc, 1, np.radians(0.02)
    )
    assert at_inf[0]
    assert census["n_behind"] == 1


def test_the_reprojection_bar_demotes(cam):
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points = np.array([[0.3, -0.2, -8.0]])
    q, uv, si, sc = _observations(cam, centres, points, [(0, 0), (1, 0), (2, 0)])
    # Rays that no longer meet: an estimate exists, and it explains none of
    # the three observations at the adjustment's own final bound.  The
    # disagreement differs per observation, since a uniform pixel shift is
    # just a shifted point and would be explained perfectly.
    bent = uv + np.array([[20.0, 0.0], [-20.0, 0.0], [0.0, 20.0]])
    kept = estimate_points(cam, q, -centres, bent, si, sc, 1, np.radians(0.02))
    assert not kept[1][0]
    cut = estimate_points(
        cam, q, -centres, bent, si, sc, 1, np.radians(0.02), bar_px=4.0
    )
    assert cut[1][0]
    assert cut[2]["n_reproj_cut"] == 1


def test_the_state_the_poses_were_given_in_is_left_alone(cam):
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points = np.array([[0.3, -0.2, -8.0], [0.0, 0.4, -9.0]])
    q, uv, si, sc = _observations(
        cam, centres, points, [(0, 0), (1, 0), (0, 1), (1, 1)]
    )
    trans = -centres
    before = (q.tobytes(), trans.tobytes(), uv.tobytes())
    pts, at_inf, census = estimate_points(
        cam, q, trans, uv, si, sc, 2, np.radians(0.02)
    )
    assert (q.tobytes(), trans.tobytes(), uv.tobytes()) == before
    assert census["n_finite"] == 2
    assert not at_inf.any()
    assert pts.shape == (2, 3)


def test_the_later_schedule_keeps_only_the_stages_the_state_still_needs():
    full = [(50.0, 5.0), (12.0, 2.0), (4.0, 1.0)]
    assert later_schedule(np.full(100, 30.0), schedule=full) == full[1:]
    assert later_schedule(np.full(100, 8.0), schedule=full) == [(4.0, 1.0)]
    assert later_schedule(np.full(100, 1.0), schedule=full) == [(4.0, 1.0)]
    assert later_schedule(np.full(100, 0.1), schedule=full) == [(4.0, 1.0)]
    assert later_schedule(np.full(100, np.inf), schedule=full) is None
