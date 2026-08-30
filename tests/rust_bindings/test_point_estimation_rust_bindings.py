# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the point-estimation binding
(``sfmtool._sfmtool.reconstruction.estimate_points``, ``VERDICT_CODES``).

The operation re-reads every track from its own observations at one geometry
and decides, per track, what those observations support. Two input forms, and
five rules each with an off position; with every rule off it is the batch
triangulation solve. Canonical camera convention throughout, so a point in
front of a camera has ``z < 0`` in that camera's frame. Rotations are built with
numpy (the test env has no scipy).
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.analysis import triangulate_batch
from sfmtool._sfmtool.geometry import CameraIntrinsics
from sfmtool._sfmtool.reconstruction import VERDICT_CODES, estimate_points

FINITE = VERDICT_CODES["finite"]
MARKED = VERDICT_CODES["marked"]
THIN = VERDICT_CODES["thin"]
BEHIND = VERDICT_CODES["behind"]
OVER_BAR = VERDICT_CODES["over_bar"]
FEW = VERDICT_CODES["few"]
FINITE_PRUNED = VERDICT_CODES["finite_pruned"]

#: Identity world-to-camera rotation, WXYZ.
IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _cam(f=500.0, w=640, h=480):
    return CameraIntrinsics.from_dict(
        {
            "model": "SIMPLE_PINHOLE",
            "width": w,
            "height": h,
            "parameters": {
                "focal_length": f,
                "principal_point_x": w / 2.0,
                "principal_point_y": h / 2.0,
            },
        }
    )


def _views(centres):
    """``(quaternions, translations)`` of cameras looking along -Z."""
    centres = np.asarray(centres, float)
    quats = np.tile(IDENTITY_Q, (len(centres), 1))
    return quats, -centres


def _observations(cam, centres, world, pairs):
    """``(uv, obs_image, obs_point)`` for the listed ``(camera, track)`` pairs."""
    centres = np.asarray(centres, float)
    world = np.asarray(world, float)
    oi = np.array([p[0] for p in pairs], np.uint32)
    op = np.array([p[1] for p in pairs], np.uint32)
    xc = world[op.astype(np.int64)] - centres[oi.astype(np.int64)]
    uv = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(xc)), float)
    return uv, oi, op


PAIR = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
WORLD = np.array([[0.2, -0.1, -5.0]])


def _call(cam, centres, world, pairs, **rules):
    quats, trans = _views(centres)
    uv, oi, op = _observations(cam, centres, world, pairs)
    return estimate_points(
        uv=uv,
        obs_image=oi,
        obs_point=op,
        camera=cam,
        quaternions_wxyz=quats,
        translations=trans,
        n_points=len(world),
        **rules,
    )


# ── Rules off is the solve ────────────────────────────────────────────────


def test_every_rule_off_is_the_batch_triangulation_solve():
    dirs = np.array([[0.0, 0.0, -1.0], [0.2, 0.0, -1.0], [-0.3, 0.1, -1.0]])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    offsets = np.array([0, 3], np.int64)
    want = triangulate_batch(dirs, centres, offsets)
    got = estimate_points(dirs=dirs, centres=centres, offsets=offsets)
    npt.assert_array_equal(got["verdicts"], [FINITE])
    npt.assert_array_equal(got["xyzw"][0][:3], want["points"][0])
    assert got["xyzw"][0][3] == 1.0
    npt.assert_array_equal(got["in_front"], want["in_front_of_all_cameras"])
    assert got["census"]["triangulation_angle_median_deg"] is None
    assert got["census"]["seen"] == 1


def test_the_two_forms_agree_on_the_same_geometry():
    cam = _cam()
    from_obs = _call(cam, PAIR, WORLD, [(0, 0), (1, 0)])
    uv, _oi, _op = _observations(cam, PAIR, WORLD, [(0, 0), (1, 0)])
    dirs = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(uv)), float)
    from_rays = estimate_points(
        dirs=dirs, centres=PAIR, offsets=np.array([0, 2], np.int64)
    )
    npt.assert_array_equal(from_obs["xyzw"], from_rays["xyzw"])
    npt.assert_array_equal(from_obs["verdicts"], from_rays["verdicts"])


# ── The rules ─────────────────────────────────────────────────────────────


def test_a_marked_track_is_not_solved():
    cam = _cam()
    out = _call(
        cam,
        PAIR,
        WORLD,
        [(0, 0), (1, 0)],
        marks=np.array([True]),
        floor_rad=np.deg2rad(0.5),
        cheirality=True,
    )
    npt.assert_array_equal(out["verdicts"], [MARKED])
    assert out["xyzw"][0][3] == 0.0
    npt.assert_allclose(np.linalg.norm(out["xyzw"][0][:3]), 1.0, atol=1e-12)
    assert out["census"]["marked"] == 1
    solved = _call(
        cam, PAIR, WORLD, [(0, 0), (1, 0)], floor_rad=np.deg2rad(0.5), cheirality=True
    )
    npt.assert_array_equal(solved["verdicts"], [FINITE])


def test_a_pair_exactly_at_the_floor_is_not_thin():
    theta = 0.2
    dirs = np.array([[0.0, 0.0, -1.0], [np.sin(theta), 0.0, -np.cos(theta)]])
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    offsets = np.array([0, 2], np.int64)
    floor = np.arccos(np.clip(dirs[0] @ dirs[1], -1.0, 1.0))
    at = estimate_points(
        dirs=dirs, centres=centres, offsets=offsets, floor_rad=float(floor)
    )
    npt.assert_array_equal(at["verdicts"], [FINITE])
    inside = estimate_points(
        dirs=dirs, centres=centres, offsets=offsets, floor_rad=float(floor) * 1.0001
    )
    npt.assert_array_equal(inside["verdicts"], [THIN])
    assert inside["census"]["thin"] == 1


def test_cheirality_off_keeps_the_point_and_reports_the_flag():
    dirs = np.array([[0.0, 0.0, -1.0], [1e-2, 0.0, -1.0]])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    offsets = np.array([0, 2], np.int64)
    kept = estimate_points(dirs=dirs, centres=centres, offsets=offsets)
    npt.assert_array_equal(kept["verdicts"], [FINITE])
    npt.assert_array_equal(kept["in_front"], [False])
    cut = estimate_points(dirs=dirs, centres=centres, offsets=offsets, cheirality=True)
    npt.assert_array_equal(cut["verdicts"], [BEHIND])
    assert cut["xyzw"][0][3] == 0.0
    assert cut["census"]["behind"] == 1


def test_the_bar_demotes_what_its_own_rows_disagree_with():
    cam = _cam()
    triple = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    quats, trans = _views(triple)
    uv, oi, op = _observations(cam, triple, WORLD, [(0, 0), (1, 0), (2, 0)])
    uv = uv + np.array([[20.0, 0.0], [-20.0, 0.0], [20.0, 0.0]])
    common = dict(
        uv=uv,
        obs_image=oi,
        obs_point=op,
        camera=cam,
        quaternions_wxyz=quats,
        translations=trans,
        n_points=1,
    )
    loose = estimate_points(bar_px=1e6, **common)
    npt.assert_array_equal(loose["verdicts"], [FINITE])
    tight = estimate_points(bar_px=0.5, **common)
    npt.assert_array_equal(tight["verdicts"], [OVER_BAR])
    assert tight["census"]["over_bar"] == 1
    assert tight["xyzw"][0][3] == 0.0


def test_the_bar_needs_the_observation_form():
    with pytest.raises(ValueError, match="bar_px needs the observation form"):
        estimate_points(
            dirs=np.zeros((2, 3)),
            centres=np.zeros((2, 3)),
            offsets=np.array([0, 2], np.int64),
            bar_px=1.0,
        )


# ── Few ───────────────────────────────────────────────────────────────────


def test_one_usable_ray_is_a_bearing_or_absent():
    dirs = np.array([[0.0, 0.0, -1.0]])
    centres = np.zeros((1, 3))
    offsets = np.array([0, 1], np.int64)
    bearing = estimate_points(
        dirs=dirs, centres=centres, offsets=offsets, few="bearing"
    )
    npt.assert_array_equal(bearing["verdicts"], [FEW])
    npt.assert_array_equal(bearing["xyzw"], [[0.0, 0.0, -1.0, 0.0]])
    absent = estimate_points(dirs=dirs, centres=centres, offsets=offsets)
    npt.assert_array_equal(absent["verdicts"], [FEW])
    assert np.isnan(absent["xyzw"]).all()
    assert absent["census"]["few"] == 1


def test_no_usable_ray_falls_back_to_the_forward_direction():
    out = estimate_points(
        dirs=np.zeros((0, 3)),
        centres=np.zeros((0, 3)),
        offsets=np.array([0, 0], np.int64),
        few="bearing",
    )
    npt.assert_array_equal(out["xyzw"], [[0.0, 0.0, -1.0, 0.0]])


def test_few_is_read_before_marks():
    out = estimate_points(
        dirs=np.array([[0.0, 0.0, -1.0]]),
        centres=np.zeros((1, 3)),
        offsets=np.array([0, 1], np.int64),
        marks=np.array([True]),
    )
    npt.assert_array_equal(out["verdicts"], [FEW])
    assert np.isnan(out["xyzw"]).all()
    assert out["census"]["marked"] == 0


def test_a_track_no_observation_names_is_a_few_track():
    cam = _cam()
    world = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, -5.0]])
    out = _call(cam, PAIR, world, [(0, 1), (1, 1)], few="bearing")
    npt.assert_array_equal(out["verdicts"], [FEW, FINITE])
    npt.assert_array_equal(out["xyzw"][0], [0.0, 0.0, -1.0, 0.0])
    assert out["census"]["seen"] == 2
    assert out["census"]["few"] == 1


# ── Contract ──────────────────────────────────────────────────────────────


def test_the_result_repeats_itself_bit_for_bit():
    cam = _cam()
    rng = np.random.default_rng(4242)
    centres = np.stack([np.linspace(-2.0, 2.0, 5), np.zeros(5), np.zeros(5)], axis=1)
    world = np.stack(
        [
            rng.uniform(-3.0, 3.0, 200),
            rng.uniform(-2.0, 2.0, 200),
            -rng.uniform(5.0, 40.0, 200),
        ],
        axis=1,
    )
    pairs = [(f, c) for c in range(200) for f in range(5)]
    kw = dict(floor_rad=np.deg2rad(0.5), cheirality=True, bar_px=2.0, few="bearing")
    a = _call(cam, centres, world, pairs, **kw)
    b = _call(cam, centres, world, pairs, **kw)
    npt.assert_array_equal(a["xyzw"], b["xyzw"])
    npt.assert_array_equal(a["verdicts"], b["verdicts"])
    assert a["census"] == b["census"]
    assert a["census"]["finite"] == 200


def test_the_verdict_code_table_is_exposed():
    assert VERDICT_CODES == {
        "finite": 0,
        "marked": 1,
        "thin": 2,
        "behind": 3,
        "over_bar": 4,
        "few": 5,
        "finite_pruned": 6,
    }


# ── The cheirality prune, read per observation ────────────────────────────

#: The revisit view of the specimen tracks: a camera off to one side, turned
#: away, whose observation is a ray far off the track's own and which the point
#: sits behind.
REVISIT_YAW_DEG = 80.0
REVISIT_CENTRE = np.array([5.0, 0.0, -5.0])
#: The point the agreeing views triangulate.
SPECIMEN = np.array([[0.0, 0.0, -5.0]])


def _yaw(deg):
    """``(quaternion_wxyz, rotation)`` of a world-to-camera yaw about +Y."""
    h = np.deg2rad(deg) / 2.0
    q = np.array([np.cos(h), 0.0, np.sin(h), 0.0])
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return q, np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _specimen(cam, n_agreeing, baseline=0.05):
    """A track of ``n_agreeing`` agreeing views plus the revisit view.

    The agreeing views sit on a short baseline and see :data:`SPECIMEN` where
    they say it is; the last row is the revisit camera's own observation, whose
    ray is ``REVISIT_YAW_DEG`` off theirs and which the point is behind.
    Returns the keyword arguments of one ``estimate_points`` call."""
    centres = np.array(
        [[baseline * k, 0.0, 0.0] for k in range(n_agreeing)] + [REVISIT_CENTRE]
    )
    quats = np.tile(IDENTITY_Q, (len(centres), 1))
    q_rev, rot_rev = _yaw(REVISIT_YAW_DEG)
    quats[-1] = q_rev
    trans = -centres.copy()
    trans[-1] = -(rot_rev @ REVISIT_CENTRE)
    xc = SPECIMEN[0] - centres[:n_agreeing]
    uv = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(xc)), float)
    # The revisit observation sits at the principal point, so its ray is that
    # camera's optical axis: a full REVISIT_YAW_DEG off the agreeing ones.
    uv = np.vstack([uv, [[cam.width / 2.0, cam.height / 2.0]]])
    return dict(
        uv=uv,
        obs_image=np.arange(len(centres), dtype=np.uint32),
        obs_point=np.zeros(len(centres), np.uint32),
        camera=cam,
        quaternions_wxyz=quats,
        translations=trans,
        n_points=1,
    )


@pytest.mark.parametrize("n_agreeing", [4, 2])
def test_one_wrong_observation_no_longer_hides_a_finite_point(n_agreeing):
    """The two specimen tracks: five observations and three, one wrong each."""
    cam = _cam()
    kw = _specimen(cam, n_agreeing)
    floor = dict(floor_rad=np.deg2rad(0.05), few="bearing")

    # The agreeing rays are within a couple of degrees of each other and the
    # revisit ray is tens of degrees away, which is the shape of the defect.
    dirs = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(kw["uv"])), float)
    _q, rot_rev = _yaw(REVISIT_YAW_DEG)
    world = dirs.copy()
    world[-1] = rot_rev.T @ dirs[-1]
    cosines = world @ world.T
    agree = np.degrees(np.arccos(np.clip(cosines[:n_agreeing, :n_agreeing], -1, 1)))
    assert agree.max() < 2.0
    off = np.degrees(np.arccos(np.clip(cosines[-1, :n_agreeing], -1, 1)))
    assert off.min() > 45.0

    # As it stands, the whole track is a bearing.
    whole = estimate_points(cheirality=True, **floor, **kw)
    npt.assert_array_equal(whole["verdicts"], [BEHIND])
    assert whole["xyzw"][0][3] == 0.0
    assert not whole["pruned"].any()

    # Read per observation, the wrong one is dropped and the point stands.
    pruned = estimate_points(cheirality=True, prune_behind=True, **floor, **kw)
    npt.assert_array_equal(pruned["verdicts"], [FINITE_PRUNED])
    assert pruned["xyzw"][0][3] == 1.0
    npt.assert_allclose(pruned["xyzw"][0][:3], SPECIMEN[0], atol=1e-9)
    npt.assert_array_equal(pruned["pruned"], [False] * n_agreeing + [True])
    npt.assert_array_equal(pruned["in_front"], [True])
    assert pruned["census"]["finite_pruned"] == 1
    assert pruned["census"]["pruned_obs"] == 1
    assert pruned["census"]["behind"] == 0
    assert pruned["census"]["finite"] == 0


def test_a_majority_behind_is_still_a_bearing():
    dirs = np.array(
        [
            [0.0, 0.0, -1.0],
            [-0.2, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [-0.1, 0.0, 1.0],
        ]
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    centres = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -10.0],
            [1.0, 0.0, -10.0],
            [-1.0, 0.0, -10.0],
        ]
    )
    offsets = np.array([0, 5], np.int64)
    common = dict(dirs=dirs, centres=centres, offsets=offsets, cheirality=True)
    whole = estimate_points(**common)
    pruned = estimate_points(prune_behind=True, **common)
    npt.assert_array_equal(pruned["verdicts"], [BEHIND])
    npt.assert_array_equal(pruned["xyzw"], whole["xyzw"])
    assert not pruned["pruned"].any()
    assert pruned["census"]["pruned_obs"] == 0


def test_half_the_track_behind_is_not_a_minority():
    # Two rays that see the point and two the point sits behind: no majority.
    dirs = np.array(
        [[0.0, 0.0, -1.0], [-0.2, 0.0, -1.0], [0.0, 0.0, 1.0], [0.2, 0.0, 1.0]]
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    centres = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -10.0],
            [1.0, 0.0, -10.0],
        ]
    )
    out = estimate_points(
        dirs=dirs,
        centres=centres,
        offsets=np.array([0, 4], np.int64),
        cheirality=True,
        prune_behind=True,
    )
    npt.assert_array_equal(out["verdicts"], [BEHIND])
    assert not out["pruned"].any()


def test_the_prune_flags_are_over_the_observations_given():
    cam = _cam()
    kw = _specimen(cam, 4)
    out = estimate_points(cheirality=True, prune_behind=True, few="bearing", **kw)
    assert out["pruned"].shape == (len(kw["obs_image"]),)
    assert out["pruned"].dtype == np.bool_
    # The rows the caller keeps are the rows the estimate was solved on.
    keep = ~out["pruned"]
    again = estimate_points(
        uv=np.ascontiguousarray(kw["uv"][keep]),
        obs_image=np.ascontiguousarray(kw["obs_image"][keep]),
        obs_point=np.ascontiguousarray(kw["obs_point"][keep]),
        camera=cam,
        quaternions_wxyz=kw["quaternions_wxyz"],
        translations=kw["translations"],
        n_points=1,
        cheirality=True,
        few="bearing",
    )
    npt.assert_array_equal(again["verdicts"], [FINITE])
    npt.assert_array_equal(again["xyzw"], out["xyzw"])


def test_the_prune_off_is_the_reading_it_always_was():
    cam = _cam()
    kw = _specimen(cam, 4)
    a = estimate_points(cheirality=True, few="bearing", **kw)
    b = estimate_points(cheirality=True, prune_behind=False, few="bearing", **kw)
    npt.assert_array_equal(a["xyzw"], b["xyzw"])
    npt.assert_array_equal(a["verdicts"], b["verdicts"])
    npt.assert_array_equal(a["pruned"], b["pruned"])
    assert a["census"] == b["census"]
    assert a["census"]["finite_pruned"] == 0
    assert a["census"]["pruned_obs"] == 0


def test_the_prune_needs_the_rule_it_reads():
    cam = _cam()
    with pytest.raises(ValueError, match="needs cheirality"):
        estimate_points(prune_behind=True, **_specimen(cam, 4))


def test_the_inputs_are_checked():
    cam = _cam()
    quats, trans = _views(PAIR)
    uv, oi, op = _observations(cam, PAIR, WORLD, [(0, 0), (1, 0)])
    good = dict(
        uv=uv,
        obs_image=oi,
        obs_point=op,
        camera=cam,
        quaternions_wxyz=quats,
        translations=trans,
        n_points=1,
    )

    def call(**over):
        return estimate_points(**{**good, **over})

    with pytest.raises(ValueError, match="not both"):
        call(dirs=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="needs n_points"):
        estimate_points(**{k: v for k, v in good.items() if k != "n_points"})
    with pytest.raises(ValueError, match="uv must have shape"):
        call(uv=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="same length"):
        call(obs_image=np.zeros(1, np.uint32))
    with pytest.raises(ValueError, match="past the 2 images"):
        call(obs_image=np.array([0, 5], np.uint32))
    with pytest.raises(ValueError, match="past the 1 points"):
        call(obs_point=np.array([0, 9], np.uint32))
    with pytest.raises(ValueError, match="one entry per track"):
        call(marks=np.ones(3, bool))
    with pytest.raises(ValueError, match="not a rotation"):
        call(quaternions_wxyz=quats * 2.0)
    with pytest.raises(ValueError, match="few must be"):
        call(few="whatever")
    with pytest.raises(ValueError, match="non-decreasing"):
        estimate_points(
            dirs=np.zeros((4, 3)),
            centres=np.zeros((4, 3)),
            offsets=np.array([0, 3, 1], np.int64),
        )
