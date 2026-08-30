# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The point-estimation kernel against the Python loop it replaced.

The reference below is the loop `seed_relax.structure.estimate_points` carried
until the kernel took over, verbatim except for the module-level names it
reached for.  It is kept HERE and nowhere else: the package ships the kernel,
and this module is the only thing that still runs the loop, so the two can be
compared on the same arrays.

The two do not agree bit for bit and are not expected to.  The rules that
decide a cluster -- the pairwise angle floor, the cheirality test, the
reprojection bar -- are evaluated identically, but the least-squares midpoint
itself is solved differently: the loop factors the normal matrix (LAPACK's
general solve, through ``numpy.linalg.solve``), while the kernel inverts it
through its symmetric eigendecomposition, which is what makes it report the
observability spectrum for free.  The two solves answer the same linear system
to the conditioning of the matrix, so the positions agree to a relative
tolerance rather than exactly, and a cluster whose cheirality or bar decision
sits exactly on the boundary could in principle differ.
"""

import math

import numpy as np
import pytest

from seed_relax.structure import estimate_points_verdicts
from sfmtool._sfmtool.reconstruction import VERDICT_CODES as CODES

FOCAL = 500.0

#: How closely the two solves have to agree on a finite position, relative to
#: the position's own magnitude.
POSITION_RTOL = 1e-9


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


# ── The Python loop, as it stood ──────────────────────────────────────────


def _angular_lsq(centres, dirs):
    """The point minimizing the squared perpendicular distance to the rays."""
    acc = np.zeros((3, 3))
    rhs = np.zeros(3)
    for c, u in zip(centres, dirs):
        p = np.eye(3) - np.outer(u, u)
        acc += p
        rhs += p @ c
    try:
        return np.linalg.solve(acc, rhs)
    except np.linalg.LinAlgError:
        return None


def _world_rays(cam, rot, uv, slot_i):
    """Unit world-frame rays of the observations, through ``cam``."""
    local = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(uv, float)), float)
    n = np.linalg.norm(local, axis=1, keepdims=True)
    local = local / np.maximum(n, 1e-12)
    return np.einsum("nji,nj->ni", rot[slot_i], local)


def reference(
    cam, rot, cen, uv, slot_i, slot_c, points, at_inf, floor_rad, bar_px=None
):
    """The loop the kernel replaced.  Returns ``(points, at_inf, census)``."""
    pts = np.asarray(points, float).copy()
    inf = np.asarray(at_inf, bool).copy()
    n_pts = len(pts)
    d = _world_rays(cam, rot, uv, slot_i)
    order = np.argsort(slot_c, kind="stable")
    sc = np.asarray(slot_c)[order]
    census = {
        "n_seen": 0,
        "n_finite": 0,
        "n_thin": 0,
        "n_behind": 0,
        "n_single": 0,
        "n_reproj_cut": 0,
        "n_rows": int(len(sc)),
        "n_points": int(n_pts),
        "tri_ang_med_deg": None,
    }
    if not len(sc):
        return pts, inf, census
    cuts = np.flatnonzero(np.diff(sc)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(sc)]))
    cos_floor = math.cos(float(floor_rad))
    tri_ang = []
    n_fin = n_thin = n_behind = n_single = n_reproj = 0

    def bearing(k, v):
        pts[k] = v / max(1e-12, float(np.linalg.norm(v)))
        inf[k] = True

    for lo, hi in zip(starts, ends):
        g = order[lo:hi]
        k = int(sc[lo])
        dk = d[g]
        good = np.isfinite(dk).all(axis=1)
        gg = g[good]
        dk = dk[good]
        if len(dk) < 2:
            n_single += 1
            bearing(k, dk[0] if len(dk) else np.array([0.0, 0.0, -1.0]))
            continue
        cosm = float((dk @ dk.T).min())
        if cosm > cos_floor:
            n_thin += 1
            bearing(k, dk.mean(axis=0))
            continue
        ck = cen[slot_i[gg]]
        p = _angular_lsq(ck, dk)
        if p is None or not np.isfinite(p).all():
            bearing(k, dk.mean(axis=0))
            continue
        if not (np.einsum("ij,ij->i", p[None, :] - ck, dk) > 0).all():
            n_behind += 1
            bearing(k, dk.mean(axis=0))
            continue
        if bar_px is not None:
            xc = np.einsum("nij,nj->ni", rot[slot_i[gg]], p[None, :] - cen[slot_i[gg]])
            pred = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(xc)), float)
            res = np.linalg.norm(np.asarray(uv, float)[gg] - pred, axis=1)
            fin = np.isfinite(res)
            if not fin.any() or float(np.median(res[fin])) > float(bar_px):
                n_reproj += 1
                bearing(k, dk.mean(axis=0))
                continue
        pts[k] = p
        inf[k] = False
        n_fin += 1
        tri_ang.append(math.degrees(math.acos(max(-1.0, min(1.0, cosm)))))
    census.update(
        {
            "n_seen": int(len(np.unique(sc))),
            "n_finite": n_fin,
            "n_thin": n_thin,
            "n_behind": n_behind,
            "n_single": n_single,
            "n_reproj_cut": n_reproj,
            "tri_ang_med_deg": float(np.median(tri_ang)) if tri_ang else None,
        }
    )
    return pts, inf, census


# ── The comparison ────────────────────────────────────────────────────────


def compare(cam, geom, uv, slot_i, slot_c, n_points, floor_rad, bar_px=None):
    """Run both and assert they agree.  Returns the kernel's own census.

    ``geom`` is ``(quats, trans, rot, cen)``: the operation reads the poses as
    the state holds them, the loop as the matrices it derived."""
    quats, trans, rot, cen = geom
    points = np.zeros((n_points, 3))
    at_inf = np.ones(n_points, bool)
    want_pts, want_inf, want_census = reference(
        cam, rot, cen, uv, slot_i, slot_c, points, at_inf, floor_rad, bar_px
    )
    got_pts, got_inf, got_census, _verdicts = estimate_points_verdicts(
        cam, quats, trans, uv, slot_i, slot_c, n_points, floor_rad, bar_px
    )
    np.testing.assert_array_equal(got_inf, want_inf)
    for key, want in want_census.items():
        got = got_census[key]
        if key == "tri_ang_med_deg":
            if want is None:
                assert got is None
            else:
                assert got == pytest.approx(want, rel=1e-12, abs=1e-12)
        else:
            assert got == want, key
    # A bearing is a unit vector either way and has to match to the last bit
    # the normalization allows; a finite position carries the solve's own
    # difference.
    scale = np.maximum(1.0, np.linalg.norm(want_pts, axis=1))
    err = np.linalg.norm(got_pts - want_pts, axis=1) / scale
    assert err.max() <= POSITION_RTOL, f"worst relative position gap {err.max():.3e}"
    return got_census, got_pts, got_inf


def _synthetic(cam, seed, n_frames=6, n_points=200, floor_deg=0.5):
    """A capture-shaped batch: frames on an arc, points in front of them."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(-0.4, 0.4, n_frames)
    cen = np.stack([6.0 * np.sin(ang), 0.2 * ang, 6.0 * np.cos(ang) - 6.0], axis=1)
    rot = np.stack([np.eye(3)] * n_frames)
    world = np.stack(
        [
            rng.uniform(-3.0, 3.0, n_points),
            rng.uniform(-2.0, 2.0, n_points),
            -rng.uniform(4.0, 60.0, n_points),
        ],
        axis=1,
    )
    si, sc = [], []
    for c in range(n_points):
        # Between one and every frame sees each point, so the single-view,
        # thin and well-observed cases all occur.
        k = int(rng.integers(1, n_frames + 1))
        for f in sorted(rng.choice(n_frames, size=k, replace=False)):
            si.append(int(f))
            sc.append(c)
    si = np.asarray(si, np.int64)
    sc = np.asarray(sc, np.int64)
    xc = np.einsum("nij,nj->ni", rot[si], world[sc] - cen[si])
    uv = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(xc)), float)
    uv += rng.normal(0.0, 0.3, uv.shape)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, 1))
    geom = (quats, -cen, rot, cen)
    return geom, uv, si, sc, world, math.radians(floor_deg)


def test_the_two_agree_on_a_capture_shaped_batch(cam):
    geom, uv, si, sc, world, floor = _synthetic(cam, 20260829)
    census, _pts, _inf = compare(cam, geom, uv, si, sc, len(world), floor)
    # The batch has to exercise more than one rule for the comparison to say
    # anything.
    assert census["n_finite"] > 0
    assert census["n_single"] > 0
    assert census["n_seen"] == len(world)


def test_the_two_agree_with_the_reprojection_bar_on(cam):
    geom, uv, si, sc, world, floor = _synthetic(cam, 7)
    # A bar tight enough that the noise cuts a real share of the batch.
    census, _pts, _inf = compare(cam, geom, uv, si, sc, len(world), floor, bar_px=0.35)
    assert census["n_reproj_cut"] > 0
    assert census["n_finite"] > 0


def test_the_two_agree_when_the_floor_refuses_almost_everything(cam):
    geom, uv, si, sc, world, _floor = _synthetic(cam, 99)
    census, _pts, _inf = compare(cam, geom, uv, si, sc, len(world), math.radians(45.0))
    assert census["n_thin"] > 0


def test_the_two_agree_with_points_behind_the_cameras(cam):
    geom, uv, si, sc, world, floor = _synthetic(cam, 3)
    # Mirroring every pixel through the principal point turns the converging
    # rays into diverging ones, so the meet lands behind the cameras.
    uv = np.stack([640.0 - uv[:, 0], 480.0 - uv[:, 1]], axis=1)
    census, _pts, _inf = compare(cam, geom, uv, si, sc, len(world), floor)
    assert census["n_behind"] > 0


def test_an_empty_batch_is_answered_rather_than_raised(cam):
    """The loop could not read an empty batch; the operation answers it.

    ``pixel_to_ray_batch`` on no pixels comes back shaped ``(0, 0)``, which the
    loop's ray rotation could not broadcast, so an empty admission raised.  The
    operation states what an unobserved cluster is instead, which is the one
    behaviour the two do not share."""
    rot = np.stack([np.eye(3)] * 2)
    cen = np.zeros((2, 3))
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
    empty_uv = np.zeros((0, 2))
    empty_i = np.zeros(0, np.int64)
    with pytest.raises(ValueError):
        reference(
            cam,
            rot,
            cen,
            empty_uv,
            empty_i,
            empty_i,
            np.array([[1.0, 2.0, 3.0]]),
            np.array([True]),
            math.radians(0.5),
        )
    pts, at_inf, census, verdicts = estimate_points_verdicts(
        cam, quats, -cen, empty_uv, empty_i, empty_i, 1, math.radians(0.5)
    )
    # Nothing names the cluster, so it has no ray and reads as the relaxation's
    # own `few = bearing` fallback.
    np.testing.assert_array_equal(pts, [[0.0, 0.0, -1.0]])
    np.testing.assert_array_equal(at_inf, [True])
    assert census["n_rows"] == 0
    assert census["n_seen"] == 0
    assert census["n_points"] == 1
    assert census["n_single"] == 1
    np.testing.assert_array_equal(verdicts, [CODES["few"]])


def test_a_cluster_no_observation_names_becomes_the_fallback_bearing(cam):
    """The one place the operation and the loop part company.

    The loop left a cluster nothing observed exactly as it came in.  The
    operation has no incoming state to leave alone: an unobserved cluster has
    no usable ray, which is what `few` decides, and under the relaxation's
    `bearing` setting that is the canonical forward direction.  Every stage of
    the relaxation indexes its state by clusters its own rows name, so the case
    does not arise there; it is stated here so it is not discovered later."""
    geom, uv, si, sc, world, floor = _synthetic(cam, 11, n_points=40)
    n = len(world) + 2
    pts, at_inf, census, verdicts = estimate_points_verdicts(
        cam, geom[0], geom[1], uv, si, sc, n, floor
    )
    np.testing.assert_array_equal(pts[-2:], [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    assert at_inf[-2:].all()
    np.testing.assert_array_equal(verdicts[-2:], [CODES["few"], CODES["few"]])
    assert census["n_seen"] == len(world)
    assert census["n_points"] == n


def test_the_verdicts_name_the_partition(cam):
    geom, uv, si, sc, world, floor = _synthetic(cam, 5)
    _pts, at_inf, census, verdicts = estimate_points_verdicts(
        cam, geom[0], geom[1], uv, si, sc, len(world), floor, bar_px=0.35
    )
    assert len(verdicts) == len(world)
    assert int((verdicts == CODES["finite"]).sum()) == census["n_finite"]
    assert int((verdicts == CODES["thin"]).sum()) == census["n_thin"]
    assert int((verdicts == CODES["behind"]).sum()) == census["n_behind"]
    assert int((verdicts == CODES["few"]).sum()) == census["n_single"]
    assert int((verdicts == CODES["over_bar"]).sum()) == census["n_reproj_cut"]
    # The relaxation never marks, so nothing reads as marked, and finite is
    # exactly the complement of the bearings.
    assert int((verdicts == CODES["marked"]).sum()) == 0
    np.testing.assert_array_equal(verdicts == CODES["finite"], ~at_inf)


def test_the_operation_repeats_itself_bit_for_bit(cam):
    geom, uv, si, sc, world, floor = _synthetic(cam, 4242)
    args = (cam, geom[0], geom[1], uv, si, sc, len(world), floor, 0.5)
    a = estimate_points_verdicts(*args)
    b = estimate_points_verdicts(*args)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]
    np.testing.assert_array_equal(a[3], b[3])
