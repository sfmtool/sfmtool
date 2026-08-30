# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The whole chain on a synthetic rotation-only member.

Needs the native extension: the relaxation's adjustment and the point
estimation both run through the kernel.
"""

import types

import numpy as np
import pytest

from seed_relax import Options, pipeline, release

F = 500.0
WIDTH, HEIGHT = 640, 480
N_FRAMES = 6
#: The frames sit on a short straight arc, a quarter unit apart.
CENTRES = np.stack([np.array([0.25 * f, 0.0, 0.0]) for f in range(N_FRAMES)])
#: The line is the hard case and the fixture keeps it: every baseline of a
#: colinear frame set carries the same direction, so the part of it the
#: perpendicular objective measures is empty for every spacing of the cameras
#: along the line.  Nothing in the directions states this spacing, and the
#: chain recovers it only because the pairs' own depths state it.
REFINE_RADIUS = 8.0
#: Feature indexes are unique per (image, cluster), which is what the fill-in
#: joins the member and the source file on.
FEATURE_STRIDE = 1000


def _camera(focal=F):
    from sfmtool._sfmtool.geometry import CameraIntrinsics

    return CameraIntrinsics.from_dict(
        {
            "model": "SIMPLE_PINHOLE",
            "width": WIDTH,
            "height": HEIGHT,
            "parameters": {
                "focal_length": float(focal),
                "principal_point_x": WIDTH / 2.0,
                "principal_point_y": HEIGHT / 2.0,
            },
        }
    )


def _spread(n, depth, seed):
    """``n`` points at one depth, spread evenly across the field."""
    k = np.arange(n)
    x = (-0.6 + 1.2 * ((k * 7 + seed) % n) / max(1, n - 1)) * depth
    y = (-0.4 + 0.8 * ((k * 3 + seed) % n) / max(1, n - 1)) * depth
    return np.stack([x, y, np.full(n, -float(depth))], axis=1)


#: Far points sit well inside the member's angular bound (a 3 px bar over a
#: 500 px focal is 0.34 degrees), so a rotation explains them; near points sit
#: well past it, which is the parallax the member's model refused.
FAR = _spread(60, 5000.0, 1)
NEAR = _spread(60, 10.0, 2)
#: The clusters the admission never held: finer features, at their own depth.
EXTRA = _spread(30, 14.0, 3)


def _project(cam, points, centre):
    xc = points - centre
    return np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(xc)), float)


def _member():
    """A rotation-only member: identity rotations, every point a bearing."""
    import seed_candidate_eval as EV

    cam = _camera()
    pts = np.concatenate([FAR, NEAR])
    obs_c, obs_i, obs_f, uv, keep = [], [], [], [], []
    for f in range(N_FRAMES):
        px = _project(cam, pts, CENTRES[f])
        ok = np.isfinite(px).all(axis=1)
        for c in np.nonzero(ok)[0]:
            obs_c.append(int(c))
            obs_i.append(f)
            obs_f.append(f * FEATURE_STRIDE + int(c))
            uv.append(px[c])
            # The model kept the far rows and refused the near ones.
            keep.append(int(c) < len(FAR))
    # The bearing the member stored: the direction from its single viewpoint.
    dirs = pts - CENTRES.mean(axis=0)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return EV.Member(
        0,
        "rotation_only",
        [f"cam/{f:03d}.jpg" for f in range(N_FRAMES)],
        cam,
        F,
        np.zeros((N_FRAMES, 3)),
        np.zeros((N_FRAMES, 3)),
        np.ones(N_FRAMES, bool),
        dirs,
        (
            np.array(obs_c, np.int64),
            np.array(obs_i, np.int64),
            np.array(uv, float),
            np.array(obs_f, np.int64),
        ),
        keep=np.array(keep, bool),
    )


def _source():
    """The selection handle the member was drawn from, plus finer clusters.

    The member's own clusters carry the admission's radius; the extra ones are
    half and a quarter of it, so they land in the rings below the floor."""
    cam = _camera()
    starts, images, features, affines = [0], [], [], []
    blocks = [
        (np.concatenate([FAR, NEAR]), 0, 2.0),
        (EXTRA, len(FAR) + len(NEAR), 1.0),
    ]
    for points, base, scale in blocks:
        for c in range(len(points)):
            for f in range(N_FRAMES):
                px = _project(cam, points[c : c + 1], CENTRES[f])[0]
                if not np.isfinite(px).all():
                    continue
                images.append(f)
                features.append(f * FEATURE_STRIDE + base + c)
                s = scale if c % 2 == 0 else 0.5 * scale
                affines.append(np.array([[s, 0.0, px[0]], [0.0, s, px[1]]]))
            starts.append(len(images))
    return types.SimpleNamespace(
        image_names=[f"cam/{f:03d}.jpg" for f in range(N_FRAMES)],
        refine_radius=REFINE_RADIUS,
        cluster_starts=np.array(starts, np.int64),
        member_images=np.array(images, np.int64),
        member_features=np.array(features, np.int64),
        member_affines=np.stack(affines),
    )


def _data(m):
    """The observation dict the hypothesis was solved on."""
    return {
        "names": list(m.names),
        "dims": [(WIDTH, HEIGHT)] * len(m.names),
        "obs_c": m.obs_c,
        "obs_i": m.obs_i,
        "obs_f": m.obs_f,
        "obs_uv": m.obs_uv,
        "adm_rank": np.arange(m.n_cl),
        "cl_quality": np.zeros(m.n_cl),
        "n_img": len(m.names),
        "n_cl": m.n_cl,
    }


@pytest.fixture(name="relaxed")
def _relaxed():
    return pipeline.run_member(_member(), _source(), Options())


def test_the_chain_places_every_frame(relaxed):
    assert relaxed.ok
    assert relaxed.census["n_placed"] == N_FRAMES
    assert len(relaxed.state["frames"]) == N_FRAMES
    assert relaxed.census["early_release"] == "held"


def test_the_near_points_graduate_and_the_far_ones_stay_bearings(relaxed):
    at_inf = np.asarray(relaxed.state["at_inf"], bool)
    clusters = np.asarray(relaxed.state["clusters"], np.int64)
    finite = set(clusters[~at_inf].tolist())
    near = set(range(len(FAR), len(FAR) + len(NEAR)))
    far = set(range(len(FAR)))
    assert len(near & finite) > 0.9 * len(near)
    assert len(far & finite) == 0


def test_the_centres_recover_the_arc(relaxed):
    from seed_relax.structure import centres_of

    _rot, cen = centres_of(relaxed.state)
    order = np.asarray(relaxed.state["frames"], np.int64)
    truth = CENTRES[order]

    # Up to a scale and a rigid motion the arc is a straight, evenly spaced
    # line, so the pairwise distance ratios are the statement.
    def ratios(x):
        d = np.linalg.norm(x[1:] - x[:-1], axis=1)
        return d / d.max()

    assert np.allclose(ratios(cen), ratios(truth), atol=1e-3)


def test_the_fill_in_adds_the_finer_clusters(relaxed):
    fill = relaxed.census["fill"]
    assert fill.get("refused") is None
    assert fill["n_candidates"] == len(EXTRA)
    assert fill["n_added"] > 0
    # No count: every candidate that lands in a band is admitted, and the only
    # candidates left out are the ones below the grid's last edge.
    assert fill["ring_cap"] is None
    assert any(r.get("n_taken") for r in fill["rings"])
    assert fill["n_added"] == sum(r["n_in_ring"] for r in fill["rings"])
    assert relaxed.member.n_cl == len(FAR) + len(NEAR) + fill["n_added"]


def test_the_perspective_chart_holds_below_the_settling_bar(relaxed):
    from seed_relax.fleet_constants import SETTLING_FINITE_COUNT

    late = relaxed.census["late_release"]
    assert late["family"] == "pinhole"
    assert not late["applied"]
    assert late["bar"] == SETTLING_FINITE_COUNT
    assert late["finite_count"] < SETTLING_FINITE_COUNT
    # Held means the base camera stands, so no spline is stated.
    assert relaxed.lens is None


def test_every_point_is_re_estimated_at_the_end(relaxed):
    retri = relaxed.census["retri"]
    assert retri["n_points"] == len(relaxed.state["at_inf"])
    assert retri["reproj_med_px"] < 1.0
    assert relaxed.census["n_finite_final"] > len(NEAR) * 0.9


def test_the_runaway_report_is_recorded_and_cuts_nothing(relaxed):
    rows = relaxed.runaway_frames
    assert len(rows) == N_FRAMES
    assert {r["frame"] for r in rows} == set(range(N_FRAMES))
    # Evenly spaced frames all sit where their neighbours do.
    assert relaxed.census["runaway"]["iso_max"] < 1.5
    assert relaxed.census["runaway"]["n_iso_over_3"] == 0


def test_two_runs_produce_the_same_arrays():
    a = pipeline.run_member(_member(), _source(), Options())
    b = pipeline.run_member(_member(), _source(), Options())
    for key in ("frames", "clusters", "quats", "trans", "points", "at_inf"):
        assert np.asarray(a.state[key]).tobytes() == np.asarray(b.state[key]).tobytes()


def test_the_writer_arrays_describe_the_extended_member(relaxed):
    m = relaxed.member
    data_x, pts, keep, res, at_inf = release.relaxed_arrays(relaxed, _data(_member()))
    assert data_x["n_cl"] == m.n_cl == len(pts) == len(at_inf)
    assert len(keep) == len(res) == len(m.obs_c)
    assert "adm_rank" not in data_x and "cl_quality" not in data_x
    assert keep.any()
    assert np.isfinite(res[keep]).all()
    # Every writable cluster carries a position, and a bearing is a unit one.
    written = np.unique(m.obs_c[keep])
    assert np.isfinite(pts[written]).all()
    bearings = written[at_inf[written]]
    assert np.allclose(np.linalg.norm(pts[bearings], axis=1), 1.0)


def test_the_release_record_reads_as_an_unqualified_relaxed_member(relaxed):
    res = release.relaxed_res(relaxed, _data(_member()), reach=0.9)
    assert res["flags"] == ["relaxed"]
    assert res["spread"] == 0.0
    assert res["kept"] == N_FRAMES
    assert 0.0 <= res["inl"] <= 1.0
    assert res["n_points_written"] <= len(res["release_pts"])
    assert res["rvec_full"].shape == (N_FRAMES, 3)
    assert res["posed_full"].all()
    opts = release.tool_options(relaxed, 7, paired_with=3, scope="capture")
    assert opts["structure_model"] == "relaxed"
    assert opts["qualified"] == "False"
    assert opts["paired_with"] == "3"
    assert opts["late_release"].startswith("held:")
    block = release.relaxation_block(relaxed)
    assert block["n_baselines"] == relaxed.census["n_baselines"]
    assert len(block["runaway"]["frames"]) == N_FRAMES
