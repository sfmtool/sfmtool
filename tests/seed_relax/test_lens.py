# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The lens promotion, the knot re-expression, and the read on bearings.

Needs the native extension, like `tests/rust_bindings/`.
"""

import numpy as np
import pytest

from seed_relax import lens, quat

F_TRUE = 300.0
WIDTH = HEIGHT = 480


def _cam(model, focal=F_TRUE, **extra):
    from sfmtool._sfmtool.geometry import CameraIntrinsics

    params = {
        "focal_length": float(focal),
        "principal_point_x": WIDTH / 2.0,
        "principal_point_y": HEIGHT / 2.0,
    }
    params.update(extra)
    return CameraIntrinsics.from_dict(
        {
            "model": model,
            "width": WIDTH,
            "height": HEIGHT,
            "parameters": params,
        }
    )


@pytest.mark.parametrize("model", ["SIMPLE_PINHOLE", "EQUIDISTANT_FISHEYE"])
def test_a_zero_spline_is_the_base_map(model):
    base = _cam(model)
    got = lens.promote(base, lens.base_focal(base), np.zeros(2), 1.2)
    assert lens.family_of(base.model) == lens.family_of(got.model)
    thetas = lens.sample_thetas(1.0)
    assert (
        float(
            np.abs(lens.radial_map(got, thetas) - lens.radial_map(base, thetas)).max()
        )
        == 0.0
    )
    rays = np.ascontiguousarray(
        np.stack(
            [np.sin(thetas), np.zeros_like(thetas), -np.cos(thetas)],
            axis=1,
        )
    )
    a = np.asarray(base.ray_to_pixel_batch(rays), float)
    b = np.asarray(got.ray_to_pixel_batch(rays), float)
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    assert ok.any()
    assert float(np.abs(a[ok] - b[ok]).max()) == 0.0


def test_a_two_knot_map_is_reproduced_on_four_knots():
    base = _cam("EQUIDISTANT_FISHEYE")
    d_max = 1.2
    thetas = lens.sample_thetas(1.0)
    shaped = lens.promote(base, F_TRUE, np.array([0.01, -0.02]), d_max)
    wider, resid = lens.refit_knots(shaped, 4, thetas, d_max)
    assert wider.parameters["bspline_coeff_count"] == 4
    assert resid < 1e-6
    assert (
        float(
            np.abs(
                lens.radial_map(wider, thetas) - lens.radial_map(shaped, thetas)
            ).max()
        )
        < 1e-6
    )


def test_a_base_camera_re_expressed_gives_exactly_zeros():
    base = _cam("SIMPLE_PINHOLE")
    thetas = lens.sample_thetas(0.6)
    got, resid = lens.refit_knots(
        lens.promote(base, F_TRUE, np.zeros(2), 1.2), 4, thetas, 1.2
    )
    coeffs = [float(got.parameters[f"bspline_c{i}"]) for i in range(4)]
    assert max(abs(c) for c in coeffs) < 1e-12
    assert resid < 1e-9


def _bearing_member(base_focal_px):
    """A rotation-only member: one rotation per frame, one bearing per point."""
    import seed_candidate_eval as EV

    n_frames, n_pts = 8, 60
    truth = _cam("EQUIDISTANT_FISHEYE", focal=F_TRUE)
    # Frames sweeping about the vertical, points spread over the field.
    angles = np.linspace(-0.35, 0.35, n_frames)
    rots = np.stack(
        [
            np.array(
                [
                    [np.cos(a), 0.0, np.sin(a)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(a), 0.0, np.cos(a)],
                ]
            )
            for a in angles
        ]
    )
    k = np.arange(n_pts)
    theta = 0.05 + 0.75 * (k % 10) / 9.0
    phi = 2.0 * np.pi * (k % 7) / 7.0
    dirs = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)],
        axis=1,
    )
    obs_c, obs_i, uv = [], [], []
    for f in range(n_frames):
        xc = dirs @ rots[f].T
        px = np.asarray(truth.ray_to_pixel_batch(np.ascontiguousarray(xc)), float)
        good = np.isfinite(px).all(axis=1)
        obs_c.extend(np.nonzero(good)[0].tolist())
        obs_i.extend([f] * int(good.sum()))
        uv.append(px[good])
    uv = np.concatenate(uv)
    return (
        EV.Member(
            0,
            "rotation_only",
            [f"img/{f:03d}.jpg" for f in range(n_frames)],
            _cam("EQUIDISTANT_FISHEYE", focal=base_focal_px),
            float(base_focal_px),
            quat.rotvecs_from_rots(rots),
            np.zeros((n_frames, 3)),
            np.ones(n_frames, bool),
            dirs,
            (
                np.array(obs_c, np.int64),
                np.array(obs_i, np.int64),
                uv,
                np.array(obs_c, np.int64),
            ),
        ),
        truth,
    )


def test_the_read_on_bearings_recovers_the_focal():
    # The base focal is 12 per cent short of the lens that made the pixels.
    m, truth = _bearing_member(0.88 * F_TRUE)
    rec, cam, rots = lens.rot_lens_ba(m, knots=2, opt_bspline=True)
    # Every point is a bearing, so the state has no finite survivors at all.
    # The solve still runs at the kernel's own degenerate floor: that floor
    # counts every trim survivor, not only the finite ones.
    assert rec.get("refused") is None
    assert cam is not None and rots is not None
    assert rec["resid_finite_frac"] > 0.99
    assert rec["reproj_med_px"] < 1.0
    thetas = lens.sample_thetas(0.8)
    got = lens.equivalent_focal(cam, thetas)
    want = lens.equivalent_focal(truth, thetas)
    assert abs(got / want - 1.0) < 0.02


def test_a_member_with_no_field_refuses():
    m, _truth = _bearing_member(F_TRUE)
    # Every observation sits on the principal point: no radius, so no spline
    # domain, so nothing to release the lens over.
    m.obs_uv[:] = np.array([WIDTH / 2.0, HEIGHT / 2.0])
    rec, cam, rots = lens.rot_lens_ba(m, knots=2)
    assert rec["refused"] == "no spline domain"
    assert cam is None and rots is None


def test_the_gate_reads_the_base_chart():
    take, why = lens.gate_early_release(_cam("EQUIDISTANT_FISHEYE"))
    assert take and "is the chart" in why
    take, why = lens.gate_early_release(_cam("SIMPLE_PINHOLE"))
    assert not take and "is not the chart" in why
