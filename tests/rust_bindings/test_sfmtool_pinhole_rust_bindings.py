# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The SFMTOOL_PINHOLE camera model through the Python surface.

``SFMTOOL_PINHOLE`` is an sfmtool-native (non-COLMAP) model: a pinhole base
plus a monotone radial spline, ``r(rho) = f * (rho + delta(rho))`` with
``bspline_coeff_count`` dimensionless coefficients ``bspline_c0..cN`` on
``[0, bspline_rho_max]``, where ``rho = hypot(rx, ry) / rz = tan(theta)``.
These tests cover the boundaries it crosses in Python — the generic
`CameraIntrinsics.from_dict` path (including its parameter-map validation),
projection batch consistency, the exact zero-spline identity with
SIMPLE_PINHOLE, pickling, the pinhole fits it (unlike its fisheye sibling)
supports, and the pycolmap interop shim, which has NO carrier for it and must
reject loudly.
"""

import copy
import math
import pickle

import numpy as np
import pytest

from sfmtool._sfmtool.geometry import CameraIntrinsics

F = 250.0
W = H = 480


def _is_coeff(key):
    """True for a coefficient key ``bspline_c<i>``, not ``bspline_coeff_count``."""
    return key.startswith("bspline_c") and key.removeprefix("bspline_c").isdigit()


BSPLINE = {
    "bspline_rho_max": 0.9,
    "bspline_coeff_count": 8.0,
    "bspline_c0": 0.0008,
    "bspline_c1": 0.0031,
    "bspline_c2": 0.0075,
    "bspline_c3": 0.0142,
    "bspline_c4": 0.0236,
    "bspline_c5": 0.0361,
    "bspline_c6": 0.052,
    "bspline_c7": 0.0718,
}

SFMTOOL = {
    "model": "SFMTOOL_PINHOLE",
    "width": W,
    "height": H,
    "parameters": {
        "focal_length": F,
        "principal_point_x": W / 2.0,
        "principal_point_y": H / 2.0,
        **BSPLINE,
    },
}

# The same intrinsics with the spline zeroed: the exact pinhole map.
ZERO_BSPLINE = {
    "model": "SFMTOOL_PINHOLE",
    "width": W,
    "height": H,
    "parameters": dict(
        SFMTOOL["parameters"],
        **{k: 0.0 for k in BSPLINE if _is_coeff(k)},
    ),
}

SIMPLE_PINHOLE = {
    "model": "SIMPLE_PINHOLE",
    "width": W,
    "height": H,
    "parameters": {
        "focal_length": F,
        "principal_point_x": W / 2.0,
        "principal_point_y": H / 2.0,
    },
}


def _ray_at(rho, phi_deg):
    """Canonical-frame unit ray at image-plane radius `rho`, azimuth `phi`."""
    theta = math.atan(rho)
    phi = math.radians(phi_deg)
    return [
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        -math.cos(theta),
    ]


def _rays():
    # Out to rho = 1.5 (theta ~ 56 deg): across the rho_max = 0.9 seam, so
    # both the Newton branch and the held-constant tail are exercised.
    return np.array(
        [
            _ray_at(r, p)
            for r in (0.0, 0.2, 0.5, 0.89, 0.9, 0.95, 1.2, 1.5)
            for p in (0.0, 71.0, 200.0)
        ],
        dtype=np.float64,
    )


def test_from_dict_accepts_the_native_model():
    cam = CameraIntrinsics.from_dict(SFMTOOL)
    assert cam.model == "SFMTOOL_PINHOLE"
    assert cam.width == W and cam.height == H
    assert cam.focal_lengths == (F, F)
    assert cam.principal_point == (W / 2.0, H / 2.0)
    # A live spline counts as distortion; a zeroed one does not.
    assert cam.has_distortion
    assert not CameraIntrinsics.from_dict(ZERO_BSPLINE).has_distortion
    assert set(cam.parameters) == set(SFMTOOL["parameters"])
    # Dict round trip through the generic path, all 8 coefficients.
    assert CameraIntrinsics.from_dict(cam.to_dict()).to_dict() == cam.to_dict()
    assert cam.to_dict()["parameters"] == SFMTOOL["parameters"]


def test_from_dict_rejects_a_gapped_bspline():
    params = dict(SFMTOOL["parameters"])
    del params["bspline_c1"]  # c2..c7 still present: a hole, not a shorter spline
    with pytest.raises(ValueError, match="bspline_c1"):
        CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))


def test_from_dict_rejects_a_missing_rho_max():
    params = dict(SFMTOOL["parameters"])
    del params["bspline_rho_max"]
    with pytest.raises(ValueError, match="bspline_rho_max"):
        CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))


def test_from_dict_rejects_the_fisheye_domain_end_key():
    # The domain end is named per model: the fisheye's key does not stand in
    # for this model's, and the read reports the one it needs.
    params = dict(SFMTOOL["parameters"])
    params["bspline_theta_max"] = params.pop("bspline_rho_max")
    with pytest.raises(ValueError, match="bspline_rho_max"):
        CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))


def test_from_dict_rejects_a_missing_coeff_count():
    # The declared length is required: the coefficient keys are never counted
    # as a fallback.
    params = dict(SFMTOOL["parameters"])
    del params["bspline_coeff_count"]
    with pytest.raises(ValueError, match="bspline_coeff_count"):
        CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))


@pytest.mark.parametrize("bad", [2.5, -1.0, float("nan"), float("inf")])
def test_from_dict_rejects_a_coeff_count_that_is_not_a_count(bad):
    params = dict(SFMTOOL["parameters"], bspline_coeff_count=bad)
    with pytest.raises(ValueError, match="bspline_coeff_count"):
        CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))


def test_from_dict_rejects_a_single_coefficient_spline():
    # A clamped cubic basis needs at least two coefficients. Zero declares the
    # empty spline and stays valid; exactly one is a length the model does not
    # define, and reading it as the identity would hide a corrupt file.
    params = {k: v for k, v in SFMTOOL["parameters"].items() if not _is_coeff(k)}
    params["bspline_coeff_count"] = 1.0
    params["bspline_c0"] = 0.0008
    with pytest.raises(ValueError, match="bspline_coeff_count"):
        CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_from_dict_rejects_a_degenerate_rho_max(bad):
    # The domain end must be a real interval: zero and negative leave the basis
    # nothing to live on, and inf/nan are not a domain at all.
    params = dict(SFMTOOL["parameters"], bspline_rho_max=bad)
    with pytest.raises(ValueError, match="bspline_rho_max"):
        CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))


def test_from_dict_rejects_a_coefficient_beyond_the_declared_length():
    # Eight declared, nine present: the stray key is named rather than read
    # as a ninth coefficient.
    params = dict(SFMTOOL["parameters"], bspline_c8=0.09)
    with pytest.raises(ValueError, match="bspline_c8"):
        CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))


def test_empty_bspline_round_trips_as_a_zero_size():
    params = {k: v for k, v in SFMTOOL["parameters"].items() if not _is_coeff(k)}
    params["bspline_coeff_count"] = 0.0
    cam = CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))
    assert cam.to_dict()["parameters"] == params
    assert not any(_is_coeff(k) for k in cam.parameters)


def test_projection_batches_round_trip_across_the_rho_max_seam():
    cam = CameraIntrinsics.from_dict(SFMTOOL)
    rays = _rays()
    pixels = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(rays)))
    assert np.isfinite(pixels).all(), "an in-front ray fell out of the domain"
    # The expanding spline pushes the periphery OUT vs the pinhole map.
    cx, cy = cam.principal_point
    radii = np.hypot(pixels[:, 0] - cx, pixels[:, 1] - cy)
    rho = np.hypot(rays[:, 0], rays[:, 1]) / -rays[:, 2]
    assert (radii >= F * rho - 1e-12).all()
    assert radii[rho > 0.5].max() > (F * rho[rho > 0.5]).max() + 1.0
    # And exactly back to the same rays through the Newton inverse.
    back = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(pixels)))
    np.testing.assert_allclose(back, rays, atol=1e-12)
    # Scalar and batch paths agree.
    u, v = cam.ray_to_pixel(rays[5])
    assert (u, v) == pytest.approx((pixels[5, 0], pixels[5, 1]), abs=0.0)


def test_behind_the_camera_has_no_projection():
    cam = CameraIntrinsics.from_dict(SFMTOOL)
    assert cam.ray_to_pixel([0.0, 0.0, 1.0]) is None
    assert cam.ray_to_pixel([0.3, -0.2, 0.5]) is None


def test_zero_bspline_equals_simple_pinhole_exactly():
    """Empty and all-zero splines short-circuit to the pinhole kernels.

    Exact equality (atol=0), not approximate: the promotion contract is that
    wrapping a solved pinhole camera in this model moves nothing.
    """
    native = CameraIntrinsics.from_dict(SIMPLE_PINHOLE)
    no_coeffs = {k: v for k, v in SFMTOOL["parameters"].items() if not _is_coeff(k)}
    no_coeffs["bspline_coeff_count"] = 0.0
    for spec in (ZERO_BSPLINE, dict(SFMTOOL, parameters=no_coeffs)):
        cam = CameraIntrinsics.from_dict(spec)
        rays = _rays()
        px_s = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(rays)))
        px_n = np.asarray(native.ray_to_pixel_batch(np.ascontiguousarray(rays)))
        np.testing.assert_array_equal(px_s, px_n)
        np.testing.assert_array_equal(
            np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(px_s))),
            np.asarray(native.pixel_to_ray_batch(np.ascontiguousarray(px_n))),
        )
        # project/unproject (the image-plane pair) as well.
        pts = np.ascontiguousarray(
            np.array([[0.0, 0.0], [0.3, -0.2], [-0.4, 0.3]], dtype=np.float64)
        )
        np.testing.assert_array_equal(
            np.asarray(cam.project_batch(pts)), np.asarray(native.project_batch(pts))
        )


def test_copy_and_dict_serialization_round_trip():
    """Copy and the dict path — the supported serialization contract.

    PyO3 camera objects are not picklable (binding-wide: the registered
    class path is not importable, so every model fails the same way);
    `to_dict`/`from_dict` is the contract for crossing process boundaries,
    and it must carry the variable-length spline losslessly.
    """
    cam = CameraIntrinsics.from_dict(SFMTOOL)
    assert copy.copy(cam) == cam
    assert copy.deepcopy(cam) == cam
    assert CameraIntrinsics.from_dict(cam.to_dict()) == cam
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(cam)


def test_best_fit_inside_pinhole_is_supported():
    # A perspective model, so the pinhole fits are defined for it (its fisheye
    # sibling raises) and run through the spline's forward and inverse maps.
    cam = CameraIntrinsics.from_dict(SFMTOOL)
    fit = cam.best_fit_inside_pinhole(W, H)
    assert fit.model == "PINHOLE"
    assert fit.principal_point == (W / 2.0, H / 2.0)
    fx, fy = fit.focal_lengths
    assert fx == fy and math.isfinite(fx) and fx > 0.0


# ── pycolmap interop: no carrier, hard error ────────────────────────────────


def test_export_to_pycolmap_is_rejected():
    from sfmtool.camera.cameras import colmap_camera_from_intrinsics

    cam = CameraIntrinsics.from_dict(SFMTOOL)
    with pytest.raises(ValueError, match="no COLMAP representation"):
        colmap_camera_from_intrinsics(cam)
    # The zero-spline variant is still this model, not its pinhole
    # equivalent: the rejection is by model, not by coefficient values.
    with pytest.raises(ValueError, match="no COLMAP representation"):
        colmap_camera_from_intrinsics(CameraIntrinsics.from_dict(ZERO_BSPLINE))


def test_model_is_deliberately_not_user_specifiable():
    from sfmtool.camera.cameras import CAMERA_MODEL_NAMES, _CAMERA_PARAM_NAMES

    assert "SFMTOOL_PINHOLE" not in CAMERA_MODEL_NAMES
    assert "SFMTOOL_PINHOLE" not in _CAMERA_PARAM_NAMES
