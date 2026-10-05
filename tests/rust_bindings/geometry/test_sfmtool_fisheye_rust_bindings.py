# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The SFMTOOL_FISHEYE camera model through the Python surface.

``SFMTOOL_FISHEYE`` is an sfmtool-native (non-COLMAP) model: an equidistant
base plus a monotone radial spline, ``r(theta) = f * (theta + delta(theta))``
with ``bspline_coeff_count`` dimensionless coefficients
``bspline_c0..cN`` on ``[0, bspline_theta_max]``. These tests cover the
boundaries it crosses in Python — the generic `CameraIntrinsics.from_dict`
path (including its parameter-map validation), projection batch
consistency, the exact zero-spline identity with EQUIDISTANT_FISHEYE,
pickling, and the pycolmap interop shim, which (unlike EQUIDISTANT_FISHEYE)
has NO carrier for it and must reject loudly.
"""

import copy
import math
import pickle

import numpy as np
import pytest

from sfmtool._sfmtool.geometry import CameraIntrinsics

F = 130.0
W = H = 480


def _is_coeff(key):
    """True for a coefficient key ``bspline_c<i>``, not ``bspline_coeff_count``."""
    return key.startswith("bspline_c") and key.removeprefix("bspline_c").isdigit()


BSPLINE = {
    "bspline_theta_max": 2.0,
    "bspline_coeff_count": 8.0,
    "bspline_c0": -0.001,
    "bspline_c1": -0.004,
    "bspline_c2": -0.01,
    "bspline_c3": -0.02,
    "bspline_c4": -0.03,
    "bspline_c5": -0.05,
    "bspline_c6": -0.07,
    "bspline_c7": -0.09,
}

SFMTOOL = {
    "model": "SFMTOOL_FISHEYE",
    "width": W,
    "height": H,
    "parameters": {
        "focal_length": F,
        "principal_point_x": W / 2.0,
        "principal_point_y": H / 2.0,
        **BSPLINE,
    },
}

# The same intrinsics with the spline zeroed: the exact equidistant map.
ZERO_BSPLINE = {
    "model": "SFMTOOL_FISHEYE",
    "width": W,
    "height": H,
    "parameters": dict(
        SFMTOOL["parameters"],
        **{k: 0.0 for k in BSPLINE if _is_coeff(k)},
    ),
}

EQUIDISTANT = {
    "model": "EQUIDISTANT_FISHEYE",
    "width": W,
    "height": H,
    "parameters": {
        "focal_length": F,
        "principal_point_x": W / 2.0,
        "principal_point_y": H / 2.0,
    },
}


def _ray_at(theta_deg, phi_deg):
    """Canonical-frame unit ray at incidence `theta` off -Z, azimuth `phi`."""
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    return [
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        -math.cos(theta),
    ]


def _rays():
    return np.array(
        [
            _ray_at(d, p)
            for d in (0.0, 30.0, 89.0, 90.0, 91.0, 105.0, 120.0)
            for p in (0.0, 71.0, 200.0)
        ],
        dtype=np.float64,
    )


def test_from_dict_accepts_the_native_model():
    cam = CameraIntrinsics.from_dict(SFMTOOL)
    assert cam.model == "SFMTOOL_FISHEYE"
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


def test_from_dict_rejects_a_missing_theta_max():
    params = dict(SFMTOOL["parameters"])
    del params["bspline_theta_max"]
    with pytest.raises(ValueError, match="bspline_theta_max"):
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
    params["bspline_c0"] = -0.001
    with pytest.raises(ValueError, match="bspline_coeff_count"):
        CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_from_dict_rejects_a_degenerate_theta_max(bad):
    # The domain end must be a real interval: zero and negative leave the basis
    # nothing to live on, and inf/nan are not a domain at all.
    params = dict(SFMTOOL["parameters"], bspline_theta_max=bad)
    with pytest.raises(ValueError, match="bspline_theta_max"):
        CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))


def test_from_dict_rejects_a_coefficient_beyond_the_declared_length():
    # Eight declared, nine present: the stray key is named rather than read
    # as a ninth coefficient.
    params = dict(SFMTOOL["parameters"], bspline_c8=-0.11)
    with pytest.raises(ValueError, match="bspline_c8"):
        CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))


def test_empty_bspline_round_trips_as_a_zero_size():
    params = {k: v for k, v in SFMTOOL["parameters"].items() if not _is_coeff(k)}
    params["bspline_coeff_count"] = 0.0
    cam = CameraIntrinsics.from_dict(dict(SFMTOOL, parameters=params))
    assert cam.to_dict()["parameters"] == params
    assert not any(_is_coeff(k) for k in cam.parameters)


def test_projection_batches_round_trip_past_ninety_degrees():
    cam = CameraIntrinsics.from_dict(SFMTOOL)
    rays = _rays()
    pixels = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(rays)))
    assert np.isfinite(pixels).all(), "a ray past 90 deg fell out of the domain"
    # The flattening spline pulls the periphery IN vs the equidistant map.
    cx, cy = cam.principal_point
    radii = np.hypot(pixels[:, 0] - cx, pixels[:, 1] - cy)
    thetas = np.arccos(np.clip(-rays[:, 2], -1.0, 1.0))
    assert (radii <= F * thetas + 1e-12).all()
    assert radii[thetas > 1.5].max() < (F * thetas[thetas > 1.5]).max() - 1.0
    # And exactly back to the same rays through the Newton inverse.
    back = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(pixels)))
    np.testing.assert_allclose(back, rays, atol=1e-12)
    # Scalar and batch paths agree.
    u, v = cam.ray_to_pixel(rays[5])
    assert (u, v) == pytest.approx((pixels[5, 0], pixels[5, 1]), abs=0.0)


def test_the_radial_map_continues_linearly_past_theta_max():
    """Past ``bspline_theta_max`` the correction follows its end tangent.

    The map is then exactly affine in theta with slope
    ``f * (1 + delta'(theta_max))``, so equally spaced angles give equally
    spaced radii -- and that slope is the spline's own, not the base
    equidistant ``f``.
    """
    cam = CameraIntrinsics.from_dict(SFMTOOL)
    theta_max_deg = math.degrees(BSPLINE["bspline_theta_max"])
    step = 8.0
    thetas = np.array([theta_max_deg + step * (i + 1) for i in range(4)])
    rays = np.array([_ray_at(t, 37.0) for t in thetas], dtype=np.float64)
    pixels = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(rays)))
    cx, cy = cam.principal_point
    radii = np.hypot(pixels[:, 0] - cx, pixels[:, 1] - cy)
    # Affine in theta: the second differences vanish.
    np.testing.assert_allclose(np.diff(radii, n=2), 0.0, atol=1e-9)
    # The flattening spline ends on a negative slope, so the tail keeps rising
    # more slowly than the equidistant base instead of resuming its rate.
    slope = np.diff(radii)[0] / math.radians(step)
    assert 0.0 < slope < F - 1.0
    # And the tail inverts back exactly.
    back = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(pixels)))
    np.testing.assert_allclose(back, rays, atol=1e-12)


def test_zero_bspline_equals_equidistant_fisheye_exactly():
    """Empty and all-zero splines short-circuit to the equidistant kernels.

    Exact equality (atol=0), not approximate: the promotion contract is that
    wrapping a solved equidistant camera in this model moves nothing.
    """
    native = CameraIntrinsics.from_dict(EQUIDISTANT)
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
        # project/unproject (the tangent-plane pair) as well, below 90 deg.
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


def test_best_fit_inside_pinhole_raises():
    cam = CameraIntrinsics.from_dict(SFMTOOL)
    with pytest.raises(ValueError, match="SFMTOOL_FISHEYE"):
        cam.best_fit_inside_pinhole(W, H)


# ── pycolmap interop: no carrier, hard error ────────────────────────────────


def test_export_to_pycolmap_is_rejected():
    from sfmtool.camera.cameras import colmap_camera_from_intrinsics

    cam = CameraIntrinsics.from_dict(SFMTOOL)
    with pytest.raises(ValueError, match="no COLMAP representation"):
        colmap_camera_from_intrinsics(cam)
    # The zero-spline variant is still this model, not its equidistant
    # equivalent: the rejection is by model, not by coefficient values.
    with pytest.raises(ValueError, match="no COLMAP representation"):
        colmap_camera_from_intrinsics(CameraIntrinsics.from_dict(ZERO_BSPLINE))


def test_model_is_deliberately_not_user_specifiable():
    from sfmtool.camera.cameras import CAMERA_MODEL_NAMES, _CAMERA_PARAM_NAMES

    assert "SFMTOOL_FISHEYE" not in CAMERA_MODEL_NAMES
    assert "SFMTOOL_FISHEYE" not in _CAMERA_PARAM_NAMES
