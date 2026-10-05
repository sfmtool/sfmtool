# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The EQUIDISTANT_FISHEYE camera model through the Python surface.

`EQUIDISTANT_FISHEYE` is an sfmtool-native (non-COLMAP) model: the
distortion-free equidistant map ``theta = r / f`` with SIMPLE_PINHOLE's
parameter list. These tests cover the three boundaries it crosses in Python —
the generic `CameraIntrinsics.from_dict` path, `.sfmr` storage, and the
pycolmap interop shim that carries it as SIMPLE_RADIAL_FISHEYE with ``k = 0``.
"""

import math

import numpy as np
import pytest

from sfmtool._sfmtool.geometry import CameraIntrinsics
from sfmtool._sfmtool.reconstruction import SfmrReconstruction

F = 130.0
W = H = 480

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

# The pre-Phase-3a convention for the same map.
K1_ZERO = {
    "model": "SIMPLE_RADIAL_FISHEYE",
    "width": W,
    "height": H,
    "parameters": dict(EQUIDISTANT["parameters"], radial_distortion_k1=0.0),
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


def test_from_dict_accepts_the_native_model():
    cam = CameraIntrinsics.from_dict(EQUIDISTANT)
    assert cam.model == "EQUIDISTANT_FISHEYE"
    assert cam.width == W and cam.height == H
    assert cam.focal_lengths == (F, F)
    assert cam.principal_point == (W / 2.0, H / 2.0)
    # No distortion coefficients at all — the map is the pure theta = r/f.
    assert not cam.has_distortion
    assert set(cam.parameters) == {
        "focal_length",
        "principal_point_x",
        "principal_point_y",
    }
    # Dict round trip through the generic path.
    assert CameraIntrinsics.from_dict(cam.to_dict()).to_dict() == cam.to_dict()


def test_from_dict_rejects_a_missing_parameter():
    bad = dict(EQUIDISTANT, parameters={"principal_point_x": 0.0})
    with pytest.raises(ValueError, match="focal_length"):
        CameraIntrinsics.from_dict(bad)


def test_projection_is_exactly_theta_over_f_past_ninety_degrees():
    cam = CameraIntrinsics.from_dict(EQUIDISTANT)
    cx, cy = cam.principal_point
    rays = np.array(
        [
            _ray_at(d, p)
            for d in (0.0, 30.0, 89.0, 90.0, 91.0, 105.0, 130.0)
            for p in (0.0, 137.0)
        ],
        dtype=np.float64,
    )
    pixels = np.asarray(cam.ray_to_pixel_batch(np.ascontiguousarray(rays)))
    assert np.isfinite(pixels).all(), "a ray past 90 deg fell out of the domain"
    radii = np.hypot(pixels[:, 0] - cx, pixels[:, 1] - cy)
    thetas = np.arccos(np.clip(-rays[:, 2], -1.0, 1.0))
    np.testing.assert_allclose(radii, F * thetas, atol=1e-9)
    # And exactly back to the same rays.
    back = np.asarray(cam.pixel_to_ray_batch(np.ascontiguousarray(pixels)))
    np.testing.assert_allclose(back, rays, atol=1e-12)


def test_matches_the_simple_radial_fisheye_k1_zero_convention():
    """The two representations parameterize the identical map.

    They agree bit-for-bit outside the polynomial family's 90-100 deg blend
    band, where SIMPLE_RADIAL_FISHEYE lerps two identical rays and
    renormalizes; 1e-12 covers that round-off.
    """
    native = CameraIntrinsics.from_dict(EQUIDISTANT)
    legacy = CameraIntrinsics.from_dict(K1_ZERO)
    rays = np.array(
        [
            _ray_at(d, p)
            for d in (0.0, 45.0, 89.0, 95.0, 105.0, 130.0)
            for p in (0.0, 71.0, 200.0)
        ],
        dtype=np.float64,
    )
    px_n = np.asarray(native.ray_to_pixel_batch(np.ascontiguousarray(rays)))
    px_l = np.asarray(legacy.ray_to_pixel_batch(np.ascontiguousarray(rays)))
    np.testing.assert_allclose(px_n, px_l, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(native.pixel_to_ray_batch(np.ascontiguousarray(px_n))),
        np.asarray(legacy.pixel_to_ray_batch(np.ascontiguousarray(px_l))),
        atol=1e-12,
    )


def test_sfmr_storage_round_trip(seoul_bull_sfmr_only, tmp_path):
    """`.sfmr` stores the native model name verbatim, like EQUIRECTANGULAR."""
    recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    cam = CameraIntrinsics.from_dict(
        dict(
            EQUIDISTANT,
            width=recon.cameras[0].width,
            height=recon.cameras[0].height,
        )
    )
    out = tmp_path / "equidistant.sfmr"
    recon.clone_with_changes(cameras=[cam] * len(recon.cameras)).save(out)

    loaded = SfmrReconstruction.load(out)
    assert [c.model for c in loaded.cameras] == ["EQUIDISTANT_FISHEYE"] * len(
        recon.cameras
    )
    assert loaded.cameras[0].to_dict()["parameters"] == cam.to_dict()["parameters"]


# ── pycolmap interop ────────────────────────────────────────────────────────


def test_export_to_pycolmap_uses_the_zero_k_carrier():
    from sfmtool.camera.cameras import colmap_camera_from_intrinsics

    cam = colmap_camera_from_intrinsics(CameraIntrinsics.from_dict(EQUIDISTANT))
    assert cam.model.name == "SIMPLE_RADIAL_FISHEYE"
    assert cam.width == W and cam.height == H
    np.testing.assert_array_equal(cam.params, [F, W / 2.0, H / 2.0, 0.0])


def test_import_claims_simple_radial_fisheye_only_at_exactly_zero_k():
    import pycolmap

    from sfmtool.camera.cameras import pycolmap_camera_to_intrinsics

    zero = pycolmap.Camera(
        model="SIMPLE_RADIAL_FISHEYE",
        width=W,
        height=H,
        params=[F, W / 2.0, H / 2.0, 0.0],
    )
    # Claiming is opt-in: the default keeps the caller's requested model, which
    # is what a freshly-initialized camera for a COLMAP solve needs.
    assert pycolmap_camera_to_intrinsics(zero).model == "SIMPLE_RADIAL_FISHEYE"

    claimed = pycolmap_camera_to_intrinsics(zero, claim_native=True)
    assert claimed.model == "EQUIDISTANT_FISHEYE"
    assert "radial_distortion_k1" not in claimed.parameters
    assert claimed.focal_lengths == (F, F)

    # Any nonzero k, however small, stays SIMPLE_RADIAL_FISHEYE.
    for k in (1e-12, -0.017):
        cam = pycolmap.Camera(
            model="SIMPLE_RADIAL_FISHEYE",
            width=W,
            height=H,
            params=[F, W / 2.0, H / 2.0, k],
        )
        out = pycolmap_camera_to_intrinsics(cam, claim_native=True)
        assert out.model == "SIMPLE_RADIAL_FISHEYE"
        assert out.parameters["radial_distortion_k1"] == pytest.approx(k)


def test_pycolmap_round_trip_is_lossless():
    from sfmtool.camera.cameras import (
        colmap_camera_from_intrinsics,
        pycolmap_camera_to_intrinsics,
    )

    original = CameraIntrinsics.from_dict(EQUIDISTANT)
    back = pycolmap_camera_to_intrinsics(
        colmap_camera_from_intrinsics(original), claim_native=True
    )
    assert back.to_dict() == original.to_dict()


class TestPixelRadiusToWorldBatch:
    """`pixel_radius_to_world_batch` — the position-anchored sizing rule.

    One rule for every model (`radius_px / sigma_min`), with two exact closed
    forms: `radius_px * |z| / f` for a pinhole and `radius_px * ||p_cam|| / f`
    for the equidistant map. The pair is the whole point — the two readings
    differ by `cos(theta)`, so a sizing rule that used the range for a pinhole
    would oversize every off-axis patch by `sec(theta)`.
    """

    PINHOLE = {
        "model": "SIMPLE_PINHOLE",
        "width": W,
        "height": H,
        "parameters": {
            "focal_length": F,
            "principal_point_x": W / 2.0,
            "principal_point_y": H / 2.0,
        },
    }

    @staticmethod
    def _points(theta_deg, rng=4.0):
        th = np.radians(np.atleast_1d(np.asarray(theta_deg, dtype=np.float64)))
        # Canonical frame: -Z is forward.
        return np.ascontiguousarray(
            np.stack([rng * np.sin(th), np.zeros_like(th), -rng * np.cos(th)], axis=1)
        )

    def test_equidistant_sizes_by_the_ray_range(self):
        cam = CameraIntrinsics.from_dict(EQUIDISTANT)
        rng, radius = 4.0, 6.0
        pts = self._points([0.0, 45.0, 100.0, 130.0], rng)
        got = np.asarray(
            cam.pixel_radius_to_world_batch(pts, np.array([radius], np.float64))
        )
        assert got == pytest.approx(radius * rng / F, rel=1e-12)

    def test_pinhole_sizes_by_the_optical_axis_depth(self):
        cam = CameraIntrinsics.from_dict(self.PINHOLE)
        rng, radius = 4.0, 6.0
        thetas = np.array([0.0, 30.0, 55.0])
        pts = self._points(thetas, rng)
        got = np.asarray(
            cam.pixel_radius_to_world_batch(pts, np.array([radius], np.float64))
        )
        want = radius * rng * np.cos(np.radians(thetas)) / F
        assert got == pytest.approx(want, rel=1e-12)
        # The equidistant (range) reading would be sec(theta) larger off axis.
        assert got[-1] < 0.6 * radius * rng / F

    def test_per_point_radii_and_scalar_broadcast_agree(self):
        cam = CameraIntrinsics.from_dict(EQUIDISTANT)
        pts = self._points([10.0, 60.0, 110.0])
        scalar = np.asarray(
            cam.pixel_radius_to_world_batch(pts, np.array([3.0], np.float64))
        )
        vector = np.asarray(
            cam.pixel_radius_to_world_batch(pts, np.full(3, 3.0, np.float64))
        )
        assert scalar == pytest.approx(vector, rel=0, abs=0)
        # Linear in the pixel budget.
        doubled = np.asarray(
            cam.pixel_radius_to_world_batch(pts, np.full(3, 6.0, np.float64))
        )
        assert doubled == pytest.approx(2.0 * scalar, rel=1e-12)

    def test_shape_and_length_errors(self):
        cam = CameraIntrinsics.from_dict(EQUIDISTANT)
        with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
            cam.pixel_radius_to_world_batch(
                np.zeros((2, 2)), np.array([1.0], np.float64)
            )
        with pytest.raises(ValueError, match="scalar or have length 3"):
            cam.pixel_radius_to_world_batch(
                self._points([0.0, 10.0, 20.0]), np.array([1.0, 2.0], np.float64)
            )

    def test_agrees_with_the_angular_sibling_through_the_range(self):
        # `pixel_radius_to_angle` is the same rule with the range factored out,
        # so `world = angle * range` at every theta for both families.
        rng, radius = 5.0, 4.0
        pts = self._points([5.0, 40.0, 85.0], rng)
        for spec in (EQUIDISTANT, self.PINHOLE):
            cam = CameraIntrinsics.from_dict(spec)
            world = np.asarray(
                cam.pixel_radius_to_world_batch(pts, np.array([radius], np.float64))
            )
            angle = np.asarray(
                cam.pixel_radius_to_angle_batch(pts, np.array([radius], np.float64))
            )
            assert world == pytest.approx(angle * rng, rel=1e-12)
