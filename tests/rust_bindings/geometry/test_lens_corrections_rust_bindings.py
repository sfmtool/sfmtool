# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-image radial corrections and rolling-shutter rates through the Python surface.

``CameraIntrinsics.with_adjustment`` applies an image's radial correction and
its rolling-shutter rates to an ``SFMTOOL_PINHOLE`` camera, and
``SfmrReconstruction`` reads each image through them. See
``specs/core/camera/sfmtool-pinhole-kernels.md`` § "The rolling-shutter step"
and ``specs/formats/sfmtool-camera-models.md`` § "Per-image radial correction".
"""

import pickle

import numpy as np
import pytest

from sfmtool.fileio import read_sfmr
from sfmtool.geometry import CameraIntrinsics
from sfmtool.reconstruction import SfmrReconstruction

RATES = (2e-2, -1.5e-2, 1e-2)


def _pinhole(coefficients=(0.002, 0.007, 0.014, 0.024)):
    params = {
        "focal_length": 500.0,
        "principal_point_x": 330.0,
        "principal_point_y": 235.0,
        "bspline_rho_max": 0.8,
        "bspline_coeff_count": float(len(coefficients)),
    }
    params.update({f"bspline_c{i}": c for i, c in enumerate(coefficients)})
    return CameraIntrinsics("SFMTOOL_PINHOLE", 640, 480, params)


def _coefficients(camera):
    p = camera.parameters
    return [p[f"bspline_c{i}"] for i in range(int(p["bspline_coeff_count"]))]


class TestWithAdjustment:
    def test_a_two_coefficient_correction_adds_to_a_two_coefficient_spline(self):
        camera = _pinhole((0.01, 0.02))
        image = camera.with_adjustment(radial_bspline=(0.003, -0.004))
        assert _coefficients(image) == pytest.approx([0.013, 0.016], abs=1e-15)
        assert image.rolling_shutter is None

    def test_the_correction_changes_the_projection_by_its_closed_form(self):
        camera = _pinhole()
        e = (0.01, -0.02)
        image = camera.with_adjustment(radial_bspline=e)
        rho_max = 0.8
        for rho in (0.1, 0.4, 0.75):
            # A point on the +x axis: its distorted radius moves by
            # e0*3t^2(1 - t) + e1*t^3, t = rho/rho_max, times the focal.
            t = rho / rho_max
            delta = e[0] * 3 * t * t * (1 - t) + e[1] * t**3
            u0, _ = camera.project(rho, 0.0)
            u1, _ = image.project(rho, 0.0)
            assert u1 - u0 == pytest.approx(500.0 * delta, abs=1e-9)

    def test_rates_set_the_rolling_shutter_and_round_trip(self):
        camera = _pinhole()
        for scanlines in ("ROWS", "COLUMNS"):
            image = camera.with_adjustment(
                rolling_shutter=RATES, rolling_shutter_scanlines=scanlines
            )
            assert image.rolling_shutter == (scanlines, RATES)
            assert image != camera
            assert "rolling_shutter" in repr(image)
            for u, v in ((0.0, 0.0), (320.0, 240.0), (639.0, 10.0), (17.0, 470.0)):
                ray = image.pixel_to_ray(u, v)
                back = image.ray_to_pixel(ray)
                assert back == pytest.approx((u, v), abs=1e-9)
            # The rates move the image's edge.
            ray = camera.pixel_to_ray(600.0, 450.0)
            assert (
                np.hypot(
                    *np.subtract(image.ray_to_pixel(ray), camera.ray_to_pixel(ray))
                )
                > 1.0
            )

    def test_a_camera_that_cannot_take_the_adjustment_is_refused(self):
        simple = CameraIntrinsics(
            "SIMPLE_PINHOLE",
            640,
            480,
            {
                "focal_length": 500.0,
                "principal_point_x": 320.0,
                "principal_point_y": 240.0,
            },
        )
        with pytest.raises(ValueError, match="SFMTOOL_PINHOLE"):
            simple.with_adjustment(radial_bspline=(0.01, 0.0))
        with pytest.raises(ValueError, match="SFMTOOL_PINHOLE"):
            simple.with_adjustment(
                rolling_shutter=RATES, rolling_shutter_scanlines="ROWS"
            )
        with pytest.raises(ValueError, match="scanlines"):
            _pinhole().with_adjustment(rolling_shutter=RATES)
        with pytest.raises(ValueError, match="ROWS"):
            _pinhole().with_adjustment(
                rolling_shutter=RATES, rolling_shutter_scanlines="X"
            )

    def test_a_value_that_is_not_finite_is_refused(self):
        # As from_dict and __setstate__ refuse a rate that is not finite,
        # rather than handing back a camera that projects to NaN.
        nan, inf = float("nan"), float("inf")
        for rates in ((nan, 0.0, 0.0), (0.0, inf, 0.0), (1e-3, 0.0, -inf)):
            with pytest.raises(ValueError, match="not finite"):
                _pinhole().with_adjustment(
                    rolling_shutter=rates, rolling_shutter_scanlines="ROWS"
                )
        with pytest.raises(ValueError, match="not finite"):
            _pinhole().with_adjustment(radial_bspline=(nan, 0.0))

    def test_the_rolling_shutter_survives_dict_and_pickle(self):
        image = _pinhole().with_adjustment(
            rolling_shutter=RATES, rolling_shutter_scanlines="COLUMNS"
        )
        d = image.to_dict()
        assert d["rolling_shutter"] == ("COLUMNS", RATES)
        assert CameraIntrinsics.from_dict(d) == image
        assert pickle.loads(pickle.dumps(image)) == image
        assert "rolling_shutter" not in _pinhole().to_dict()
        assert pickle.loads(pickle.dumps(_pinhole())) == _pinhole()


def _corrected(path, workspace_dir):
    """The reconstruction at ``path`` with its cameras an ``SFMTOOL_PINHOLE``
    refit, image 2 carrying a radial correction and image 3 rates on rows."""
    data = read_sfmr(path)
    data["cameras"] = [c.refit("SFMTOOL_PINHOLE")[0] for c in data["cameras"]]
    n = len(data["image_names"])
    radial = np.zeros((n, 2))
    radial[2] = (0.004, -0.006)
    rates = np.zeros((n, 3))
    rates[3] = RATES
    data["per_image_adjustments"] = [
        ["radial_bspline", "rolling_shutter"] for _ in data["cameras"]
    ]
    data["rolling_shutter_scanlines"] = ["ROWS" for _ in data["cameras"]]
    data["radial_bspline_adjustments"] = radial
    data["rolling_shutter_rates_xyz"] = rates
    return SfmrReconstruction.from_data(workspace_dir, data), radial, rates


class TestTheReconstruction:
    def test_getters_and_each_images_intrinsics(self, seoul_bull_sfmr_only, tmp_path):
        plain = SfmrReconstruction.load(seoul_bull_sfmr_only)
        assert plain.radial_bspline_adjustments is None
        assert plain.rolling_shutter_rates_xyz is None
        assert plain.rolling_shutter_scanlines == [None for _ in plain.cameras]

        recon, radial, rates = _corrected(
            seoul_bull_sfmr_only, seoul_bull_sfmr_only.parent
        )
        np.testing.assert_array_equal(recon.radial_bspline_adjustments, radial)
        np.testing.assert_array_equal(recon.rolling_shutter_rates_xyz, rates)
        assert recon.rolling_shutter_scanlines == ["ROWS" for _ in recon.cameras]
        assert recon.per_image_adjustments == [
            ["radial_bspline", "rolling_shutter"] for _ in recon.cameras
        ]
        camera = recon.cameras[recon.camera_indexes[2]]
        assert recon.image_intrinsics(2) == camera.with_adjustment(
            radial_bspline=tuple(radial[2])
        )
        image = recon.image_intrinsics(3)
        assert image.rolling_shutter == ("ROWS", RATES)
        assert recon.image_intrinsics(0) == recon.cameras[recon.camera_indexes[0]]

        # A file keeps them, and a load reads each image through them again.
        out = tmp_path / "corrected.sfmr"
        recon.save(out)
        loaded = SfmrReconstruction.load(out)
        np.testing.assert_array_equal(loaded.radial_bspline_adjustments, radial)
        np.testing.assert_array_equal(loaded.rolling_shutter_rates_xyz, rates)
        assert loaded.rolling_shutter_scanlines == recon.rolling_shutter_scanlines
        for i in range(recon.image_count):
            assert loaded.image_intrinsics(i) == recon.image_intrinsics(i)

    def test_a_camera_list_refuses_a_rolling_shutter(self, seoul_bull_sfmr_only):
        plain = SfmrReconstruction.load(seoul_bull_sfmr_only)
        pinhole = plain.cameras[0].refit("SFMTOOL_PINHOLE")[0]
        image = pinhole.with_adjustment(
            rolling_shutter=RATES, rolling_shutter_scanlines="ROWS"
        )
        with pytest.raises(ValueError, match="rolling shutter"):
            plain.clone_with_changes(cameras=[image for _ in plain.cameras])


class TestNoInputReachesAPanic:
    """Every path that sets a rolling shutter, pairs an image with a camera or
    changes a scanlines setting is checked, and raises ``ValueError`` rather
    than letting a projection panic."""

    def test_setstate_refuses_rates_on_another_model(self):
        cam = CameraIntrinsics(
            "PINHOLE",
            640,
            480,
            {
                "focal_length_x": 500.0,
                "focal_length_y": 500.0,
                "principal_point_x": 320.0,
                "principal_point_y": 240.0,
            },
        )
        with pytest.raises(ValueError, match="SFMTOOL_PINHOLE"):
            cam.__setstate__(("ROWS", (1e-3, 0.0, 0.0)))
        assert cam.rolling_shutter is None
        cam.project(0.1, 0.1)
        # Zero rates on any model are no rolling shutter to apply.
        cam.__setstate__(("ROWS", (0.0, 0.0, 0.0)))
        with pytest.raises(ValueError, match="not finite"):
            _pinhole().__setstate__(("ROWS", (float("nan"), 0.0, 0.0)))
        d = cam.to_dict()
        d["rolling_shutter"] = ("COLUMNS", (1e-3, 0.0, 0.0))
        with pytest.raises(ValueError, match="SFMTOOL_PINHOLE"):
            CameraIntrinsics.from_dict(d)

    def test_clone_with_changes_checks_every_image_against_its_camera(
        self, seoul_bull_sfmr_only
    ):
        recon, _, _ = _corrected(seoul_bull_sfmr_only, seoul_bull_sfmr_only.parent)
        plain = SfmrReconstruction.load(seoul_bull_sfmr_only)
        # The corrected images' cameras replaced by the plain model.
        with pytest.raises(ValueError, match="image 2"):
            recon.clone_with_changes(cameras=list(plain.cameras))
        # An image moved onto a new camera with no scanlines setting.
        cameras = list(recon.cameras)
        indexes = np.asarray(recon.camera_indexes).copy()
        indexes[3] = len(cameras)
        with pytest.raises(ValueError, match="image 3"):
            recon.clone_with_changes(
                cameras=cameras + [cameras[0]], camera_indexes=indexes
            )
        # The reconstruction itself is untouched and still reads every image.
        for i in range(recon.image_count):
            recon.image_intrinsics(i)

    def test_an_empty_flag_list_keeps_the_scanlines(self, seoul_bull_sfmr_only):
        recon, _, _ = _corrected(seoul_bull_sfmr_only, seoul_bull_sfmr_only.parent)
        cleared = recon.clone_with_changes(per_image_adjustments=[])
        assert cleared.per_image_adjustments == [[] for _ in recon.cameras]
        assert cleared.rolling_shutter_scanlines == recon.rolling_shutter_scanlines
        assert cleared.image_intrinsics(3) == recon.image_intrinsics(3)
