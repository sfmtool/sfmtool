# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The per-image releases of ``sfm xform --bundle-adjust``: the
``image_focal``, ``image_principal_point``, ``image_radial_bspline``,
``image_rolling_shutter`` and ``scanlines`` keys, their routing to sfmtool's
adjustment, and the lines the run prints. See
``specs/cli/reconstruction/xform/xform-command.md`` § "--bundle-adjust".
"""

import click
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import BundleAdjustTransform
from sfmtool.xform._arg_parser import parse_transform_args
from sfmtool.xform._bundle_adjust import format_image_prior_widths, format_scanlines


def _pinhole(path):
    """The reconstruction at ``path`` with its camera switched to
    ``SFMTOOL_PINHOLE``, with no scanlines setting."""
    recon = SfmrReconstruction.load(path)
    switched, _ = recon.switch_camera_model("SFMTOOL_PINHOLE", coeff_count=4)
    return switched


def _simple_pinhole(path):
    recon = SfmrReconstruction.load(path)
    switched, _ = recon.switch_camera_model("SIMPLE_PINHOLE")
    return switched


class TestTheKeys:
    def test_each_key_reaches_the_transform(self):
        (ba,) = parse_transform_args(
            [
                "--bundle-adjust",
                "image_focal=on,image_principal_point=true,"
                "image_radial_bspline=1,image_rolling_shutter=yes,scanlines=Columns",
            ]
        )
        assert ba.image_focal and ba.image_principal_point
        assert ba.image_radial_bspline and ba.image_rolling_shutter
        assert ba.scanlines == "COLUMNS"
        assert ba.description() == (
            "Bundle adjustment (release: focal,distortion, image_focal=on, "
            "image_principal_point=on, image_radial_bspline=on, "
            "image_rolling_shutter=on, scanlines=columns)"
        )
        (unknown,) = parse_transform_args(
            ["--bundle-adjust", "image_rolling_shutter=on,scanlines=UNKNOWN"]
        )
        assert unknown.scanlines == "UNKNOWN"

    def test_the_bare_option_releases_nothing_per_image(self):
        (bare,) = parse_transform_args(["--bundle-adjust"])
        assert not bare.releases_per_image
        assert bare.scanlines is None
        assert bare.description() == "Bundle adjustment (release: focal,distortion)"

    def test_bad_values_are_refused_at_parse_time(self):
        with pytest.raises(click.UsageError, match="is not rows, columns or unknown"):
            parse_transform_args(
                ["--bundle-adjust", "image_rolling_shutter=on,scanlines=sideways"]
            )
        with pytest.raises(click.UsageError, match="only with image_rolling_shutter"):
            parse_transform_args(["--bundle-adjust", "scanlines=rows"])
        # The error names the key and the values it takes, not a function.
        with pytest.raises(click.UsageError) as refused:
            parse_transform_args(["--bundle-adjust", "image_focal=maybe"])
        assert str(refused.value) == (
            "Invalid value for --bundle-adjust key 'image_focal': 'maybe' is not a "
            "valid boolean (on, off, true, false, yes, no, 1 or 0)"
        )
        with pytest.raises(click.UsageError, match="not a valid camera list"):
            parse_transform_args(["--bundle-adjust", "cameras=one"])
        with pytest.raises(ValueError, match="scanlines must be one of"):
            BundleAdjustTransform(image_rolling_shutter=True, scanlines="rows")


class TestRouting:
    def test_a_per_image_release_takes_sfmtools_path(
        self, seoul_bull_ground_truth_sfmr, capsys
    ):
        """A reconstruction of COLMAP models goes through pycolmap bare, and
        through sfmtool's adjustment with a per-image release, which also
        accepts ``cross=off`` there."""
        recon = _simple_pinhole(seoul_bull_ground_truth_sfmr)
        capsys.readouterr()
        result = BundleAdjustTransform(image_focal=True, cross=False).apply(recon)
        out = capsys.readouterr().out
        assert "sfmtool; a per-image release, which pycolmap cannot hold" in out
        assert "released: focal, focal per image" in out
        assert "per-image widths focal " in out
        assert result.per_image_adjustments == [["focal"]]
        assert np.any(np.asarray(result.focal_adjustments) != 0.0)

    def test_a_camera_that_cannot_take_it_is_refused_naming_it(
        self, seoul_bull_ground_truth_sfmr, tmp_path
    ):
        recon = _simple_pinhole(seoul_bull_ground_truth_sfmr)
        with pytest.raises(
            ValueError,
            match="radial correction cannot be released per image on camera 0, "
            "a SIMPLE_PINHOLE",
        ):
            BundleAdjustTransform(image_radial_bspline=True).apply(recon)
        path = seoul_bull_ground_truth_sfmr.parent / "simple.sfmr"
        recon.save(path)
        result = CliRunner().invoke(
            main,
            [
                "xform",
                str(path),
                str(tmp_path / "out.sfmr"),
                "--bundle-adjust",
                "image_rolling_shutter=on",
            ],
        )
        assert result.exit_code != 0
        assert (
            "the rolling shutter cannot be released per image on camera 0, a "
            "SIMPLE_PINHOLE" in result.output
        )
        assert not (tmp_path / "out.sfmr").exists()

    def test_a_held_camera_releases_nothing_per_image(
        self, seoul_bull_ground_truth_sfmr
    ):
        recon = _pinhole(seoul_bull_ground_truth_sfmr)
        indexes = np.arange(recon.image_count, dtype=np.uint32) % 2
        rig = recon.clone_with_changes(
            cameras=[recon.cameras[0], recon.cameras[0]], camera_indexes=indexes
        )
        result = BundleAdjustTransform(cameras=[1], image_radial_bspline=True).apply(
            rig
        )
        assert result.per_image_adjustments == [[], ["radial_bspline"]]
        assert result.cameras[0] == rig.cameras[0]
        radial = np.asarray(result.radial_bspline_adjustments)
        assert not np.any(radial[indexes == 0])
        assert np.any(radial[indexes == 1])


class TestARealRun:
    @pytest.mark.parametrize(
        "key, stored, released",
        [
            ("image_focal", "focal", "focal per image"),
            ("image_principal_point", "principal_point", "principal point per image"),
            (
                "image_radial_bspline",
                "radial_bspline",
                "radial distortion per image",
            ),
        ],
    )
    def test_each_field_is_released_and_reported(
        self, seoul_bull_ground_truth_sfmr, tmp_path, key, stored, released
    ):
        # Beside the copy, whose workspace marker it resolves.
        path = seoul_bull_ground_truth_sfmr.parent / "pinhole.sfmr"
        _pinhole(seoul_bull_ground_truth_sfmr).save(path)
        out_path = tmp_path / "adjusted.sfmr"
        result = CliRunner().invoke(
            main,
            ["xform", str(path), str(out_path), "--bundle-adjust", f"{key}=on"],
        )
        assert result.exit_code == 0, result.output
        assert "released: focal, " in result.output
        assert released in result.output
        (widths,) = [
            line for line in result.output.splitlines() if "per-image widths" in line
        ]
        assert widths.strip().startswith("Camera 0 per-image widths ")
        assert ", derived, " in widths and widths.endswith("beyond two widths")
        adjusted = SfmrReconstruction.load(out_path)
        assert adjusted.per_image_adjustments == [[stored]]
        assert "Rolling-shutter" not in result.output
        assert "rolling shutter on" not in result.output

    def test_the_rates_on_given_scanlines(self, seoul_bull_ground_truth_sfmr, capsys):
        recon = _pinhole(seoul_bull_ground_truth_sfmr)
        capsys.readouterr()
        result = BundleAdjustTransform(
            image_rolling_shutter=True, scanlines="COLUMNS"
        ).apply(recon)
        out = capsys.readouterr().out
        assert "released: focal, distortion, rolling shutter per image" in out
        assert " mrad per short side, derived, " in out
        assert "    Camera 0 rolling shutter on columns\n" in out
        assert result.per_image_adjustments == [["rolling_shutter"]]
        assert result.rolling_shutter_scanlines == ["COLUMNS"]

    def test_unknown_scanlines_race_and_a_stored_setting_is_used(
        self, seoul_bull_ground_truth_sfmr, capsys
    ):
        """A camera with no stored setting races rows against columns; one
        with a stored setting is solved on it, unless ``scanlines=unknown``
        asks for the race."""
        recon = _pinhole(seoul_bull_ground_truth_sfmr)
        assert recon.rolling_shutter_scanlines == [None]
        capsys.readouterr()
        raced = BundleAdjustTransform(image_rolling_shutter=True).apply(recon)
        out = capsys.readouterr().out
        (line,) = [x for x in out.splitlines() if "rolling shutter on" in x]
        assert " chosen: cost " in line and " with rows, " in line
        assert line.endswith("observations, both solves run to the end")
        chosen = raced.rolling_shutter_scanlines[0]
        assert f"rolling shutter on {chosen.lower()} chosen" in line

        stored = recon.clone_with_changes(rolling_shutter_scanlines=["ROWS"])
        BundleAdjustTransform(image_rolling_shutter=True).apply(stored)
        assert "    Camera 0 rolling shutter on rows\n" in capsys.readouterr().out
        BundleAdjustTransform(image_rolling_shutter=True, scanlines="UNKNOWN").apply(
            stored
        )
        assert " chosen: cost " in capsys.readouterr().out


def _camera(**overrides):
    """A ``report["cameras"]`` entry with every per-image field released and
    two images beyond two widths, image 4 for two parameters."""
    camera = {
        "camera": 0,
        "image_focal_released": True,
        "image_principal_point_released": True,
        "image_radial_bspline_released": True,
        "image_rolling_shutter_released": True,
        "rolling_shutter_scanlines": "ROWS",
        "scanlines_race": None,
        "image_adjustments": {
            "prior": {
                "focal": 0.0042,
                "principal_point": 0.0131,
                "radial_bspline": 0.0012,
                "rolling_shutter": 0.002,
            },
            "prior_derived": True,
            "near_prior": [
                {"image": 4, "parameter": "focal", "widths": 2.5},
                {"image": 4, "parameter": "rolling_shutter", "widths": 2.1},
                {"image": 9, "parameter": "radial_bspline", "widths": 3.0},
            ],
        },
    }
    camera.update(overrides)
    return camera


class TestThePrintout:
    def test_the_widths_line(self):
        assert format_image_prior_widths(_camera(), 1400.0, 1536.0) == (
            "per-image widths focal 0.42 %, radial distortion 0.00120 of the focal "
            "(1.68 px), principal point 1.31 % of the short side (20.12 px), rolling "
            "shutter 2.00 mrad per short side, derived, 2 images beyond two widths"
        )
        camera = _camera(
            image_focal_released=False,
            image_principal_point_released=False,
            image_rolling_shutter_released=False,
        )
        camera["image_adjustments"]["prior_derived"] = False
        camera["image_adjustments"]["near_prior"] = []
        assert format_image_prior_widths(camera, 1400.0, 1536.0) == (
            "per-image widths radial distortion 0.00120 of the focal (1.68 px), "
            "fixed, no image beyond two widths"
        )
        assert format_image_prior_widths(_camera(image_adjustments=None), 1, 1) is None

    def test_the_scanlines_line(self):
        assert format_scanlines(_camera()) == "rolling shutter on rows"
        assert format_scanlines(_camera(image_rolling_shutter_released=False)) is None
        race = {
            "rows_cost": 49350.0,
            "columns_cost": 52349.0,
            "observations": 24931,
            "sigma_px": 0.4,
            "stopped_early": False,
            "stop": None,
        }
        assert format_scanlines(_camera(scanlines_race=race)) == (
            "rolling shutter on rows chosen: cost 49350.0 with rows, 52349.0 with "
            "columns over 24931 observations, both solves run to the end"
        )
        stopped = dict(
            race,
            rows_cost=16552.7,
            columns_cost=17506.1,
            observations=24918,
            stopped_early=True,
            stop={
                "stopped": "COLUMNS",
                "round": 1,
                "iteration": 12,
                "gap": 953.4,
                "winner_cost": 16000.0,
            },
        )
        assert format_scanlines(_camera(scanlines_race=stopped)) == (
            "rolling shutter on rows chosen: columns stopped at round 1, iteration "
            "12: cost 16552.7 with rows, 17506.1 with columns (gap 953.4 over 24918 "
            "observations)"
        )
