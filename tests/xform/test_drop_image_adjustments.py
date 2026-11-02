# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""``sfm xform --drop-image-adjustments`` and the ``principal_point`` key of
``--bundle-adjust``: the parsing, the refusals, the fold into the camera and
the chain that drops each image's principal-point offset and then solves one
per camera. See ``specs/cli/reconstruction/xform/xform-command.md``.
"""

import click
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import read_sfmr_metadata
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import BundleAdjustTransform, DropImageAdjustmentsTransform
from sfmtool.xform._arg_parser import parse_transform_args


def _principal_point(recon, camera=0):
    return np.asarray(recon.cameras[camera].principal_point)


def _with_offsets(path):
    """The reconstruction at ``path`` with its camera varying its principal
    point and focal per image: alternating offsets of ±0.01 short sides, which
    average to a known mean, and focal scales of ±1 %."""
    recon = SfmrReconstruction.load(path)
    n = recon.image_count
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    offsets = np.stack([0.01 * sign + 0.002, np.zeros(n)], axis=1)
    return recon.clone_with_changes(
        principal_point_adjustments_xy=offsets,
        focal_adjustments=0.01 * sign,
        per_image_adjustments=[["focal", "principal_point"]],
    )


class TestParsing:
    def test_the_keys_reach_the_transform(self):
        (drop, ba) = parse_transform_args(
            [
                "--drop-image-adjustments",
                "parameters=principal_point+focal,cameras=1+0",
                "--bundle-adjust",
                "principal_point=on,cameras=0",
            ]
        )
        # The names keep the order a camera's per_image_adjustments lists.
        assert drop.parameters == ["focal", "principal_point"]
        assert drop.cameras == [0, 1]
        assert drop.description() == (
            "Drop per-image adjustments (parameters=focal+principal_point, cameras=0+1)"
        )
        assert ba.release_principal_point
        assert ba.description() == (
            "Bundle adjustment (release: focal,principal_point,distortion, cameras=0)"
        )
        (joined,) = parse_transform_args(
            ["--drop-image-adjustments=parameters=rolling_shutter"]
        )
        assert joined.parameters == ["rolling_shutter"] and joined.cameras is None

    def test_the_principal_point_is_held_without_the_key(self):
        (bare,) = parse_transform_args(["--bundle-adjust"])
        assert not bare.release_principal_point
        (off,) = parse_transform_args(["--bundle-adjust", "principal_point=off"])
        assert not off.release_principal_point

    @pytest.mark.parametrize(
        "value, message",
        [
            ("parameters=pp", "'pp' is not a per-image parameter; the parameters"),
            ("parameters=", "names no per-image parameter to drop"),
            ("cameras=0", "needs parameters="),
            ("parameters=focal,scale=2", "Unknown --drop-image-adjustments key"),
            ("parameters=focal,cameras=one", "not a valid camera list"),
        ],
    )
    def test_bad_values_are_refused_at_parse_time(self, value, message):
        with pytest.raises(click.UsageError, match=message):
            parse_transform_args(["--drop-image-adjustments", value])

    def test_a_missing_value_and_a_bad_boolean_are_refused(self):
        with pytest.raises(click.UsageError, match="requires an argument"):
            parse_transform_args(["--drop-image-adjustments"])
        with pytest.raises(click.UsageError, match="not a valid boolean"):
            parse_transform_args(["--bundle-adjust", "principal_point=maybe"])


class TestTheDrop:
    def test_the_mean_is_folded_into_the_camera(
        self, seoul_bull_ground_truth_sfmr, capsys
    ):
        recon = _with_offsets(seoul_bull_ground_truth_sfmr)
        before = _principal_point(recon)
        short_side = min(recon.cameras[0].width, recon.cameras[0].height)
        mean = np.asarray(recon.principal_point_adjustments_xy).mean(axis=0)
        capsys.readouterr()
        dropped = DropImageAdjustmentsTransform(["principal_point"]).apply(recon)
        assert "Camera 0: dropped principal_point per image" in capsys.readouterr().out
        assert dropped.per_image_adjustments == [["focal"]]
        assert dropped.principal_point_adjustments_xy is None
        np.testing.assert_allclose(
            _principal_point(dropped), before + mean * short_side, rtol=0, atol=1e-9
        )
        # The focal it was not asked to drop is kept.
        np.testing.assert_array_equal(
            dropped.focal_adjustments, recon.focal_adjustments
        )

    def test_a_camera_that_varies_none_of_them_is_unchanged(
        self, seoul_bull_ground_truth_sfmr, capsys
    ):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        capsys.readouterr()
        same = DropImageAdjustmentsTransform(["focal", "rolling_shutter"]).apply(recon)
        assert "Camera 0: varies none of them per image" in capsys.readouterr().out
        assert same.cameras[0] == recon.cameras[0]
        assert same.per_image_adjustments == [[]]

    def test_an_unknown_camera_is_refused_naming_it(
        self, seoul_bull_ground_truth_sfmr, tmp_path
    ):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        with pytest.raises(click.UsageError, match=r"names camera\(s\) \[3\]"):
            DropImageAdjustmentsTransform(["focal"], cameras=[0, 3]).apply(recon)
        out_path = tmp_path / "out.sfmr"
        result = CliRunner().invoke(
            main,
            [
                "xform",
                str(seoul_bull_ground_truth_sfmr),
                str(out_path),
                "--drop-image-adjustments",
                "parameters=focal,cameras=3",
            ],
        )
        assert result.exit_code != 0
        assert "this reconstruction has 1 camera(s)" in result.output
        assert not out_path.exists()


class TestTheChain:
    def test_the_principal_point_moves_with_the_key_and_is_held_without_it(
        self, seoul_bull_ground_truth_sfmr, tmp_path
    ):
        """On the pycolmap path, a SIMPLE_RADIAL camera's principal point is
        held by the bare option and solved with ``principal_point=on``."""
        source = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        results = {}
        for name, value in (("held", ""), ("released", "principal_point=on")):
            out_path = tmp_path / f"{name}.sfmr"
            args = ["xform", str(seoul_bull_ground_truth_sfmr), str(out_path)]
            args += [f"--bundle-adjust={value}"] if value else ["--bundle-adjust"]
            result = CliRunner().invoke(main, args)
            assert result.exit_code == 0, result.output
            results[name] = _principal_point(SfmrReconstruction.load(out_path))
        np.testing.assert_array_equal(results["held"], _principal_point(source))
        assert np.abs(results["released"] - _principal_point(source)).max() > 1e-3

    def test_drop_the_offsets_then_solve_one_principal_point_per_camera(
        self, seoul_bull_ground_truth_sfmr, tmp_path
    ):
        """The Kerry Park workflow on sfmtool's adjustment: drop each image's
        principal-point offset and focal scale, then release the camera's
        shared principal point. The output varies nothing per image, and its
        principal point is solved, away from the folded one."""
        recon = _with_offsets(seoul_bull_ground_truth_sfmr)
        switched, _ = recon.switch_camera_model("SFMTOOL_PINHOLE", coeff_count=4)
        # Beside the copy, whose workspace marker it resolves.
        path = seoul_bull_ground_truth_sfmr.parent / "offsets.sfmr"
        switched.save(path)
        folded = DropImageAdjustmentsTransform(["focal", "principal_point"]).apply(
            switched
        )
        out_path = tmp_path / "solved.sfmr"
        result = CliRunner().invoke(
            main,
            [
                "xform",
                str(path),
                str(out_path),
                "--drop-image-adjustments",
                "parameters=principal_point+focal",
                "--bundle-adjust=principal_point=on",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (
            "[1/2] Drop per-image adjustments (parameters=focal+principal_point)"
            in (result.output)
        )
        assert "Camera 0: dropped focal, principal_point per image" in result.output
        assert "released: focal, principal point, distortion" in result.output
        solved = SfmrReconstruction.load(out_path)
        assert solved.per_image_adjustments == [[]]
        assert solved.principal_point_adjustments_xy is None
        assert np.abs(_principal_point(solved) - _principal_point(folded)).max() > 1e-3
        assert read_sfmr_metadata(out_path)["tool_options"]["transforms"] == [
            "Drop per-image adjustments (parameters=focal+principal_point)",
            "Bundle adjustment (release: focal,principal_point,distortion)",
        ]

    def test_the_transform_releases_the_principal_point_on_sfmtools_path(
        self, seoul_bull_ground_truth_sfmr, capsys
    ):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        switched, _ = recon.switch_camera_model("SFMTOOL_PINHOLE", coeff_count=4)
        capsys.readouterr()
        held = BundleAdjustTransform().apply(switched)
        assert "; principal point (" not in capsys.readouterr().out
        np.testing.assert_array_equal(
            _principal_point(held), _principal_point(switched)
        )
        BundleAdjustTransform(release_principal_point=True).apply(switched)
        assert "; principal point (" in capsys.readouterr().out
