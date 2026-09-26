# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for SwitchCameraModelTransform and ``--camera-model``."""

import click
import numpy as np
import pytest

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import SwitchCameraModelTransform
from sfmtool.xform._arg_parser import parse_camera_model_params

from .conftest import apply_transforms_to_file


def test_switch_simple_radial_to_radial_reproduces_the_copy(
    seoul_bull_ground_truth_sfmr, tmp_path
):
    """A target that contains the source's model fits to the copied parameters,
    with the new term at zero."""
    source_recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    assert all(c.model == "SIMPLE_RADIAL" for c in source_recon.cameras)

    output_path = tmp_path / "radial.sfmr"
    apply_transforms_to_file(
        seoul_bull_ground_truth_sfmr,
        output_path,
        [SwitchCameraModelTransform("RADIAL")],
    )

    result = SfmrReconstruction.load(output_path)
    assert len(result.cameras) == len(source_recon.cameras)
    for src, dst in zip(source_recon.cameras, result.cameras):
        assert dst.model == "RADIAL"
        src_p = src.to_dict()["parameters"]
        dst_p = dst.to_dict()["parameters"]
        assert dst_p["focal_length"] == pytest.approx(src_p["focal_length"])
        assert dst_p["principal_point_x"] == src_p["principal_point_x"]
        assert dst_p["principal_point_y"] == src_p["principal_point_y"]
        assert dst_p["radial_distortion_k1"] == pytest.approx(
            src_p["radial_distortion_k1"]
        )
        assert dst_p["radial_distortion_k2"] == pytest.approx(0.0, abs=1e-12)
        assert dst.width == src.width
        assert dst.height == src.height


def test_switch_to_opencv_keeps_the_lens(seoul_bull_ground_truth_sfmr, tmp_path):
    """A split-focal target gets both focals from the single one, and the terms
    the source does not have stay at zero because the fit needs none."""
    output_path = tmp_path / "opencv.sfmr"
    apply_transforms_to_file(
        seoul_bull_ground_truth_sfmr,
        output_path,
        [SwitchCameraModelTransform("OPENCV")],
    )
    source_recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    result = SfmrReconstruction.load(output_path)
    for src, dst in zip(source_recon.cameras, result.cameras):
        assert dst.model == "OPENCV"
        src_p = src.to_dict()["parameters"]
        dst_p = dst.to_dict()["parameters"]
        assert dst_p["focal_length_x"] == pytest.approx(src_p["focal_length"])
        assert dst_p["focal_length_y"] == pytest.approx(src_p["focal_length"])
        assert dst_p["radial_distortion_k1"] == pytest.approx(
            src_p["radial_distortion_k1"]
        )
        for name in (
            "radial_distortion_k2",
            "tangential_distortion_p1",
            "tangential_distortion_p2",
        ):
            assert dst_p[name] == pytest.approx(0.0, abs=1e-12)


def test_switch_preserves_points_and_poses(seoul_bull_ground_truth_sfmr, tmp_path):
    """Only cameras and the stored point errors change."""
    output_path = tmp_path / "switched.sfmr"
    apply_transforms_to_file(
        seoul_bull_ground_truth_sfmr,
        output_path,
        [SwitchCameraModelTransform("SFMTOOL_PINHOLE", coeff_count=4)],
    )
    original = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    switched = SfmrReconstruction.load(output_path)

    assert switched.cameras[0].model == "SFMTOOL_PINHOLE"
    assert switched.image_count == original.image_count
    assert switched.point_count == original.point_count
    assert switched.observation_count == original.observation_count
    assert (switched.positions == original.positions).all()
    assert (switched.translations == original.translations).all()
    assert np.allclose(
        switched.quaternions_wxyz, original.quaternions_wxyz, rtol=0, atol=1e-12
    )
    assert (switched.keypoints_xy == original.keypoints_xy).all()


def test_switch_reads_sift_files_without_inline_keypoints(
    seoul_bull_workspace, tmp_path
):
    """A ``sift_files`` reconstruction measures its observations from the
    ``.sift`` files."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    switched, report = recon.switch_camera_model("RADIAL")
    entry = report["cameras"][0]
    assert entry["observations"]["observations"] > 0
    assert entry["observations"]["after"]["median_px"] == pytest.approx(
        entry["observations"]["before"]["median_px"], abs=1e-6
    )
    assert switched.cameras[0].model == "RADIAL"


def test_switch_without_pixels_is_refused(seoul_bull_sfmr_only):
    """With neither inline keypoints nor its ``.sift`` files, there is nothing
    to compare the models on, and the switch says so."""
    recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    with pytest.raises(ValueError, match="could not be read"):
        recon.switch_camera_model("RADIAL")


def test_switch_from_equidistant_fisheye_source(seoul_bull_ground_truth_sfmr, tmp_path):
    """An ``EQUIDISTANT_FISHEYE`` switched to ``SIMPLE_RADIAL_FISHEYE`` fits to
    the same focal with a zero coefficient: the carrier's identical map."""
    from sfmtool._sfmtool.geometry import CameraIntrinsics

    source = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    equidistant = [
        CameraIntrinsics.from_dict(
            {
                "model": "EQUIDISTANT_FISHEYE",
                "width": c.width,
                "height": c.height,
                "parameters": {
                    "focal_length": 130.0,
                    "principal_point_x": c.width / 2.0,
                    "principal_point_y": c.height / 2.0,
                },
            }
        )
        for c in source.cameras
    ]
    # Beside the source, so it resolves the same workspace.
    staged = seoul_bull_ground_truth_sfmr.parent / "equidistant.sfmr"
    source.clone_with_changes(cameras=equidistant).save(staged)

    output_path = tmp_path / "carrier.sfmr"
    apply_transforms_to_file(
        staged,
        output_path,
        [SwitchCameraModelTransform("SIMPLE_RADIAL_FISHEYE")],
    )
    result = SfmrReconstruction.load(output_path)
    for dst in result.cameras:
        p = dst.to_dict()["parameters"]
        assert dst.model == "SIMPLE_RADIAL_FISHEYE"
        assert p["focal_length"] == pytest.approx(130.0)
        assert p["principal_point_x"] == pytest.approx(dst.width / 2.0)
        assert p["radial_distortion_k1"] == pytest.approx(0.0, abs=1e-12)


def test_switch_prints_the_report(seoul_bull_ground_truth_sfmr, capsys):
    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    SwitchCameraModelTransform("SFMTOOL_PINHOLE", coeff_count=4).apply(recon)
    out = capsys.readouterr().out
    assert "SIMPLE_RADIAL -> SFMTOOL_PINHOLE" in out
    assert "fit over θ ≤" in out
    assert "before: median" in out
    assert "after:  median" in out


def test_switch_names_the_target_vocabulary():
    for name in ("SFMTOOL_FISHEYE", "sfmtool_pinhole", "EQUIDISTANT_FISHEYE", "radial"):
        SwitchCameraModelTransform(name)
    with pytest.raises(ValueError, match="Unknown camera model"):
        SwitchCameraModelTransform("NOT_A_MODEL")
    # Equirectangular is not a lens model a camera can be refitted to.
    with pytest.raises(ValueError, match="Unknown camera model"):
        SwitchCameraModelTransform("EQUIRECTANGULAR")
    with pytest.raises(ValueError, match="coeffs= applies only"):
        SwitchCameraModelTransform("RADIAL", coeff_count=4)


def test_camera_model_option_parses_its_keys():
    t = parse_camera_model_params(
        "SFMTOOL_FISHEYE,coeffs=6,fit_to=80,spline_domain=140,cameras=0+2"
    )
    assert t.target_model == "SFMTOOL_FISHEYE"
    assert t.coeff_count == 6
    assert t.theta_fit_deg == 80.0
    assert t.spline_domain_deg == 140.0
    assert t.cameras == [0, 2]
    assert t.description() == (
        "Switch camera model to SFMTOOL_FISHEYE,coeffs=6,fit_to=80,"
        "spline_domain=140,cameras=0+2"
    )

    plain = parse_camera_model_params("radial")
    assert plain.target_model == "RADIAL"
    assert plain.cameras is None

    with pytest.raises(click.UsageError, match="Unknown --camera-model key"):
        parse_camera_model_params("SFMTOOL_FISHEYE,knots=4")
    with pytest.raises(click.UsageError, match="Invalid --camera-model parameter"):
        parse_camera_model_params("RADIAL,coeffs=4")


def test_bundle_adjust_after_a_switch_to_a_spline_releases_it(
    seoul_bull_ground_truth_sfmr, tmp_path, capsys
):
    """pycolmap knows neither spline model, so ``--bundle-adjust`` on a spline
    camera runs sfmtool's own adjustment with the focal and the spline
    released."""
    from sfmtool.xform import BundleAdjustTransform

    output_path = tmp_path / "adjusted.sfmr"
    apply_transforms_to_file(
        seoul_bull_ground_truth_sfmr,
        output_path,
        [
            SwitchCameraModelTransform("SFMTOOL_PINHOLE", coeff_count=4),
            BundleAdjustTransform(),
        ],
    )
    out = capsys.readouterr().out
    assert "sfmtool; a camera has a spline model" in out
    assert "released: focal, distortion" in out
    result = SfmrReconstruction.load(output_path)
    assert result.cameras[0].model == "SFMTOOL_PINHOLE"


def test_bundle_adjust_option_takes_a_coefficient_count():
    """``--bundle-adjust`` bare keeps working; ``coeffs=N`` is its one key."""
    from sfmtool.xform import BundleAdjustTransform
    from sfmtool.xform._arg_parser import parse_transform_args

    (bare,) = parse_transform_args(["--bundle-adjust"])
    assert isinstance(bare, BundleAdjustTransform)
    assert bare.coeff_count is None
    (spaced,) = parse_transform_args(["--bundle-adjust", "coeffs=12"])
    (joined,) = parse_transform_args(["--bundle-adjust=coeffs=12"])
    assert spaced.coeff_count == joined.coeff_count == 12
    assert "coeffs=12" in spaced.description()
    # A following option is not taken as the value.
    bare, scale = parse_transform_args(["--bundle-adjust", "--scale", "2"])
    assert bare.coeff_count is None and scale.scale == 2.0
    with pytest.raises(click.UsageError, match="Unknown --bundle-adjust key"):
        parse_transform_args(["--bundle-adjust", "knots=4"])
    with pytest.raises(click.UsageError, match="not a valid int"):
        parse_transform_args(["--bundle-adjust", "coeffs=many"])


def test_bundle_adjust_refits_the_spline_to_a_new_coefficient_count(
    seoul_bull_ground_truth_sfmr, tmp_path, capsys
):
    """A camera switched to six spline coefficients is refitted to eight over
    its whole domain before the solve, which then starts from the refit."""
    from sfmtool.xform import BundleAdjustTransform

    output_path = tmp_path / "adjusted.sfmr"
    apply_transforms_to_file(
        seoul_bull_ground_truth_sfmr,
        output_path,
        [
            SwitchCameraModelTransform("SFMTOOL_FISHEYE", coeff_count=6),
            BundleAdjustTransform(coeff_count=8),
        ],
    )
    out = capsys.readouterr().out
    assert "spline refitted 6 -> 8 coefficients before the solve" in out
    result = SfmrReconstruction.load(output_path)
    (camera,) = result.cameras
    assert camera.model == "SFMTOOL_FISHEYE"
    params = camera.to_dict()["parameters"]
    assert params["bspline_coeff_count"] == 8


def test_bundle_adjust_coefficients_need_a_spline_camera(seoul_bull_ground_truth_sfmr):
    from sfmtool.xform import BundleAdjustTransform

    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    with pytest.raises(click.UsageError, match="apply only to a reconstruction"):
        BundleAdjustTransform(coeff_count=8).apply(recon)


def test_bundle_adjust_option_takes_a_domain():
    from sfmtool.xform._arg_parser import parse_transform_args

    (both,) = parse_transform_args(["--bundle-adjust", "coeffs=12,domain=108.5"])
    assert (both.coeff_count, both.spline_domain_deg) == (12, 108.5)
    assert "coeffs=12, domain=108.5" in both.description()
    (domain,) = parse_transform_args(["--bundle-adjust=domain=100"])
    assert (domain.coeff_count, domain.spline_domain_deg) == (None, 100.0)
    with pytest.raises(click.UsageError, match="not a valid float"):
        parse_transform_args(["--bundle-adjust", "domain=wide"])


def test_bundle_adjust_moves_the_spline_domain(
    seoul_bull_ground_truth_sfmr, tmp_path, capsys
):
    """A new domain end is one refit with the count, over the whole new
    domain, and the outermost observation is printed beside it."""
    from sfmtool.xform import BundleAdjustTransform

    output_path = tmp_path / "adjusted.sfmr"
    apply_transforms_to_file(
        seoul_bull_ground_truth_sfmr,
        output_path,
        [
            SwitchCameraModelTransform("SFMTOOL_FISHEYE", coeff_count=6),
            BundleAdjustTransform(coeff_count=8, spline_domain_deg=40.0),
        ],
    )
    out = capsys.readouterr().out
    assert "spline refitted 6 -> 8 coefficients, domain " in out
    assert "-> 40.0° before the solve" in out
    # The switch and the adjustment both print how far out the keypoints
    # reach; the ground truth sits beside no .sift file, so only observed.
    assert out.count("outermost keypoint: ") >= 2
    assert "° observed" in out and "detected" not in out.split("outermost")[-1]
    result = SfmrReconstruction.load(output_path)
    params = result.cameras[0].to_dict()["parameters"]
    assert params["bspline_coeff_count"] == 8
    assert params["bspline_theta_max"] == pytest.approx(np.radians(40.0))


def test_the_switch_report_names_the_outermost_keypoint(seoul_bull_ground_truth_sfmr):
    from sfmtool.xform._switch_camera_model import format_camera_report

    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    _, report = recon.switch_camera_model("SFMTOOL_PINHOLE", coeff_count=4)
    (entry,) = report["cameras"]
    outermost = entry["outermost"]
    assert outermost["detected"] is None and outermost["detected_images"] == 0
    observed = outermost["observed"]
    assert 0.0 < observed["radius_px"] and 0.0 < observed["theta_deg"] < 90.0
    lines = format_camera_report(entry)
    assert lines[-1].startswith("  outermost keypoint: ")
    assert lines[-1].endswith("° observed")
