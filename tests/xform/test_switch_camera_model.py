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
