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


def test_bundle_adjust_option_parses_bare_and_with_keys():
    """``--bundle-adjust`` bare, followed by another option, and with an
    unknown key."""
    from sfmtool.xform import BundleAdjustTransform
    from sfmtool.xform._arg_parser import parse_transform_args

    (bare,) = parse_transform_args(["--bundle-adjust"])
    assert isinstance(bare, BundleAdjustTransform)
    assert bare.description() == "Bundle adjustment (refine: focal,extra)"
    # A following option is not taken as the value.
    bare, scale = parse_transform_args(["--bundle-adjust", "--scale", "2"])
    assert bare.cameras is None and scale.scale == 2.0
    with pytest.raises(click.UsageError, match="Unknown --bundle-adjust key"):
        parse_transform_args(["--bundle-adjust", "knots=4"])


def test_camera_model_refits_a_spline_camera_to_a_new_count_and_domain(
    seoul_bull_ground_truth_sfmr, tmp_path, capsys
):
    """A camera switched to six spline coefficients, switched again to its own
    model with eight coefficients on a 108° domain, is refitted over the whole
    new domain; a following adjustment refines the refitted spline."""
    from sfmtool.xform._arg_parser import parse_transform_args

    switched_path = tmp_path / "switched.sfmr"
    apply_transforms_to_file(
        seoul_bull_ground_truth_sfmr,
        switched_path,
        parse_transform_args(["--camera-model", "SFMTOOL_FISHEYE,coeffs=6"]),
    )
    source = SfmrReconstruction.load(switched_path)
    source_params = source.cameras[0].to_dict()["parameters"]
    assert source_params["bspline_coeff_count"] == 6
    capsys.readouterr()

    output_path = tmp_path / "refitted.sfmr"
    apply_transforms_to_file(
        switched_path,
        output_path,
        parse_transform_args(
            ["--camera-model", "SFMTOOL_FISHEYE,coeffs=8,spline_domain=108,cameras=0"]
        ),
    )
    out = capsys.readouterr().out
    assert "SFMTOOL_FISHEYE -> SFMTOOL_FISHEYE" in out
    assert "fit over the whole spline domain θ ≤ 108.0°" in out
    result = SfmrReconstruction.load(output_path)
    params = result.cameras[0].to_dict()["parameters"]
    assert params["bspline_coeff_count"] == 8
    assert params["bspline_theta_max"] == pytest.approx(np.radians(108.0))
    # Exactly the switch's own refit of that camera, poses untouched.
    expected, report = source.switch_camera_model(
        "SFMTOOL_FISHEYE", cameras=[0], coeff_count=8, spline_domain_deg=108.0
    )
    assert result.cameras[0] == expected.cameras[0]
    fit = report["cameras"][0]["fit"]
    assert fit["theta_fit_source"] == "spline_domain"
    assert fit["theta_fit_deg"] == pytest.approx(108.0)
    np.testing.assert_array_equal(result.translations, source.translations)

    # With no domain given the domain end is kept exactly.
    kept, _ = source.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=8)
    kept_params = kept.cameras[0].to_dict()["parameters"]
    assert kept_params["bspline_theta_max"] == source_params["bspline_theta_max"]

    # The adjustment then refines the refitted spline.
    adjusted_path = tmp_path / "adjusted.sfmr"
    apply_transforms_to_file(
        output_path, adjusted_path, parse_transform_args(["--bundle-adjust"])
    )
    out = capsys.readouterr().out
    assert "released: focal, distortion" in out
    adjusted = SfmrReconstruction.load(adjusted_path)
    assert adjusted.cameras[0].to_dict()["parameters"]["bspline_coeff_count"] == 8


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


def _two_camera_rig(recon, second):
    """``recon`` with its odd images taken through ``second``, appended to the
    camera table."""
    indexes = np.arange(recon.image_count, dtype=np.uint32) % 2
    return recon.clone_with_changes(
        cameras=[recon.cameras[0], second], camera_indexes=indexes
    )


def test_bundle_adjust_option_takes_cameras():
    from sfmtool.xform._arg_parser import parse_transform_args

    (limited,) = parse_transform_args(["--bundle-adjust", "cameras=2+0"])
    assert limited.cameras == [0, 2]
    assert "cameras=0+2" in limited.description()
    (bare,) = parse_transform_args(["--bundle-adjust"])
    assert bare.cameras is None
    with pytest.raises(click.UsageError, match="not a valid"):
        parse_transform_args(["--bundle-adjust", "cameras=one"])


def test_bundle_adjust_cameras_releases_those_and_holds_the_rest(
    seoul_bull_ground_truth_sfmr, capsys
):
    """On a spline rig, ``cameras=1`` releases camera 1's focal and spline and
    holds camera 0 exactly."""
    from sfmtool.xform import BundleAdjustTransform

    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    switched, _ = recon.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=6)
    rig = _two_camera_rig(switched, switched.cameras[0])
    capsys.readouterr()

    result = BundleAdjustTransform(cameras=[1]).apply(rig)

    out = capsys.readouterr().out
    assert "Camera 0 (" in out and "released: none" in out
    assert "released: focal, distortion" in out
    assert result.cameras[0] == rig.cameras[0]
    assert result.cameras[1] != rig.cameras[1]


def test_bundle_adjust_cameras_holds_a_camera_the_solve_cannot_release(
    seoul_bull_ground_truth_sfmr,
):
    """A rig mixing a spline camera with an ``OPENCV_FISHEYE`` one is refused
    bare, because the solve cannot release the ``OPENCV_FISHEYE`` focal, and
    runs with the release limited to the spline camera."""
    from sfmtool.xform import BundleAdjustTransform

    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    spline, _ = recon.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=6)
    opencv, _ = recon.switch_camera_model("OPENCV_FISHEYE")
    rig = _two_camera_rig(spline, opencv.cameras[0])

    with pytest.raises(ValueError, match="camera 1, a OPENCV_FISHEYE"):
        BundleAdjustTransform().apply(rig)
    result = BundleAdjustTransform(cameras=[0]).apply(rig)
    assert result.cameras[1] == rig.cameras[1]
    with pytest.raises(click.UsageError, match=r"camera\(s\) \[2\]"):
        BundleAdjustTransform(cameras=[0, 2]).apply(rig)


def test_bundle_adjust_cameras_holds_colmap_cameras_through_pycolmap(
    seoul_bull_ground_truth_sfmr,
):
    """A COLMAP-model rig goes through pycolmap, and ``cameras=0`` holds camera
    1's intrinsics constant there."""
    from sfmtool.xform import BundleAdjustTransform

    recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
    rig = _two_camera_rig(recon, recon.cameras[0])

    result = BundleAdjustTransform(cameras=[0]).apply(rig)

    held = rig.cameras[1].to_dict()["parameters"]
    after = result.cameras[1].to_dict()["parameters"]
    assert after == pytest.approx(held, rel=0, abs=1e-12)
    moved = result.cameras[0].to_dict()["parameters"]
    assert moved != pytest.approx(rig.cameras[0].to_dict()["parameters"], abs=1e-12)
