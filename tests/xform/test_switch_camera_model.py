# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for SwitchCameraModelTransform."""

import pytest

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import SwitchCameraModelTransform

from .conftest import apply_transforms_to_file


def test_switch_simple_radial_to_radial(seoul_bull_sfmr_only, tmp_path):
    """The motivating case: upgrade SIMPLE_RADIAL → RADIAL to expose a k2
    term for bundle adjustment to refine."""
    source_recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    # The 17-image solve uses SIMPLE_RADIAL by default.
    assert all(c.model == "SIMPLE_RADIAL" for c in source_recon.cameras)

    output_path = tmp_path / "radial.sfmr"
    apply_transforms_to_file(
        seoul_bull_sfmr_only,
        output_path,
        [SwitchCameraModelTransform("RADIAL")],
    )

    result = SfmrReconstruction.load(output_path)
    assert len(result.cameras) == len(source_recon.cameras)

    for src, dst in zip(source_recon.cameras, result.cameras):
        assert dst.model == "RADIAL"
        src_p = src.to_dict()["parameters"]
        dst_p = dst.to_dict()["parameters"]
        # Shared parameters carry over.
        assert dst_p["focal_length"] == pytest.approx(src_p["focal_length"])
        assert dst_p["principal_point_x"] == pytest.approx(src_p["principal_point_x"])
        assert dst_p["principal_point_y"] == pytest.approx(src_p["principal_point_y"])
        assert dst_p["radial_distortion_k1"] == pytest.approx(
            src_p["radial_distortion_k1"]
        )
        # The new parameter initializes to zero.
        assert dst_p["radial_distortion_k2"] == 0.0
        # Image dimensions preserved.
        assert dst.width == src.width
        assert dst.height == src.height


def test_switch_simple_radial_to_opencv_pads_new_params_with_zero(
    seoul_bull_sfmr_only, tmp_path
):
    """Converting to a split-focal model duplicates focal_length into fx/fy
    and zeros out k2/p1/p2."""
    output_path = tmp_path / "opencv.sfmr"
    apply_transforms_to_file(
        seoul_bull_sfmr_only,
        output_path,
        [SwitchCameraModelTransform("OPENCV")],
    )

    source_recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    result = SfmrReconstruction.load(output_path)

    for src, dst in zip(source_recon.cameras, result.cameras):
        assert dst.model == "OPENCV"
        src_p = src.to_dict()["parameters"]
        dst_p = dst.to_dict()["parameters"]
        # Single focal_length split into fx/fy.
        assert dst_p["focal_length_x"] == pytest.approx(src_p["focal_length"])
        assert dst_p["focal_length_y"] == pytest.approx(src_p["focal_length"])
        # k1 carries over; the new terms initialize to zero.
        assert dst_p["radial_distortion_k1"] == pytest.approx(
            src_p["radial_distortion_k1"]
        )
        assert dst_p["radial_distortion_k2"] == 0.0
        assert dst_p["tangential_distortion_p1"] == 0.0
        assert dst_p["tangential_distortion_p2"] == 0.0


def test_switch_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown camera model"):
        SwitchCameraModelTransform("NOT_A_MODEL")


def test_switch_is_case_insensitive():
    SwitchCameraModelTransform("radial")
    SwitchCameraModelTransform("Radial")
    SwitchCameraModelTransform("RADIAL")


def test_switch_preserves_points_and_poses(seoul_bull_sfmr_only, tmp_path):
    """Only cameras change; points, poses, and observations are untouched."""
    import numpy as np

    from .conftest import load_reconstruction_data

    output_path = tmp_path / "switched.sfmr"
    apply_transforms_to_file(
        seoul_bull_sfmr_only,
        output_path,
        [SwitchCameraModelTransform("RADIAL")],
    )

    original = load_reconstruction_data(seoul_bull_sfmr_only)
    switched = load_reconstruction_data(output_path)

    assert switched["image_count"] == original["image_count"]
    assert switched["point_count"] == original["point_count"]
    assert switched["observation_count"] == original["observation_count"]
    assert (switched["positions"] == original["positions"]).all()
    assert (switched["translations"] == original["translations"]).all()
    # Loading a .sfmr re-normalizes quaternions (reconstruction.rs:440), so the
    # extra save→load cycle that `switched` goes through can perturb the bytes
    # by a few ULPs even though the transform itself never touches poses.
    assert np.allclose(
        switched["quaternions_wxyz"], original["quaternions_wxyz"], rtol=0, atol=1e-12
    )


def test_switch_from_equidistant_fisheye_source(seoul_bull_sfmr_only, tmp_path):
    """An sfmtool-native model works as a SOURCE without being a target.

    The transform reads the source's parameters generically (`to_dict`), so
    `EQUIDISTANT_FISHEYE`'s focal and principal point carry into any COLMAP
    target and the target's extra distortion terms initialize to zero.
    """
    from sfmtool._sfmtool.geometry import CameraIntrinsics

    source = SfmrReconstruction.load(seoul_bull_sfmr_only)
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
    staged = tmp_path / "equidistant.sfmr"
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
        # The carrier's k initializes to the zero the native model implies.
        assert p["radial_distortion_k1"] == 0.0


def test_native_models_are_not_switch_targets():
    """`--camera-model` and the switch target vocabulary are COLMAP-only.

    Both read `_CAMERA_PARAM_NAMES`, which feeds `pycolmap.CameraModelId`;
    sfmtool's own models (`EQUIRECTANGULAR`, `EQUIDISTANT_FISHEYE`) are
    deliberately absent, and a native model is produced by the pipeline that
    solves it, not by asking for a model switch.
    """
    for native in ("EQUIRECTANGULAR", "EQUIDISTANT_FISHEYE"):
        with pytest.raises(ValueError, match="Unknown camera model"):
            SwitchCameraModelTransform(native)
