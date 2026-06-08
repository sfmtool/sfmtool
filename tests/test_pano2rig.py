# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for pano2rig and insv2rig modules."""

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.rig.pano2rig import (
    _cubemap_rotations,
    convert_panoramas,
    default_face_size,
    extract_perspective_face,
    find_panorama_images,
    write_pano_camrig,
)
from sfmtool.cli import main


# =============================================================================
# Helpers
# =============================================================================


def _make_test_equirect(width=400, height=200):
    """Create a synthetic equirectangular image with a gradient pattern."""
    u = np.linspace(0, 255, width, dtype=np.uint8)
    v = np.linspace(0, 255, height, dtype=np.uint8)
    h_grad = np.tile(u, (height, 1))
    v_grad = np.tile(v[:, np.newaxis], (1, width))
    return np.stack([h_grad, v_grad, np.full_like(h_grad, 128)], axis=-1)


# =============================================================================
# Rotation tests
# =============================================================================


class TestRotations:
    def test_cubemap_has_6_faces(self):
        rotations = _cubemap_rotations()
        assert len(rotations) == 6

    def test_cubemap_front_is_identity(self):
        rotations = _cubemap_rotations()
        R = rotations[0].to_rotation_matrix()
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_cubemap_rotations_are_orthogonal(self):
        for q in _cubemap_rotations():
            R = q.to_rotation_matrix()
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)

    def test_cubemap_axes_are_orthogonal_pairs(self):
        """Front/back, left/right, top/bottom should have opposite optical axes."""
        rotations = _cubemap_rotations()
        axes = [q.to_rotation_matrix().T @ np.array([0, 0, 1.0]) for q in rotations]
        # front (+Z) and back (-Z)
        np.testing.assert_allclose(axes[0] + axes[2], 0, atol=1e-14)
        # right (+X) and left (-X)
        np.testing.assert_allclose(axes[1] + axes[3], 0, atol=1e-14)
        # top (+Y) and bottom (-Y)
        np.testing.assert_allclose(axes[4] + axes[5], 0, atol=1e-14)


# =============================================================================
# Face extraction tests
# =============================================================================


class TestFaceExtraction:
    def test_face_shape(self):
        equirect = _make_test_equirect(400, 200)
        rotations = _cubemap_rotations()
        face = extract_perspective_face(equirect, rotations[0], 100)
        assert face.shape == (100, 100, 3)

    def test_face_not_all_black(self):
        equirect = _make_test_equirect(400, 200)
        rotations = _cubemap_rotations()
        for R in rotations:
            face = extract_perspective_face(equirect, R, 100)
            assert face.mean() > 0, "Face should not be all black"

    def test_front_face_center_matches_equirect_center(self):
        width, height = 400, 200
        equirect = np.zeros((height, width, 3), dtype=np.uint8)
        cx, cy = width // 2, height // 2
        equirect[cy - 5 : cy + 5, cx - 5 : cx + 5] = 255

        rotations = _cubemap_rotations()
        face_size = 100
        face = extract_perspective_face(equirect, rotations[0], face_size)

        fc = face_size // 2
        center_brightness = face[fc, fc].mean()
        assert center_brightness > 200, (
            f"Center should be bright, got {center_brightness}"
        )

    def test_grayscale_input(self):
        equirect = np.random.randint(0, 256, (200, 400), dtype=np.uint8)
        rotations = _cubemap_rotations()
        face = extract_perspective_face(equirect, rotations[0], 50)
        assert face.shape == (50, 50), "Grayscale face should be 2D"

    def test_different_face_sizes(self):
        equirect = _make_test_equirect(800, 400)
        R = _cubemap_rotations()[0]
        for size in [32, 64, 128, 256]:
            face = extract_perspective_face(equirect, R, size)
            assert face.shape[:2] == (size, size)


# =============================================================================
# Default face size tests
# =============================================================================


class TestDefaultFaceSize:
    def test_standard_panorama(self):
        assert default_face_size(4000) == 1000

    def test_small_panorama(self):
        assert default_face_size(800) == 200


# =============================================================================
# Find panorama images tests
# =============================================================================


class TestFindPanoramaImages:
    def test_finds_images(self, tmp_path):
        (tmp_path / "pano_01.jpg").write_bytes(b"fake")
        (tmp_path / "pano_02.png").write_bytes(b"fake")
        (tmp_path / "readme.txt").write_text("not an image")
        images = find_panorama_images(tmp_path)
        assert len(images) == 2
        assert images[0].name == "pano_01.jpg"
        assert images[1].name == "pano_02.png"

    def test_empty_dir(self, tmp_path):
        images = find_panorama_images(tmp_path)
        assert len(images) == 0


# =============================================================================
# Integration: convert_panoramas
# =============================================================================


class TestConvertPanoramas:
    def _create_test_panoramas(self, tmp_path, count=2, width=400, height=200):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for i in range(count):
            img = _make_test_equirect(width, height)
            cv2.imwrite(str(input_dir / f"pano_{i:03d}.jpg"), img)
        return input_dir

    def test_cubemap_conversion(self, tmp_path):
        input_dir = self._create_test_panoramas(tmp_path)
        output_dir = tmp_path / "output"

        num_panos, face_size, face_names = convert_panoramas(input_dir, output_dir)

        assert num_panos == 2
        assert face_size == 100  # 400 / 4
        assert len(face_names) == 6

        for name in face_names:
            face_dir = output_dir / name
            assert face_dir.is_dir()
            images = list(face_dir.glob("*.jpg"))
            assert len(images) == 2

    def test_custom_face_size(self, tmp_path):
        input_dir = self._create_test_panoramas(tmp_path)
        output_dir = tmp_path / "output"

        _, face_size, _ = convert_panoramas(input_dir, output_dir, face_size=64)
        assert face_size == 64

        face_img = cv2.imread(str(output_dir / "front" / "frame_000000.jpg"))
        assert face_img.shape[:2] == (64, 64)

    def test_empty_input_raises(self, tmp_path):
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="No panorama images found"):
            convert_panoramas(input_dir, output_dir)


# =============================================================================
# Write cubemap .camrig tests
# =============================================================================


class TestPano2rigCamrig:
    """`write_pano_camrig` builds the six-face cubemap `.camrig`."""

    _FACE_NAMES = ["front", "right", "back", "left", "top", "bottom"]

    def test_writes_a_valid_cubemap_rig(self, tmp_path):
        from sfmtool._sfmtool import read_camrig, verify_camrig

        camrig_path = tmp_path / "cubemap.camrig"
        write_pano_camrig(
            camrig_path,
            rig_name="cubemap",
            face_names=self._FACE_NAMES,
            rotations=_cubemap_rotations(),
            face_size=512,
        )
        assert camrig_path.exists()
        valid, errors = verify_camrig(str(camrig_path))
        assert valid, errors

        data = read_camrig(str(camrig_path))
        meta = data["metadata"]
        assert meta["rig_type"] == "cubemap"
        assert meta["sensor_count"] == 6
        assert meta["camera_count"] == 1
        assert meta["rig_attributes"] == {}
        assert data["sensor_image_patterns"] == [
            f"{name}/frame_%06d.jpg" for name in self._FACE_NAMES
        ]

    def test_camera_is_a_square_90deg_pinhole(self, tmp_path):
        from sfmtool._sfmtool import read_camrig

        camrig_path = tmp_path / "cubemap.camrig"
        write_pano_camrig(
            camrig_path,
            rig_name="cubemap",
            face_names=self._FACE_NAMES,
            rotations=_cubemap_rotations(),
            face_size=512,
        )
        cam = read_camrig(str(camrig_path))["cameras"][0]
        assert cam["model"] == "PINHOLE"
        assert cam["width"] == 512 and cam["height"] == 512
        # 90° FOV over a 512-px square: fx = fy = 256, principal point at centre.
        assert cam["parameters"]["focal_length_x"] == 256.0
        assert cam["parameters"]["focal_length_y"] == 256.0
        assert cam["parameters"]["principal_point_x"] == 256.0
        assert cam["parameters"]["principal_point_y"] == 256.0

    def test_cocentric_zero_translations(self, tmp_path):
        from sfmtool._sfmtool import read_camrig

        camrig_path = tmp_path / "cubemap.camrig"
        write_pano_camrig(
            camrig_path,
            rig_name="cubemap",
            face_names=self._FACE_NAMES,
            rotations=_cubemap_rotations(),
            face_size=256,
        )
        data = read_camrig(str(camrig_path))
        np.testing.assert_array_equal(data["translations_xyz"], 0)
        # Front (sensor 0) sits at the identity sensor_from_rig pose.
        np.testing.assert_allclose(
            data["quaternions_wxyz"][0], [1, 0, 0, 0], atol=1e-14
        )


# =============================================================================
# CLI tests
# =============================================================================


class TestPano2rigCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["pano2rig", "--help"])
        assert result.exit_code == 0
        assert "equirectangular" in result.output.lower()

    def test_basic_invocation(self, tmp_path):
        from sfmtool._workspace import init_workspace

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        img = _make_test_equirect(400, 200)
        cv2.imwrite(str(input_dir / "test_pano.jpg"), img)

        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()
        init_workspace(workspace_dir)

        output_dir = workspace_dir / "images"

        runner = CliRunner()
        result = runner.invoke(
            main, ["pano2rig", str(input_dir), "-o", str(output_dir)]
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert (workspace_dir / ".sfm-workspace.json").exists()
        assert (output_dir / "cubemap.camrig").exists()
        assert (output_dir / "front").is_dir()
        assert (output_dir / "front" / "frame_000000.jpg").exists()


class TestInsv2rigCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["insv2rig", "--help"])
        assert result.exit_code == 0
        assert "insta360" in result.output.lower()

    def test_output_required(self, tmp_path):
        fake_insv = tmp_path / "test.insv"
        fake_insv.write_bytes(b"fake")
        runner = CliRunner()
        result = runner.invoke(main, ["insv2rig", str(fake_insv)])
        assert result.exit_code != 0


class TestInsv2rigCamrig:
    """`write_insv_camrig` builds the back-to-back fisheye `.camrig`."""

    def test_writes_a_valid_fisheye_360_rig(self, tmp_path):
        from sfmtool.rig.insv2rig import write_insv_camrig
        from sfmtool._sfmtool import read_camrig, verify_camrig

        camrig_path = tmp_path / "recording.camrig"
        write_insv_camrig(
            camrig_path,
            rig_name="insv2_x5",
            sensor_names=["fisheye_left", "fisheye_right"],
            frame_pattern="frame_%06d.jpg",
            camera_model="OPENCV_FISHEYE",
            camera_params=[1031.7, 1029.7, 1920.0, 1920.0, 0.042, -0.011, 0.01, -0.003],
            width=3840,
            height=3840,
            quaternions_wxyz=[[1, 0, 0, 0], [0, 0, 1, 0]],
            translations_xyz=[[0, 0, 0], [0, 0, -0.0307]],
            baseline_m=0.0307,
        )
        assert camrig_path.exists()
        valid, errors = verify_camrig(str(camrig_path))
        assert valid, errors

        data = read_camrig(str(camrig_path))
        meta = data["metadata"]
        assert meta["rig_type"] == "fisheye_360"
        assert meta["sensor_count"] == 2
        assert meta["camera_count"] == 1
        assert meta["rig_attributes"] == {"baseline_m": 0.0307}
        assert data["sensor_image_patterns"] == [
            "fisheye_left/frame_%06d.jpg",
            "fisheye_right/frame_%06d.jpg",
        ]
        cam = data["cameras"][0]
        assert cam["model"] == "OPENCV_FISHEYE"
        assert cam["width"] == 3840 and cam["height"] == 3840
        np.testing.assert_allclose(data["quaternions_wxyz"][1], [0, 0, 1, 0])
        np.testing.assert_allclose(data["translations_xyz"][1], [0, 0, -0.0307])

    def test_rejects_wrong_param_count(self, tmp_path):
        from sfmtool.rig.insv2rig import write_insv_camrig

        with pytest.raises(ValueError, match="expects 8 values"):
            write_insv_camrig(
                tmp_path / "bad.camrig",
                rig_name="insv2_x5",
                sensor_names=["fisheye_left", "fisheye_right"],
                frame_pattern="frame_%06d.jpg",
                camera_model="OPENCV_FISHEYE",
                camera_params=[1.0, 2.0, 3.0],
                width=3840,
                height=3840,
                quaternions_wxyz=[[1, 0, 0, 0], [0, 0, 1, 0]],
                translations_xyz=[[0, 0, 0], [0, 0, -0.0307]],
                baseline_m=0.0307,
            )
