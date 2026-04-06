# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for pano2rig and insv2rig modules."""

import json

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._pano2rig import (
    _cubemap_rotations,
    build_rig_frame_data,
    convert_panoramas,
    default_face_size,
    extract_perspective_face,
    find_panorama_images,
    generate_rig_config_json,
    write_rig_config,
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
# Rig config JSON tests
# =============================================================================


class TestRigConfigJson:
    def test_cubemap_config(self):
        names = ["front", "right", "back", "left", "top", "bottom"]
        config = generate_rig_config_json(names)
        assert len(config["cameras"]) == 6
        assert config["cameras"][0]["ref_sensor"] is True
        assert config["cameras"][0]["image_prefix"] == "front/"
        assert config["cameras"][1]["ref_sensor"] is False

    def test_prefix_base(self):
        names = ["front", "right"]
        config = generate_rig_config_json(names, prefix_base="images/seq1/")
        assert config["cameras"][0]["image_prefix"] == "images/seq1/front/"
        assert config["cameras"][1]["image_prefix"] == "images/seq1/right/"

    def test_camera_intrinsics(self):
        names = ["front", "right"]
        intrinsics = {"model": "PINHOLE"}
        config = generate_rig_config_json(names, camera_intrinsics=intrinsics)
        assert config["camera_intrinsics"] == {"model": "PINHOLE"}
        assert len(config["cameras"]) == 2

    def test_no_camera_intrinsics_by_default(self):
        names = ["front", "right"]
        config = generate_rig_config_json(names)
        assert "camera_intrinsics" not in config


# =============================================================================
# Rig frame data tests
# =============================================================================


class TestBuildRigFrameData:
    def test_basic_structure(self):
        names = ["front", "right", "back", "left", "top", "bottom"]
        rotations = _cubemap_rotations()
        data = build_rig_frame_data(names, rotations, num_panoramas=5)

        assert data["rigs_metadata"]["rig_count"] == 1
        assert data["rigs_metadata"]["sensor_count"] == 6
        assert data["rigs_metadata"]["rigs"][0]["name"] == "pano2rig"
        assert data["rigs_metadata"]["rigs"][0]["ref_sensor_name"] == "front"

        assert data["sensor_camera_indexes"].shape == (6,)
        assert data["sensor_quaternions_wxyz"].shape == (6, 4)
        assert data["sensor_translations_xyz"].shape == (6, 3)

        assert data["frames_metadata"]["frame_count"] == 5
        assert data["rig_indexes"].shape == (5,)
        assert data["image_sensor_indexes"].shape == (30,)
        assert data["image_frame_indexes"].shape == (30,)

    def test_front_sensor_is_identity(self):
        names = ["front", "right"]
        rotations = _cubemap_rotations()[:2]
        data = build_rig_frame_data(names, rotations, num_panoramas=1)

        np.testing.assert_allclose(
            data["sensor_quaternions_wxyz"][0], [1, 0, 0, 0], atol=1e-14
        )

    def test_translations_are_zero(self):
        names = ["front", "right"]
        rotations = _cubemap_rotations()[:2]
        data = build_rig_frame_data(names, rotations, num_panoramas=3)

        np.testing.assert_allclose(data["sensor_translations_xyz"], 0)

    def test_image_mapping(self):
        names = ["a", "b", "c"]
        rotations = _cubemap_rotations()[:3]
        data = build_rig_frame_data(names, rotations, num_panoramas=2)

        # 3 sensors * 2 panos = 6 images
        np.testing.assert_array_equal(data["image_sensor_indexes"], [0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(data["image_frame_indexes"], [0, 1, 0, 1, 0, 1])


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

        face_img = cv2.imread(str(output_dir / "front" / "pano_000.jpg"))
        assert face_img.shape[:2] == (64, 64)

    def test_empty_input_raises(self, tmp_path):
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="No panorama images found"):
            convert_panoramas(input_dir, output_dir)


# =============================================================================
# Write rig config tests
# =============================================================================


class TestWriteRigConfig:
    def test_creates_rig_config(self, tmp_path):
        from sfmtool._workspace import init_workspace

        init_workspace(tmp_path)

        face_names = ["front", "right", "back", "left", "top", "bottom"]
        write_rig_config(tmp_path, face_names)

        rig_config_path = tmp_path / "rig_config.json"
        assert rig_config_path.exists()
        rig_config = json.loads(rig_config_path.read_text())
        assert len(rig_config) == 1
        assert len(rig_config[0]["cameras"]) == 6

    def test_workspace_relative_prefixes(self, tmp_path):
        from sfmtool._workspace import init_workspace

        init_workspace(tmp_path)

        output_dir = tmp_path / "images"
        output_dir.mkdir()
        face_names = ["front", "right", "back", "left", "top", "bottom"]
        write_rig_config(output_dir, face_names)

        rig_config = json.loads((tmp_path / "rig_config.json").read_text())
        assert rig_config[0]["cameras"][0]["image_prefix"] == "images/front/"

    def test_appends_to_existing_rig_config(self, tmp_path):
        from sfmtool._workspace import init_workspace

        init_workspace(tmp_path)

        face_names_1 = ["front", "right", "back", "left", "top", "bottom"]
        output_dir_1 = tmp_path / "images" / "pano1"
        output_dir_1.mkdir(parents=True)
        write_rig_config(output_dir_1, face_names_1)

        rig_config = json.loads((tmp_path / "rig_config.json").read_text())
        assert len(rig_config) == 1

        face_names_2 = ["a", "b", "c"]
        output_dir_2 = tmp_path / "images" / "pano2"
        output_dir_2.mkdir(parents=True)
        write_rig_config(output_dir_2, face_names_2)

        rig_config = json.loads((tmp_path / "rig_config.json").read_text())
        assert len(rig_config) == 2
        assert len(rig_config[0]["cameras"]) == 6
        assert len(rig_config[1]["cameras"]) == 3
        assert rig_config[1]["cameras"][0]["image_prefix"] == "images/pano2/a/"


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
        assert (workspace_dir / "rig_config.json").exists()
        assert (output_dir / "front").is_dir()


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
