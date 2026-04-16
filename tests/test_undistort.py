# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for image undistortion functionality."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._sfmtool import SfmrReconstruction
from sfmtool._undistort_images import undistort_reconstruction_images
from sfmtool.cli import main


# =============================================================================
# CLI tests
# =============================================================================


class TestUndistortCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["undistort", "--help"])
        assert result.exit_code == 0
        assert "Undistort all images" in result.output
        assert "--fit" in result.output

    def test_non_sfmr_rejected(self, tmp_path):
        recon = tmp_path / "recon.txt"
        recon.write_bytes(b"fake")
        runner = CliRunner()
        result = runner.invoke(main, ["undistort", str(recon)])
        assert result.exit_code != 0
        assert ".sfmr" in result.output

    def test_missing_file_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["undistort", str(tmp_path / "nonexistent.sfmr")])
        assert result.exit_code != 0


# =============================================================================
# Core undistortion tests
# =============================================================================


class TestUndistortReconstructionImages:
    def test_basic_undistortion(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Test undistorting images from a reconstruction."""
        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(sfmr_path)
        output_dir = tmp_path / "undistorted"

        image_count, output_dir_str = undistort_reconstruction_images(
            recon=recon, output_dir=output_dir
        )

        output_dir_path = Path(output_dir_str)
        assert output_dir_path.exists()
        assert image_count > 0

        # Verify JSON metadata exists and is valid
        json_path = output_dir_path / "undistorted_cameras.json"
        assert json_path.exists()

        with open(json_path) as f:
            metadata = json.load(f)

        assert "cameras" in metadata
        assert "images" in metadata
        assert len(metadata["cameras"]) > 0
        assert len(metadata["images"]) == image_count

        # Verify all cameras are PINHOLE (distortion removed)
        for cam in metadata["cameras"]:
            assert cam["model"] == "PINHOLE", (
                f"Expected PINHOLE model after undistortion, got {cam['model']}"
            )
            assert "width" in cam
            assert "height" in cam
            assert len(cam["parameters"]) == 4  # fx, fy, cx, cy only

        # Verify image entries reference valid camera indices
        num_cameras = len(metadata["cameras"])
        for img in metadata["images"]:
            assert "name" in img
            assert "camera_index" in img
            assert 0 <= img["camera_index"] < num_cameras

        # Verify image files were created and are valid
        for image_name in recon.image_names:
            undistorted_path = output_dir_path / image_name
            assert undistorted_path.exists(), f"Missing undistorted image: {image_name}"

            img = cv2.imread(str(undistorted_path))
            assert img is not None, f"Failed to load undistorted image: {image_name}"
            assert img.shape[0] > 0 and img.shape[1] > 0

    def test_fit_outside(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Test undistortion with fit=outside produces valid output."""
        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(sfmr_path)
        output_dir = tmp_path / "undistorted_outside"

        image_count, output_dir_str = undistort_reconstruction_images(
            recon=recon, output_dir=output_dir, fit="outside"
        )

        output_dir_path = Path(output_dir_str)
        assert image_count > 0

        json_path = output_dir_path / "undistorted_cameras.json"
        with open(json_path) as f:
            metadata = json.load(f)

        for cam in metadata["cameras"]:
            assert cam["model"] == "PINHOLE"

    def test_progress_callback(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Test that progress callback is called correctly."""
        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(sfmr_path)
        output_dir = tmp_path / "undistorted"

        progress_calls = []

        def progress_callback(current, total, image_name):
            progress_calls.append((current, total, image_name))

        image_count, _ = undistort_reconstruction_images(
            recon=recon, output_dir=output_dir, progress_callback=progress_callback
        )

        # Should be called once per image + final call
        assert len(progress_calls) >= image_count

        # Verify progress increases monotonically
        for i in range(len(progress_calls) - 1):
            current, total, _ = progress_calls[i]
            assert current <= total
            if i > 0:
                prev_current = progress_calls[i - 1][0]
                assert current >= prev_current

    def test_image_dimensions_preserved(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Test that undistorted images maintain reasonable dimensions."""
        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(sfmr_path)
        output_dir = tmp_path / "undistorted"

        _, output_dir_str = undistort_reconstruction_images(
            recon=recon, output_dir=output_dir
        )
        output_dir_path = Path(output_dir_str)

        with open(output_dir_path / "undistorted_cameras.json") as f:
            metadata = json.load(f)

        workspace_dir = Path(recon.workspace_dir)

        for img_entry in metadata["images"]:
            cam = metadata["cameras"][img_entry["camera_index"]]
            original_image = workspace_dir / img_entry["name"]
            if original_image.exists():
                orig = cv2.imread(str(original_image))
                orig_height, orig_width = orig.shape[:2]

                # Dimensions should match since we pass the source dimensions
                assert cam["width"] == orig_width
                assert cam["height"] == orig_height

    def test_actually_modifies_images(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Test that undistortion actually modifies pixel values for distorted cameras."""
        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(sfmr_path)
        output_dir = tmp_path / "undistorted"

        # Check if any camera has distortion parameters
        cameras = recon.cameras
        has_distortion = any(cam.has_distortion for cam in cameras)

        if not has_distortion:
            pytest.skip("Test reconstruction has no lens distortion")

        _, output_dir_str = undistort_reconstruction_images(
            recon=recon, output_dir=output_dir
        )
        output_dir_path = Path(output_dir_str)

        workspace_dir = Path(recon.workspace_dir)
        image_name = recon.image_names[0]

        orig_array = cv2.imread(str(workspace_dir / image_name))
        undist_array = cv2.imread(str(output_dir_path / image_name))

        assert orig_array.shape == undist_array.shape

        pixel_diff = np.mean(
            np.abs(orig_array.astype(float) - undist_array.astype(float))
        )
        assert pixel_diff > 0.1, "Expected pixel differences after undistortion"

    def test_inside_and_outside_produce_different_focals(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Test that inside and outside fit modes produce different focal lengths."""
        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(sfmr_path)

        cameras = recon.cameras
        has_distortion = any(cam.has_distortion for cam in cameras)
        if not has_distortion:
            pytest.skip("Test reconstruction has no lens distortion")

        inside_dir = tmp_path / "inside"
        outside_dir = tmp_path / "outside"

        undistort_reconstruction_images(
            recon=recon, output_dir=inside_dir, fit="inside"
        )
        undistort_reconstruction_images(
            recon=recon, output_dir=outside_dir, fit="outside"
        )

        with open(inside_dir / "undistorted_cameras.json") as f:
            inside_meta = json.load(f)
        with open(outside_dir / "undistorted_cameras.json") as f:
            outside_meta = json.load(f)

        inside_fx = inside_meta["cameras"][0]["parameters"]["focal_length_x"]
        outside_fx = outside_meta["cameras"][0]["parameters"]["focal_length_x"]

        # They should differ for distorted cameras
        assert inside_fx != outside_fx


# =============================================================================
# E2E CLI test
# =============================================================================


class TestUndistortE2E:
    def test_undistort_command(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Run the undistort command on a real reconstruction."""
        sfmr_path = sfmrfile_reconstruction_with_17_images

        runner = CliRunner()
        result = runner.invoke(main, ["undistort", str(sfmr_path)])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Successfully undistorted" in result.output

        # Clean up the output directory created next to the fixture sfmr
        import shutil

        output_dir = sfmr_path.parent / f"{sfmr_path.stem}_undistorted"
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_undistort_command_fit_outside(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Run the undistort command with --fit outside."""
        sfmr_path = sfmrfile_reconstruction_with_17_images

        runner = CliRunner()
        result = runner.invoke(main, ["undistort", str(sfmr_path), "--fit", "outside"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Successfully undistorted" in result.output

        # Clean up
        import shutil

        output_dir = sfmr_path.parent / f"{sfmr_path.stem}_undistorted"
        if output_dir.exists():
            shutil.rmtree(output_dir)
