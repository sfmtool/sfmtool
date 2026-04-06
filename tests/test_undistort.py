# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for image undistortion functionality."""

import json
from pathlib import Path

import numpy as np
import pycolmap
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
        assert "image_sizes" in metadata
        assert len(metadata["cameras"]) > 0
        assert len(metadata["image_sizes"]) == image_count

        # Verify all cameras are PINHOLE (distortion removed)
        for cam in metadata["cameras"]:
            assert cam["model"] == "PINHOLE", (
                f"Expected PINHOLE model after undistortion, got {cam['model']}"
            )
            assert "width" in cam
            assert "height" in cam
            assert len(cam["parameters"]) == 4  # fx, fy, cx, cy only

        # Verify image files were created and are valid
        for image_name in recon.image_names:
            undistorted_path = output_dir_path / image_name
            assert undistorted_path.exists(), f"Missing undistorted image: {image_name}"

            bitmap = pycolmap.Bitmap.read(str(undistorted_path), as_rgb=True)
            assert bitmap is not None, f"Failed to load undistorted image: {image_name}"

            array = bitmap.to_array()
            assert array.shape[0] > 0 and array.shape[1] > 0

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
        image_names = recon.image_names

        for i, size_info in enumerate(metadata["image_sizes"]):
            original_image = workspace_dir / image_names[i]
            if original_image.exists():
                orig_bitmap = pycolmap.Bitmap.read(str(original_image), as_rgb=True)
                orig_array = orig_bitmap.to_array()
                orig_height, orig_width = orig_array.shape[:2]

                undist_width = size_info["width"]
                undist_height = size_info["height"]

                # With min_scale=1.0 and max_scale=1.0, dimensions should be very close
                assert abs(undist_width - orig_width) <= orig_width * 0.2
                assert abs(undist_height - orig_height) <= orig_height * 0.2

    def test_actually_modifies_images(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Test that undistortion actually modifies pixel values for distorted cameras."""
        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(sfmr_path)
        output_dir = tmp_path / "undistorted"

        # Check if any camera has distortion parameters
        cameras = recon.cameras
        has_distortion = any(
            cam.model not in ["PINHOLE", "SIMPLE_PINHOLE"] for cam in cameras
        )

        if not has_distortion:
            pytest.skip("Test reconstruction has no lens distortion")

        _, output_dir_str = undistort_reconstruction_images(
            recon=recon, output_dir=output_dir
        )
        output_dir_path = Path(output_dir_str)

        workspace_dir = Path(recon.workspace_dir)
        image_name = recon.image_names[0]

        orig_bitmap = pycolmap.Bitmap.read(str(workspace_dir / image_name), as_rgb=True)
        undist_bitmap = pycolmap.Bitmap.read(
            str(output_dir_path / image_name), as_rgb=True
        )

        orig_array = orig_bitmap.to_array()
        undist_array = undist_bitmap.to_array()

        assert orig_array.shape == undist_array.shape

        pixel_diff = np.mean(
            np.abs(orig_array.astype(float) - undist_array.astype(float))
        )
        assert pixel_diff > 0.1, "Expected pixel differences after undistortion"


# =============================================================================
# E2E CLI test
# =============================================================================


class TestUndistortE2E:
    def test_undistort_command(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Run the undistort command on a real reconstruction."""
        sfmr_path = sfmrfile_reconstruction_with_17_images

        # Copy sfmr to tmp_path so output goes there
        import shutil

        local_sfmr = tmp_path / "test.sfmr"
        shutil.copy(sfmr_path, local_sfmr)

        # We also need the workspace to be accessible — the sfmr contains
        # workspace_dir pointing to the fixture's workspace. Copy that too.
        recon = SfmrReconstruction.load(sfmr_path)
        workspace_dir = Path(recon.workspace_dir)
        local_workspace = tmp_path / "workspace"
        shutil.copytree(workspace_dir, local_workspace)

        # Re-save the sfmr with updated workspace_dir
        # Actually, let's just run against the fixture's sfmr directly
        runner = CliRunner()
        result = runner.invoke(main, ["undistort", str(sfmr_path)])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert "Successfully undistorted" in result.output

        # Clean up the output directory created next to the fixture sfmr
        output_dir = sfmr_path.parent / f"{sfmr_path.stem}_undistorted"
        if output_dir.exists():
            shutil.rmtree(output_dir)
