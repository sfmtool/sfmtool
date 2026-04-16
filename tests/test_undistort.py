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
from sfmtool._sift_file import SiftReader
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
# Helper: run undistort and return all outputs
# =============================================================================


def _run_undistort(sfmr_path, tmp_path, fit="inside", **kwargs):
    """Run undistort and return (recon, output_dir, sfmr_path, undistorted_recon)."""
    from sfmtool._workspace import find_workspace_for_path, load_workspace_config

    recon = SfmrReconstruction.load(sfmr_path)
    output_dir = tmp_path / "undistorted"

    source_workspace_dir = find_workspace_for_path(sfmr_path)
    source_workspace_config = None
    source_sfmr_path = None
    if source_workspace_dir is not None:
        source_workspace_config = load_workspace_config(source_workspace_dir)
        try:
            source_sfmr_path = (
                sfmr_path.resolve()
                .relative_to(source_workspace_dir.resolve())
                .as_posix()
            )
        except ValueError:
            source_sfmr_path = sfmr_path.name

    image_count, output_dir_str, sfmr_out_path = undistort_reconstruction_images(
        recon=recon,
        output_dir=output_dir,
        fit=fit,
        source_workspace_config=source_workspace_config,
        source_sfmr_path=source_sfmr_path,
        **kwargs,
    )

    output_dir_path = Path(output_dir_str)
    undistorted_recon = None
    if sfmr_out_path:
        undistorted_recon = SfmrReconstruction.load(Path(sfmr_out_path))

    return recon, output_dir_path, sfmr_out_path, undistorted_recon


# =============================================================================
# Core undistortion tests (updated for 3-tuple return)
# =============================================================================


class TestUndistortReconstructionImages:
    def test_basic_undistortion(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Test undistorting images from a reconstruction."""
        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(sfmr_path)
        output_dir = tmp_path / "undistorted"

        image_count, output_dir_str, sfmr_out_path = undistort_reconstruction_images(
            recon=recon, output_dir=output_dir
        )

        output_dir_path = Path(output_dir_str)
        assert output_dir_path.exists()
        assert image_count > 0

        # Verify .sfmr was created and is loadable
        assert sfmr_out_path is not None
        undist_recon = SfmrReconstruction.load(Path(sfmr_out_path))

        assert undist_recon.camera_count > 0
        assert undist_recon.image_count == image_count

        # Verify all cameras are PINHOLE (distortion removed)
        for cam in undist_recon.cameras:
            assert cam.model == "PINHOLE", (
                f"Expected PINHOLE model after undistortion, got {cam.model}"
            )

        # Verify camera indexes are valid
        cam_idxs = np.asarray(undist_recon.camera_indexes)
        assert np.all(cam_idxs < undist_recon.camera_count)

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

        image_count, output_dir_str, sfmr_out_path = undistort_reconstruction_images(
            recon=recon, output_dir=output_dir, fit="outside"
        )

        assert image_count > 0
        assert sfmr_out_path is not None

        undist_recon = SfmrReconstruction.load(Path(sfmr_out_path))
        for cam in undist_recon.cameras:
            assert cam.model == "PINHOLE"

    def test_progress_callback(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Test that progress callback is called correctly."""
        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon = SfmrReconstruction.load(sfmr_path)
        output_dir = tmp_path / "undistorted"

        progress_calls = []

        def progress_callback(current, total, image_name):
            progress_calls.append((current, total, image_name))

        image_count, _, _ = undistort_reconstruction_images(
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

        _, output_dir_str, sfmr_out_path = undistort_reconstruction_images(
            recon=recon, output_dir=output_dir
        )

        undist_recon = SfmrReconstruction.load(Path(sfmr_out_path))
        workspace_dir = Path(recon.workspace_dir)
        cam_idxs = np.asarray(undist_recon.camera_indexes)
        cameras = undist_recon.cameras

        for i, image_name in enumerate(undist_recon.image_names):
            cam = cameras[cam_idxs[i]]
            original_image = workspace_dir / image_name
            if original_image.exists():
                orig = cv2.imread(str(original_image))
                orig_height, orig_width = orig.shape[:2]

                # Dimensions should match since we pass the source dimensions
                assert cam.width == orig_width
                assert cam.height == orig_height

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

        _, output_dir_str, _ = undistort_reconstruction_images(
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

        _, _, inside_sfmr_path = undistort_reconstruction_images(
            recon=recon, output_dir=inside_dir, fit="inside"
        )
        _, _, outside_sfmr_path = undistort_reconstruction_images(
            recon=recon, output_dir=outside_dir, fit="outside"
        )

        inside_recon = SfmrReconstruction.load(Path(inside_sfmr_path))
        outside_recon = SfmrReconstruction.load(Path(outside_sfmr_path))

        inside_cam = inside_recon.cameras[0].to_dict()
        outside_cam = outside_recon.cameras[0].to_dict()

        inside_fx = inside_cam["parameters"]["focal_length_x"]
        outside_fx = outside_cam["parameters"]["focal_length_x"]

        # They should differ for distorted cameras
        assert inside_fx != outside_fx


# =============================================================================
# New workspace / .sfmr / .sift tests
# =============================================================================


class TestUndistortWorkspace:
    def test_workspace_created(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Workspace config exists with correct feature_tool."""
        _, output_dir, _, _ = _run_undistort(
            sfmrfile_reconstruction_with_17_images, tmp_path
        )
        config_path = output_dir / ".sfm-workspace.json"
        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)

        assert config["feature_tool"] == "sfmtool-undistort"
        assert config["version"] == 1
        assert "feature_prefix_dir" in config
        assert config["feature_type"] == "sift-sfmtool-undistort"

    def test_sfmr_output(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """.sfmr exists, is loadable, has pinhole cameras and same image count."""
        recon, output_dir, sfmr_path, undistorted_recon = _run_undistort(
            sfmrfile_reconstruction_with_17_images, tmp_path
        )

        assert sfmr_path is not None
        sfmr_file = Path(sfmr_path)
        assert sfmr_file.exists()
        assert sfmr_file.name == "undistorted.sfmr"
        assert sfmr_file.parent.name == "sfmr"

        assert undistorted_recon is not None
        assert undistorted_recon.image_count == recon.image_count

        # All cameras should be PINHOLE
        for cam in undistorted_recon.cameras:
            assert cam.model == "PINHOLE"

    def test_sift_files_created(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """One .sift file per image, readable."""
        recon, output_dir, _, undistorted_recon = _run_undistort(
            sfmrfile_reconstruction_with_17_images, tmp_path
        )

        with open(output_dir / ".sfm-workspace.json") as f:
            config = json.load(f)

        feature_prefix_dir = config["feature_prefix_dir"]

        for image_name in recon.image_names:
            image_basename = Path(image_name).name
            sift_path = (
                output_dir
                / Path(image_name).parent
                / feature_prefix_dir
                / (image_basename + ".sift")
            )
            assert sift_path.exists(), f"Missing .sift file: {sift_path}"

            with SiftReader(sift_path) as reader:
                positions = reader.read_positions()
                assert positions.dtype == np.float32
                assert positions.ndim == 2
                assert positions.shape[1] == 2

    def test_feature_positions_in_bounds(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """All feature positions are within pinhole image dimensions."""
        recon, output_dir, _, undistorted_recon = _run_undistort(
            sfmrfile_reconstruction_with_17_images, tmp_path
        )

        with open(output_dir / ".sfm-workspace.json") as f:
            config = json.load(f)

        feature_prefix_dir = config["feature_prefix_dir"]

        for i, image_name in enumerate(recon.image_names):
            cam = undistorted_recon.cameras[undistorted_recon.camera_indexes[i]]
            image_basename = Path(image_name).name
            sift_path = (
                output_dir
                / Path(image_name).parent
                / feature_prefix_dir
                / (image_basename + ".sift")
            )

            with SiftReader(sift_path) as reader:
                positions = reader.read_positions()

            if len(positions) > 0:
                assert np.all(positions[:, 0] >= 0)
                assert np.all(positions[:, 0] < cam.width)
                assert np.all(positions[:, 1] >= 0)
                assert np.all(positions[:, 1] < cam.height)

    def test_affine_shapes_transformed(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Affine shapes differ from source (transformation applied)."""
        from sfmtool._workspace import find_workspace_for_path, load_workspace_config

        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon, output_dir, _, _ = _run_undistort(sfmr_path, tmp_path)

        source_workspace_dir = find_workspace_for_path(sfmr_path)
        source_config = load_workspace_config(source_workspace_dir)

        with open(output_dir / ".sfm-workspace.json") as f:
            undist_config = json.load(f)

        # Compare first image's affine shapes
        image_name = recon.image_names[0]
        image_path = Path(recon.workspace_dir) / image_name

        with SiftReader.for_image(
            image_path,
            feature_tool=source_config["feature_tool"],
            feature_options=source_config["feature_options"],
        ) as reader:
            src_shapes = reader.read_affine_shapes()

        image_basename = Path(image_name).name
        undist_sift_path = (
            output_dir
            / Path(image_name).parent
            / undist_config["feature_prefix_dir"]
            / (image_basename + ".sift")
        )
        with SiftReader(undist_sift_path) as reader:
            undist_shapes = reader.read_affine_shapes()

        # If distortion exists, shapes should differ
        cameras = recon.cameras
        has_distortion = any(cam.has_distortion for cam in cameras)
        if has_distortion and len(src_shapes) > 0 and len(undist_shapes) > 0:
            # Shapes won't be directly comparable element-wise (some may be dropped),
            # but at least the first few should differ
            min_len = min(len(src_shapes), len(undist_shapes))
            if min_len > 0:
                assert not np.allclose(
                    src_shapes[:min_len], undist_shapes[:min_len], atol=1e-6
                )

    def test_feature_count_leq_original(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Undistorted features <= original feature count per image."""
        from sfmtool._workspace import find_workspace_for_path, load_workspace_config

        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon, output_dir, _, _ = _run_undistort(sfmr_path, tmp_path)

        source_workspace_dir = find_workspace_for_path(sfmr_path)
        source_config = load_workspace_config(source_workspace_dir)

        with open(output_dir / ".sfm-workspace.json") as f:
            undist_config = json.load(f)

        for image_name in recon.image_names:
            image_path = Path(recon.workspace_dir) / image_name

            with SiftReader.for_image(
                image_path,
                feature_tool=source_config["feature_tool"],
                feature_options=source_config["feature_options"],
            ) as reader:
                src_count = reader.metadata["feature_count"]

            image_basename = Path(image_name).name
            undist_sift_path = (
                output_dir
                / Path(image_name).parent
                / undist_config["feature_prefix_dir"]
                / (image_basename + ".sift")
            )
            with SiftReader(undist_sift_path) as reader:
                undist_count = reader.metadata["feature_count"]

            assert undist_count <= src_count

    def test_descriptors_unchanged(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """Surviving descriptors match source descriptors."""
        from sfmtool._workspace import find_workspace_for_path, load_workspace_config

        sfmr_path = sfmrfile_reconstruction_with_17_images
        recon, output_dir, _, _ = _run_undistort(sfmr_path, tmp_path)

        source_workspace_dir = find_workspace_for_path(sfmr_path)
        source_config = load_workspace_config(source_workspace_dir)

        with open(output_dir / ".sfm-workspace.json") as f:
            undist_config = json.load(f)

        # Check first image
        image_name = recon.image_names[0]
        image_path = Path(recon.workspace_dir) / image_name

        with SiftReader.for_image(
            image_path,
            feature_tool=source_config["feature_tool"],
            feature_options=source_config["feature_options"],
        ) as reader:
            src_descriptors = reader.read_descriptors()

        image_basename = Path(image_name).name
        undist_sift_path = (
            output_dir
            / Path(image_name).parent
            / undist_config["feature_prefix_dir"]
            / (image_basename + ".sift")
        )
        with SiftReader(undist_sift_path) as reader:
            undist_descriptors = reader.read_descriptors()

        # Undistorted descriptors should be a subset of source
        # Since features are filtered but order is preserved, each undistorted
        # descriptor should match the corresponding source descriptor
        assert len(undist_descriptors) <= len(src_descriptors)
        if len(undist_descriptors) > 0:
            # The first undistorted descriptor should match some source descriptor
            # (it's the first kept feature from the source)
            found = False
            for src_desc in src_descriptors:
                if np.array_equal(undist_descriptors[0], src_desc):
                    found = True
                    break
            assert found, "First undistorted descriptor not found in source"

    def test_tracks_valid(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Track feature indexes are valid, observation counts sum matches."""
        recon, output_dir, _, undistorted_recon = _run_undistort(
            sfmrfile_reconstruction_with_17_images, tmp_path
        )

        track_img_idxs = np.asarray(undistorted_recon.track_image_indexes)
        track_pt3d_idxs = np.asarray(undistorted_recon.track_point_ids)
        obs_counts = np.asarray(undistorted_recon.observation_counts)

        # All image indexes valid
        assert np.all(track_img_idxs < undistorted_recon.image_count)

        # All point indexes valid
        assert np.all(track_pt3d_idxs < undistorted_recon.point_count)

        # observation_counts sum == total observations
        assert int(obs_counts.sum()) == len(track_img_idxs)

        # Each observation count matches actual count per point
        computed_counts = np.bincount(
            track_pt3d_idxs, minlength=undistorted_recon.point_count
        )
        np.testing.assert_array_equal(obs_counts, computed_counts)

    def test_no_orphan_points(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Every 3D point has at least one observation."""
        _, _, _, undistorted_recon = _run_undistort(
            sfmrfile_reconstruction_with_17_images, tmp_path
        )

        obs_counts = np.asarray(undistorted_recon.observation_counts)
        assert np.all(obs_counts > 0), "Found 3D points with zero observations"

    def test_poses_preserved(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Quaternions and translations are unchanged by undistortion."""
        recon, _, _, undistorted_recon = _run_undistort(
            sfmrfile_reconstruction_with_17_images, tmp_path
        )

        np.testing.assert_array_almost_equal(
            np.asarray(recon.quaternions_wxyz),
            np.asarray(undistorted_recon.quaternions_wxyz),
            decimal=10,
        )
        np.testing.assert_array_almost_equal(
            np.asarray(recon.translations),
            np.asarray(undistorted_recon.translations),
            decimal=10,
        )

    def test_3d_points_subset(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Output 3D points are a subset of input 3D points."""
        recon, _, _, undistorted_recon = _run_undistort(
            sfmrfile_reconstruction_with_17_images, tmp_path
        )

        assert undistorted_recon.point_count <= recon.point_count

        # Each output point position should exist in the input
        src_positions = np.asarray(recon.positions)
        dst_positions = np.asarray(undistorted_recon.positions)

        for pos in dst_positions:
            dists = np.linalg.norm(src_positions - pos, axis=1)
            assert np.min(dists) < 1e-10, f"Output point {pos} not found in input"

    def test_thumbnail_128x128(self, sfmrfile_reconstruction_with_17_images, tmp_path):
        """Thumbnails in .sift files are 128x128x3."""
        recon, output_dir, _, _ = _run_undistort(
            sfmrfile_reconstruction_with_17_images, tmp_path
        )

        with open(output_dir / ".sfm-workspace.json") as f:
            config = json.load(f)

        feature_prefix_dir = config["feature_prefix_dir"]
        image_name = recon.image_names[0]
        image_basename = Path(image_name).name
        sift_path = (
            output_dir
            / Path(image_name).parent
            / feature_prefix_dir
            / (image_basename + ".sift")
        )

        with SiftReader(sift_path) as reader:
            thumbnail = reader.read_thumbnail()

        assert thumbnail.shape == (128, 128, 3)
        assert thumbnail.dtype == np.uint8

    def test_camera_indexes_valid(
        self, sfmrfile_reconstruction_with_17_images, tmp_path
    ):
        """All camera_indexes reference valid cameras."""
        _, _, _, undistorted_recon = _run_undistort(
            sfmrfile_reconstruction_with_17_images, tmp_path
        )

        cam_indexes = np.asarray(undistorted_recon.camera_indexes)
        num_cameras = undistorted_recon.camera_count
        assert np.all(cam_indexes < num_cameras)


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
