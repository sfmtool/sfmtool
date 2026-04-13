# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `sfm inspect` CLI command."""

from pathlib import Path

from click.testing import CliRunner

from sfmtool.cli import main


def test_inspect_default_summary(sfmrfile_reconstruction_with_17_images):
    """Default inspect prints summary with all expected sections."""
    sfmr_path = str(sfmrfile_reconstruction_with_17_images)
    result = CliRunner().invoke(main, ["inspect", sfmr_path])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Metadata:" in out
    assert "Workspace:" in out
    assert "Reconstruction summary:" in out
    assert "Images: 17" in out
    assert "Cameras:" in out
    assert "Camera 0:" in out
    assert "Parameter" in out
    assert "Value" in out
    assert "Rig configuration:" in out
    assert "3D Point statistics:" in out
    assert "Position range:" in out
    assert "Reprojection error:" in out
    assert "Observation statistics:" in out
    assert "Nearest neighbor distances:" in out
    assert "Completed in" in out


def test_inspect_coviz(sfmrfile_reconstruction_with_17_images):
    """--coviz prints covisibility graph."""
    sfmr_path = str(sfmrfile_reconstruction_with_17_images)
    result = CliRunner().invoke(main, ["inspect", "--coviz", sfmr_path])
    assert result.exit_code == 0, result.output
    assert "ovisibility" in result.output


def test_inspect_images(sfmrfile_reconstruction_with_17_images):
    """--images prints per-image connectivity table."""
    sfmr_path = str(sfmrfile_reconstruction_with_17_images)
    result = CliRunner().invoke(main, ["inspect", "--images", sfmr_path])
    assert result.exit_code == 0, result.output
    assert "seoul_bull_sculpture" in result.output


def test_inspect_metrics(sfmrfile_reconstruction_with_17_images):
    """--metrics prints per-image metrics table."""
    sfmr_path = str(sfmrfile_reconstruction_with_17_images)
    result = CliRunner().invoke(main, ["inspect", "--metrics", sfmr_path])
    assert result.exit_code == 0, result.output
    assert "MeanErr" in result.output
    assert "seoul_bull_sculpture" in result.output


def test_inspect_metrics_with_range(sfmrfile_reconstruction_with_17_images):
    """--metrics --range filters to subset of images."""
    sfmr_path = str(sfmrfile_reconstruction_with_17_images)
    result = CliRunner().invoke(
        main, ["inspect", "--metrics", "--range", "1-5", sfmr_path]
    )
    assert result.exit_code == 0, result.output
    assert "5 of 17 images" in result.output


def test_inspect_non_sfmr_file(tmp_path: Path):
    """Passing a non-.sfmr file raises an error."""
    p = tmp_path / "input.txt"
    p.touch()
    result = CliRunner().invoke(main, ["inspect", str(p)])
    assert result.exit_code != 0
    assert ".sfmr" in result.output


def test_inspect_mutually_exclusive_flags(sfmrfile_reconstruction_with_17_images):
    """Multiple mode flags at once are rejected."""
    sfmr_path = str(sfmrfile_reconstruction_with_17_images)
    result = CliRunner().invoke(main, ["inspect", "--coviz", "--metrics", sfmr_path])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_inspect_range_without_metrics(sfmrfile_reconstruction_with_17_images):
    """--range without --metrics is rejected."""
    sfmr_path = str(sfmrfile_reconstruction_with_17_images)
    result = CliRunner().invoke(main, ["inspect", "--range", "1-5", sfmr_path])
    assert result.exit_code != 0
    assert "--range" in result.output
