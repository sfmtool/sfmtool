# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

from click.testing import CliRunner

from sfmtool.cli import main
from tests.conftest import TEST_DATA_DIR


def test_sift_workspace_mode(isolated_seoul_bull_image: Path):
    """Test that sift --extract uses workspace configuration when --tool is not specified."""
    workspace_dir = isolated_seoul_bull_image.parent
    workspace_config_file = workspace_dir / ".sfm-workspace.json"

    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0
    assert workspace_config_file.exists()

    result = CliRunner().invoke(
        main, ["sift", "--extract", str(isolated_seoul_bull_image)]
    )
    assert result.exit_code == 0, result.output
    assert "Using workspace:" in result.output
    assert "Feature tool: colmap" in result.output

    config = json.load(open(workspace_dir / ".sfm-workspace.json"))
    sift_dir = workspace_dir / config["feature_prefix_dir"]
    assert sift_dir.exists()


def test_sift_workspace_with_dsp(isolated_seoul_bull_image: Path):
    """Test workspace mode with DSP enabled."""
    workspace_dir = isolated_seoul_bull_image.parent
    workspace_config_file = workspace_dir / ".sfm-workspace.json"

    result = CliRunner().invoke(main, ["ws", "init", "--dsp", str(workspace_dir)])
    assert result.exit_code == 0

    with open(workspace_config_file) as f:
        config = json.load(f)
    options = config["feature_options"]
    assert options["domain_size_pooling"] is True

    result = CliRunner().invoke(
        main, ["sift", "--extract", str(isolated_seoul_bull_image)]
    )
    assert result.exit_code == 0, result.output
    assert "Using workspace:" in result.output


def test_sift_cli_override_workspace(isolated_seoul_bull_image: Path):
    """Test that --tool overrides workspace configuration."""
    workspace_dir = isolated_seoul_bull_image.parent

    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0

    result = CliRunner().invoke(
        main,
        ["sift", "--extract", "--tool", "opencv", str(isolated_seoul_bull_image)],
    )
    assert result.exit_code == 0, result.output
    assert "Using workspace:" not in result.output

    features_dir = workspace_dir / "features"
    opencv_dirs = list(features_dir.glob("sift-opencv-*"))
    assert len(opencv_dirs) == 1


def test_sift_no_workspace_error(isolated_seoul_bull_image: Path):
    """Test error when no workspace found and --tool not specified."""
    result = CliRunner().invoke(
        main, ["sift", "--extract", str(isolated_seoul_bull_image)]
    )
    assert result.exit_code != 0
    assert "No workspace found" in result.output
    assert "Initialize a workspace with 'sfm ws init'" in result.output
    assert "Specify --tool explicitly" in result.output


def test_sift_dsp_without_tool_error(isolated_seoul_bull_image: Path):
    """Test that --dsp without --tool raises an error."""
    workspace_dir = isolated_seoul_bull_image.parent
    CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])

    result = CliRunner().invoke(
        main, ["sift", "--extract", "--dsp", str(isolated_seoul_bull_image)]
    )
    assert result.exit_code != 0
    assert "--dsp/--no-dsp can only be used with --tool" in result.output
    assert "reinitialize the workspace with 'sfm ws init --dsp'" in result.output


def test_sift_dsp_with_opencv_error(isolated_seoul_bull_image: Path):
    """Test that --dsp with opencv raises an error."""
    result = CliRunner().invoke(
        main,
        [
            "sift",
            "--extract",
            "--tool",
            "opencv",
            "--dsp",
            str(isolated_seoul_bull_image),
        ],
    )
    assert result.exit_code != 0
    assert "--dsp/--no-dsp option is only supported for COLMAP" in result.output


def test_sift_workspace_opencv(isolated_seoul_bull_image: Path):
    """Test workspace mode with OpenCV tool."""
    workspace_dir = isolated_seoul_bull_image.parent

    result = CliRunner().invoke(
        main, ["ws", "init", "--feature-tool", "opencv", str(workspace_dir)]
    )
    assert result.exit_code == 0

    result = CliRunner().invoke(
        main, ["sift", "--extract", str(isolated_seoul_bull_image)]
    )
    assert result.exit_code == 0, result.output
    assert "Using workspace:" in result.output
    assert "Feature tool: opencv" in result.output


def test_sift_workspace_detection_from_subdirectory(tmp_path):
    """Test that workspace is detected from parent directories."""
    workspace_dir = tmp_path / "workspace"
    subdir = workspace_dir / "subdir"
    images_dir = subdir / "images"
    images_dir.mkdir(parents=True)

    test_image_src = (
        TEST_DATA_DIR
        / "images"
        / "seoul_bull_sculpture"
        / "seoul_bull_sculpture_01.jpg"
    )
    test_image = images_dir / "test.jpg"
    shutil.copy(test_image_src, test_image)

    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0

    result = CliRunner().invoke(main, ["sift", "--extract", str(test_image)])
    assert result.exit_code == 0, result.output
    assert "Using workspace:" in result.output
    assert str(workspace_dir) in result.output
