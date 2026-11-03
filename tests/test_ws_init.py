# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main


def test_init_basic(tmp_path: Path):
    """Test basic workspace initialization."""
    runner = CliRunner()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    result = runner.invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0

    config_file = workspace_dir / ".sfm-workspace.json"
    assert config_file.exists()

    with open(config_file) as f:
        config = json.load(f)
    assert config["feature_tool"] == "sfmtool"
    assert "feature_options" in config


def test_init_existing_workspace_protection(tmp_path: Path):
    """Test that init fails if a workspace already exists and --force is not used."""
    runner = CliRunner()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    runner.invoke(main, ["ws", "init", str(workspace_dir)])

    # Without --force should fail
    result = runner.invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code != 0
    assert "A workspace already exists" in result.output
    assert "Use --force to overwrite" in result.output

    # With --force should succeed
    result = runner.invoke(main, ["ws", "init", "--force", str(workspace_dir)])
    assert result.exit_code == 0
    assert "Initialized workspace" in result.output


def test_init_nested_workspace_protection(tmp_path: Path):
    """Test that init fails if target is inside an existing workspace."""
    runner = CliRunner()
    parent_dir = tmp_path / "parent"
    child_dir = parent_dir / "child"
    child_dir.mkdir(parents=True)

    runner.invoke(main, ["ws", "init", str(parent_dir)])

    # Without --force
    result = runner.invoke(main, ["ws", "init", str(child_dir)])
    assert result.exit_code != 0
    assert "Cannot create nested workspace" in result.output
    assert str(parent_dir.resolve()) in result.output
    assert "Use --force to create" in result.output

    # With --force
    result = runner.invoke(main, ["ws", "init", "--force", str(child_dir)])
    assert result.exit_code == 0
    assert "Initialized workspace" in result.output


def test_init_opencv(tmp_path: Path):
    """Test initialization with OpenCV tool."""
    runner = CliRunner()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    result = runner.invoke(
        main, ["ws", "init", "--feature-tool", "opencv", str(workspace_dir)]
    )
    assert result.exit_code == 0

    config_file = workspace_dir / ".sfm-workspace.json"
    with open(config_file) as f:
        config = json.load(f)
    assert config["feature_tool"] == "opencv"


def test_init_dsp_validation():
    """Test that --dsp requires colmap."""
    runner = CliRunner()
    result = runner.invoke(main, ["ws", "init", "--feature-tool", "opencv", "--dsp"])
    assert result.exit_code != 0
    assert "The --dsp/--no-dsp option is only supported for COLMAP" in result.output


def test_init_sfmtool_max_features(tmp_path: Path):
    """--max-features is honored by the sfmtool backend and written to the config."""
    runner = CliRunner()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    result = runner.invoke(
        main,
        [
            "ws",
            "init",
            "--feature-tool",
            "sfmtool",
            "--max-features",
            "2500",
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0

    with open(workspace_dir / ".sfm-workspace.json") as f:
        config = json.load(f)
    assert config["feature_tool"] == "sfmtool"
    assert config["feature_options"]["max_num_features"] == 2500


def test_init_max_features_opencv_rejected():
    """--max-features is only supported for COLMAP and sfmtool, not opencv."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["ws", "init", "--feature-tool", "opencv", "--max-features", "500"],
    )
    assert result.exit_code != 0
    assert "only supported for COLMAP and sfmtool" in result.output


def test_init_writes_version_2(tmp_path: Path):
    result = CliRunner().invoke(main, ["ws", "init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / ".sfm-workspace.json").read_text())
    assert config["version"] == 2
    assert config["feature_options"]["image_to_gray"] == {
        "formula": "0.2126*R + 0.7152*G + 0.0722*B"
    }
    assert "image_to_gray: 0.2126*R + 0.7152*G + 0.0722*B" in result.output
    assert f"feature_prefix_dir: {config['feature_prefix_dir']}" in result.output


def test_init_image_to_gray_and_quantize_bits(tmp_path: Path):
    result = CliRunner().invoke(
        main,
        ["ws", "init", "--image-to-gray", "G", "--quantize-bits", "8", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / ".sfm-workspace.json").read_text())
    assert config["feature_options"]["image_to_gray"] == {
        "formula": "G",
        "quantize_bits": 8,
    }
    assert "image_to_gray: G, rounded to 8 bits" in result.output

    # --force with other flags writes fresh options and a new folder.
    result = CliRunner().invoke(main, ["ws", "init", "--force", str(tmp_path)])
    assert result.exit_code == 0, result.output
    again = json.loads((tmp_path / ".sfm-workspace.json").read_text())
    assert again["feature_options"]["image_to_gray"] == {
        "formula": "0.2126*R + 0.7152*G + 0.0722*B"
    }
    assert again["feature_prefix_dir"] != config["feature_prefix_dir"]


@pytest.mark.parametrize(
    "args, message",
    [
        (
            ["--feature-tool", "colmap", "--image-to-gray", "G"],
            "--image-to-gray option is only supported for sfmtool",
        ),
        (
            ["--feature-tool", "opencv", "--quantize-bits", "8"],
            "--quantize-bits option is only supported for sfmtool",
        ),
        (["--quantize-bits", "0"], "must be an integer from 1 to 16, got 0"),
        (["--quantize-bits", "17"], "must be an integer from 1 to 16, got 17"),
        (["--image-to-gray", "R +"], "--image-to-gray: "),
        (["--image-to-gray", "R - G"], "--image-to-gray: "),
    ],
)
def test_init_image_to_gray_errors(tmp_path: Path, args, message):
    result = CliRunner().invoke(main, ["ws", "init", *args, str(tmp_path / "ws")])
    assert result.exit_code != 0
    assert message in result.output
    assert not (tmp_path / "ws" / ".sfm-workspace.json").exists()
