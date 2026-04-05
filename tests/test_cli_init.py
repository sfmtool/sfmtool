# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from click.testing import CliRunner

from sfmtool.cli import main


def test_init_basic(tmp_path: Path):
    """Test basic workspace initialization."""
    runner = CliRunner()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    result = runner.invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0

    config_file = workspace_dir / ".sfm-workspace.json"
    assert config_file.exists()

    with open(config_file) as f:
        config = json.load(f)
    assert config["feature_tool"] == "colmap"
    assert "feature_options" in config


def test_init_existing_workspace_protection(tmp_path: Path):
    """Test that init fails if a workspace already exists and --force is not used."""
    runner = CliRunner()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    runner.invoke(main, ["init", str(workspace_dir)])

    # Without --force should fail
    result = runner.invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code != 0
    assert "A workspace already exists" in result.output
    assert "Use --force to overwrite" in result.output

    # With --force should succeed
    result = runner.invoke(main, ["init", "--force", str(workspace_dir)])
    assert result.exit_code == 0
    assert "Initialized workspace" in result.output


def test_init_nested_workspace_protection(tmp_path: Path):
    """Test that init fails if target is inside an existing workspace."""
    runner = CliRunner()
    parent_dir = tmp_path / "parent"
    child_dir = parent_dir / "child"
    child_dir.mkdir(parents=True)

    runner.invoke(main, ["init", str(parent_dir)])

    # Without --force
    result = runner.invoke(main, ["init", str(child_dir)])
    assert result.exit_code != 0
    assert "Cannot create nested workspace" in result.output
    assert str(parent_dir.resolve()) in result.output
    assert "Use --force to create" in result.output

    # With --force
    result = runner.invoke(main, ["init", "--force", str(child_dir)])
    assert result.exit_code == 0
    assert "Initialized workspace" in result.output


def test_init_opencv(tmp_path: Path):
    """Test initialization with OpenCV tool."""
    runner = CliRunner()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    result = runner.invoke(
        main, ["init", "--feature-tool", "opencv", str(workspace_dir)]
    )
    assert result.exit_code == 0

    config_file = workspace_dir / ".sfm-workspace.json"
    with open(config_file) as f:
        config = json.load(f)
    assert config["feature_tool"] == "opencv"


def test_init_dsp_validation():
    """Test that --dsp requires colmap."""
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--feature-tool", "opencv", "--dsp"])
    assert result.exit_code != 0
    assert "The --dsp/--no-dsp option is only supported for COLMAP" in result.output
