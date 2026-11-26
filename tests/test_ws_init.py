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


@pytest.mark.parametrize("tool", ["sfmtool", "colmap", "opencv"])
def test_init_writes_the_mipmap_options(tmp_path: Path, tool):
    """Every tool's new workspace reads photographs with a side over 4096 px
    from mipmaps at quality 95, the object written after the folder's name,
    and the command says so."""
    from sfmtool.workspace import new_workspace_mipmap_options

    result = CliRunner().invoke(
        main, ["ws", "init", "--feature-tool", tool, str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    config = json.loads((tmp_path / ".sfm-workspace.json").read_text())
    expected = {
        "out_of_core_long_side": 4096,
        "jpeg_quality": 95,
        "full_resolution_jpeg_quality": 95,
    }
    assert list(config.items())[-1] == ("mipmap_options", expected)
    assert list(config["mipmap_options"]) == list(expected)
    assert new_workspace_mipmap_options() == expected
    assert (
        "  mipmap_options: photographs with a side over 4096 px are read from "
        "mipmaps, JPEG quality 95 (95 at full resolution)" in result.output
    )


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


def _python_written_marker(max_num_features=None, image_to_gray=None) -> dict:
    """The record ``init_workspace`` built for the sfmtool tool before its
    writer moved to Rust: the old ``get_default_sfmtool_feature_options``
    literal and the record around it, keys in the order it wrote them, with
    the ``mipmap_options`` every new workspace records since after them."""
    from sfmtool.workspace import feature_prefix_dir, feature_type_for_tool

    options = {
        "octave_layers": 3,
        "sigma": 1.6,
        "blur_radius_factor": 2.25,
        "input_sigma": 0.5,
        "double_image": True,
        "double_image_below_long_side": 4096,
        "contrast_threshold": 0.0067,
        "edge_threshold": 10.0,
        "max_num_features": 8192 if max_num_features is None else max_num_features,
        "orientation_bins": 36,
        "peak_ratio": 0.8,
        "descriptor_width": 4,
        "descriptor_bins": 8,
        "descriptor_magnification": 3.0,
        "descriptor_clamp": 0.2,
        "image_to_gray": (
            {"formula": "0.2126*R + 0.7152*G + 0.0722*B"}
            if image_to_gray is None
            else dict(image_to_gray)
        ),
    }
    feature_type = feature_type_for_tool("sfmtool", options)
    return {
        "version": 2,
        "feature_tool": "sfmtool",
        "feature_type": feature_type,
        "feature_options": options,
        "feature_prefix_dir": feature_prefix_dir("sfmtool", feature_type, options),
        "mipmap_options": {
            "out_of_core_long_side": 4096,
            "jpeg_quality": 95,
            "full_resolution_jpeg_quality": 95,
        },
    }


@pytest.mark.parametrize(
    "kwargs, old",
    [
        ({}, {}),
        (
            {
                "max_num_features": 4000,
                "image_to_gray_formula": "G",
                "quantize_bits": 6,
            },
            {
                "max_num_features": 4000,
                "image_to_gray": {"formula": "G", "quantize_bits": 6},
            },
        ),
        (
            {"quantize_bits": 8},
            {
                "image_to_gray": {
                    "formula": "0.2126*R + 0.7152*G + 0.0722*B",
                    "quantize_bits": 8,
                }
            },
        ),
    ],
)
def test_init_workspace_writes_what_the_python_writer_wrote(
    tmp_path: Path, kwargs, old
):
    """The Rust writer ``init_workspace`` calls for the sfmtool tool writes the
    text the Python writer's ``json.dump(record, f, indent=2)`` wrote, with
    ``\n`` line breaks on every platform, and returns the same record."""
    from sfmtool._workspace import init_workspace
    from sfmtool.sift.extract_sfmtool import get_default_sfmtool_feature_options

    expected = _python_written_marker(**old)
    workspace_dir = tmp_path / "new" / "workspace"
    returned = init_workspace(workspace_dir, **kwargs)
    written = (workspace_dir / ".sfm-workspace.json").read_bytes()
    assert written == json.dumps(expected, indent=2).encode()
    assert list(returned.items()) == list(expected.items())
    assert list(returned["feature_options"]) == list(expected["feature_options"])
    assert get_default_sfmtool_feature_options(**old) == expected["feature_options"]
