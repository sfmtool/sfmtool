# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
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
    assert "Feature tool: sfmtool" in result.output

    config = json.load(open(workspace_dir / ".sfm-workspace.json"))
    sift_dir = workspace_dir / config["feature_prefix_dir"]
    assert sift_dir.exists()


def test_sift_workspace_with_dsp(isolated_seoul_bull_image: Path):
    """Test workspace mode with DSP enabled."""
    workspace_dir = isolated_seoul_bull_image.parent
    workspace_config_file = workspace_dir / ".sfm-workspace.json"

    result = CliRunner().invoke(
        main, ["ws", "init", "--feature-tool", "colmap", "--dsp", str(workspace_dir)]
    )
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


def test_sift_draw_with_tool_override_in_workspace(
    isolated_seoul_bull_image: Path, tmp_path: Path
):
    """--draw --tool reads the features --extract --tool wrote, not the workspace's."""
    workspace_dir = isolated_seoul_bull_image.parent
    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0

    # Only the opencv features exist; the workspace's sfmtool ones were never
    # extracted.
    result = CliRunner().invoke(
        main,
        ["sift", "--extract", "--tool", "opencv", str(isolated_seoul_bull_image)],
    )
    assert result.exit_code == 0, result.output

    draw_dir = tmp_path / "drawn"
    result = CliRunner().invoke(
        main,
        [
            "sift",
            "--draw",
            str(draw_dir),
            "--tool",
            "opencv",
            str(isolated_seoul_bull_image),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Error" not in result.output
    assert (draw_dir / isolated_seoul_bull_image.name).exists()


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


def test_sift_workspace_sfmtool(isolated_seoul_bull_image: Path):
    """Test workspace mode with the sfmtool tool."""
    workspace_dir = isolated_seoul_bull_image.parent
    workspace_config_file = workspace_dir / ".sfm-workspace.json"

    result = CliRunner().invoke(
        main, ["ws", "init", "--feature-tool", "sfmtool", str(workspace_dir)]
    )
    assert result.exit_code == 0, result.output

    with open(workspace_config_file) as f:
        config = json.load(f)
    assert config["feature_tool"] == "sfmtool"
    assert config["feature_type"] == "sift-sfmtool"

    result = CliRunner().invoke(
        main, ["sift", "--extract", str(isolated_seoul_bull_image)]
    )
    assert result.exit_code == 0, result.output
    assert "Using workspace:" in result.output
    assert "Feature tool: sfmtool" in result.output

    sift_dir = workspace_dir / config["feature_prefix_dir"]
    assert sift_dir.exists()
    assert (sift_dir / (isolated_seoul_bull_image.name + ".sift")).is_file()


def test_sift_cli_override_workspace_sfmtool(isolated_seoul_bull_image: Path):
    """Test that --tool sfmtool overrides workspace configuration."""
    workspace_dir = isolated_seoul_bull_image.parent

    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0

    result = CliRunner().invoke(
        main,
        ["sift", "--extract", "--tool", "sfmtool", str(isolated_seoul_bull_image)],
    )
    assert result.exit_code == 0, result.output
    assert "Using workspace:" not in result.output

    features_dir = workspace_dir / "features"
    sfmtool_dirs = list(features_dir.glob("sift-sfmtool-*"))
    assert len(sfmtool_dirs) == 1


def test_ws_init_sfmtool_rejects_colmap_only_options(tmp_path):
    """COLMAP-only knobs are rejected for the sfmtool tool."""
    result = CliRunner().invoke(
        main, ["ws", "init", "--feature-tool", "sfmtool", "--dsp", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert "--dsp/--no-dsp option is only supported for COLMAP" in result.output


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


def _mipmap_workspace(root: Path, with_mipmap_options: bool) -> tuple[Path, Path]:
    """An `sfmtool` workspace at `root` with a 300 x 200 noise photograph and
    a 60 x 40 one in `images/`, its `.sfm-workspace.json` given
    `mipmap_options` with a threshold of 64 px when `with_mipmap_options`, so
    the first is out of core and the second in core, and with no
    `mipmap_options` otherwise, in place of the object `sfm ws init` writes."""
    result = CliRunner().invoke(
        main, ["ws", "init", "--feature-tool", "sfmtool", str(root)]
    )
    assert result.exit_code == 0, result.output
    config_file = root / ".sfm-workspace.json"
    config = json.loads(config_file.read_text())
    del config["mipmap_options"]
    if with_mipmap_options:
        config["mipmap_options"] = {
            "out_of_core_long_side": 64,
            "jpeg_quality": 95,
            "full_resolution_jpeg_quality": 95,
        }
    config_file.write_text(json.dumps(config))
    images = root / "images"
    images.mkdir()
    rng = np.random.default_rng(7)
    for name, (width, height) in (("big.png", (300, 200)), ("small.png", (60, 40))):
        pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(images / name), cv2.GaussianBlur(pixels, (5, 5), 1.5))
    return images / "big.png", images / "small.png"


def test_sift_extract_makes_the_mipmaps_of_out_of_core_photographs(tmp_path):
    """`sfm sift --extract` in a workspace with `mipmap_options` writes the
    mip level files of the out-of-core photograph and none for the in-core
    one, and counts them; in a workspace without the object it writes no
    mipmap directory and says nothing of mipmaps. The `.sift` files are the
    same bytes in both."""
    sift_bytes = {}
    for with_options in (True, False):
        root = tmp_path / ("with" if with_options else "without")
        big, small = _mipmap_workspace(root, with_options)
        result = CliRunner().invoke(main, ["sift", "--extract", str(root / "images")])
        assert result.exit_code == 0, result.output
        prefix = json.loads((root / ".sfm-workspace.json").read_text())[
            "feature_prefix_dir"
        ]
        sift_bytes[with_options] = [
            (big.parent / prefix / (p.name + ".sift")).read_bytes()
            for p in (big, small)
        ]
        mipmaps = root / ".sfm-mipmaps" / "images"
        if with_options:
            assert "New mipmaps: 1 / 1 out-of-core image(s)" in result.output
            assert sorted(p.name for p in mipmaps.iterdir()) == [
                f"big.png-mip{level}-q95.jpg" for level in (0, 1)
            ]
        else:
            assert "mipmap" not in result.output.lower()
            assert not (root / ".sfm-mipmaps").exists()
    assert sift_bytes[True] == sift_bytes[False]


def test_sift_extract_warns_of_each_mipmap_an_invalid_object_stops(tmp_path):
    """An invalid `mipmap_options` makes no mipmap: `sfm sift --extract`
    warns of each photograph, naming the key at fault, and still writes every
    `.sift` file."""
    big, small = _mipmap_workspace(tmp_path, with_mipmap_options=True)
    config_file = tmp_path / ".sfm-workspace.json"
    config = json.loads(config_file.read_text())
    config["mipmap_options"]["jpeg_quality"] = 0
    config_file.write_text(json.dumps(config))

    result = CliRunner().invoke(main, ["sift", "--extract", str(big), str(small)])

    assert result.exit_code == 0, result.output
    warnings = [
        line
        for line in result.output.splitlines()
        if line.startswith("Warning: no mipmap was made:")
    ]
    assert len(warnings) == 2
    assert all("mipmap_options.jpeg_quality" in line for line in warnings)
    assert "New mipmaps" not in result.output
    assert not (tmp_path / ".sfm-mipmaps").exists()
    prefix = config["feature_prefix_dir"]
    for photograph in (big, small):
        assert (photograph.parent / prefix / (photograph.name + ".sift")).is_file()
