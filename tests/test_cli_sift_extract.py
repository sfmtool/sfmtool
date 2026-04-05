# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool._sift_file import (
    SiftReader,
    get_feature_tool_xxh128,
    get_feature_type_for_tool,
)
from sfmtool._extract_sift_colmap import get_colmap_feature_options

# Compute the hash of the default feature tool metadata (COLMAP without DSP)
_default_options = get_colmap_feature_options()
_default_feature_type = get_feature_type_for_tool("colmap", _default_options)
EXPECTED_FEATURE_TOOL_HASH = get_feature_tool_xxh128(
    "colmap", _default_feature_type, _default_options
)


def test_default_feature_tool():
    """Sanity check that we're getting a consistent hash for the feature tool."""
    assert len(EXPECTED_FEATURE_TOOL_HASH) == 32
    assert all(c in "0123456789abcdef" for c in EXPECTED_FEATURE_TOOL_HASH)


def test_sift_cli_no_paths():
    """Tests that 'sift --extract' fails with --extract but no paths."""
    result = CliRunner().invoke(main, ["sift", "--extract"])
    assert result.exit_code != 0
    assert "Must provide a list of paths to process" in result.output


@pytest.mark.parametrize("provide_dir", [True, False])
def test_sift_cli_with_file(isolated_seoul_bull_image: Path, provide_dir: bool):
    """Tests 'sift --extract' with a single image file."""
    expected_sift_path = (
        isolated_seoul_bull_image.parent
        / "features"
        / f"sift-colmap-{EXPECTED_FEATURE_TOOL_HASH}"
        / (isolated_seoul_bull_image.name + ".sift")
    )

    if provide_dir:
        extract_path = str(isolated_seoul_bull_image.parent)
    else:
        extract_path = str(isolated_seoul_bull_image)

    # Run the command once, and it should create the .sift file
    result = CliRunner().invoke(
        main, ["sift", "--extract", "--tool", "colmap", extract_path]
    )
    assert result.exit_code == 0, result.output
    assert "Given 1 path(s) to process, expanded to 1 filename(s):" in result.output
    assert "test_image.jpg (1 file)" in result.output
    assert "New SIFT feature extraction: 1 / 1 image(s)" in result.output
    assert "test_image.jpg.sift (1 file)" in result.output

    assert expected_sift_path.exists()
    assert expected_sift_path.is_file()
    with SiftReader(expected_sift_path) as reader:
        assert reader.metadata["image_name"] == isolated_seoul_bull_image.name

    # Run the command a second time, and it should leave the .sift file untouched
    result = CliRunner().invoke(
        main, ["sift", "--extract", "--tool", "colmap", extract_path]
    )
    assert result.exit_code == 0, result.output
    assert "Given 1 path(s) to process, expanded to 1 filename(s)" in result.output
    assert (
        "Existing SIFT features already processed for 1 / 1 image(s)" in result.output
    )
    assert "New SIFT feature extraction: 0 / 1 image(s)" in result.output
    assert "test_image.jpg (1 file)" in result.output
    assert "test_image.jpg.sift (1 file)" in result.output

    # Touch the last-modified time the image, and it should re-compute the .sift file
    isolated_seoul_bull_image.touch()
    result = CliRunner().invoke(
        main, ["sift", "--extract", "--tool", "colmap", extract_path]
    )
    assert result.exit_code == 0, result.output
    assert "Given 1 path(s) to process, expanded to 1 filename(s)" in result.output
    assert "New SIFT feature extraction: 1 / 1 image(s)" in result.output
    assert "test_image.jpg (1 file)" in result.output
    assert "test_image.jpg.sift (1 file)" in result.output


def test_sift_cli_with_empty_directory(tmp_path):
    """Tests the sift command with an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = CliRunner().invoke(
        main, ["sift", "--extract", "--tool", "colmap", str(empty_dir)]
    )
    assert result.exit_code != 0
    assert "There were no image files to process" in result.output


def test_sift_cli_with_range(isolated_seoul_bull_17_images: list[Path]):
    """Tests 'sift --extract' with the --range option."""
    image_dir = isolated_seoul_bull_17_images[0].parent
    range_expr = "5-7,10"

    result = CliRunner().invoke(
        main,
        [
            "sift",
            "--extract",
            "--tool",
            "colmap",
            "--range",
            range_expr,
            str(image_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "expanded to 4 filename(s)" in result.output
    assert "New SIFT feature extraction: 4 / 4 image(s)" in result.output

    # Check that the correct .sift files were created
    for i in [5, 6, 7, 10]:
        assert (
            image_dir
            / "features"
            / f"sift-colmap-{EXPECTED_FEATURE_TOOL_HASH}"
            / f"seoul_bull_sculpture_{i:02d}.jpg.sift"
        ).exists()
