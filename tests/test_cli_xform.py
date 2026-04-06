# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from sfmtool.cli import main


def test_xform_no_transforms(tmp_path: Path):
    """Test that xform without any transforms raises an error."""
    # Create a dummy .sfmr file (doesn't need to be valid for this test)
    input_path = tmp_path / "input.sfmr"
    input_path.touch()
    output_path = tmp_path / "output.sfmr"
    args = ["xform", str(input_path), str(output_path)]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "At least one transformation must be specified" in result.output


def test_xform_non_sfmr_input(tmp_path: Path):
    """Test that xform with non-.sfmr input raises an error."""
    input_path = tmp_path / "input.txt"
    input_path.touch()
    output_path = tmp_path / "output.sfmr"
    args = ["xform", str(input_path), str(output_path), "--scale", "2.0"]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "Input path must be a .sfmr file" in result.output


def test_xform_non_sfmr_output(tmp_path: Path):
    """Test that xform with non-.sfmr output raises an error."""
    input_path = tmp_path / "input.sfmr"
    input_path.touch()
    output_path = tmp_path / "output.txt"
    args = ["xform", str(input_path), str(output_path), "--scale", "2.0"]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "Output path must be a .sfmr file" in result.output


def test_xform_on_reconstruction(isolated_seoul_bull_17_images: list[Path]):
    """Test xform scale on a real reconstruction."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # Build a reconstruction
    result = CliRunner().invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    output_sfmr = workspace_dir / "test_solve.sfmr"
    result = CliRunner().invoke(
        main,
        ["solve", "-i", "--output", str(output_sfmr), str(workspace_dir)],
    )
    assert result.exit_code == 0, result.output
    assert output_sfmr.exists()

    # Now test xform with scale
    scaled_sfmr = workspace_dir / "scaled.sfmr"
    args = ["xform", str(output_sfmr), str(scaled_sfmr), "--scale", "2.0"]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert scaled_sfmr.exists()
    assert "Scale by 2.000" in result.output
    assert "Transformation complete" in result.output

    # Verify the scaled reconstruction
    from sfmtool._sfmtool import SfmrReconstruction

    original = SfmrReconstruction.load(output_sfmr)
    scaled = SfmrReconstruction.load(scaled_sfmr)
    assert scaled.image_count == original.image_count
    assert scaled.point_count == original.point_count


def test_xform_remove_short_tracks(isolated_seoul_bull_17_images: list[Path]):
    """Test xform with short track removal."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    result = CliRunner().invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    output_sfmr = workspace_dir / "test_solve.sfmr"
    result = CliRunner().invoke(
        main,
        ["solve", "-i", "--output", str(output_sfmr), str(workspace_dir)],
    )
    assert result.exit_code == 0, result.output

    # Remove short tracks
    filtered_sfmr = workspace_dir / "filtered.sfmr"
    args = ["xform", str(output_sfmr), str(filtered_sfmr), "--remove-short-tracks", "3"]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert filtered_sfmr.exists()
    assert "Remove tracks with length <= 3" in result.output

    from sfmtool._sfmtool import SfmrReconstruction

    original = SfmrReconstruction.load(output_sfmr)
    filtered = SfmrReconstruction.load(filtered_sfmr)
    assert filtered.point_count <= original.point_count
    assert filtered.image_count == original.image_count


def test_xform_chained_transforms(isolated_seoul_bull_17_images: list[Path]):
    """Test xform with multiple chained transforms."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    result = CliRunner().invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    output_sfmr = workspace_dir / "test_solve.sfmr"
    result = CliRunner().invoke(
        main,
        ["solve", "-i", "--output", str(output_sfmr), str(workspace_dir)],
    )
    assert result.exit_code == 0, result.output

    # Chain: remove short tracks -> scale -> translate
    chained_sfmr = workspace_dir / "chained.sfmr"
    args = [
        "xform",
        str(output_sfmr),
        str(chained_sfmr),
        "--remove-short-tracks",
        "2",
        "--scale",
        "0.5",
        "--translate",
        "1,2,3",
    ]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert chained_sfmr.exists()
    assert "Applying 3 transformation(s)" in result.output
