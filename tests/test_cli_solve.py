# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from click.testing import CliRunner

from sfmtool.cli import main


def test_solve_no_mode(isolated_seoul_bull_image: Path):
    """Test that solve without -i or -g raises an error."""
    result = CliRunner().invoke(main, ["solve", str(isolated_seoul_bull_image)])
    assert result.exit_code != 0
    assert "Must specify either --incremental" in result.output


def test_solve_both_modes(isolated_seoul_bull_image: Path):
    """Test that specifying both modes raises an error."""
    result = CliRunner().invoke(
        main, ["solve", "-i", "-g", str(isolated_seoul_bull_image)]
    )
    assert result.exit_code != 0
    assert "Cannot specify both" in result.output


def test_solve_no_paths():
    """Test that solve without paths raises an error."""
    result = CliRunner().invoke(main, ["solve", "-i"])
    assert result.exit_code != 0
    assert "Must provide image paths" in result.output


def test_solve_incremental(isolated_seoul_bull_17_images: list[Path]):
    """Test incremental SfM on a small set of images."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # Initialize workspace
    result = CliRunner().invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Extract SIFT features
    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Run incremental SfM
    output_path = workspace_dir / "test_solve.sfmr"
    result = CliRunner().invoke(
        main,
        [
            "solve",
            "-i",
            "--output",
            str(output_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Running incremental SfM" in result.output
    assert "Found reconstruction" in result.output
    assert output_path.exists()

    # Verify the .sfmr file can be loaded
    from sfmtool._sfmtool import SfmrReconstruction

    recon = SfmrReconstruction.load(output_path)
    assert recon.image_count > 0
    assert recon.point_count > 0
    assert recon.camera_count > 0


def test_solve_global(isolated_seoul_bull_17_images: list[Path]):
    """Test global SfM (GLOMAP) on a small set of images."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # Initialize workspace
    result = CliRunner().invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Extract SIFT features
    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Run global SfM
    output_path = workspace_dir / "test_global.sfmr"
    result = CliRunner().invoke(
        main,
        [
            "solve",
            "-g",
            "--output",
            str(output_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Running global SfM" in result.output
    assert "Found reconstruction" in result.output
    assert output_path.exists()


def test_solve_from_matches(isolated_seoul_bull_17_images: list[Path]):
    """Test solving from a pre-computed .matches file."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # Initialize workspace and extract SIFT
    result = CliRunner().invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Run exhaustive matching first
    matches_path = workspace_dir / "test.matches"
    result = CliRunner().invoke(
        main,
        [
            "match",
            "--exhaustive",
            "--output",
            str(matches_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert matches_path.exists()

    # Solve from the .matches file
    output_path = workspace_dir / "from_matches.sfmr"
    result = CliRunner().invoke(
        main,
        [
            "solve",
            "-i",
            "--output",
            str(output_path),
            str(matches_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()
