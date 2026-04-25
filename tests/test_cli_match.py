# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from click.testing import CliRunner

from sfmtool.cli import main


def test_match_no_method(isolated_seoul_bull_image: Path):
    """Test that match without a method flag raises an error."""
    result = CliRunner().invoke(main, ["match", str(isolated_seoul_bull_image)])
    assert result.exit_code != 0
    assert "Must specify a matching method" in result.output


def test_match_multiple_methods(isolated_seoul_bull_image: Path):
    """Test that specifying multiple methods raises an error."""
    result = CliRunner().invoke(
        main,
        ["match", "--exhaustive", "--sequential", str(isolated_seoul_bull_image)],
    )
    assert result.exit_code != 0
    assert "Cannot specify more than one matching method" in result.output

    result = CliRunner().invoke(
        main,
        ["match", "--exhaustive", "--flow", str(isolated_seoul_bull_image)],
    )
    assert result.exit_code != 0
    assert "Cannot specify more than one matching method" in result.output


def test_match_no_paths():
    """Test that match without paths raises an error."""
    result = CliRunner().invoke(main, ["match", "--exhaustive"])
    assert result.exit_code != 0
    assert "Must provide image paths" in result.output


def test_match_exhaustive(isolated_seoul_bull_17_images: list[Path]):
    """Test exhaustive matching on a small set of images."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # Initialize workspace
    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Extract SIFT features first
    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Run exhaustive matching
    result = CliRunner().invoke(main, ["match", "--exhaustive", str(workspace_dir)])
    assert result.exit_code == 0, result.output
    assert "Running exhaustive matching" in result.output
    assert "Done:" in result.output
    assert "pairs" in result.output
    assert "matches" in result.output

    # Check that a .matches file was created
    tvg_dir = workspace_dir / "tvg-matches"
    matches_dir = workspace_dir / "matches"
    matches_files = []
    if tvg_dir.exists():
        matches_files.extend(tvg_dir.glob("*.matches"))
    if matches_dir.exists():
        matches_files.extend(matches_dir.glob("*.matches"))
    assert len(matches_files) == 1

    # Verify the .matches file can be read back
    from sfmtool._sfmtool import read_matches

    matches_data = read_matches(str(matches_files[0]))
    assert matches_data["metadata"]["matching_method"] == "exhaustive"
    assert matches_data["metadata"]["matching_tool"] == "colmap"
    assert matches_data["metadata"]["image_count"] == 17
    assert matches_data["metadata"]["match_count"] > 0


def test_match_sequential(isolated_seoul_bull_17_images: list[Path]):
    """Test sequential matching on a small set of images."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # Initialize workspace
    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Extract SIFT features
    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Run sequential matching with small overlap
    result = CliRunner().invoke(
        main,
        ["match", "--sequential", "--sequential-overlap", "3", str(workspace_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "Running sequential matching" in result.output
    assert "Done:" in result.output


def test_match_with_output_path(isolated_seoul_bull_17_images: list[Path]):
    """Test matching with a custom output path."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    output_path = workspace_dir / "custom_output.matches"
    result = CliRunner().invoke(
        main,
        [
            "match",
            "--exhaustive",
            "--output",
            str(output_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_match_flow(isolated_seoul_bull_17_images: list[Path]):
    """Test flow-based matching on a small set of images."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # Initialize workspace
    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Extract SIFT features
    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Run flow matching with small window
    output_path = workspace_dir / "flow_test.matches"
    result = CliRunner().invoke(
        main,
        [
            "match",
            "--flow",
            "--flow-skip",
            "3",
            "--output",
            str(output_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Running flow matching" in result.output
    assert "Done:" in result.output

    # Verify the .matches file
    from sfmtool._sfmtool import read_matches

    matches_data = read_matches(str(output_path))
    assert matches_data["metadata"]["matching_method"] == "flow"
    assert matches_data["metadata"]["matching_tool"] == "sfmtool-flow"
    assert matches_data["metadata"]["matching_options"]["flow_preset"] == "default"
    assert matches_data["metadata"]["matching_options"]["flow_skip"] == 3
    assert matches_data["metadata"]["match_count"] > 0


def test_match_with_range(isolated_seoul_bull_17_images: list[Path]):
    """Test matching with a range expression."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Extract features for the subset first
    result = CliRunner().invoke(
        main,
        ["sift", "--extract", "--range", "1-5", str(workspace_dir)],
    )
    assert result.exit_code == 0, result.output

    # Match the subset
    output_path = workspace_dir / "range_test.matches"
    result = CliRunner().invoke(
        main,
        [
            "match",
            "--exhaustive",
            "--range",
            "1-5",
            "--output",
            str(output_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    from sfmtool._sfmtool import read_matches

    matches_data = read_matches(str(output_path))
    assert matches_data["metadata"]["image_count"] == 5


def test_match_merge(isolated_seoul_bull_17_images: list[Path]):
    """Test merging two .matches files."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    # Initialize workspace and extract features
    result = CliRunner().invoke(main, ["ws", "init", str(workspace_dir)])
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    # Create two matches files with different methods
    seq_path = workspace_dir / "seq.matches"
    result = CliRunner().invoke(
        main,
        [
            "match",
            "--sequential",
            "--sequential-overlap",
            "3",
            "--output",
            str(seq_path),
            str(workspace_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    exh_path = workspace_dir / "exh.matches"
    result = CliRunner().invoke(
        main,
        ["match", "--exhaustive", "--output", str(exh_path), str(workspace_dir)],
    )
    assert result.exit_code == 0, result.output

    # Merge them
    merged_path = workspace_dir / "merged.matches"
    result = CliRunner().invoke(
        main,
        [
            "match",
            "--merge",
            str(seq_path),
            str(exh_path),
            "--output",
            str(merged_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Merging 2 .matches files" in result.output
    assert "Merged:" in result.output

    # Verify the merged file
    from sfmtool._sfmtool import read_matches

    merged = read_matches(str(merged_path))
    seq = read_matches(str(seq_path))
    exh = read_matches(str(exh_path))

    assert merged["metadata"]["matching_method"] == "merged"
    assert merged["metadata"]["image_count"] == 17
    # Merged should have at least as many pairs as either input
    assert merged["metadata"]["image_pair_count"] >= seq["metadata"]["image_pair_count"]
    assert merged["metadata"]["image_pair_count"] >= exh["metadata"]["image_pair_count"]
    # Merged should have at least as many matches as the larger input
    assert merged["metadata"]["match_count"] >= max(
        seq["metadata"]["match_count"], exh["metadata"]["match_count"]
    )
    # Both inputs have TVGs (from COLMAP), so merged should preserve them
    assert merged["has_two_view_geometries"] is True
    assert merged["tvg_metadata"]["inlier_count"] > 0


def test_match_merge_errors(tmp_path: Path):
    """Test merge validation errors."""
    result = CliRunner().invoke(main, ["match", "--merge"])
    assert result.exit_code != 0
    assert "Must provide .matches file paths" in result.output

    # Single file (need at least 2)
    dummy = tmp_path / "a.matches"
    dummy.write_bytes(b"")
    result = CliRunner().invoke(main, ["match", "--merge", str(dummy)])
    assert result.exit_code != 0
    assert "at least 2" in result.output

    # Missing --output
    dummy2 = tmp_path / "b.matches"
    dummy2.write_bytes(b"")
    result = CliRunner().invoke(main, ["match", "--merge", str(dummy), str(dummy2)])
    assert result.exit_code != 0
    assert "--output" in result.output or "-o" in result.output
