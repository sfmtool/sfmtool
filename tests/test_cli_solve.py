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


def test_resolve_output_path_explicit(tmp_path: Path):
    """Explicit --output: recon 0 gets the exact path, subsequent recons
    get `-N` before the suffix."""
    from sfmtool._isfm import _resolve_output_path

    explicit = tmp_path / "mine.sfmr"
    images = [Path("img_1.jpg"), Path("img_2.jpg")]

    p0 = _resolve_output_path(
        order=0,
        explicit_output=explicit,
        recon_image_paths=images,
        sfmr_dir=None,
        workspace_dir=tmp_path,
    )
    p1 = _resolve_output_path(
        order=1,
        explicit_output=explicit,
        recon_image_paths=images,
        sfmr_dir=None,
        workspace_dir=tmp_path,
    )
    p2 = _resolve_output_path(
        order=2,
        explicit_output=explicit,
        recon_image_paths=images,
        sfmr_dir=None,
        workspace_dir=tmp_path,
    )
    assert p0 == explicit
    assert p1 == tmp_path / "mine-1.sfmr"
    assert p2 == tmp_path / "mine-2.sfmr"


def test_resolve_output_path_auto_per_recon(tmp_path: Path):
    """Auto-name: each reconstruction gets its own descriptor reflecting the
    images actually present in that reconstruction."""
    from sfmtool._isfm import _resolve_output_path

    sfmr_dir = tmp_path / "sfmr"

    # Recon 0: three frames that form a small sub-range (1-3).
    recon0_images = [Path(f"KerryPark_{n}.jpg") for n in (1, 2, 3)]
    p0 = _resolve_output_path(
        order=0,
        explicit_output=None,
        recon_image_paths=recon0_images,
        sfmr_dir=sfmr_dir,
        workspace_dir=tmp_path,
    )
    assert "KerryPark_1-3" in p0.name
    # Touch the file so the counter advances for the next call.
    p0.parent.mkdir(parents=True, exist_ok=True)
    p0.touch()

    # Recon 1: the other 98 frames. Descriptor reflects 4..101, not the
    # full input range.
    recon1_images = [Path(f"KerryPark_{n}.jpg") for n in range(4, 102)]
    p1 = _resolve_output_path(
        order=1,
        explicit_output=None,
        recon_image_paths=recon1_images,
        sfmr_dir=sfmr_dir,
        workspace_dir=tmp_path,
    )
    assert "KerryPark_4-101" in p1.name
    assert p1 != p0


def test_solve_from_matches_with_range(isolated_seoul_bull_17_images: list[Path]):
    """`--range` on the .matches solve path restricts which images/pairs are
    written to the COLMAP DB."""
    workspace_dir = isolated_seoul_bull_17_images[0].parent

    result = CliRunner().invoke(main, ["init", str(workspace_dir)])
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(main, ["sift", "--extract", str(workspace_dir)])
    assert result.exit_code == 0, result.output

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

    output_path = workspace_dir / "from_matches_range.sfmr"
    # The 17-image fixture is numbered 1..17; keep 5..12 (8 images).
    result = CliRunner().invoke(
        main,
        [
            "solve",
            "-i",
            "--range",
            "5-12",
            "--output",
            str(output_path),
            str(matches_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # Summary line reflects the filtered counts, not the full .matches payload.
    assert "filtered from 17" in result.output
    # 8 images → C(8, 2) = 28 possible pairs (upper bound).
    import re

    m = re.search(r"Pairs: (\d+) \(filtered from (\d+)\)", result.output)
    assert m is not None, result.output
    kept_pairs, full_pairs = int(m.group(1)), int(m.group(2))
    assert 0 < kept_pairs <= 28
    assert kept_pairs < full_pairs
