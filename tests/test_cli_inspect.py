# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified `sfm inspect` CLI command."""

from pathlib import Path

from click.testing import CliRunner

from sfmtool.cli import main


# ── .sfmr reconstructions ────────────────────────────────────────────────────


def test_inspect_sfmr_default(sfmrfile_reconstruction_with_17_images):
    """Default inspect of a .sfmr prints the compact summary block."""
    sfmr_path = str(sfmrfile_reconstruction_with_17_images)
    result = CliRunner().invoke(main, ["inspect", sfmr_path])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Format:" in out
    assert ".sfmr version" in out
    assert "Images:" in out
    assert "17" in out
    assert "Integrity:" in out
    assert "OK" in out


def test_inspect_sfmr_verbose(sfmrfile_reconstruction_with_17_images):
    """Verbose inspect of a .sfmr prints the full report."""
    sfmr_path = str(sfmrfile_reconstruction_with_17_images)
    result = CliRunner().invoke(main, ["inspect", "-v", sfmr_path])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Reconstruction summary:" in out
    assert "3D Point statistics:" in out
    assert "Nearest neighbor distances:" in out


# ── .sift feature files ──────────────────────────────────────────────────────


def _find_sift_file(workspace_dir: Path) -> Path:
    sift_files = sorted(workspace_dir.rglob("*.sift"))
    assert sift_files, f"no .sift files under {workspace_dir}"
    return sift_files[0]


def test_inspect_sift_default(sfmrfile_reconstruction_with_17_images):
    """Default inspect of a .sift prints the compact summary block."""
    sift_path = _find_sift_file(sfmrfile_reconstruction_with_17_images.parent)
    result = CliRunner().invoke(main, ["inspect", str(sift_path)])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Format:" in out
    assert ".sift" in out
    assert "Features:" in out
    assert "Integrity:" in out


def test_inspect_sift_verbose(sfmrfile_reconstruction_with_17_images):
    """Verbose inspect of a .sift prints hashes and top features."""
    sift_path = _find_sift_file(sfmrfile_reconstruction_with_17_images.parent)
    result = CliRunner().invoke(main, ["inspect", "-v", str(sift_path)])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Content hash" in out
    assert "Top 5 features" in out


# ── .matches files ───────────────────────────────────────────────────────────


def test_inspect_matches(sfmrfile_reconstruction_with_17_images, tmp_path: Path):
    """Inspect a .matches file produced by exhaustive matching."""
    image_dir = sfmrfile_reconstruction_with_17_images.parent / "test_17_image"
    matches_path = tmp_path / "test.matches"
    match_result = CliRunner().invoke(
        main, ["match", "--exhaustive", str(image_dir), "-o", str(matches_path)]
    )
    assert match_result.exit_code == 0, match_result.output

    result = CliRunner().invoke(main, ["inspect", str(matches_path)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert ".matches version" in out
    assert "Image pairs:" in out
    assert "Two-view geom:" in out

    verbose = CliRunner().invoke(main, ["inspect", "-v", str(matches_path)])
    assert verbose.exit_code == 0, verbose.output
    assert "Descriptor distance" in verbose.output


# ── .camrig camera rigs ──────────────────────────────────────────────────────


def _build_camrig(out: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "camrig",
            "spherical-tiles",
            str(out),
            "--n",
            "150",
            "--equirect-width",
            "256",
        ],
    )
    assert result.exit_code == 0, result.output


def test_inspect_camrig_default(tmp_path: Path):
    """Default inspect of a .camrig prints the compact summary block."""
    camrig_path = tmp_path / "tiles.camrig"
    _build_camrig(camrig_path)

    result = CliRunner().invoke(main, ["inspect", str(camrig_path)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Rig type:" in out
    assert "spherical_tiles" in out
    assert "Sensors:" in out
    assert "Integrity:" in out


def test_inspect_camrig_verbose(tmp_path: Path):
    """Verbose inspect of a .camrig prints attributes and content hash."""
    camrig_path = tmp_path / "tiles.camrig"
    _build_camrig(camrig_path)

    result = CliRunner().invoke(main, ["inspect", "-v", str(camrig_path)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Rig attributes:" in out
    assert "Content hash:" in out


def test_inspect_camrig_rejects_corrupt(tmp_path: Path):
    """A corrupt .camrig file fails inspection."""
    fake = tmp_path / "junk.camrig"
    fake.write_bytes(b"not a zip archive")
    result = CliRunner().invoke(main, ["inspect", str(fake)])
    assert result.exit_code != 0


# ── image files ──────────────────────────────────────────────────────────────


def test_inspect_image_default(isolated_seoul_bull_image: Path):
    """Default inspect of an image prints dimensions and size."""
    result = CliRunner().invoke(main, ["inspect", str(isolated_seoul_bull_image)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "JPEG image" in out
    assert "Dimensions:" in out
    assert "File size:" in out


def test_inspect_image_verbose(isolated_seoul_bull_image: Path):
    """Verbose inspect of an image prints the inferred camera."""
    result = CliRunner().invoke(main, ["inspect", "-v", str(isolated_seoul_bull_image)])
    assert result.exit_code == 0, result.output
    assert "Inferred camera" in result.output


# ── dispatch errors ──────────────────────────────────────────────────────────


def test_inspect_unsupported_type(tmp_path: Path):
    """An unsupported extension is rejected with a helpful message."""
    p = tmp_path / "input.txt"
    p.touch()
    result = CliRunner().invoke(main, ["inspect", str(p)])
    assert result.exit_code != 0
    assert "unsupported file type" in result.output
