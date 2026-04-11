# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `sfm discontinuity` CLI command."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main


SEOUL_BULL_DIR = (
    Path(__file__).parent.parent / "test-data" / "images" / "seoul_bull_sculpture"
)


@pytest.fixture
def runner():
    return CliRunner()


# --- Error handling ---


def test_no_paths(runner):
    """No arguments produces a usage error."""
    result = runner.invoke(main, ["discontinuity"])
    assert result.exit_code != 0
    assert "Must provide" in result.output


def test_sfmr_not_implemented(runner, tmp_path):
    """Passing a .sfmr file gives a not-implemented error."""
    fake_sfmr = tmp_path / "test.sfmr"
    fake_sfmr.touch()
    result = runner.invoke(main, ["discontinuity", str(fake_sfmr)])
    assert result.exit_code != 0
    assert "not yet implemented" in result.output


def test_no_images_found(runner, tmp_path):
    """Empty directory produces an error."""
    result = runner.invoke(main, ["discontinuity", str(tmp_path)])
    assert result.exit_code != 0
    assert "No image files found" in result.output


def test_single_image_no_sequence(runner, tmp_path):
    """A single image can't form a sequence — error."""
    import shutil

    src = SEOUL_BULL_DIR / "seoul_bull_sculpture_01.jpg"
    shutil.copy(src, tmp_path / "img_01.jpg")
    result = runner.invoke(main, ["discontinuity", str(tmp_path)])
    assert result.exit_code != 0
    assert "No numbered sequences" in result.output


def test_non_sequential_images(runner, tmp_path):
    """Images without numbered names produce a no-sequence error."""
    import shutil

    for name in ("alpha.jpg", "beta.jpg", "gamma.jpg"):
        shutil.copy(SEOUL_BULL_DIR / "seoul_bull_sculpture_01.jpg", tmp_path / name)
    result = runner.invoke(main, ["discontinuity", str(tmp_path)])
    assert result.exit_code != 0
    assert "No numbered sequences" in result.output


# --- Analysis with stride 1 (default) ---


def test_stride_1_analysis(runner):
    """Default stride=1 on 3 frames: checks output structure, tile grids,
    histograms, stride-2 comparison, and summary."""
    result = runner.invoke(
        main,
        [
            "discontinuity",
            str(SEOUL_BULL_DIR),
            "-r",
            "1-3",
            "--no-adaptive",
            "--initial-stride",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output

    out = result.output
    # Sequence detection
    assert "Found 3 images in 1 sequence" in out
    assert "Analyzing sequence:" in out

    # Every frame sampled (stride=1 advances by 1)
    assert "Frame 1:" in out
    assert "Frame 2:" in out

    # Local and stride flow reported
    assert "Local:" in out
    assert "Stride:" in out

    # Stride-2 comparison: from frame 1 targets frame 3
    assert "seoul_bull_sculpture_03.jpg" in out

    # Tile magnitude grids and difference
    assert "tile magnitudes" in out.lower()
    assert "Difference" in out

    # In-bounds percentage
    assert "in bounds" in out.lower()

    # Direction histogram box characters
    assert "\u2502" in out

    # Summary present
    assert "Summary:" in out


# --- Fixed stride ---


def test_fixed_stride(runner):
    """--no-adaptive with stride 2: samples at frame 1 and 3, no stride changes."""
    result = runner.invoke(
        main,
        [
            "discontinuity",
            str(SEOUL_BULL_DIR),
            "-r",
            "1-4",
            "--initial-stride",
            "2",
            "--no-adaptive",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Frame 1:" in result.output
    assert "Frame 3:" in result.output
    # No stride change messages
    assert "\u2193 stride" not in result.output
    assert "\u2191 stride" not in result.output


# --- Adaptive stride ---


def test_adaptive_stride(runner):
    """Adaptive stride on 8 frames runs to completion with summary."""
    result = runner.invoke(
        main,
        ["discontinuity", str(SEOUL_BULL_DIR), "-r", "1-8"],
    )
    assert result.exit_code == 0, result.output
    assert "Found 8 images in 1 sequence" in result.output
    assert "Frame 1:" in result.output
    assert "Summary:" in result.output


# --- Flow image saving ---


def test_save_flow_dir(runner, tmp_path):
    """--save-flow-dir saves valid flow color images with correct naming."""
    flow_dir = tmp_path / "flows"
    result = runner.invoke(
        main,
        [
            "discontinuity",
            str(SEOUL_BULL_DIR),
            "-r",
            "1-3",
            "--no-adaptive",
            "--initial-stride",
            "1",
            "--save-flow-dir",
            str(flow_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert flow_dir.exists()

    saved_files = sorted(flow_dir.glob("*.jpg"))
    assert len(saved_files) > 0

    # Naming convention: {seq_name}_from_{N}_to_{M}.jpg
    names = [f.name for f in saved_files]
    assert any("seoul_bull_sculpture" in n for n in names)
    assert any("_from_" in n and "_to_" in n for n in names)

    # Files are valid JPEGs with nonzero content
    for jpg in saved_files:
        assert jpg.stat().st_size > 1000, f"{jpg.name} is suspiciously small"
