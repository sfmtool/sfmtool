# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `sfm analyze` CLI command."""

from pathlib import Path

from click.testing import CliRunner

from sfmtool.cli import main


def test_analyze_requires_mode(seoul_bull_workspace):
    """analyze with no mode flag is rejected."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", sfmr_path])
    assert result.exit_code != 0
    assert "analysis mode" in result.output


def test_analyze_coviz(seoul_bull_workspace):
    """--coviz prints covisibility graph."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", "--coviz", sfmr_path])
    assert result.exit_code == 0, result.output
    assert "ovisibility" in result.output


def test_analyze_images(seoul_bull_workspace):
    """--images prints per-image connectivity table."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", "--images", sfmr_path])
    assert result.exit_code == 0, result.output
    assert "seoul_bull_sculpture" in result.output


def test_analyze_metrics(seoul_bull_workspace):
    """--metrics prints per-image metrics table."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", "--metrics", sfmr_path])
    assert result.exit_code == 0, result.output
    assert "MeanErr" in result.output
    assert "seoul_bull_sculpture" in result.output


def test_analyze_metrics_with_range(seoul_bull_workspace):
    """--metrics --range filters to subset of images."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(
        main, ["analyze", "--metrics", "--range", "1-5", sfmr_path]
    )
    assert result.exit_code == 0, result.output
    assert "5 of 17 images" in result.output


def test_analyze_depth_reliability(seoul_bull_workspace):
    """--depth-reliability prints the inverse-depth z-score report."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", "--depth-reliability", sfmr_path])
    assert result.exit_code == 0, result.output
    assert "Depth reliability" in result.output
    assert "Inverse-depth z" in result.output


def test_analyze_non_sfmr_file(tmp_path: Path):
    """Passing a non-.sfmr file raises an error."""
    p = tmp_path / "input.txt"
    p.touch()
    result = CliRunner().invoke(main, ["analyze", "--coviz", str(p)])
    assert result.exit_code != 0
    assert ".sfmr" in result.output


def test_analyze_mutually_exclusive_flags(seoul_bull_workspace):
    """Multiple mode flags at once are rejected."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["analyze", "--coviz", "--metrics", sfmr_path])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_analyze_range_without_metrics(seoul_bull_workspace):
    """--range without --metrics is rejected."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(
        main, ["analyze", "--coviz", "--range", "1-5", sfmr_path]
    )
    assert result.exit_code != 0
    assert "--range" in result.output
