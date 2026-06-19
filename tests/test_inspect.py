# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified `sfm inspect` CLI command."""

from pathlib import Path

from click.testing import CliRunner

from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.analyze.summary import _print_tool_options
from sfmtool.cli import main


# ── tool options rendering ───────────────────────────────────────────────────


def test_print_tool_options_renders_transforms_and_keys(capsys):
    """An operation's recorded parameters are surfaced: transforms first, then
    the remaining keys in sorted order."""
    _print_tool_options(
        {
            "transforms": [
                "Find points at infinity (eps=0.1, min_views=2)",
                "Bundle adjustment (refine: focal)",
            ],
            "algorithm": "global",
            "rig_aware": True,
        }
    )
    out = capsys.readouterr().out
    assert "Tool options:" in out
    assert "transforms:" in out
    assert "Find points at infinity (eps=0.1, min_views=2)" in out
    assert "Bundle adjustment (refine: focal)" in out
    assert "algorithm: global" in out
    assert "rig_aware: True" in out


def test_print_tool_options_empty_prints_nothing(capsys):
    """A reconstruction with no recorded parameters prints no block."""
    _print_tool_options({})
    _print_tool_options(None)
    assert capsys.readouterr().out == ""


# ── .sfmr reconstructions ────────────────────────────────────────────────────


def test_inspect_sfmr_default(seoul_bull_workspace):
    """Default inspect of a .sfmr prints the compact summary block."""
    sfmr_path = str(seoul_bull_workspace)
    result = CliRunner().invoke(main, ["inspect", sfmr_path])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Format:" in out
    assert ".sfmr version" in out
    assert "Images:" in out
    assert "17" in out
    assert "Integrity:" in out
    assert "OK" in out


def test_inspect_sfmr_verbose(seoul_bull_workspace):
    """Verbose inspect of a .sfmr prints the full report."""
    sfmr_path = str(seoul_bull_workspace)
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


def test_inspect_sift_default(seoul_bull_workspace):
    """Default inspect of a .sift prints the compact summary block."""
    sift_path = _find_sift_file(seoul_bull_workspace.parent)
    result = CliRunner().invoke(main, ["inspect", str(sift_path)])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Format:" in out
    assert ".sift" in out
    assert "Features:" in out
    assert "Integrity:" in out


def test_inspect_sift_verbose(seoul_bull_workspace):
    """Verbose inspect of a .sift prints hashes and top features."""
    sift_path = _find_sift_file(seoul_bull_workspace.parent)
    result = CliRunner().invoke(main, ["inspect", "-v", str(sift_path)])
    assert result.exit_code == 0, result.output

    out = result.output
    assert "Content hash" in out
    assert "Top 5 features" in out


# ── .matches files ───────────────────────────────────────────────────────────


def test_inspect_matches(seoul_bull_workspace, tmp_path: Path):
    """Inspect a .matches file produced by exhaustive matching."""
    image_dir = seoul_bull_workspace.parent / "test_17_image"
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


# ── pt3d_ point IDs ──────────────────────────────────────────────────────────


def _point_id(sfmr_path, index):
    """Build a pt3d_ ID for the given .sfmr and point index."""
    recon = SfmrReconstruction.load(str(sfmr_path))
    return f"pt3d_{recon.content_xxh128[:8]}_{index}"


def test_inspect_point_summary(seoul_bull_workspace):
    """A pt3d_ ID resolves to its .sfmr and prints the point summary."""
    sfmr = seoul_bull_workspace
    point_id = _point_id(sfmr, 0)
    result = CliRunner().invoke(main, ["inspect", point_id, str(sfmr.parent)])
    assert result.exit_code == 0, result.output
    assert point_id in result.output
    assert "Observations" in result.output
    assert sfmr.name in result.output


def test_inspect_point_verbose(seoul_bull_workspace):
    """--verbose adds the full triangulation analysis."""
    sfmr = seoul_bull_workspace
    point_id = _point_id(sfmr, 0)
    result = CliRunner().invoke(main, ["inspect", point_id, str(sfmr.parent), "-v"])
    assert result.exit_code == 0, result.output
    assert "Triangulation analysis:" in result.output
    assert "Inverse-depth z" in result.output
    assert "incidence" in result.output


def test_inspect_point_unknown_hash(seoul_bull_workspace):
    """A hash matching no .sfmr is a clear error."""
    sfmr = seoul_bull_workspace
    result = CliRunner().invoke(main, ["inspect", "pt3d_deadbeef_0", str(sfmr.parent)])
    assert result.exit_code != 0
    assert "no .sfmr" in result.output


def test_inspect_point_index_out_of_range(seoul_bull_workspace):
    """An index beyond the point count is rejected."""
    sfmr = seoul_bull_workspace
    point_id = _point_id(sfmr, 10_000_000)
    result = CliRunner().invoke(main, ["inspect", point_id, str(sfmr.parent)])
    assert result.exit_code != 0
    assert "out of range" in result.output


def test_inspect_workspace_arg_rejected_for_file(
    seoul_bull_workspace,
):
    """The second (workspace) argument is only valid with a point ID."""
    sfmr = seoul_bull_workspace
    result = CliRunner().invoke(main, ["inspect", str(sfmr), str(sfmr.parent)])
    assert result.exit_code != 0
    assert "point ID" in result.output
