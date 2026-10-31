# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--refine-normals`` xform operation.

Covers the option's value (it takes no keys), the summary line, and runs over
the seoul_bull ground truth and an ``embedded_patches`` conversion of the
seoul_bull workspace: finite points' normals turn, no position or keypoint
moves, points at infinity are untouched, and bitmaps are rendered again only
where the reconstruction stores them. See
specs/cli/reconstruction/xform/refine-normals-command.md.
"""

from unittest.mock import patch

import click
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import AddPatchBitmapsTransform, RefineNormalsTransform
from sfmtool.xform._arg_parser import parse_refine_normals_params
from sfmtool.xform._refine_normals import _summary

from ..conftest import SEOUL_BULL_GROUND_TRUTH


# ── The option's value ──────────────────────────────────────────────────────


@pytest.mark.parametrize("param", ["", ",", " , "])
def test_parse_without_keys(param):
    assert isinstance(parse_refine_normals_params(param), RefineNormalsTransform)


@pytest.mark.parametrize(
    "param",
    [
        "angular_range_deg=25,init_steps=7",
        "bitmaps=false",
        "resolution=32",
        "quality=fine",
        "k_neighbors=8",
        "init_steps",
    ],
)
def test_any_key_is_a_usage_error_naming_the_change(param):
    with pytest.raises(click.UsageError) as raised:
        parse_refine_normals_params(param)
    message = str(raised.value)
    assert "takes no keys" in message
    assert param in message
    assert "grown-plane normal" in message
    assert "--add-patch-bitmaps" in message


def test_cli_key_is_rejected_before_loading(seoul_bull_workspace):
    output_sfmr = seoul_bull_workspace.with_name("rejected_key.sfmr")
    result = CliRunner().invoke(
        main,
        [
            "xform",
            str(seoul_bull_workspace),
            str(output_sfmr),
            "--to-embedded-patches",
            "--refine-normals",
            "bitmaps=false",
        ],
    )
    assert result.exit_code == 2
    assert "takes no keys" in result.output
    assert not output_sfmr.exists()


# ── The summary line ────────────────────────────────────────────────────────


def _report(**counts) -> dict:
    report = {
        "read": 0,
        "turned": 0,
        "unchanged": 0,
        "low_confidence": 0,
        "none_joined": 0,
        "weak_turn": 0,
        "kept": 0,
        "at_infinity": 0,
        "not_asked": 0,
        "median_turn_deg": float("nan"),
        "mean_agreement": float("nan"),
        "points": [],
    }
    report.update(counts)
    return report


def test_summary_reports_counts_median_turn_and_mean_agreement():
    line = _summary(
        _report(
            read=266,
            turned=258,
            low_confidence=168,
            none_joined=2,
            weak_turn=6,
            at_infinity=14,
            median_turn_deg=17.04,
            mean_agreement=0.8376,
        )
    )
    assert line == (
        "  Turned 258 of 266 normals (median turn 17.0°, mean agreement 0.84; "
        "168 low-confidence); left unchanged: 2 no view joined, "
        "6 large turn on weak evidence, 0 refused or with no bitmap; "
        "14 at infinity skipped"
    )


def test_summary_with_nothing_turned_has_no_median_or_agreement():
    line = _summary(_report(read=3, none_joined=3))
    assert line == (
        "  Turned 0 of 3 normals; left unchanged: 3 no view joined, "
        "0 large turn on weak evidence, 0 refused or with no bitmap"
    )


# ── Runs over real reconstructions ──────────────────────────────────────────


@pytest.fixture(scope="module")
def ground_truth_refined():
    """The seoul_bull ground truth (embedded patches, 14 points at infinity),
    read in place and never written, and the transform's output."""
    recon = SfmrReconstruction.load(str(SEOUL_BULL_GROUND_TRUTH))
    return recon, RefineNormalsTransform().apply(recon)


def test_refine_normals_turns_finite_normals_and_moves_nothing(ground_truth_refined):
    recon, out = ground_truth_refined
    assert out.point_count == recon.point_count
    assert out.observation_count == recon.observation_count
    np.testing.assert_array_equal(
        np.asarray(out.positions_xyzw), np.asarray(recon.positions_xyzw)
    )
    np.testing.assert_array_equal(
        np.asarray(out.keypoints_xy), np.asarray(recon.keypoints_xy)
    )
    np.testing.assert_array_equal(
        np.asarray(out.track_image_indexes), np.asarray(recon.track_image_indexes)
    )

    finite = ~np.asarray(recon.point_is_at_infinity, dtype=bool)
    before = np.asarray(recon.normals)[finite]
    after = np.asarray(out.normals)[finite]
    turned = (after != before).any(axis=1)
    # Most finite points are turned (258 of 266 on this file).
    assert turned.mean() > 0.8
    cos_turn = np.clip((after[turned] * before[turned]).sum(axis=1), -1.0, 1.0)
    assert np.degrees(np.arccos(cos_turn)).max() > 5.0


def test_refine_normals_leaves_points_at_infinity_untouched(ground_truth_refined):
    recon, out = ground_truth_refined
    at_infinity = np.asarray(recon.point_is_at_infinity, dtype=bool)
    assert at_infinity.any()
    np.testing.assert_array_equal(
        np.asarray(out.normals)[at_infinity], np.asarray(recon.normals)[at_infinity]
    )
    np.testing.assert_array_equal(
        np.asarray(out.colors)[at_infinity], np.asarray(recon.colors)[at_infinity]
    )


def test_refine_normals_prints_its_summary(capsys):
    recon = SfmrReconstruction.load(str(SEOUL_BULL_GROUND_TRUTH))
    RefineNormalsTransform().apply(recon)
    out = capsys.readouterr().out
    assert "Refining the normals of 280 points across 17 images" in out
    assert "  Turned " in out
    assert "median turn" in out
    assert "mean agreement" in out
    assert "14 at infinity skipped" in out


def _embedded(workspace) -> SfmrReconstruction:
    return SfmrReconstruction.load(workspace).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )


def test_refine_normals_renders_bitmaps_only_where_stored(seoul_bull_workspace):
    """A value with no bitmaps gets none; one that stores them has the
    rewritten points' bitmaps rendered again and every other row kept."""
    recon = _embedded(seoul_bull_workspace)
    assert recon.patch_bitmaps is None
    assert RefineNormalsTransform().apply(recon).patch_bitmaps is None

    with_bitmaps = AddPatchBitmapsTransform().apply(recon)
    out = RefineNormalsTransform().apply(with_bitmaps)
    before = np.asarray(with_bitmaps.patch_bitmaps)
    after = np.asarray(out.patch_bitmaps)
    assert after.shape == before.shape
    rewritten = (np.asarray(out.normals) != np.asarray(with_bitmaps.normals)).any(
        axis=1
    )
    assert rewritten.any() and not rewritten.all()
    np.testing.assert_array_equal(after[~rewritten], before[~rewritten])
    assert (after[rewritten] != before[rewritten]).any(axis=(1, 2, 3)).mean() > 0.9


def test_missing_image_is_hard_error(seoul_bull_workspace):
    recon = _embedded(seoul_bull_workspace)
    from pathlib import Path

    (Path(recon.workspace_dir) / recon.image_names[0]).unlink()
    with pytest.raises(FileNotFoundError):
        RefineNormalsTransform().apply(recon)


def test_cli_refine_normals(seoul_bull_workspace):
    """``--to-embedded-patches --refine-normals`` writes turned normals."""
    input_sfmr = seoul_bull_workspace
    output_sfmr = input_sfmr.with_name("refined.sfmr")
    result = CliRunner().invoke(
        main,
        [
            "xform",
            str(input_sfmr),
            str(output_sfmr),
            "--to-embedded-patches",
            "--refine-normals",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "mean agreement" in result.output

    converted = _embedded(input_sfmr)
    refined = SfmrReconstruction.load(output_sfmr)
    assert refined.point_count == converted.point_count
    assert not np.array_equal(
        np.asarray(converted.normals), np.asarray(refined.normals)
    )


def test_cli_refine_normals_bare_before_other_option(seoul_bull_workspace):
    """A bare --refine-normals followed by another option leaves that option
    intact. The refinement is stubbed: this checks the option reading."""
    input_sfmr = seoul_bull_workspace
    output_sfmr = input_sfmr.with_name("refined_bare.sfmr")
    args = [
        "xform",
        str(input_sfmr),
        str(output_sfmr),
        "--to-embedded-patches",
        "--refine-normals",
        "--scale",
        "2.0",
    ]
    captured: list[RefineNormalsTransform] = []

    def _stub_apply(self, recon):
        captured.append(self)
        return recon

    with patch.object(RefineNormalsTransform, "apply", _stub_apply):
        result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    original = SfmrReconstruction.load(input_sfmr)
    refined = SfmrReconstruction.load(output_sfmr)
    np.testing.assert_allclose(
        np.asarray(refined.positions), np.asarray(original.positions) * 2.0, rtol=1e-5
    )


def test_refine_normals_rejects_sift_files(seoul_bull_workspace):
    """A sift_files input is refused before any image is read, with a pointer
    to the conversion."""
    output_sfmr = seoul_bull_workspace.with_name("rejected.sfmr")
    result = CliRunner().invoke(
        main, ["xform", str(seoul_bull_workspace), str(output_sfmr), "--refine-normals"]
    )
    assert result.exit_code != 0
    assert "embedded_patches" in result.output
    assert not output_sfmr.exists()


def test_refine_normals_rejects_wrong_chain_order(seoul_bull_workspace):
    """The gate checks the reconstruction as it stands at the step, so
    ``--refine-normals`` before ``--to-embedded-patches`` is refused."""
    output_sfmr = seoul_bull_workspace.with_name("wrong_order.sfmr")
    result = CliRunner().invoke(
        main,
        [
            "xform",
            str(seoul_bull_workspace),
            str(output_sfmr),
            "--refine-normals",
            "--to-embedded-patches",
        ],
    )
    assert result.exit_code != 0
    assert "embedded_patches" in result.output
    assert not output_sfmr.exists()
