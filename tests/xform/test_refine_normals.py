# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--refine-normals`` xform operation.

Covers the option's value (empty, or ``method=``), the summary lines of both
steps, and runs over the seoul_bull ground truth and an ``embedded_patches`` conversion of the
seoul_bull workspace: finite points' normals turn, no position or keypoint
moves, points at infinity are untouched, and bitmaps are rendered again only
where the reconstruction stores them. See
specs/cli/reconstruction/xform/refine-normals-command.md.
"""

import signal
import threading
from contextlib import contextmanager
from unittest.mock import patch

import click
import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.xform import AddPatchBitmapsTransform, RefineNormalsTransform
from sfmtool.xform._arg_parser import parse_refine_normals_params
from sfmtool.xform import _refine_normals
from sfmtool.xform._refine_normals import _summary

from ..conftest import SEOUL_BULL_GROUND_TRUTH, SEOUL_BULL_POINT_STEP, every_nth_point


# ── The option's value ──────────────────────────────────────────────────────


@pytest.mark.parametrize("param", ["", ",", " , "])
def test_parse_without_keys_is_the_grown_plane(param):
    transform = parse_refine_normals_params(param)
    assert isinstance(transform, RefineNormalsTransform)
    assert transform.method == "grown_plane"
    assert transform.description() == "Refine normals (grown plane)"


@pytest.mark.parametrize("method", ["grown_plane", "pairwise_flow"])
def test_parse_method(method):
    transform = parse_refine_normals_params(f"method={method}")
    assert transform.method == method


def test_pairwise_flow_description():
    transform = parse_refine_normals_params("method=pairwise_flow,")
    assert transform.description() == "Refine normals (pairwise flow)"


@pytest.mark.parametrize("value", ["pairwise-flow", "flow", "Grown_Plane", ""])
def test_unknown_method_is_a_usage_error(value):
    with pytest.raises(click.UsageError) as raised:
        parse_refine_normals_params(f"method={value}")
    message = str(raised.value)
    assert f"'{value}' is not grown_plane or pairwise_flow" in message


@pytest.mark.parametrize(
    "param, error",
    [
        ("method", "expected key=value"),
        ("method=grown_plane,method=pairwise_flow", "Duplicate --refine-normals key"),
    ],
)
def test_malformed_method_is_a_usage_error(param, error):
    with pytest.raises(click.UsageError, match=error):
        parse_refine_normals_params(param)


@pytest.mark.parametrize(
    "param",
    [
        "angular_range_deg=25,init_steps=7",
        "bitmaps=false",
        "resolution=32",
        "quality=fine",
        "k_neighbors=8",
        "init_steps",
        "method=pairwise_flow,bitmaps=false",
    ],
)
def test_any_other_key_is_a_usage_error_naming_the_change(param):
    with pytest.raises(click.UsageError) as raised:
        parse_refine_normals_params(param)
    message = str(raised.value)
    assert "takes only the key method" in message
    # The message quotes the other keys, and not the method.
    other = param.removeprefix("method=pairwise_flow,")
    assert f"was given '{other}'." in message
    assert "grown-plane normal" in message
    assert "pairwise-flow normal" in message
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
    assert "takes only the key method" in result.output
    assert not output_sfmr.exists()


def test_cli_unknown_method_is_rejected_before_loading(seoul_bull_workspace):
    output_sfmr = seoul_bull_workspace.with_name("rejected_method.sfmr")
    result = CliRunner().invoke(
        main,
        [
            "xform",
            str(seoul_bull_workspace),
            str(output_sfmr),
            "--to-embedded-patches",
            "--refine-normals",
            "method=pairwise-flow",
        ],
    )
    assert result.exit_code == 2
    assert "'pairwise-flow' is not grown_plane or pairwise_flow" in result.output
    assert "Loading reconstruction" not in result.output
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
        "single_pair": 0,
        "undetermined": 0,
        "unstable": 0,
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
        ),
        "grown_plane",
    )
    assert line == (
        "  Turned 258 of 266 normals (median turn 17.0°, mean agreement 0.84; "
        "168 low-confidence); left unchanged: 2 no view joined, "
        "6 large turn on weak evidence, 0 refused or with no bitmap; "
        "14 at infinity skipped"
    )


def test_summary_with_nothing_turned_has_no_median_or_agreement():
    line = _summary(_report(read=3, none_joined=3), "grown_plane")
    assert line == (
        "  Turned 0 of 3 normals; left unchanged: 3 no view joined, "
        "0 large turn on weak evidence, 0 refused or with no bitmap"
    )


def test_pairwise_flow_summary_reports_its_own_outcomes():
    line = _summary(
        _report(
            read=40,
            turned=30,
            single_pair=4,
            undetermined=6,
            unstable=3,
            kept=1,
            at_infinity=2,
            median_turn_deg=8.24,
            # The grown plane's figures, which the pairwise line leaves out.
            low_confidence=7,
            none_joined=8,
            weak_turn=9,
            mean_agreement=0.5,
        ),
        "pairwise_flow",
    )
    assert line == (
        "  Turned 30 of 40 normals (median turn 8.2°; 4 from a single pair); "
        "left unchanged: 6 undetermined, 3 unstable, "
        "1 refused or with no bitmap; 2 at infinity skipped"
    )


def test_pairwise_flow_summary_with_nothing_turned_has_no_median():
    line = _summary(_report(read=2, undetermined=2), "pairwise_flow")
    assert line == (
        "  Turned 0 of 2 normals; left unchanged: 2 undetermined, 0 unstable, "
        "0 refused or with no bitmap"
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


def test_pairwise_flow_turns_finite_normals_and_skips_infinity(capsys):
    """``method="pairwise_flow"`` reaches the binding: on every seventieth point
    of the ground truth, and point 86, which is at infinity, the run writes
    normals other than the grown plane's, moves no point, leaves the point at
    infinity alone and prints the pairwise-flow summary. The pairwise-flow
    normal costs seconds of one thread per point, so the run reads 5 of the
    280 points."""
    recon = every_nth_point(
        SfmrReconstruction.load(str(SEOUL_BULL_GROUND_TRUTH)), 70, keep=[86]
    )
    at_infinity = np.asarray(recon.point_is_at_infinity, dtype=bool)
    assert at_infinity.sum() == 1

    out = RefineNormalsTransform("pairwise_flow").apply(recon)
    printed = capsys.readouterr().out
    grown = RefineNormalsTransform().apply(recon)

    assert "Refining the normals of 5 points across 17 images" in printed
    assert "  Turned " in printed
    assert "from a single pair" in printed
    assert " undetermined, " in printed
    assert " unstable, " in printed
    assert "mean agreement" not in printed
    assert "no view joined" not in printed
    assert "1 at infinity skipped" in printed

    np.testing.assert_array_equal(
        np.asarray(out.positions_xyzw), np.asarray(recon.positions_xyzw)
    )
    np.testing.assert_array_equal(
        np.asarray(out.keypoints_xy), np.asarray(recon.keypoints_xy)
    )
    before = np.asarray(recon.normals)
    after = np.asarray(out.normals)
    np.testing.assert_array_equal(after[at_infinity], before[at_infinity])
    assert (after[~at_infinity] != before[~at_infinity]).any(axis=1).any()
    assert not np.array_equal(after, np.asarray(grown.normals))


def _embedded(workspace) -> SfmrReconstruction:
    return SfmrReconstruction.load(workspace).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )


def test_refine_normals_renders_bitmaps_only_where_stored(seoul_bull_normals_refined):
    """A value with no bitmaps gets none; one that stores them has the
    rewritten points' bitmaps rendered again and every other row kept."""
    recon, refined = seoul_bull_normals_refined
    assert recon.patch_bitmaps is None
    assert refined.patch_bitmaps is None

    # Bitmaps at R = 12 rather than the default 24: the refinement runs at the
    # value's patch resolution, so this pass costs about a third as much, and
    # it still rewrites most points. The fixture's point at infinity is one it
    # does not rewrite.
    with_bitmaps = AddPatchBitmapsTransform(resolution=12).apply(recon)
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
    """``--to-embedded-patches --refine-normals`` writes turned normals. It
    reads every eighth point of the workspace's reconstruction, saved beside
    it."""
    input_sfmr = seoul_bull_workspace.with_name("subset.sfmr")
    every_nth_point(
        SfmrReconstruction.load(seoul_bull_workspace), SEOUL_BULL_POINT_STEP
    ).save(str(input_sfmr), operation="test subset")
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


def test_the_run_counts_every_point_on_its_progress_counter():
    """The transform hands the binding the counter its poller echoes, which
    climbs to every point, the one at infinity included."""
    recon = every_nth_point(
        SfmrReconstruction.load(str(SEOUL_BULL_GROUND_TRUTH)), 70, keep=[86]
    )
    counters = []
    poll = _refine_normals._poll_progress

    @contextmanager
    def _recording(log, total, **kwargs):
        assert total == recon.point_count
        with poll(log, total, **kwargs) as counter:
            counters.append(counter)
            yield counter

    with patch.object(_refine_normals, "_poll_progress", _recording):
        RefineNormalsTransform().apply(recon)
    assert len(counters) == 1
    assert counters[0].value == recon.point_count


def test_cli_ctrl_c_stops_a_pairwise_flow_run_and_writes_nothing(tmp_path):
    """Ctrl-C part way through ``--refine-normals method=pairwise_flow`` over
    every point of the ground truth, about a minute of work, stops the run:
    SIGINT is raised once the run has counted its first point, so it lands
    inside the binding, and the command aborts without writing. The ground
    truth is read in place; the output would go to a temporary folder."""
    output_sfmr = tmp_path / "interrupted.sfmr"
    poll = _refine_normals._poll_progress

    def _interrupt_once_counting(counter):
        # Give up after a minute rather than wait forever on a run that
        # failed before counting; the assertions then fail.
        for _ in range(6000):
            if counter.value:
                signal.raise_signal(signal.SIGINT)
                return
            threading.Event().wait(0.01)

    threads = []

    @contextmanager
    def _interrupting(log, total, **kwargs):
        with poll(log, total, **kwargs) as counter:
            thread = threading.Thread(target=_interrupt_once_counting, args=(counter,))
            threads.append(thread)
            thread.start()
            yield counter

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with patch.object(_refine_normals, "_poll_progress", _interrupting):
            result = CliRunner().invoke(
                main,
                [
                    "xform",
                    str(SEOUL_BULL_GROUND_TRUTH),
                    str(output_sfmr),
                    "--refine-normals",
                    "method=pairwise_flow",
                ],
            )
        for thread in threads:
            thread.join()
    finally:
        signal.signal(signal.SIGINT, previous)

    assert len(threads) == 1
    assert result.exit_code == 1, result.output
    assert "Aborted!" in result.output
    assert "Turned " not in result.output
    assert not output_sfmr.exists()


def test_cli_refine_normals_method_reaches_the_transform(seoul_bull_workspace):
    """``--refine-normals method=pairwise_flow`` builds a pairwise-flow step,
    separated or joined. The refinement is stubbed: this checks the option
    reading."""
    output_sfmr = seoul_bull_workspace.with_name("refined_method.sfmr")
    captured: list[RefineNormalsTransform] = []

    def _stub_apply(self, recon):
        captured.append(self)
        return recon

    with patch.object(RefineNormalsTransform, "apply", _stub_apply):
        result = CliRunner().invoke(
            main,
            [
                "xform",
                str(seoul_bull_workspace),
                str(output_sfmr),
                "--to-embedded-patches",
                "--refine-normals",
                "method=pairwise_flow",
                "--refine-normals=method=pairwise_flow",
                "--refine-normals",
            ],
        )

    assert result.exit_code == 0, result.output
    assert [t.method for t in captured] == [
        "pairwise_flow",
        "pairwise_flow",
        "grown_plane",
    ]
    assert "Refine normals (pairwise flow)" in result.output


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
