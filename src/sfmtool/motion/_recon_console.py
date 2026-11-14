# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Console rendering for `sfm motion` reconstruction analysis.

Pure presentation: each sequence's per-frame table and discontinuity table,
printed together in that sequence's section, and the end-of-run totals and
verdict.  All computation lives in
`recon_discontinuity.py`; these helpers only format already-computed
per-sequence result dicts for the terminal, so the printed output and the
JSON report in `report.py` are both driven by the same computed data.
"""

import click
import numpy as np

from .constants import (
    BREAK_ROTATION_FACTOR,
    BREAK_SHARED_FRACTION,
    BREAK_WINDOW,
    OBS_WINDOW,
    OBS_Z_THRESHOLD,
    OVERLAP_DROP_THRESHOLD,
    OVERLAP_WINDOW,
    STEP_RATIO_THRESHOLD,
    STEP_RATIO_WINDOW,
)
from .._pose_math import rotation_angle_deg
from .recon_discontinuity import _break_edges, _flag_frame
from .report import _signals_from_evidence


def print_frame_table(seq_result: dict) -> None:
    """Render the per-frame discontinuity table for one sequence.

    Reads only already-computed fields from `seq_result`; per-row flags are
    recomputed via the shared `_flag_frame` so the displayed "Flag" column
    cannot drift from the analysis or the JSON report.
    """
    extrap_results = seq_result["extrap_results"]
    seq_image_names = seq_result["seq_image_names"]
    step_ratios = seq_result["step_ratios"]
    overlap_drops = seq_result["overlap_drops"]
    obs_z_scores = seq_result["obs_z_scores"]
    median_trans = seq_result["median_trans"]
    median_rot = seq_result["median_rot"]
    trans_threshold = seq_result["trans_threshold"]
    rot_threshold = seq_result["rot_threshold"]

    click.echo(
        f"\n  Successive motion --"
        f" median translation: {median_trans:.4f},"
        f" median rotation: {median_rot:.2f}°"
    )
    click.echo("  Signal thresholds:")
    click.echo(
        f"    Pose extrapolation:  trans>{trans_threshold:.4f} (3x median step)"
        f"  rot>{rot_threshold:.1f}°"
    )
    click.echo(
        f"    Step-size ratio  (w={STEP_RATIO_WINDOW}):  "
        f"max(pre/post, post/pre) > {STEP_RATIO_THRESHOLD}"
    )
    click.echo(
        f"    Coviz drop       (w={OVERLAP_WINDOW}):  "
        f"baseline/cross-overlap > {OVERLAP_DROP_THRESHOLD}"
    )
    click.echo(
        f"    Obs-count outlier (w={OBS_WINDOW}):  z-score < -{OBS_Z_THRESHOLD:g}"
    )
    click.echo("")
    header = (
        f"  {'Frame':>7}  {'Image':<40}  "
        f"{'L.trans':>7}  {'L.rot':>7}  "
        f"{'R.trans':>7}  {'R.rot':>7}  "
        f"{'StepR':>6}  {'CovR':>6}  {'ObsZ':>6}  {'Flag'}"
    )
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))

    for er in extrap_results:
        i = er["seq_idx"]
        frame_num = er["frame_number"]
        name = seq_image_names[i]
        if len(name) > 40:
            name = "..." + name[-37:]

        lt = er["left_trans_err"]
        lr = er["left_rot_err"]
        rt = er["right_trans_err"]
        rr = er["right_rot_err"]

        lt_s = f"{lt:.4f}" if lt is not None else "-"
        lr_s = f"{lr:.2f}°" if lr is not None else "-"
        rt_s = f"{rt:.4f}" if rt is not None else "-"
        rr_s = f"{rr:.2f}°" if rr is not None else "-"

        # StepR and CovR come from the landing edge (i-1, i) so they annotate
        # the frame that the edge lands on.  ObsZ is the per-frame z-score.
        landing_edge = i - 1
        step_r = step_ratios[landing_edge] if landing_edge >= 0 else None
        cov_r = overlap_drops[landing_edge] if landing_edge >= 0 else None
        obs_z = obs_z_scores[i] if i < len(obs_z_scores) else None

        step_r_s = f"{step_r:.2f}" if step_r is not None else "-"
        cov_r_s = (
            f"{cov_r:.2f}"
            if cov_r is not None and cov_r != float("inf")
            else ("inf" if cov_r == float("inf") else "-")
        )
        obs_z_s = f"{obs_z:+.1f}" if obs_z is not None else "-"

        flags = _flag_frame(
            left_trans_err=lt,
            left_rot_err=lr,
            right_trans_err=rt,
            right_rot_err=rr,
            step_ratio=step_r,
            overlap_drop=cov_r,
            obs_z=obs_z,
            trans_threshold=trans_threshold,
            rot_threshold=rot_threshold,
        )
        flag_str = ",".join(flags) if flags else ""

        click.echo(
            f"  {frame_num:>7}  {name:<40}  "
            f"{lt_s:>7}  {lr_s:>7}  "
            f"{rt_s:>7}  {rr_s:>7}  "
            f"{step_r_s:>6}  {cov_r_s:>6}  {obs_z_s:>6}  {flag_str}"
        )


def _count(n: int, singular: str, plural: str) -> str:
    """``n`` followed by the singular or the plural noun, as it reads."""
    return f"{n} {singular if n == 1 else plural}"


# The discontinuity table's columns before `Signals`, as (header, width). A
# column whose values can be wrapped in `< >` has a header with a trailing
# space, so the header ends over a value's last digit, not over its bracket.
_EDGE_COLUMNS = (
    ("Edge", 14),
    ("Dist", 8),
    ("Rot", 7),
    ("StepR ", 8),
    ("CovR ", 8),
    ("ObsZ(A/B)", 13),
    ("SharedPts", 9),
    ("Err(A)", 8),
    ("Err(B)", 8),
)


def _edge_table_header() -> str:
    left = "  ".join(f"{label:>{width}}" for label, width in _EDGE_COLUMNS)
    return f"    {left}  {'Signals':<7}  {'NearPts':>7}  {'RotR':>6}  Break"


def _print_edge_row(
    seq_result: dict,
    edge: tuple[int, int],
    evidence: list[str],
) -> None:
    """Render one row of a sequence's discontinuity table."""
    a, b = edge
    seq_img_indexes = seq_result["seq_image_indexes"]
    seq_frm_numbers = seq_result["seq_frame_numbers"]
    seq_centers = seq_result["seq_centers"]
    seq_quats = seq_result["seq_quats"]
    pair_counts = seq_result["pair_counts"]
    step_ratios = seq_result["step_ratios"]
    overlap_drops = seq_result["overlap_drops"]
    obs_z_scores = seq_result["obs_z_scores"]
    reproj_errors = seq_result["core_edge_reproj_errors"]

    img_a = seq_img_indexes[a]
    img_b = seq_img_indexes[b]
    frame_a = seq_frm_numbers[a]
    frame_b = seq_frm_numbers[b]

    dist = float(np.linalg.norm(seq_centers[b] - seq_centers[a]))
    rot = rotation_angle_deg(seq_quats[a], seq_quats[b])

    shared = pair_counts.get((img_a, img_b), 0)
    err_a = reproj_errors.get(img_a, float("nan"))
    err_b = reproj_errors.get(img_b, float("nan"))
    err_a_s = f"{err_a:.3f}px" if not np.isnan(err_a) else "N/A"
    err_b_s = f"{err_b:.3f}px" if not np.isnan(err_b) else "N/A"

    # Edge (a, b) corresponds to edge index a in the step/overlap arrays.
    step_r = step_ratios[a] if a < len(step_ratios) else None
    cov_r = overlap_drops[a] if a < len(overlap_drops) else None
    obs_a = obs_z_scores[a] if a < len(obs_z_scores) else None
    obs_b = obs_z_scores[b] if b < len(obs_z_scores) else None

    # In the legend's order, P S C O.
    fired = _signals_from_evidence(evidence)
    signals = [code for code in "PSCO" if code in fired]

    def _fmt_ratio(value, fired):
        if value is None:
            return "- "
        s = "inf" if value == float("inf") else f"{value:.2f}"
        return f"<{s}>" if fired else f" {s} "

    def _fmt_z(z, highlight):
        if z is None:
            return "-"
        s = f"{z:+.1f}"
        return f"<{s}>" if highlight and z < -OBS_Z_THRESHOLD else s

    has_o = "O" in signals
    verdict = seq_result["breaks"][edge]
    ratio = verdict["rotation_ratio"]
    ratio_s = "inf" if ratio == float("inf") else f"{ratio:.2f}"
    break_s = "BREAK" if verdict["is_break"] else "-"

    # Dist and Rot are the step between the edge's two images. No signal has
    # a threshold on them (the pose signal's are on the extrapolation
    # residuals of the frame table), so they are never wrapped in `< >`.
    values = (
        f"{frame_a}->{frame_b}",
        f"{dist:.4f}",
        f"{rot:.2f}°",
        _fmt_ratio(step_r, "S" in signals),
        _fmt_ratio(cov_r, "C" in signals),
        f"{_fmt_z(obs_a, has_o)}/{_fmt_z(obs_b, has_o)}",
        shared,
        err_a_s,
        err_b_s,
    )
    left = "  ".join(
        f"{value:>{width}}" for value, (_, width) in zip(values, _EDGE_COLUMNS)
    )
    click.echo(
        f"    {left}  {' '.join(signals):<7}  "
        f"{verdict['nearby_shared_points']:>7}  {ratio_s:>6}  {break_s}"
    )


def print_sequence_discontinuities(seq_result: dict) -> None:
    """Render one sequence's discontinuities: its break rule and edge table.

    Printed right after the sequence's frame table, so that with several
    sequences each sequence's section holds all of its own results.
    """
    core_edges = seq_result["core_edges"]
    name = seq_result["sequence_name"]
    if not core_edges:
        click.echo(f"\n  {name}: no discontinuities")
        return

    n_breaks = sum(v["is_break"] for v in seq_result["breaks"].values())
    click.echo(
        f"\n  {name}: {_count(len(core_edges), 'discontinuity', 'discontinuities')}"
        f", {_count(n_breaks, 'break', 'breaks')}"
    )
    ref = seq_result["break_reference"]
    click.echo(
        f"    Break rule: RotR > {BREAK_ROTATION_FACTOR:g} and SharedPts < "
        f"{BREAK_SHARED_FRACTION * ref['median_shared_points']:.1f} and "
        f"NearPts < "
        f"{BREAK_SHARED_FRACTION * ref['median_nearby_shared_points']:.1f}, "
        f"where expected rotation = {ref['rotation_per_frame_deg']:.3f}° per "
        f"frame number x max(gap, {ref['median_frame_gap']:g})"
    )
    header = _edge_table_header()
    click.echo(header)
    click.echo("    " + "-" * (len(header) - 4))
    for edge, evidence in sorted(core_edges.items()):
        _print_edge_row(seq_result, edge, evidence)


def verdict_line(all_sequence_results: list[dict]) -> str:
    """The one-line break verdict printed at the end of the summary."""
    found = _break_edges(all_sequence_results)
    if not found:
        return "Verdict: no breaks"
    several = len(all_sequence_results) > 1
    edges = ", ".join(
        f"{name} {a}->{b}" if several else f"{a}->{b}" for name, a, b in found
    )
    noun = "break" if len(found) == 1 else "breaks"
    return f"Verdict: {len(found)} {noun}: {edges}"


def print_summary(all_sequence_results: list[dict]) -> None:
    """Render the totals over every sequence, then the verdict line.

    Each sequence's discontinuity table is printed in its own section by
    `print_sequence_discontinuities`; this only counts them.
    """
    click.echo("")
    click.echo("=" * 80)
    click.echo("Summary")
    click.echo("=" * 80)
    click.echo("")

    several = len(all_sequence_results) > 1
    total = 0
    total_breaks = 0
    single_signal = 0
    multi_signal = 0
    for s in all_sequence_results:
        n_edges = len(s["core_edges"])
        n_breaks = sum(v["is_break"] for v in s["breaks"].values())
        total += n_edges
        total_breaks += n_breaks
        if several:
            click.echo(
                f"  {s['sequence_name']}: {s['frame_count']} frames, "
                f"{_count(n_edges, 'discontinuity', 'discontinuities')}, "
                f"{_count(n_breaks, 'break', 'breaks')}"
            )
        # Counted by the distinct signal codes, as the JSON report's
        # `confidence` is, so the two cannot disagree.
        for evidence in s["core_edges"].values():
            if len(_signals_from_evidence(evidence)) >= 2:
                multi_signal += 1
            else:
                single_signal += 1

    if total == 0:
        click.echo("No discontinuities in any sequence.")
    else:
        sequences = _count(len(all_sequence_results), "sequence", "sequences")
        click.echo(
            f"Total: {_count(total, 'discontinuity', 'discontinuities')} in "
            f"{sequences}, {_count(total_breaks, 'break', 'breaks')}."
        )
        click.echo(f"  Found by one signal:          {single_signal}")
        click.echo(f"  Found by two or more signals: {multi_signal}")
        for line in (
            "A discontinuity is an edge where at least one signal found the",
            "camera motion irregular; the two counts above say how many signals",
            "agree on that, not whether it is a break. A break is a",
            "discontinuity that passes the break rule, a place where the solve put",
            "the parts of the camera path on either side in inconsistent places;",
            "the number of signals is not part of that rule.",
            "Legend: P=pose residual, S=step-size ratio, C=coviz drop, "
            "O=obs-count outlier.",
            "< > marks a value past its signal's threshold. Dist and Rot are "
            "the step between the edge's two images.",
            f"NearPts = points seen both in the {BREAK_WINDOW} images up to the "
            f"edge and the {BREAK_WINDOW} after it. RotR = rotation / expected "
            "rotation. BREAK = RotR, SharedPts and NearPts all past the break "
            "rule.",
        ):
            click.echo(f"  {line}")

    click.echo(verdict_line(all_sequence_results))
