# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Console rendering for `sfm motion` reconstruction analysis.

Pure presentation: the per-frame discontinuity table (one per sequence) and
the end-of-run summary table.  All computation lives in
`recon_discontinuity.py`; these helpers only format already-computed
per-sequence result dicts for the terminal, so the printed output and the
JSON report in `report.py` are both driven by the same computed data.
"""

import click
import numpy as np

from .constants import (
    OBS_WINDOW,
    OBS_Z_THRESHOLD,
    OVERLAP_DROP_THRESHOLD,
    OVERLAP_WINDOW,
    STEP_RATIO_THRESHOLD,
    STEP_RATIO_WINDOW,
)
from .recon_discontinuity import _flag_frame, _rotation_angle_deg


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
        f"    Obs-count outlier (w={OBS_WINDOW}):  "
        f"|z-score| > {OBS_Z_THRESHOLD} (low tail)"
    )
    click.echo("")
    click.echo(
        f"  {'Frame':>7}  {'Image':<40}  "
        f"{'L.trans':>7}  {'L.rot':>6}  "
        f"{'R.trans':>7}  {'R.rot':>6}  "
        f"{'StepR':>6}  {'CovR':>6}  {'ObsZ':>6}  {'Flag'}"
    )
    click.echo("  " + "-" * 120)

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
            f"{lt_s:>7}  {lr_s:>6}  "
            f"{rt_s:>7}  {rr_s:>6}  "
            f"{step_r_s:>6}  {cov_r_s:>6}  {obs_z_s:>6}  {flag_str}"
        )


def _print_summary_edge_row(
    seq_result: dict,
    edge: tuple[int, int],
    evidence: list[str],
) -> None:
    """Render one edge row in the summary table for a flagged discontinuity."""
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
    rot = _rotation_angle_deg(seq_quats[a], seq_quats[b])

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

    # Determine which signals contributed to this edge.
    has_p = any(
        " L.t" in e or " L.r" in e or " R.t" in e or " R.r" in e for e in evidence
    )
    has_s = any(" Step" in e for e in evidence)
    has_c = any(" Cov" in e for e in evidence)
    has_o = any(" Obs" in e for e in evidence)
    has_pose_t = any(" L.t" in e or " R.t" in e for e in evidence)
    has_pose_r = any(" L.r" in e or " R.r" in e for e in evidence)

    dist_s = f"<{dist:.4f}>" if has_pose_t else f" {dist:.4f} "
    rot_s = f"<{rot:.2f}°>" if has_pose_r else f" {rot:.2f}° "

    if step_r is None:
        step_r_s = "-"
    elif has_s:
        step_r_s = f"<{step_r:.2f}>"
    else:
        step_r_s = f" {step_r:.2f} "

    if cov_r is None:
        cov_r_s = "-"
    elif cov_r == float("inf"):
        cov_r_s = "<inf>" if has_c else " inf "
    elif has_c:
        cov_r_s = f"<{cov_r:.2f}>"
    else:
        cov_r_s = f" {cov_r:.2f} "

    def _fmt_z(z, highlight):
        if z is None:
            return "-"
        s = f"{z:+.1f}"
        return f"<{s}>" if highlight and z < -OBS_Z_THRESHOLD else s

    obs_a_s = _fmt_z(obs_a, has_o)
    obs_b_s = _fmt_z(obs_b, has_o)
    obs_pair = f"{obs_a_s}/{obs_b_s}"

    signals_parts = []
    if has_p:
        signals_parts.append("P")
    if has_s:
        signals_parts.append("S")
    if has_c:
        signals_parts.append("C")
    if has_o:
        signals_parts.append("O")
    signals_str = " ".join(signals_parts)

    edge_str = f"{frame_a}->{frame_b}"
    click.echo(
        f"    {edge_str:>14}  {dist_s:>10}  {rot_s:>9}  "
        f"{step_r_s:>8}  {cov_r_s:>8}  {obs_pair:>11}  "
        f"{shared:>9}  {err_a_s:>8}  {err_b_s:>8}  {signals_str}"
    )


def print_summary(all_sequence_results: list[dict]) -> None:
    """Render the end-of-run summary table grouped across all sequences."""
    click.echo("")
    click.echo("=" * 80)
    click.echo("Summary")
    click.echo("=" * 80)

    total_discontinuities = sum(len(s["core_edges"]) for s in all_sequence_results)

    for s in all_sequence_results:
        core_edges = s["core_edges"]
        click.echo(
            f"\n  {s['sequence_name']}: {s['frame_count']} frames, "
            f"{len(core_edges)} discontinuity(s)"
        )

        if not core_edges:
            click.echo("    No pose discontinuities detected.")
            continue

        click.echo(
            f"    {'Edge':>14}  {'Dist':>8}  {'Rot':>7}  "
            f"{'StepR':>6}  {'CovR':>6}  {'ObsZ(A/B)':>11}  "
            f"{'SharedPts':>9}  {'Err(A)':>8}  {'Err(B)':>8}  {'Signals'}"
        )
        click.echo("    " + "-" * 105)

        for edge, evidence in sorted(core_edges.items()):
            _print_summary_edge_row(s, edge, evidence)

    click.echo("")

    # Footer: partition detections by signal-agreement level.  An edge counts
    # as "high confidence" when at least two distinct signals fire — whether
    # that's two primary signals (P/S/C) or one primary plus an Obs outlier on
    # an endpoint.  A single fire is "low confidence".
    single_signal = 0
    multi_signal = 0
    for s in all_sequence_results:
        for evidence in s["core_edges"].values():
            codes = set()
            for e in evidence:
                if any(t in e for t in (" L.t", " L.r", " R.t", " R.r")):
                    codes.add("P")
                if " Step" in e:
                    codes.add("S")
                if " Cov" in e:
                    codes.add("C")
                if " Obs" in e:
                    codes.add("O")
            if len(codes) >= 2:
                multi_signal += 1
            else:
                single_signal += 1

    if total_discontinuities == 0:
        click.echo("No pose discontinuities detected in any sequence.")
    else:
        click.echo(
            f"Total: {total_discontinuities} discontinuity(s) "
            f"across {len(all_sequence_results)} sequence(s)."
        )
        click.echo(f"  Single-signal (low confidence):  {single_signal}")
        click.echo(f"  Multi-signal  (high confidence): {multi_signal}")
        click.echo(
            "  Legend: P=pose residual, S=step-size ratio, "
            "C=coviz drop, O=obs-count outlier."
        )
