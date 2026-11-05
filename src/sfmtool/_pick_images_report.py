# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The text `sfm pick-images` prints from the report `pick_images` returns.

The report names frames by their position in their sequence; every line here
names them by file number, through the command's sequences. See
specs/cli/image-feature/pick-images-command.md § "Report".
"""

import math

import numpy as np

from ._pick_images import PickSequence

# How many items a line lists before it gives the rest as a count.
_SHOWN = 8
# How many rows the motion table divides a sequence into.
_MOTION_ROWS = 10
# The images in each suggested slice of a written list.
SLICE_IMAGES = 30


def _items(items: list[str]) -> str:
    text = ", ".join(items[:_SHOWN])
    if len(items) > _SHOWN:
        text += f", ... and {len(items) - _SHOWN} more"
    return text


def _pct(value) -> str:
    """A step or a motion in percent; `None` is a value the report could not
    measure (too few matches)."""
    return "unmeasured" if value is None else f"{value:.2f}%"


def _share(value) -> str:
    return "unmeasured" if value is None else f"{value:.2f}"


def _span(first: int, last: int) -> str:
    return str(first) if first == last else f"{first}-{last}"


def _not_a_video_reasons(video: dict, frame_count: int) -> str:
    if frame_count < 4:
        return f"{frame_count} frames, too few to tell a video from photographs"
    reasons = []
    if not video["step_small"]:
        reasons.append(f"steps 1 frame apart median {_pct(video['step_1_median'])}")
    if not video["step_steady"]:
        change = video["step_change_median"]
        reasons.append(
            "neighbouring samples' steps erratic"
            if change is None
            else f"neighbouring samples' steps differ by a factor of {math.exp(change):.1f}"
        )
    if not video["motion_adds_up"]:
        ratio = video["ratio_3_to_1_median"]
        reasons.append(
            "3 frames apart unmeasured"
            if ratio is None
            else f"3 frames apart {ratio:.1f} times as far"
        )
    if not video["neighbours_match"]:
        reasons.append(
            f"F-inlier share 1 frame apart {_share(video['f_inlier_share_1_median'])}"
        )
    return ", ".join(reasons)


def _not_solvable_reasons(check: dict) -> str:
    reasons = []
    if check["panorama"]:
        reasons.append(
            f"panorama: 30 frames apart, {check['h_over_f_30_median']:.3f} of the "
            "F-matrix inliers fit one homography and it is a pure rotation "
            f"(error {check['rotation_error_30_median']:.3f})"
        )
    if check["moving_scene"]:
        reasons.append(
            f"moving scene: only {check['f_inlier_share_10_median']:.2f} of the "
            "matches 10 frames apart fit one camera motion"
        )
    if check["barely_moves"]:
        reasons.append(
            "the camera barely moves: "
            f"{check['total_motion_percent']:.0f}% of the long side in all"
        )
    return "; ".join(reasons)


def verdict_line(report: dict, sequences: list[PickSequence]) -> str:
    """The `Verdict:` line, naming each sequence that stopped the run."""
    verdict = report["verdict"]
    if verdict == "solvable":
        return "Verdict: solvable"
    if verdict == "not_a_video":
        parts = [
            f"{s.name}: {_not_a_video_reasons(r['video'], len(s.numbers))}"
            for s, r in zip(sequences, report["sequences"])
            if not r["video"]["is_video"]
        ]
        return (
            "Verdict: not a video: "
            + "; ".join(parts)
            + "; sfm pick-images picks from video sequences only"
        )
    parts = [
        f"{s.name}: {_not_solvable_reasons(r['solvability'])}"
        for s, r in zip(sequences, report["sequences"])
        if r["solvability"] is not None and not r["solvability"]["solvable"]
    ]
    return "Verdict: not solvable: " + "; ".join(parts)


def _counts(report: dict, sequence: int, stages: tuple[str, ...], extracted: dict):
    """How many frames of `sequence` the given stages read, and how many of
    them had a `.sift` file written."""
    read = written = 0
    for frame in report["frames"]:
        if frame["id"]["sequence"] == sequence and frame["stage"] in stages:
            read += 1
            written += bool(extracted.get(frame["id"]["frame"]))
    return read, written


def baseline_lines(
    report: dict,
    index: int,
    sequence: PickSequence,
    extracted: dict,
    label: str,
) -> list[str]:
    """The baseline section of one sequence."""
    seq = report["sequences"][index]
    base = seq["baseline"]
    numbers = sequence.numbers
    read, written = _counts(report, index, ("baseline_near", "baseline_far"), extracted)
    lines = [
        f"Baseline{label}: {len(base['samples'])} frames sampled, "
        f"{base['frames_read']} frames read ({written} extracted, "
        f"{read - written} reused)"
    ]
    medians = [s["sharpness_median"] for s in base["sensors"]]
    if sequence.sensors is None:
        sharpness = f"median {medians[0]:.4g}"
    else:
        sharpness = "median " + ", ".join(
            f"{m:.4g} ({p})" for m, p in zip(medians, sequence.sensors)
        )
    blurred = base["blurred_samples"]
    lines.append(
        f"  Sharpness: {sharpness}; {blurred} sampled "
        f"frame{'' if blurred == 1 else 's'} blurred"
    )
    lines.append(
        f"  Motion per frame: median {_pct(base['rate_median'])}, "
        f"p90 {_pct(base['rate_p90'])} of the long side; "
        "total "
        + (
            "unmeasured"
            if base["total_motion_percent"] is None
            else f"{base['total_motion_percent']:.0f}%"
        )
    )
    video = seq["video"]
    signs = sum(
        video[k]
        for k in ("motion_adds_up", "step_small", "step_steady", "neighbours_match")
    )
    change = video["step_change_median"]
    ratio = video["ratio_3_to_1_median"]
    lines.append(
        f"  {'A video' if video['is_video'] else 'Not a video'} ({signs} of 4 signs): "
        f"1 frame apart step {_pct(video['step_1_median'])}, "
        f"F-inlier share {_share(video['f_inlier_share_1_median'])}; 3 frames apart"
    )
    lines.append(
        "    "
        + ("unmeasured" if ratio is None else f"{ratio:.1f} times as far")
        + "; neighbouring samples' steps within a factor of "
        + ("unmeasured" if change is None else f"{math.exp(change):.1f}")
    )
    check = seq["solvability"]
    if check is not None:
        lines.append(
            f"  10 frames apart: F-inlier share {_share(check['f_inlier_share_10_median'])}"
        )
        error = check["rotation_error_30_median"]
        lines.append(
            f"  30 frames apart: H/F {_share(check['h_over_f_30_median'])}, "
            f"rotation error {'unmeasured' if error is None else f'{error:.3f}'}"
        )
    regions = []
    for region in base["flagged_regions"]:
        why = [
            name
            for name, flag in (
                ("blurred", region["blurred"]),
                ("weak matches", region["weak_matches"]),
            )
            if flag
        ]
        regions.append(
            f"{_span(numbers[region['first']], numbers[region['last']])} "
            f"({', '.join(why)})"
        )
    lines.append(f"  Flagged regions: {_items(regions) if regions else 'none'}")
    return lines


def _motion_rows(walk: dict, numbers: list[int]) -> list[str]:
    """The motion table: the sequence in up to ten runs of positions, each
    with the median step, rate and stride of the steps that start in it."""
    steps = walk["steps"]
    if not steps:
        return ["    no steps"]
    length = len(numbers)
    rows = []
    for row in range(_MOTION_ROWS):
        first = row * length // _MOTION_ROWS
        last = (row + 1) * length // _MOTION_ROWS - 1
        if last < first:
            continue
        inside = [s for s in steps if first <= s["from"] <= last]
        if not inside:
            continue
        step = np.median([s["stats"]["step_percent"] for s in inside])
        rate = np.median([s["rate"] for s in inside])
        stride = np.median([s["to"] - s["from"] for s in inside])
        frames = f"{numbers[first]}-{numbers[last]}"
        rows.append(f"    frames {frames:>13}  {step:6.1f}%  {rate:6.2f}%  {stride:5g}")
    return rows


def walk_lines(
    report: dict,
    index: int,
    sequence: PickSequence,
    extracted: dict,
    label: str,
    target_step: float,
    step_band: float,
) -> list[str]:
    """The walk section of one sequence."""
    walk = report["sequences"][index]["walk"]
    numbers = sequence.numbers
    steps = walk["steps"]

    def number(position: int) -> int:
        return numbers[position]

    def step_text(step: dict) -> str:
        return f"{number(step['from'])}->{number(step['to'])}"

    low, high = target_step * (1 - step_band), target_step * (1 + step_band)
    breaks = walk["breaks"]
    segments = walk["segments"]
    lines = [
        f"Walk{label}: target step {target_step:g}% (band {low:.3g}-{high:.3g}%), "
        f"{len(breaks)} break{'s' if len(breaks) != 1 else ''}, "
        f"{len(segments)} segment{'s' if len(segments) != 1 else ''}"
    ]
    read, written = _counts(report, index, ("walk",), extracted)
    lines.append(
        f"  Chosen: {len(walk['chosen'])} frames; read: {walk['frames_used']} frames "
        f"({written} extracted, {read - written} reused, "
        f"{walk['frames_from_baseline']} from the baseline)"
    )
    lines.append(
        "  Motion along the capture (median step, rate per frame, frames per step):"
    )
    lines.extend(_motion_rows(walk, numbers))

    for reason, heading in (
        ("median", "Blurred frames swapped"),
        ("neighbours", "Frames blurred against their neighbours swapped"),
    ):
        swaps = [
            f"{number(s['candidate'])}->{number(s['chosen'])}"
            for s in walk["swaps"]
            if s["reason"] == reason
        ]
        lines.append(
            f"  {heading} ({len(swaps)}): {_items(swaps)}"
            if swaps
            else f"  {heading}: none"
        )

    gaps = walk["gaps"]
    if gaps:
        lines.append(f"  Gaps filled in ({len(gaps)}):")
        for gap in gaps[:_SHOWN]:
            run = steps[gap["first_step"] : gap["end_step"]]
            percents = [s["stats"]["step_percent"] for s in run]
            shares = [
                s["stats"]["f_inlier_share"]
                for s in run
                if s["stats"]["f_inlier_share"] is not None
            ]
            share = f", F-inlier share down to {min(shares):.2f}" if shares else ""
            low_step, high_step = f"{min(percents):.1f}", f"{max(percents):.1f}"
            steps_text = (
                f"step {low_step}%"
                if low_step == high_step
                else f"steps {low_step}-{high_step}%"
            )
            lines.append(
                f"    {_span(number(run[0]['from']), number(run[-1]['to']))}: "
                f"{len(run)} frame{'s' if len(run) != 1 else ''}, {steps_text}" + share
            )
        if len(gaps) > _SHOWN:
            lines.append(f"    ... and {len(gaps) - _SHOWN} more")
    else:
        lines.append("  Gaps filled in: none")

    kept = [
        f"{number(k['frame'])}, the sharpest of "
        f"{_span(number(k['searched_first']), number(k['searched_last']))} "
        f"({k['sharpness_ratio']:.2f} of the median)"
        for k in walk["kept_blurred"]
    ]
    if kept:
        lines.append(f"  Kept though blurred ({len(kept)}):")
        lines.extend(f"    {k}" for k in kept[:_SHOWN])
        if len(kept) > _SHOWN:
            lines.append(f"    ... and {len(kept) - _SHOWN} more")
    else:
        lines.append("  Kept though blurred: none")

    break_texts = []
    for b in breaks:
        after = number(b["after"])
        if b["resumes_at"] is None:
            break_texts.append(
                f"after {after}: no later frame matches its next frame, so the "
                "sequence ends there"
            )
        else:
            resumes = number(b["resumes_at"])
            stuck = number(b["resumes_at"] - 1)
            break_texts.append(
                f"{after}->{resumes}: frames {_span(after, stuck)} do not match "
                "their next frame"
            )
    lines.append(
        f"  Breaks ({len(break_texts)}): " + "; ".join(break_texts)
        if break_texts
        else "  Breaks: none"
    )

    weak = []
    for point in walk["weak_points"]:
        if point["kind"] == "step":
            step = steps[point["step"]]
            stats = step["stats"]
            why = []
            if point["few_matches"]:
                why.append(f"{stats['matches']} matches")
            if point["low_inlier_share"]:
                why.append(f"F-inlier share {_share(stats['f_inlier_share'])}")
            weak.append(f"{step_text(step)} ({', '.join(why)})")
        else:
            segment = segments[point["segment"]]
            chosen = walk["chosen"][
                segment["start"] : segment["start"] + segment["len"]
            ]
            weak.append(
                f"the segment {_span(number(chosen[0]), number(chosen[-1]))} "
                f"({len(chosen)} frame{'s' if len(chosen) != 1 else ''})"
            )
    lines.append(
        f"  Weak points ({len(weak)}): {_items(weak)}"
        if weak
        else "  Weak points: none"
    )
    return lines


def slice_suggestion(name: str, image_count: int) -> str | None:
    """The line suggesting three slices of a written list as quick test
    solves, or `None` for a list of `2 * SLICE_IMAGES` images or fewer."""
    if image_count <= 2 * SLICE_IMAGES:
        return None
    middle = (image_count - SLICE_IMAGES) // 2
    return (
        f"  For a quick test solve, try -r '{name}[:{SLICE_IMAGES}]', "
        f"-r '{name}[{middle}:{middle + SLICE_IMAGES}]' or "
        f"-r '{name}[-{SLICE_IMAGES}:]'"
    )


def report_lines(
    report: dict,
    sequences: list[PickSequence],
    extracted: list[dict],
    target_step: float,
    step_band: float,
) -> list[str]:
    """Every section of the report before the list is written: the baseline
    of each sequence, the verdict, and the walk of each sequence.

    `extracted[i]` maps a frame position of sequence `i` to whether a `.sift`
    file was written for it; the last line counts the frames that had a
    `.sift` file written and were never read, `report["frames_extracted_unread"]`."""
    several = len(sequences) > 1
    lines: list[str] = []
    for i, sequence in enumerate(sequences):
        label = f" of {sequence.name}" if several else ""
        lines.extend(baseline_lines(report, i, sequence, extracted[i], label))
    lines.append(verdict_line(report, sequences))
    for i, sequence in enumerate(sequences):
        if report["sequences"][i]["walk"] is None:
            continue
        label = f" of {sequence.name}" if several else ""
        lines.append("")
        lines.extend(
            walk_lines(report, i, sequence, extracted[i], label, target_step, step_band)
        )
    unread = report["frames_extracted_unread"]
    lines.append("")
    lines.append(
        f"Extracted and never read: {unread} frame{'' if unread == 1 else 's'}"
    )
    return lines
