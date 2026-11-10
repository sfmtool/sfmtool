# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The text `sfm pick-images` prints from the report `pick_images` returns.

The report names frames by their position in their sequence; every line here
names them by file number, through the command's sequences. See
specs/cli/image-feature/pick-images-command.md § "Report".
"""

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
        # A step whose frames match nothing has no step or rate (`None`).
        measured = [s for s in inside if s["rate"] is not None]
        stride = np.median([s["to"] - s["from"] for s in inside])
        frames = f"{numbers[first]}-{numbers[last]}"
        if measured:
            step = np.median([s["stats"]["step_percent"] for s in measured])
            rate = np.median([s["rate"] for s in measured])
            motion = f"{step:6.1f}%  {rate:6.2f}%"
        else:
            motion = f"{'unmeasured':>16}"
        rows.append(f"    frames {frames:>13}  {motion}  {stride:5g}")
    return rows


def _vote_model(vote: dict) -> str:
    """The camera model of a pilot vote's verdict, in words."""
    if vote["camera_model"] == "EquidistantFisheye":
        return (
            "unconfirmed equidistant fisheye"
            if vote["confirmed"] is False
            else "equidistant fisheye"
        )
    return "pinhole"


def pilot_source(vote: dict, sequence: PickSequence) -> str:
    """What `sequence`'s camera source is after its pilot vote: the vote's
    camera, with its model, focal and pool spread, or why the vote gave it
    none, followed by the note of a `.camrig` that covers only some of its
    frames."""
    text = _vote_source(vote)
    if sequence.camera_source != PickSequence.camera_source:
        text = f"{text}; {sequence.camera_source}"
    return text


def _vote_source(vote: dict) -> str:
    """The camera a pilot vote gave, or why it gave none."""
    outcome = vote["outcome"]
    status = outcome["status"]
    if status == "used":
        return (
            f"the pilot vote's {_vote_model(vote)} camera, focal "
            f"{vote['focal_px']:.1f} px, pool spread {vote['pool_spread']:.3f}"
        )
    if status == "wide_spread":
        return (
            f"the pilot vote's pool spread {vote['pool_spread']:.3f} is above "
            f"{outcome['max_pool_spread']:g} ({_vote_model(vote)}, focal "
            f"{vote['focal_px']:.1f} px)"
        )
    if status == "no_consensus":
        pooled = vote["n_pool"]
        return (
            f"the pilot vote reached no consensus ({pooled} pooled "
            f"vote{'' if pooled == 1 else 's'})"
        )
    return (
        f"the pilot walk chose {len(vote['frames'])} frames, and a vote needs "
        f"at least {outcome['min_frames']}"
    )


def _pilot_line(vote: dict) -> str:
    """The line giving what the pilot vote read and how long it took."""
    return (
        f"  Pilot vote: {len(vote['frames'])} frames chosen, "
        f"{vote['frames_read']} read, {vote['features']} features in "
        f"{vote['clusters']} clusters; took {vote['seconds']:.1f} s"
    )


def _density_term_line(walk: dict, sequence: PickSequence) -> str:
    """The line saying whether the local density term was on, with the
    camera it measured with and how many steps it shortened, or why it was
    off."""
    term = walk["density_term"]
    if term["status"] == "on":
        steps = walk["steps"]
        gained = sum(s["density"]["gain"] > 1 for s in steps)
        # The pilot vote's camera is at the frames' size, and its source
        # names its focal.
        focal = (
            ""
            if walk["pilot_vote"] is not None
            else f" (focal {term['focal_px']:.1f} px at the frames' size)"
        )
        return (
            f"  Local density term: on, with {sequence.camera_source}{focal}; "
            f"{gained} of {len(steps)} steps walked with a gain above 1"
        )
    reason = term["reason"]
    if reason == "max_gain_one":
        why = "--max-density-gain is 1"
    elif reason == "no_camera":
        why = sequence.camera_source
    elif reason == "camera_shape":
        (cw, ch), (fw, fh) = term["camera"], term["frames"]
        why = (
            f"{sequence.camera_source} is {cw} x {ch}, of another aspect than the "
            f"frames' {fw} x {fh}"
        )
    else:
        why = f"{sequence.camera_source} cannot be measured with: {term['problem']}"
    return f"  Local density term: off, {why}"


def _density_rows(walk: dict, numbers: list[int]) -> list[str]:
    """The table of each step's target, the gain and median measure it came
    from, and the step's parallax P50 and P90/P50 and measure."""
    rows = [
        "  Local density by step (target, gain, median measure of the last three "
        "steps; the step's parallax P50, P90/P50 and measure):",
        f"    {'frames':>15}  {'target':>7}  {'gain':>5}  {'median':>6}  "
        f"{'P50 deg':>8}  {'P90/P50':>7}  {'measure':>7}",
    ]
    for step in walk["steps"]:
        density = step["density"]
        frames = f"{numbers[step['from']]}-{numbers[step['to']]}"
        split = density["essential_split"]
        if split["status"] == "measured":
            p50, p90 = split["parallax_p50_deg"], split["parallax_p90_deg"]
            ratio = f"{p90 / p50:7.2f}" if p50 else f"{'-':>7}"
            parallax = f"{p50:8.3f}  {ratio}  {density['measure']:7.2f}"
        else:
            parallax = f"not measured ({split['reason']})"
        rows.append(
            f"    {frames:>15}  {density['target_percent']:6.2f}%  "
            f"{density['gain']:5.2f}  {density['median_measure']:6.2f}  {parallax}"
        )
    return rows


def walk_lines(
    report: dict,
    index: int,
    sequence: PickSequence,
    extracted: dict,
    label: str,
    target_step: float,
    step_band: float,
    min_step_matches: int,
) -> list[str]:
    """The section of one sequence."""
    walk = report["sequences"][index]
    numbers = sequence.numbers
    steps = walk["steps"]
    low, high = target_step * (1 - step_band), target_step * (1 + step_band)
    lines = [
        f"Walk{label}: target step {target_step:g}% (band {low:.3g}-{high:.3g}%), "
        f"match floor {min_step_matches}"
    ]
    written = sum(
        bool(extracted.get(frame["id"]["frame"]))
        for frame in report["frames"]
        if frame["id"]["sequence"] == index
    )
    read = walk["frames_read"]
    lines.append(
        f"  Chosen: {len(walk['chosen'])} of {len(numbers)} frames; read: {read} "
        f"frames ({written} extracted, {read - written} reused)"
    )
    lines.append(_density_term_line(walk, sequence))
    if walk["pilot_vote"] is not None:
        lines.append(_pilot_line(walk["pilot_vote"]))
    lines.append(
        "  Motion along the capture (median step, rate per frame, frames per step):"
    )
    lines.extend(_motion_rows(walk, numbers))
    at_floor = [s for s in steps if s["kind"] == "at_floor"]
    next_frame = sum(s["to"] == s["from"] + 1 for s in at_floor)
    if at_floor:
        lines.append(
            f"  Steps stopped at the match floor: {len(at_floor)} of {len(steps)}, "
            f"{next_frame} of them at the next frame"
        )
    else:
        lines.append("  Steps stopped at the match floor: none")
    swaps = [
        f"{numbers[s['candidate']]}->{numbers[s['chosen']]}" for s in walk["swaps"]
    ]
    heading = "Frames blurred against their neighbours swapped"
    lines.append(
        f"  {heading} ({len(swaps)}): {_items(swaps)}"
        if swaps
        else f"  {heading}: none"
    )
    if walk["density_term"]["status"] == "on" and steps:
        lines.extend(_density_rows(walk, numbers))
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
    min_step_matches: int,
) -> list[str]:
    """Every section of the report before the list is written: the walk of
    each sequence, then the count of frames extracted and never read.

    `extracted[i]` maps a frame position of sequence `i` to whether a `.sift`
    file was written for it; the last line counts the frames that had a
    `.sift` file written and were never read, `report["frames_extracted_unread"]`."""
    several = len(sequences) > 1
    lines: list[str] = []
    for i, sequence in enumerate(sequences):
        label = f" of {sequence.name}" if several else ""
        if i:
            lines.append("")
        lines.extend(
            walk_lines(
                report,
                i,
                sequence,
                extracted[i],
                label,
                target_step,
                step_band,
                min_step_matches,
            )
        )
    unread = report["frames_extracted_unread"]
    lines.append("")
    lines.append(
        f"Extracted and never read: {unread} frame{'' if unread == 1 else 's'}"
    )
    return lines
