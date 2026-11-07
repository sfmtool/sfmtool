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
