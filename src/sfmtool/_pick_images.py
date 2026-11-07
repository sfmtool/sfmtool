# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The frames `sfm pick-images` collects, the extraction it makes before
the walk for the feature tools the walk does not extract with, and the list
entries it writes.

The walk itself is `sfmtool.analysis.pick_images`
(specs/core/analysis/pick-images.md), which extracts the `.sift` files of the
`sfmtool` feature tool itself. This module groups a command's image files
into sequences by folder and by the name apart from the file number
`number_from_filename` reads, which is the number `-r` selects a file by, so
the pattern each sequence's list entry uses is built from the same rule; and
for the other feature tools it extracts every frame's `.sift` files through
`image_files_to_sift_files` with the workspace's settings. See
specs/cli/image-feature/pick-images-command.md.
"""

import os
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ._filenames import file_number_span


class PickImagesInputError(Exception):
    """The command's image files cannot be grouped into sequences; the
    message names every file at fault."""


@dataclass
class PickSequence:
    """One sequence of the capture: its frames' file numbers in increasing
    order, each frame's images (one per sensor, the reference sensor's first),
    and the list pattern of each sensor."""

    patterns: list[str]
    numbers: list[int]
    images: list[list[Path]]
    # The sensors' `image_prefix` values for a rig sequence, in the order of
    # each frame's images; `None` for a sequence of single images.
    sensors: list[str] | None = None

    @property
    def name(self) -> str:
        """What the report and the messages call the sequence: its first
        pattern."""
        return self.patterns[0]


# How many file names an error lists before it gives the rest as a count.
_NAMES_SHOWN = 10


def _names(names: list[str]) -> str:
    shown = ", ".join(names[:_NAMES_SHOWN])
    if len(names) > _NAMES_SHOWN:
        shown += f" and {len(names) - _NAMES_SHOWN} more"
    return shown


def _escape(text: str) -> str:
    """`text` as literal pattern text: `%` doubled, as the pattern grammar
    reads `%%`."""
    return text.replace("%", "%%")


def frame_field(digit_counts: set[int]) -> str:
    """The frame field for a sequence whose numbers are written with these
    counts of digits: `%0Nd` when every number has the same `N > 1` digits,
    `%d` otherwise."""
    if len(digit_counts) == 1:
        (count,) = digit_counts
        if count > 1:
            return f"%0{count}d"
    return "%d"


@dataclass
class _Group:
    rig: int | None
    prefix: str
    folder: str
    head: str
    tail: str
    # number -> {sensor index -> [workspace-relative path, ...]}
    frames: dict[int, dict[int, list[str]]] = field(default_factory=dict)
    digit_counts: set[int] = field(default_factory=set)


def collect_sequences(
    image_paths: list[Path],
    workspace_dir: Path,
    rig_configs: list[dict] | None,
) -> list[PickSequence]:
    """Group `image_paths`, all under `workspace_dir`, into sequences.

    A sequence is the files of one folder whose names are the same apart from
    the number `number_from_filename` reads, ordered by that number. With
    `rig_configs` (a workspace's `rig_config.json`), an image under a sensor's
    `image_prefix` belongs to that rig, and the images of one rig that share a
    frame key, the path after the prefix, are one frame, as `sfm solve`
    groups them; the sequence is then formed from the frame keys. The
    sequences are ordered by their first pattern.

    Raises `PickImagesInputError` naming the files with no number, the files
    of one sequence that share a number, the rig frames that lack a sensor
    another frame of their sequence has, and the names a pattern cannot hold.
    """
    from .rig.config import _infer_frame_key, _match_image_to_sensor

    root = Path(os.path.abspath(workspace_dir))
    groups: dict[tuple, _Group] = {}
    no_number: list[str] = []
    wildcard: list[str] = []
    for path in image_paths:
        rel = Path(os.path.relpath(os.path.abspath(path), root)).as_posix()
        rig = sensor = None
        prefix = ""
        if rig_configs:
            match = _match_image_to_sensor(rel, rig_configs)
            if match is not None:
                rig, sensor = match
                prefix = rig_configs[rig]["cameras"][sensor]["image_prefix"]
        key = _infer_frame_key(rel, prefix) if rig is not None else rel
        folder, slash, name = key.rpartition("/")
        span = file_number_span(name)
        if span is None:
            no_number.append(rel)
            continue
        if "*" in rel:
            wildcard.append(rel)
            continue
        start, end = span
        head, digits, tail = name[:start], name[start:end], name[end:]
        group_key = (rig, folder + slash, head, tail)
        group = groups.get(group_key)
        if group is None:
            group = groups[group_key] = _Group(rig, prefix, folder + slash, head, tail)
        group.frames.setdefault(int(digits), {}).setdefault(
            sensor if sensor is not None else 0, []
        ).append(rel)
        group.digit_counts.add(len(digits))

    problems: list[str] = []
    if no_number:
        problems.append(
            f"{len(no_number)} image file(s) have no file number, so -r cannot "
            f"select them and they have no place in a sequence: {_names(no_number)}"
        )
    if wildcard:
        problems.append(
            f"{len(wildcard)} image file(s) have a '*' in their path, which a named "
            f"image list pattern reads as a wildcard: {_names(wildcard)}"
        )
    sequences: list[PickSequence] = []
    for group in groups.values():
        sequence, group_problems = _sequence(group, rig_configs)
        problems.extend(group_problems)
        if sequence is not None:
            sequences.append(sequence)
    if problems:
        raise PickImagesInputError("\n".join(problems))
    sequences.sort(key=lambda s: s.patterns)
    return sequences


def _sequence(
    group: _Group, rig_configs: list[dict] | None
) -> tuple[PickSequence | None, list[str]]:
    """The sequence of one group of files, or the problems that stop it."""
    sensors = sorted({s for frame in group.frames.values() for s in frame})
    if group.rig is not None:
        cameras = rig_configs[group.rig]["cameras"]
        # The reference sensor first, then the others in the file's order.
        sensors.sort(key=lambda s: (not cameras[s].get("ref_sensor", False), s))
    field_text = frame_field(group.digit_counts)
    name_pattern = _escape(group.folder + group.head) + field_text + _escape(group.tail)

    problems: list[str] = []
    duplicates = [
        paths
        for frame in group.frames.values()
        for paths in frame.values()
        if len(paths) > 1
    ]
    if duplicates:
        problems.append(
            f"{len(duplicates)} file number(s) of the sequence "
            f"{_escape(group.prefix) + name_pattern} name more than one file, so "
            "the files have no order: "
            + "; ".join(" and ".join(paths) for paths in duplicates[:_NAMES_SHOWN])
            + (
                f" and {len(duplicates) - _NAMES_SHOWN} more"
                if len(duplicates) > _NAMES_SHOWN
                else ""
            )
        )
    numbers = sorted(group.frames)
    if group.rig is None:
        if problems:
            return None, problems
        return (
            PickSequence(
                patterns=[name_pattern],
                numbers=numbers,
                images=[[Path(group.frames[n][0][0])] for n in numbers],
            ),
            problems,
        )

    cameras = rig_configs[group.rig]["cameras"]
    prefixes = [cameras[s]["image_prefix"] for s in sensors]
    incomplete = []
    for n in numbers:
        frame = group.frames[n]
        missing = [cameras[s]["image_prefix"] for s in sensors if s not in frame]
        if missing:
            present = frame[min(frame)][0]
            incomplete.append(f"{present} (no image under {', '.join(missing)})")
    if incomplete:
        problems.append(
            f"{len(incomplete)} rig frame(s) of {_escape(prefixes[0]) + name_pattern} "
            "lack the image of a sensor that other frames have: " + _names(incomplete)
        )
    if problems:
        return None, problems
    return (
        PickSequence(
            patterns=[_escape(p) + name_pattern for p in prefixes],
            numbers=numbers,
            images=[[Path(group.frames[n][s][0]) for s in sensors] for n in numbers],
            sensors=prefixes,
        ),
        problems,
    )


def sift_path(image_path: Path, feature_prefix_dir: str) -> Path:
    """Where `image_files_to_sift_files` puts the `.sift` file of an image
    for a workspace's `feature_prefix_dir`."""
    return image_path.parent / feature_prefix_dir / (image_path.name + ".sift")


def capture_sequences(
    sequences: list[PickSequence], workspace_dir: Path, feature_prefix_dir: str
):
    """The `CaptureSequence` of each sequence, with absolute image paths."""
    from .analysis import CaptureFrame, CaptureImage, CaptureSequence

    root = Path(os.path.abspath(workspace_dir))
    capture = []
    for sequence in sequences:
        frames = []
        for number, images in zip(sequence.numbers, sequence.images):
            frames.append(
                CaptureFrame(
                    number,
                    [
                        CaptureImage(root / p, sift_path(root / p, feature_prefix_dir))
                        for p in images
                    ],
                )
            )
        capture.append(CaptureSequence(sequence.patterns, frames))
    return capture


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def extract_every_frame(capture, config: dict, settings_source: str) -> list[set[int]]:
    """Extract the `.sift` files of every frame of `capture` with
    `image_files_to_sift_files` and the workspace's settings, as
    `sfm sift --extract` does in a workspace, before a walk that reads them.

    Returns, per sequence, the positions of the frames that had a `.sift`
    file written: one whose modification time changed or that appeared."""
    from .sift.extract import image_files_to_sift_files

    images = [
        image
        for sequence in capture
        for frame in sequence.frames
        for image in frame.images
    ]
    before = [_mtime(image.sift_path) for image in images]
    image_files_to_sift_files(
        [image.image_path for image in images],
        feature_tool=config["feature_tool"],
        feature_options=config["feature_options"],
        feature_prefix_dir=config["feature_prefix_dir"],
        settings_source=settings_source,
    )
    written: list[set[int]] = []
    at = 0
    for sequence in capture:
        positions = set()
        for position, frame in enumerate(sequence.frames):
            count = len(frame.images)
            if any(
                _mtime(image.sift_path) != mtime
                for image, mtime in zip(
                    images[at : at + count], before[at : at + count]
                )
            ):
                positions.add(position)
            at += count
        written.append(positions)
    return written


@contextmanager
def echo_frames_read(progress, echo: Callable[[str], None], interval: float = 10.0):
    """While the body runs, echo every `interval` seconds how many frames the
    walk has read so far, from the `ProgressCounter` it sets, from a thread of
    its own: the walk runs with the GIL released."""
    stop = threading.Event()

    def loop():
        while not stop.wait(interval):
            echo(f"  ... {progress.value} frames read so far")

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def _range_pieces(numbers: list[int]) -> list:
    """`numbers` as ranges that each parse back from their text, halving the
    numbers until each piece's range does. A range expression holds a bounded
    number of sub-ranges, and the walk's numbers can need more."""
    from .reconstruction import RangeExpr

    whole = RangeExpr.from_list(numbers)
    try:
        RangeExpr(str(whole))
        return [whole]
    except ValueError:
        if len(numbers) == 1:
            raise
    middle = len(numbers) // 2
    return _range_pieces(numbers[:middle]) + _range_pieces(numbers[middle:])


def list_entries(sequence: PickSequence, numbers: list[int]) -> list[tuple]:
    """The list entries that hold the images of `sequence`'s frames numbered
    `numbers`: one entry per sensor, its pattern with the range of the
    numbers. A range too long for one range expression is split into several
    entries of the same pattern, in increasing order of number. A sequence
    with no chosen frames has no entries."""
    if not numbers:
        return []
    pieces = _range_pieces(sorted(numbers))
    return [(pattern, piece) for pattern in sequence.patterns for piece in pieces]
