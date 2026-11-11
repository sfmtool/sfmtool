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
the pattern each sequence's list entry uses is built from the same rule. A
rig's images are grouped into frames by the multi-sensor `.camrig` that
covers them, found as `sfm solve` finds it, and each sequence takes the
camera of its reference sensor from the `.camrig` that covers its frames,
for the walk's local density term; the walk finds the camera of a sequence
with none by its pilot vote.
For the other feature tools it extracts every frame's `.sift` files through
`image_files_to_sift_files` with the workspace's settings. See
specs/cli/image-feature/pick-images-command.md.
"""

import os
import threading
from collections.abc import Callable, Iterable
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
    # What names each sensor of a rig sequence, in the order of each frame's
    # images: its image pattern as the `.camrig` stores it; `None` for a
    # sequence of single images.
    sensors: list[str] | None = None
    # The camera dict of the reference sensor, from the `.camrig` that covers
    # the sequence's frames, or `None`.
    camera: dict | None = None
    # Where `camera` comes from, or why there is none.
    camera_source: str = "no .camrig covers its frames"

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


def _zero_padded(digits: str) -> bool:
    """Whether a number written as `digits` has a leading zero."""
    return len(digits) > 1 and digits[0] == "0"


def _writes(width: int, digits: str) -> bool:
    """Whether printf's `%0Nd` with `N = width` (`%d` for a width of 1)
    writes a number as `digits`, as the `.camrig` pattern matcher decides it."""
    from ._sfmtool.fileio import camrig_pattern_matches

    field_text = f"%0{width}d" if width > 1 else "%d"
    return camrig_pattern_matches(field_text, digits, False)


def frame_field(numbers: Iterable[str]) -> tuple[str, list[str]]:
    """The frame field for a sequence whose file numbers are written as
    `numbers`, the digits as the names hold them, and the numbers of
    `numbers` it does not write, sorted.

    A frame field matches a number only as printf writes it with the field,
    so the field is chosen to write every number: `%0Nd` when a number is
    zero-padded to `N` digits, or, with no zero-padded number, when every
    number has the same `N > 1` digits, and `%d` otherwise. `%04d` so writes
    `0001` to `9999` and `10000` on. Numbers zero-padded to different widths,
    or a number shorter than the padded width (`01` beside `5`), have no
    field that writes them all; the field is then the `%0Nd` that writes the
    most of them, the wider on a tie, and the others are returned.
    """
    numbers = set(numbers)
    padded = {len(digits) for digits in numbers if _zero_padded(digits)}
    if padded:
        width = max(
            padded,
            key=lambda w: (sum(_writes(w, digits) for digits in numbers), w),
        )
    else:
        lengths = {len(digits) for digits in numbers}
        width = lengths.pop() if len(lengths) == 1 else 1
    field_text = f"%0{width}d" if width > 1 else "%d"
    return field_text, sorted(d for d in numbers if not _writes(width, d))


@dataclass
class _Group:
    folder: str
    head: str
    tail: str
    # number -> [workspace-relative path, ...]
    frames: dict[int, list[str]] = field(default_factory=dict)
    # The number as the names write it -> [workspace-relative path, ...]
    numbers: dict[str, list[str]] = field(default_factory=dict)


def find_rigs(image_paths: list[Path], workspace_dir: Path):
    """The rigs that group `image_paths` into rig frames, and the cameras of
    the other images, found as `sfm solve` finds them:
    `(camrig_rigs, camrig_cameras, notes)`.

    Each multi-sensor `.camrig` covering any of the images is returned in
    `camrig_rigs`, in path order, as a `CamrigRig` whose assignments hold the
    images it covers; single-sensor ones, each a camera only, define no rig,
    and are returned in `camrig_cameras` as a `CamrigCamera` with the images
    it covers. The images no multi-sensor `.camrig` covers are single images.
    `notes` are the lines to print about the rigs.

    Raises `PickImagesInputError` when the workspace root holds a
    `rig_config.json` (`_refuse_rig_config`), when an image is matched by
    more than one `.camrig`, which `sfm solve` refuses, or when a `.camrig`
    cannot be used.
    """
    from .camrig.resolver import (
        CamrigCamera,
        CamrigRig,
        CamrigSolveError,
        _build_rig_assignments,
        find_covering_camrigs,
    )
    from .rig.config import RigConfigError, _refuse_rig_config

    try:
        _refuse_rig_config(workspace_dir)
        hits = find_covering_camrigs(image_paths, workspace_dir, images="images given")
    except (RigConfigError, CamrigSolveError) as e:
        raise PickImagesInputError(str(e)) from None

    notes = []
    rigs = []
    cameras = []
    for camrig_file, data, covered in hits:
        sensor_count = data["metadata"]["sensor_count"]
        if sensor_count == 1:  # a camera, not a rig
            camera = data["cameras"][data["camera_indexes"][0]]
            cameras.append(
                CamrigCamera(camrig_file=camrig_file, camera=camera, images=covered)
            )
            continue
        notes.append(f"  Camera rig: {camrig_file} ({sensor_count} sensors)")
        try:
            assignments = _build_rig_assignments(camrig_file, data)
        except (CamrigSolveError, ValueError) as e:
            raise PickImagesInputError(str(e)) from None
        rigs.append(
            CamrigRig(
                camrig_file=camrig_file,
                data=data,
                assignments={p: a for p, a in assignments.items() if p in covered},
            )
        )
    return rigs, cameras, notes


def collect_sequences(
    image_paths: list[Path],
    workspace_dir: Path,
    camrig_rigs=(),
    camrig_cameras=(),
) -> list[PickSequence]:
    """Group `image_paths`, all under `workspace_dir`, into sequences.

    A sequence is the files of one folder whose names are the same apart from
    the number `number_from_filename` reads, ordered by that number. Each of
    `camrig_rigs` (the `CamrigRig`s from `find_rigs`) makes the images its
    assignments hold one rig sequence instead, framed by the `.camrig`'s frame
    indexes, with the camera of its reference sensor, and the other images
    form sequences as above, each with the camera of the one of
    `camrig_cameras` (the `CamrigCamera`s from `find_rigs`) that covers all
    its images, if one does. The sequences are ordered by their first
    pattern.

    Raises `PickImagesInputError` naming the files with no number, the files
    of one sequence that share a number, the rig frames that lack a sensor
    another frame of their sequence has, and the names a pattern cannot hold.
    """
    problems: list[str] = []
    sequences: list[PickSequence] = []
    in_rig: set[Path] = set()
    for camrig_rig in camrig_rigs:
        rig_images = [
            p for p in image_paths if Path(p).resolve() in camrig_rig.assignments
        ]
        in_rig.update(Path(p).resolve() for p in rig_images)
        sequence, rig_problems = _camrig_sequence(rig_images, workspace_dir, camrig_rig)
        problems.extend(rig_problems)
        if sequence is not None:
            sequences.append(sequence)
    if in_rig:
        image_paths = [p for p in image_paths if Path(p).resolve() not in in_rig]

    root = Path(os.path.abspath(workspace_dir))
    groups: dict[tuple, _Group] = {}
    no_number: list[str] = []
    wildcard: list[str] = []
    for path in image_paths:
        rel = Path(os.path.relpath(os.path.abspath(path), root)).as_posix()
        folder, slash, name = rel.rpartition("/")
        span = file_number_span(name)
        if span is None:
            no_number.append(rel)
            continue
        if "*" in rel:
            wildcard.append(rel)
            continue
        start, end = span
        head, digits, tail = name[:start], name[start:end], name[end:]
        group_key = (folder + slash, head, tail)
        group = groups.get(group_key)
        if group is None:
            group = groups[group_key] = _Group(folder + slash, head, tail)
        group.frames.setdefault(int(digits), []).append(rel)
        group.numbers.setdefault(digits, []).append(rel)

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
    for group in groups.values():
        sequence, group_problems = _sequence(group)
        problems.extend(group_problems)
        if sequence is not None:
            _camera_of(sequence, root, camrig_cameras)
            sequences.append(sequence)
    if problems:
        raise PickImagesInputError("\n".join(problems))
    sequences.sort(key=lambda s: s.patterns)
    return sequences


def _sequence(group: _Group) -> tuple[PickSequence | None, list[str]]:
    """The sequence of one group of files, or the problems that stop it."""
    field_text, unwritten = frame_field(group.numbers)
    name_pattern = _escape(group.folder + group.head) + field_text + _escape(group.tail)

    problems: list[str] = []
    if unwritten:
        files = sorted(rel for digits in unwritten for rel in group.numbers[digits])
        problems.append(
            f"the numbers of the sequence {name_pattern} "
            "are zero-padded in ways no one frame field writes, and a frame "
            "field matches a number only as printf writes it, so "
            f"{len(files)} file(s) do not match {name_pattern}: {_names(files)}"
        )
    duplicates = [paths for paths in group.frames.values() if len(paths) > 1]
    if duplicates:
        problems.append(
            f"{len(duplicates)} file number(s) of the sequence {name_pattern} "
            "name more than one file, so the files have no order: "
            + "; ".join(" and ".join(paths) for paths in duplicates[:_NAMES_SHOWN])
            + (
                f" and {len(duplicates) - _NAMES_SHOWN} more"
                if len(duplicates) > _NAMES_SHOWN
                else ""
            )
        )
    if problems:
        return None, problems
    numbers = sorted(group.frames)
    return (
        PickSequence(
            patterns=[name_pattern],
            numbers=numbers,
            images=[[Path(group.frames[n][0])] for n in numbers],
        ),
        problems,
    )


def _camera_of(sequence: PickSequence, root: Path, camrig_cameras) -> None:
    """Give `sequence`, of single images under `root`, the camera of the
    single-sensor `.camrig` of `camrig_cameras` that covers all its images,
    or say how much of it they cover."""
    images = {(root / i[0]).resolve() for i in sequence.images}
    for camrig_camera in camrig_cameras:
        covered = len(images & camrig_camera.images)
        if covered == len(images):
            sequence.camera = camrig_camera.camera
            sequence.camera_source = f"the camera of {camrig_camera.camrig_file}"
            return
        if covered:
            sequence.camera_source = (
                f"{camrig_camera.camrig_file} covers only {covered} of its "
                f"{len(images)} frames"
            )
            return


def _camrig_sequence(
    image_paths: list[Path], workspace_dir: Path, camrig_rig
) -> tuple[PickSequence | None, list[str]]:
    """The rig sequence of images a multi-sensor `.camrig` covers, or the
    problems that stop it.

    A frame is the images of one frame index, its sensors ordered by index,
    so the lowest-indexed sensor present, the reference `sfm solve` uses,
    comes first. Each sensor's list pattern is its `.camrig` pattern behind
    the `.camrig`'s folder, relative to the workspace, so it matches the same
    files and reads the same frame number from them.
    """
    from ._sfmtool.fileio import camrig_pattern_matches

    root = Path(workspace_dir).resolve()
    folder = camrig_rig.camrig_file.parent.relative_to(root).as_posix()
    prefix = "" if folder == "." else _escape(folder + "/")
    stored = camrig_rig.data["sensor_image_patterns"]
    frames: dict[int, dict[int, str]] = {}
    for path in image_paths:
        resolved = Path(path).resolve()
        sensor, frame = camrig_rig.assignments[resolved]
        frames.setdefault(frame, {})[sensor] = resolved.relative_to(root).as_posix()
    sensors = sorted({s for frame in frames.values() for s in frame})
    patterns = [prefix + stored[s] for s in sensors]
    numbers = sorted(frames)

    problems: list[str] = []
    if "*" in prefix:
        problems.append(
            f"The folder of {camrig_rig.camrig_file} has a '*' in its path, which "
            "a named image list pattern reads as a wildcard."
        )
    # The `.camrig` matches letter case as the file system does, and a named
    # image list pattern matches it exactly.
    other_case = [
        rel
        for frame in frames.values()
        for s, rel in frame.items()
        if not camrig_pattern_matches(prefix + stored[s], rel, False)
    ]
    if other_case:
        problems.append(
            f"{len(other_case)} image file(s) match the patterns of "
            f"{camrig_rig.camrig_file} only when letter case is ignored, and a "
            f"named image list pattern matches case exactly: {_names(other_case)}"
        )
    incomplete = []
    for n in numbers:
        frame = frames[n]
        missing = [patterns[i] for i, s in enumerate(sensors) if s not in frame]
        if missing:
            incomplete.append(f"{frame[min(frame)]} (no image of {', '.join(missing)})")
    if incomplete:
        problems.append(
            f"{len(incomplete)} rig frame(s) of {patterns[0]} lack the image of a "
            "sensor that other frames have: " + _names(incomplete)
        )
    if problems:
        return None, problems
    sequence = PickSequence(
        patterns=patterns,
        numbers=numbers,
        images=[[Path(frames[n][s]) for s in sensors] for n in numbers],
        sensors=[stored[s] for s in sensors],
    )
    # A rig given none of its images is a sequence of no frames, with no
    # reference sensor to take a camera from.
    if sensors:
        data = camrig_rig.data
        sequence.camera = data["cameras"][data["camera_indexes"][sensors[0]]]
        sequence.camera_source = (
            f"the camera of {camrig_rig.camrig_file}, sensor {stored[sensors[0]]}"
        )
    return sequence, problems


def sift_path(image_path: Path, feature_prefix_dir: str) -> Path:
    """Where `image_files_to_sift_files` puts the `.sift` file of an image
    for a workspace's `feature_prefix_dir`."""
    return image_path.parent / feature_prefix_dir / (image_path.name + ".sift")


def capture_sequences(
    sequences: list[PickSequence], workspace_dir: Path, feature_prefix_dir: str
):
    """The `CaptureSequence` of each sequence, with absolute image paths and
    its camera."""
    from .analysis import CaptureFrame, CaptureImage, CaptureSequence
    from .geometry import CameraIntrinsics

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
        camera = sequence.camera
        capture.append(
            CaptureSequence(
                sequence.patterns,
                frames,
                camera=None if camera is None else CameraIntrinsics.from_dict(camera),
            )
        )
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
    entries of the same pattern, in increasing order of number."""
    pieces = _range_pieces(sorted(numbers))
    return [(pattern, piece) for pattern in sequence.patterns for piece in pieces]
