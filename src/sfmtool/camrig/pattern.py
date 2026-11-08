# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Expand `.camrig` image patterns against the filesystem.

A `.camrig` sensor image pattern is a relative forward-slash path that may
contain frame fields (`%d`, `%0Nd`), the `%%` escape, and glob wildcards
(`*`, `**`) — see `specs/formats/camrig-file-format.md`. Both `sfm camrig
create` (matching the images a new rig describes) and `sfm solve` (matching
the images a discovered rig covers) need to expand a pattern against the
filesystem.

The pattern *grammar* — what counts as a frame field, how `*` / `**` map — is
owned by the `sfmtool-camrig-format` Rust crate. This module only walks the
filesystem and delegates every per-path decision to the `camrig_pattern_*`
PyO3 bindings, so the grammar has exactly one implementation.

Matching is a two-pass operation: a pattern is globbed *loosely* (every frame
field widened to `*`) so `pathlib.glob` can enumerate candidates, then each
candidate is confirmed by the Rust `camrig_pattern_matches` so a frame
field matches a number only as printf writes it with that field and `*` /
`**` respect path-segment boundaries.
"""

import sys
from pathlib import Path

from .._sfmtool.fileio import (
    camrig_pattern_frame_index,
    camrig_pattern_matches,
    camrig_pattern_matches_any_padding,
    camrig_pattern_to_glob,
)

# `pathlib.glob` uses the OS-default case sensitivity, which is
# case-insensitive on Windows and macOS. The strict confirm must match, or it
# would reject hits the loose glob accepted. Linux globs case-sensitively.
_CASE_INSENSITIVE = sys.platform in ("win32", "darwin")


def match_pattern(root: Path, pattern: str) -> list[Path]:
    """Resolved absolute paths of the files under `root` that `pattern` matches.

    `pattern` is globbed loosely, then every hit is confirmed against the
    pattern's exact grammar by the Rust `camrig_pattern_matches` binding so a
    frame field matches a number only as printf writes it. Only files are
    returned; the result is sorted.
    """
    return match_pattern_and_padding_misses(root, pattern)[0]


def match_pattern_and_padding_misses(
    root: Path, pattern: str
) -> tuple[list[Path], list[Path]]:
    """The files under `root` that `pattern` matches, as `match_pattern`
    returns them, and the files it misses only because their number is padded
    otherwise than its frame field writes it (`cam_007.jpg` for
    `cam_%04d.jpg`), resolved and sorted the same way, from one glob.
    """
    root = Path(root)
    matches: list[Path] = []
    misses: list[Path] = []
    for hit in root.glob(camrig_pattern_to_glob(pattern)):
        if not hit.is_file():
            continue
        rel = hit.relative_to(root).as_posix()
        if camrig_pattern_matches(pattern, rel, _CASE_INSENSITIVE):
            matches.append(hit.resolve())
        elif camrig_pattern_matches_any_padding(pattern, rel, _CASE_INSENSITIVE):
            misses.append(hit.resolve())
    return sorted(matches), sorted(misses)


def match_pattern_with_frames(root: Path, pattern: str) -> list[tuple[Path, int]]:
    """`(path, frame_index)` pairs for files under `root` matching `pattern`.

    `pattern` is expected to carry a frame field (`%d` / `%0Nd`); the frame
    index is the integer that field captures. Multi-sensor rigs pair frames
    across sensors by this index, so every sensor pattern must yield one. A
    matched file from which no frame index can be captured raises `ValueError`
    rather than being silently dropped.
    """
    root = Path(root)
    result: list[tuple[Path, int]] = []
    for path in match_pattern(root, pattern):
        rel = path.relative_to(root).as_posix()
        frame = camrig_pattern_frame_index(pattern, rel, _CASE_INSENSITIVE)
        if frame is None:
            raise ValueError(
                f"pattern {pattern!r} matched {rel} but captured no frame "
                "index; a multi-sensor rig pattern must carry a frame field"
            )
        result.append((path, frame))
    return result
