# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""SfM reconstruction filename generation utilities."""

import re
import textwrap
from datetime import datetime
from pathlib import Path

from ._filenames import group_by_directory
from ._path_summary import summarize_path_list, summarize_paths_by_sequence
from ._sfmtool.reconstruction import RangeExpr


def get_next_sfm_filename(
    base_path: str | Path,
    image_paths: list[str | Path] | None = None,
    operation: str = "solve",
) -> Path:
    """Generate the next .sfmr filename with date prefix, operation, and image descriptor.

    Format: {YYYYMMDD}-##-{operation}[-descriptor].sfmr
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now().astimezone()
    date_prefix = now.strftime("%Y%m%d")

    descriptor = _generate_image_descriptor(image_paths) if image_paths else ""

    pattern = re.compile(rf"^{re.escape(date_prefix)}-(\d{{2,}})(?:-.*)?\.sfmr$")
    max_counter = -1
    if base_path.exists():
        for filename in base_path.iterdir():
            if filename.is_file():
                match = pattern.match(filename.name)
                if match:
                    counter = int(match.group(1))
                    max_counter = max(max_counter, counter)

    next_counter = max_counter + 1
    counter_str = f"{next_counter:02d}" if next_counter < 100 else str(next_counter)

    if descriptor:
        filename = f"{date_prefix}-{counter_str}-{operation}-{descriptor}.sfmr"
    else:
        filename = f"{date_prefix}-{counter_str}-{operation}.sfmr"

    return base_path / filename


# The most UTF-8 bytes an image descriptor takes, so that a generated name stays
# well inside the 255-byte (or 255 UTF-16 unit) limit file systems put on one
# path component.
_MAX_DESCRIPTOR_BYTES = 100


def _generate_image_descriptor(image_paths: list[str | Path]) -> str:
    """Generate a descriptor string characterizing the source images.

    Each directory holding images is its own sequence, named by its path
    relative to the deepest directory that holds all of them. Images in one
    directory are described by their file names: `{prefix}_{range}` for one
    numbered sequence, `{first stem}-total-{n}-images` otherwise. Images in
    several directories are described by `_describe_directories`, which always
    ends the descriptor with `-total-{n}-images`.

    The result is at most `_MAX_DESCRIPTOR_BYTES` UTF-8 bytes long. A range
    that would be longer becomes `{prefix}_{first}-{last}-total-{n}-images`,
    and a prefix or stem that still does not fit is cut short.
    """
    if not image_paths:
        return ""

    groups = _group_by_directory(image_paths)
    if len(groups) > 1:
        return _describe_directories(groups, len(image_paths))

    filenames = next(iter(groups.values()))
    single = _single_sequence(filenames)
    if single is not None:
        prefix, indexes = single
        range_str = _range_string(indexes)
        descriptor = f"{prefix}_{range_str}"
        if len(descriptor.encode("utf-8")) <= _MAX_DESCRIPTOR_BYTES:
            return descriptor
        suffix = f"_{indexes[0]}-{indexes[-1]}-total-{len(indexes)}-images"
        return _truncate_utf8(prefix, _MAX_DESCRIPTOR_BYTES - len(suffix)) + suffix

    first_name = Path(filenames[0]).stem
    suffix = f"-total-{len(image_paths)}-images"
    return _truncate_utf8(first_name, _MAX_DESCRIPTOR_BYTES - len(suffix)) + suffix


def _group_by_directory(
    image_paths: list[str | Path],
) -> dict[tuple[str, ...], list[str]]:
    """Group the images' file names by directory, in input order.

    Each key is the parts of the directory's path relative to the deepest
    directory holding every image; the images in that deepest directory itself
    have the key `()`.
    """
    return {
        key: [path.name for path in paths]
        for key, paths in group_by_directory(image_paths).items()
    }


def _single_sequence(filenames: list[str]) -> tuple[str, list[int]] | None:
    """The prefix and sorted frame numbers when `filenames` are one numbered sequence."""
    summaries = summarize_paths_by_sequence(filenames)
    if len(summaries) != 1 or not summaries[0].index_set:
        return None
    prefix = summaries[0].path.split("%")[0].rstrip("_-")
    return prefix, sorted(summaries[0].index_set)


def _range_string(indexes: list[int]) -> str:
    """The range expression for `indexes`, with its `:` step written as `x`."""
    return str(RangeExpr.from_list(indexes)).replace(":", "x")


def _describe_directories(
    groups: dict[tuple[str, ...], list[str]], image_count: int
) -> str:
    """Describe images held in several directories.

    Each directory is written as its relative path with the parts joined by
    `_`, so it can be part of a file name. The directories are listed in
    sorted order, joined by `+`, followed by `-total-{image_count}-images`.
    The first of these that fits in
    `_MAX_DESCRIPTOR_BYTES` is used:

    1. each directory with its sequence and range, `{dir}-{prefix}_{range}`;
    2. each directory with its sequence's first and last frame,
       `{dir}-{prefix}_{first}-{last}`;
    3. each directory alone, `{dir}`;
    4. as many directories as fit, then `+{k}-more`, with the first
       directory cut short when even it does not fit.

    A directory whose images are not one numbered sequence is named alone
    at every step. Images directly in the common directory list their
    sequence without a directory, or `{first stem}` when they are not one.
    """
    suffix = f"-total-{image_count}-images"
    budget = _MAX_DESCRIPTOR_BYTES - len(suffix)

    names = sorted(groups)
    sequences = {name: _single_sequence(groups[name]) for name in names}

    def label(name: tuple[str, ...], detail: str) -> str:
        return f"{'_'.join(name)}-{detail}" if name else detail

    def bare(name: tuple[str, ...]) -> str:
        return "_".join(name) if name else Path(groups[name][0]).stem

    def full(name: tuple[str, ...]) -> str:
        seq = sequences[name]
        if seq is None:
            return bare(name)
        return label(name, f"{seq[0]}_{_range_string(seq[1])}")

    def ends(name: tuple[str, ...]) -> str:
        seq = sequences[name]
        if seq is None:
            return bare(name)
        prefix, indexes = seq
        return label(name, f"{prefix}_{indexes[0]}-{indexes[-1]}")

    for describe in (full, ends, bare):
        joined = "+".join(describe(name) for name in names)
        if len(joined.encode("utf-8")) <= budget:
            return joined + suffix

    # Name as many directories as fit, followed by how many more there are.
    for shown in range(len(names) - 1, 0, -1):
        joined = "+".join(bare(name) for name in names[:shown])
        more = f"+{len(names) - shown}-more"
        if len((joined + more).encode("utf-8")) <= budget:
            return joined + more + suffix
    more = f"+{len(names) - 1}-more"
    first = _truncate_utf8(bare(names[0]), budget - len(more))
    return first + more + suffix


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Cut `text` to at most `max_bytes` UTF-8 bytes, never splitting a character."""
    return text.encode("utf-8")[: max(max_bytes, 0)].decode("utf-8", errors="ignore")


def get_image_hint_message(
    recon,
    missing_image: str | None = None,
) -> str:
    """Create a helpful error message with available images in a reconstruction.

    Args:
        recon: SfmrReconstruction object
        missing_image: Optional image name that was not found

    Returns:
        A multiline string containing the error message and hints
    """
    from difflib import get_close_matches

    image_names = recon.image_names

    error_msg = ""
    if missing_image:
        error_msg = f"Image '{missing_image}' not found in reconstruction.\n"

        basenames = [Path(name).name for name in image_names]
        similar_names = get_close_matches(
            Path(missing_image).name, basenames, n=3, cutoff=0.6
        )

        if similar_names:
            error_msg += "\nDid you mean one of these?\n"
            for name in similar_names:
                full_name = next(n for n in image_names if Path(n).name == name)
                error_msg += f"  - {full_name}\n"

    if error_msg:
        error_msg += "\n"
    error_msg += "Image in reconstruction:\n"
    error_msg += textwrap.indent(summarize_path_list(image_names), "  ")

    return error_msg
