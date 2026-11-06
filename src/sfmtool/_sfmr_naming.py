# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""SfM reconstruction filename generation utilities."""

import re
import textwrap
from datetime import datetime
from pathlib import Path

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

    The result is at most `_MAX_DESCRIPTOR_BYTES` UTF-8 bytes long. A range
    that would be longer becomes `{prefix}_{first}-{last}-total-{n}-images`,
    and a prefix or stem that still does not fit is cut short.
    """
    if not image_paths:
        return ""

    filenames = [Path(p).name for p in image_paths]
    summaries = summarize_paths_by_sequence(filenames)

    if len(summaries) == 1 and summaries[0].index_set:
        summary = summaries[0]
        prefix = summary.path.split("%")[0].rstrip("_-")
        indexes = sorted(summary.index_set)
        range_str = str(RangeExpr.from_list(indexes))
        range_str = range_str.replace(":", "x")
        descriptor = f"{prefix}_{range_str}"
        if len(descriptor.encode("utf-8")) <= _MAX_DESCRIPTOR_BYTES:
            return descriptor
        suffix = f"_{indexes[0]}-{indexes[-1]}-total-{len(indexes)}-images"
        return _truncate_utf8(prefix, _MAX_DESCRIPTOR_BYTES - len(suffix)) + suffix

    first_name = Path(filenames[0]).stem
    suffix = f"-total-{len(image_paths)}-images"
    return _truncate_utf8(first_name, _MAX_DESCRIPTOR_BYTES - len(suffix)) + suffix


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
