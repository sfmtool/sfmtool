# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""SfM reconstruction filename generation utilities."""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from deadline.job_attachments.api import summarize_paths_by_sequence
from openjd.model import IntRangeExpr


def get_next_sfm_filename(
    base_path: str | Path,
    image_paths: Optional[list[str | Path]] = None,
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


def _generate_image_descriptor(image_paths: list[str | Path]) -> str:
    """Generate a descriptor string characterizing the source images."""
    if not image_paths:
        return ""

    filenames = [Path(p).name for p in image_paths]
    summaries = summarize_paths_by_sequence(filenames)

    if len(summaries) == 1 and summaries[0].index_set:
        summary = summaries[0]
        prefix = summary.path.split("%")[0].rstrip("_-")
        range_str = str(IntRangeExpr.from_list(sorted(summary.index_set)))
        range_str = range_str.replace(":", "x")
        return f"{prefix}_{range_str}"

    first_image = Path(filenames[0])
    first_name = first_image.stem
    total_count = len(image_paths)
    return f"{first_name}-total-{total_count}-images"
