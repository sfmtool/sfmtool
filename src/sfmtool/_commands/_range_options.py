# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The shared `--range` / `--filter-points` export options.

`to-colmap-bin` and `to-nerfstudio` both subset a reconstruction by image file
number before exporting it, with the same two options, the same guard, and the
same filter. All three live here so the option help, the error text and the
filter cannot drift apart.

The filter takes its own `echo` callable rather than choosing one, because the
two commands report through different sinks (`print` and `click.echo`) and the
point of this module is to leave their output untouched.
"""

from collections.abc import Callable

import click
import numpy as np

from .._filenames import number_from_filename
from .._sfmtool.reconstruction import RangeExpr


def range_options(f: Callable) -> Callable:
    """Attach the `--range` and `--filter-points` options to a command."""
    f = click.option(
        "--filter-points",
        "filter_points",
        is_flag=True,
        default=False,
        help="With --range, also drop 3D points that have no remaining observations. "
        "Default is to keep all 3D points.",
    )(f)
    f = click.option(
        "--range",
        "-r",
        "range_expr",
        default=None,
        help="Export only images whose file number matches this range expression "
        "(e.g. '10-50' or '0-9,20-29'). Observations on excluded images are dropped.",
    )(f)
    return f


def validate_range_options(range_expr: str | None, filter_points: bool) -> None:
    """Reject `--filter-points` without a `--range` to filter against."""
    if filter_points and range_expr is None:
        raise click.UsageError("--filter-points requires --range")


def apply_range_filter(
    recon,
    range_expr_str: str,
    filter_points: bool,
    echo: Callable[[str], None],
):
    """Subset the reconstruction by image file number range."""
    range_numbers = set(RangeExpr(range_expr_str))

    image_names = recon.image_names
    keep_indices: list[int] = []
    for i, name in enumerate(image_names):
        file_number = number_from_filename(name)
        if file_number is not None and file_number in range_numbers:
            keep_indices.append(i)

    if not keep_indices:
        available_numbers = sorted(
            n
            for n in (number_from_filename(name) for name in image_names)
            if n is not None
        )
        raise ValueError(
            f"No images remain after applying range filter '{range_expr_str}'. "
            f"Available file numbers: {available_numbers}"
        )

    echo(
        f"  Applied range filter '{range_expr_str}': "
        f"keeping {len(keep_indices)} of {len(image_names)} images"
    )

    indices_arr = np.array(keep_indices, dtype=np.uint32)
    return recon.subset_by_image_indices(
        indices_arr, drop_orphaned_points=filter_points
    )
