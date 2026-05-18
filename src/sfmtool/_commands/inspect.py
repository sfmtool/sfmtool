# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Unified file inspection command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._inspect_summary import (
    print_camrig_summary,
    print_image_summary,
    print_matches_summary,
    print_sfmr_summary,
    print_sift_summary,
)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

_SUMMARY_DISPATCH = {
    ".sfmr": print_sfmr_summary,
    ".sift": print_sift_summary,
    ".matches": print_matches_summary,
    ".camrig": print_camrig_summary,
}


@click.command("inspect")
@timed_command
@click.help_option("--help", "-h")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--verbose",
    "-v",
    "verbose",
    is_flag=True,
    help="Print detailed information instead of a short summary.",
)
def inspect(path, verbose):
    """Inspect an sfmtool file or image and print a summary.

    The file type is determined by extension:

    \b
      .sfmr     reconstruction
      .sift     feature file
      .matches  feature matches
      .camrig   camera rig
      image     .png / .jpg / .jpeg

    Without --verbose, prints a short summary. With --verbose, prints the
    full detail available for that file type.

    For deep-analysis reports on a reconstruction (covisibility, frustum,
    depth, per-image metrics), use `sfm analyze`.

    Examples:

        sfm inspect reconstruction.sfmr

        sfm inspect image_001.sift --verbose

        sfm inspect matches.matches

        sfm inspect rig.camrig

        sfm inspect photo.jpg -v
    """
    path = Path(path)
    suffix = path.suffix.lower()

    summary_fn = _SUMMARY_DISPATCH.get(suffix)
    if summary_fn is None and suffix in _IMAGE_SUFFIXES:
        summary_fn = print_image_summary
    if summary_fn is None:
        raise click.UsageError(
            f"Cannot inspect '{path.name}': unsupported file type '{suffix}'. "
            "Supported types: .sfmr, .sift, .matches, .camrig, and image "
            "files (.png, .jpg, .jpeg)."
        )

    try:
        summary_fn(path, verbose=verbose)
    except Exception as e:
        raise click.ClickException(str(e))
