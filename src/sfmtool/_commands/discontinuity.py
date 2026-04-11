# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Discontinuity analysis command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._filenames import expand_paths


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


@click.command("discontinuity")
@timed_command
@click.help_option("--help", "-h")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--range",
    "-r",
    "range_expr",
    help="A range expression of file numbers to use from the input directories.",
)
@click.option(
    "--initial-stride",
    type=click.IntRange(min=1),
    default=1,
    metavar="N",
    help="Initial stride for adaptive sampling. Default: 1.",
)
@click.option(
    "--min-stride",
    type=click.IntRange(min=1),
    default=1,
    metavar="N",
    help="Minimum stride. Default: 1.",
)
@click.option(
    "--max-stride",
    type=click.IntRange(min=2),
    default=32,
    metavar="N",
    help="Maximum stride. Default: 32.",
)
@click.option(
    "--no-adaptive",
    is_flag=True,
    default=False,
    help="Disable adaptive stride adjustment. Keep the stride fixed.",
)
@click.option(
    "--save-flow-dir",
    type=click.Path(),
    default=None,
    help="Directory to save optical flow color images.",
)
def discontinuity(
    paths,
    range_expr,
    initial_stride,
    min_stride,
    max_stride,
    no_adaptive,
    save_flow_dir,
):
    """Analyze image sequences or reconstructions for discontinuities.

    Input can be image paths/directories (analyzes frame-to-frame optical flow)
    or a single .sfmr file (analyzes camera motion with flow cross-check).

    \b
    Image sequence mode:
        Detects numbered sequences among the input images, then uses adaptive-stride
        optical flow to find discontinuities within each sequence.

    \b
    Examples:
        sfm discontinuity images/
        sfm discontinuity images/ --initial-stride 16
        sfm discontinuity images/ -r 1-100
    """
    from deadline.job_attachments.api import summarize_paths_by_sequence

    from .._discontinuity import analyze_image_sequence
    from .._filenames import number_from_filename

    if not paths:
        raise click.UsageError(
            "Must provide image paths, directories, or a .sfmr file."
        )

    # Check if input is a .sfmr file
    if len(paths) == 1 and paths[0].endswith(".sfmr"):
        from .._discontinuity import analyze_reconstruction
        from .._sfmtool import SfmrReconstruction

        sfmr_path = Path(paths[0])
        recon = SfmrReconstruction.load(sfmr_path)

        numbers = None
        if range_expr:
            from openjd.model import IntRangeExpr

            numbers = set(IntRangeExpr.from_str(range_expr))

        analyze_reconstruction(recon, range_numbers=numbers)
        return

    # Image sequence mode
    numbers = None
    if range_expr:
        from openjd.model import IntRangeExpr

        numbers = IntRangeExpr.from_str(range_expr)

    image_paths = expand_paths(paths, extensions=IMAGE_EXTENSIONS, numbers=numbers)
    if not image_paths:
        raise click.ClickException("No image files found in the provided paths.")

    image_paths.sort(key=lambda p: (number_from_filename(p) or 0, p.name))

    # Detect sequences
    filenames = [p.name for p in image_paths]
    summaries = summarize_paths_by_sequence(filenames)

    numbered_sequences = [s for s in summaries if s.index_set]
    if not numbered_sequences:
        raise click.ClickException(
            "No numbered sequences detected in the provided images. "
            "Discontinuity analysis requires sequential image data."
        )

    click.echo(
        f"Found {len(image_paths)} images in {len(numbered_sequences)} sequence(s)"
    )

    # Build a filename-to-path lookup
    name_to_path = {p.name: p for p in image_paths}

    for seq in numbered_sequences:
        # Get the ordered image paths for this sequence
        sorted_indexes = sorted(seq.index_set)
        # Reconstruct filenames from the sequence pattern
        seq_paths = []
        seq_frame_numbers = []
        for idx in sorted_indexes:
            # The pattern uses %d-style formatting
            fname = seq.path % idx
            if fname in name_to_path:
                seq_paths.append(name_to_path[fname])
                seq_frame_numbers.append(idx)

        if len(seq_paths) < 2:
            click.echo(
                f"\nSkipping sequence {seq.path}: only {len(seq_paths)} image(s)"
            )
            continue

        click.echo(f"\nAnalyzing sequence: {seq.path} ({len(seq_paths)} frames)")

        # Derive a base name from the sequence pattern (e.g. "seoul_bull_sculpture")
        seq_base = seq.path.rsplit(".", 1)[0].split("%")[0].rstrip("_")

        analyze_image_sequence(
            seq_paths,
            frame_numbers=seq_frame_numbers,
            initial_stride=initial_stride,
            min_stride=min_stride,
            max_stride=max_stride,
            adaptive=not no_adaptive,
            save_flow_dir=Path(save_flow_dir) if save_flow_dir else None,
            sequence_name=seq_base,
        )
