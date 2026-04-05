# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Align reconstructions command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._multi_align import align_command


@click.command("align")
@timed_command
@click.help_option("--help", "-h")
@click.argument("reference_sfmr", type=click.Path(exists=True), required=True)
@click.argument(
    "align_sfmr_paths", nargs=-1, type=click.Path(exists=True), required=True
)
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    type=click.Path(file_okay=False),
    help="Output directory for aligned reconstructions (required)",
    required=True,
)
@click.option(
    "--method",
    type=click.Choice(["cameras", "points"], case_sensitive=False),
    default="points",
    help="Alignment method: points (default) uses corresponding 3D points, cameras uses camera poses",
)
@click.option(
    "--confidence",
    type=click.FloatRange(0.0, 1.0),
    default=0.7,
    help="Confidence threshold for image matches (camera method only, default: 0.7)",
)
@click.option(
    "--max-error",
    type=float,
    default=0.1,
    help="Maximum acceptable error threshold (default: 0.1)",
)
@click.option(
    "--iterative",
    is_flag=True,
    help="Enable iterative refinement of alignment transforms",
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Generate visualization data for alignment quality",
)
@click.option(
    "--ransac/--no-ransac",
    "use_ransac",
    default=True,
    help="Enable/disable RANSAC outlier rejection for point-based alignment (default: enabled)",
)
@click.option(
    "--ransac-percentile",
    type=click.FloatRange(0.0, 100.0),
    default=95.0,
    help="Percentile of correspondence distances to use as RANSAC threshold (default: 95.0).",
)
@click.option(
    "--ransac-iterations",
    type=click.IntRange(min=1),
    default=1000,
    help="Number of RANSAC iterations for point-based alignment (default: 1000)",
)
def align(
    reference_sfmr,
    align_sfmr_paths,
    output_dir,
    method,
    confidence,
    max_error,
    iterative,
    visualize,
    use_ransac,
    ransac_percentile,
    ransac_iterations,
):
    """Align multiple .sfmr reconstructions.

    This command aligns ALIGN_SFMR files to the REFERENCE_SFMR coordinate frame.
    The reference file and all aligned files are saved to OUTPUT_DIR.

    Two alignment methods are available:
    - points (default): Aligns using corresponding 3D points found via shared features.
      Uses RANSAC by default to reject outlier correspondences.

    - cameras: Aligns using matched camera positions and orientations.

    REFERENCE_SFMR is the reconstruction that defines the reference coordinate frame.
    ALIGN_SFMR_PATHS are the reconstructions to align to the reference frame.

    All files must be .sfmr files.

    Examples:

        sfm align reference.sfmr recon1.sfmr -o aligned/

        sfm align reference.sfmr recon1.sfmr -o aligned/ --method cameras

        sfm align reference.sfmr recon1.sfmr -o aligned/ --no-ransac
    """
    if len(align_sfmr_paths) < 1:
        raise click.UsageError("Need at least 1 reconstruction to align to reference")

    ransac_options_provided = (
        not use_ransac or ransac_percentile != 95.0 or ransac_iterations != 1000
    )
    if ransac_options_provided and method != "points":
        raise click.UsageError(
            "RANSAC options (--ransac/--no-ransac, --ransac-percentile, --ransac-iterations) "
            "can only be used with --method points"
        )

    reference_path = Path(reference_sfmr)
    if reference_path.suffix.lower() != ".sfmr":
        raise click.UsageError(
            f"Reference reconstruction must be a .sfmr file: {reference_sfmr}"
        )

    align_paths = []
    for p in align_sfmr_paths:
        path = Path(p)
        if path.suffix.lower() != ".sfmr":
            raise click.UsageError(f"Reconstruction path must be a .sfmr file: {p}")
        align_paths.append(path)

    output_dir = Path(output_dir)

    try:
        align_command(
            reference_path=reference_path,
            align_paths=align_paths,
            output_dir=output_dir,
            method=method,
            confidence=confidence,
            max_error=max_error,
            iterative=iterative,
            visualize=visualize,
            use_ransac=use_ransac,
            ransac_percentile=ransac_percentile,
            ransac_iterations=ransac_iterations,
        )
    except Exception as e:
        raise click.ClickException(str(e))
