# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Structure from motion solve command."""

import os
import tempfile
from pathlib import Path

import click
from deadline.job_attachments.api import summarize_paths_by_sequence
from openjd.model import IntRangeExpr

from .._cli_utils import timed_command
from .._filenames import expand_paths, number_from_filename


@click.command("solve")
@timed_command
@click.help_option("--help", "-h")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--colmap-dir",
    "colmap_dir",
    type=click.Path(file_okay=False),
    help="Directory for the COLMAP database and intermediate files.",
)
@click.option(
    "--max-features",
    "max_feature_count",
    type=click.IntRange(min=1),
    help="Maximum number of features to use from each image.",
)
@click.option(
    "--range",
    "-r",
    "range_expr",
    help="A range expression of file numbers to use from the input directories.",
)
@click.option(
    "--incremental",
    "-i",
    "incremental",
    is_flag=True,
    help="Run incremental structure from motion.",
)
@click.option(
    "--global",
    "-g",
    "global_mode",
    is_flag=True,
    help="Run global structure from motion using GLOMAP.",
)
@click.option(
    "--sfmr-dir",
    "sfmr_dir",
    type=click.Path(),
    help="Directory for .sfmr files. Default: workspace/sfmr.",
)
@click.option(
    "--seed",
    "-s",
    "random_seed",
    type=int,
    default=None,
    help="Random seed for reproducible reconstructions.",
)
@click.option(
    "--output",
    "-o",
    "output_sfm_file",
    type=click.Path(),
    help="Output .sfmr file path.",
)
@click.option(
    "--seq-overlap",
    "seq_overlap",
    type=str,
    help="Sequential overlap mode: 'WINDOW,OVERLAP' (e.g., '100,20').",
)
@click.option(
    "--refine-rig/--no-refine-rig",
    "refine_rig",
    default=True,
    help="Refine sensor-from-rig poses during bundle adjustment.",
)
@click.option(
    "--flow-match",
    "flow_match",
    is_flag=True,
    help="Use optical flow-based matching instead of exhaustive descriptor matching.",
)
@click.option(
    "--flow-preset",
    "flow_preset",
    type=click.Choice(["fast", "default", "high_quality"]),
    default="default",
    help="Optical flow quality preset for --flow-match. Default: default.",
)
@click.option(
    "--flow-skip",
    "flow_wide_baseline_skip",
    type=click.IntRange(min=1),
    default=5,
    help="Sliding window size for --flow-match. Default: 5.",
)
@click.option(
    "--camera-model",
    "camera_model",
    type=click.Choice(
        [
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "SIMPLE_RADIAL",
            "RADIAL",
            "OPENCV",
            "OPENCV_FISHEYE",
            "SIMPLE_RADIAL_FISHEYE",
            "RADIAL_FISHEYE",
            "THIN_PRISM_FISHEYE",
            "RAD_TAN_THIN_PRISM_FISHEYE",
        ],
        case_sensitive=False,
    ),
    default=None,
    help="Camera model to use (overrides auto-detection).",
)
def solve(
    paths,
    colmap_dir,
    max_feature_count,
    range_expr,
    incremental,
    global_mode,
    sfmr_dir,
    random_seed,
    output_sfm_file,
    seq_overlap,
    refine_rig,
    flow_match,
    flow_preset,
    flow_wide_baseline_skip,
    camera_model,
):
    """Run structure from motion on images or a .matches file.

    Input can be image paths/directories (runs feature matching internally)
    or a single .matches file (uses pre-computed matches).

    Examples:
        # From images (incremental)
        sfm solve -i path/to/images/

        # From a pre-computed .matches file
        sfm solve -i tvg-matches/my.matches

        # Use global SfM (GLOMAP)
        sfm solve -g path/to/images/

        # Sequential overlap mode
        sfm solve -i path/to/images/ --seq-overlap 100,20
    """
    from ..cli import deduce_workspace

    if not paths:
        raise click.UsageError("Must provide image paths or a .matches file.")

    if incremental and global_mode:
        raise click.UsageError(
            "Cannot specify both --incremental and --global. Choose one mode."
        )

    if not incremental and not global_mode:
        raise click.UsageError(
            "Must specify either --incremental (-i) or --global (-g) mode."
        )

    if seq_overlap and output_sfm_file:
        raise click.UsageError(
            "--seq-overlap cannot be used with --output. "
            "Sequential overlap mode only supports automatic filename generation."
        )

    if colmap_dir:
        colmap_dir = Path(colmap_dir)

    # Detect whether input is a single .matches file or image paths
    matches_file = None
    if len(paths) == 1 and paths[0].endswith(".matches"):
        matches_file = Path(paths[0])
        if not matches_file.exists():
            raise click.UsageError(f"Matches file not found: {matches_file}")
    elif any(p.endswith(".matches") for p in paths):
        raise click.UsageError(
            "Cannot mix .matches files with image paths. "
            "Provide either a single .matches file or image paths/directories."
        )

    if matches_file is not None:
        if seq_overlap:
            raise click.UsageError("--seq-overlap cannot be used with a .matches file.")
        if flow_match:
            raise click.UsageError("--flow-match cannot be used with a .matches file.")

        try:
            if colmap_dir:
                _run_sfm(
                    [],
                    None,
                    colmap_dir,
                    max_feature_count,
                    incremental,
                    sfmr_dir,
                    random_seed,
                    output_sfm_file=output_sfm_file,
                    refine_rig=refine_rig,
                    camera_model=camera_model,
                    matches_file=matches_file,
                )
            else:
                with tempfile.TemporaryDirectory(prefix="colmap_") as temp_colmap_dir:
                    _run_sfm(
                        [],
                        None,
                        Path(temp_colmap_dir),
                        max_feature_count,
                        incremental,
                        sfmr_dir,
                        random_seed,
                        output_sfm_file=output_sfm_file,
                        refine_rig=refine_rig,
                        camera_model=camera_model,
                        matches_file=matches_file,
                    )
        except Exception as e:
            raise click.ClickException(str(e))
        return

    # Standard path: solve from image paths
    paths = [Path(p) for p in paths]

    numbers = None
    if range_expr:
        numbers = IntRangeExpr.from_str(range_expr)

    filenames = expand_paths(
        paths, extensions=(".png", ".jpg", ".jpeg"), numbers=numbers
    )

    if not filenames:
        raise click.UsageError("No image files to process in the provided directories.")

    absolute_paths = [Path(os.path.normpath(os.path.abspath(p))) for p in filenames]

    workspace_dir = deduce_workspace({p.parent for p in absolute_paths})

    if seq_overlap:
        _run_sequential_overlap_sfm(
            absolute_paths,
            workspace_dir,
            colmap_dir,
            max_feature_count,
            incremental,
            sfmr_dir,
            random_seed,
            seq_overlap,
            refine_rig,
            camera_model=camera_model,
        )
        return

    matching_mode = "flow" if flow_match else "exhaustive"

    try:
        if colmap_dir:
            _run_sfm(
                absolute_paths,
                workspace_dir,
                colmap_dir,
                max_feature_count,
                incremental,
                sfmr_dir,
                random_seed,
                output_sfm_file=output_sfm_file,
                refine_rig=refine_rig,
                camera_model=camera_model,
                matching_mode=matching_mode,
                flow_preset=flow_preset,
                flow_wide_baseline_skip=flow_wide_baseline_skip,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="colmap_") as temp_colmap_dir:
                _run_sfm(
                    absolute_paths,
                    workspace_dir,
                    Path(temp_colmap_dir),
                    max_feature_count,
                    incremental,
                    sfmr_dir,
                    random_seed,
                    output_sfm_file=output_sfm_file,
                    refine_rig=refine_rig,
                    camera_model=camera_model,
                    matching_mode=matching_mode,
                    flow_preset=flow_preset,
                    flow_wide_baseline_skip=flow_wide_baseline_skip,
                )
    except Exception as e:
        raise click.ClickException(str(e))


def _run_sequential_overlap_sfm(
    image_paths: list[Path],
    workspace_dir: Path,
    colmap_dir: Path | None,
    max_feature_count: int | None,
    incremental: bool,
    sfmr_dir: str | Path | None,
    random_seed: int | None,
    seq_overlap: str,
    refine_rig: bool = True,
    camera_model: str | None = None,
):
    """Run sequential overlapping SfM solves."""
    try:
        parts = seq_overlap.split(",")
        if len(parts) != 2:
            raise ValueError("Format must be 'WINDOW,OVERLAP' (e.g., '100,20')")
        window_size = int(parts[0])
        overlap_size = int(parts[1])

        if window_size <= 0:
            raise ValueError("Window size must be positive")
        if overlap_size < 0:
            raise ValueError("Overlap size cannot be negative")
        if overlap_size >= window_size:
            raise ValueError("Overlap size must be less than window size")
    except ValueError as e:
        raise click.ClickException(f"Invalid --seq-overlap format: {e}")

    filenames = [p.name for p in image_paths]
    summaries = summarize_paths_by_sequence(filenames)

    numbered_sequences = [s for s in summaries if s.index_set]
    if len(numbered_sequences) == 0:
        raise click.ClickException(
            "No numbered sequence detected in the provided images. "
            "Sequential overlap mode requires a single numbered sequence."
        )
    if len(numbered_sequences) > 1:
        raise click.ClickException(
            f"Found {len(numbered_sequences)} numbered sequences. "
            "Sequential overlap mode requires exactly one numbered sequence."
        )

    sequence = numbered_sequences[0]
    sorted_numbers = sorted(sequence.index_set)

    number_to_path = {}
    for path in image_paths:
        file_num = number_from_filename(path.name)
        if file_num is not None and file_num in sequence.index_set:
            number_to_path[file_num] = path

    windows = []
    start_idx = 0
    while start_idx < len(sorted_numbers):
        end_idx = min(start_idx + window_size, len(sorted_numbers))
        window_numbers = sorted_numbers[start_idx:end_idx]
        windows.append(window_numbers)

        if end_idx == len(sorted_numbers):
            break
        start_idx += window_size - overlap_size

    click.echo(
        f"\nSequential overlap mode: window_size={window_size}, overlap={overlap_size}"
    )
    click.echo(
        f"Detected sequence with {len(sorted_numbers)} images: "
        f"{IntRangeExpr.from_list(sorted_numbers)}"
    )
    click.echo(f"Will perform {len(windows)} solves")
    click.echo()

    for i, window_numbers in enumerate(windows, 1):
        window_paths = [number_to_path[num] for num in window_numbers]
        window_range = IntRangeExpr.from_list(window_numbers)

        click.echo(
            f"=== Solve {i}/{len(windows)}: "
            f"{len(window_numbers)} images ({window_range}) ==="
        )

        if colmap_dir:
            solve_colmap_dir = colmap_dir / f"solve_{i:03d}"
            solve_colmap_dir.mkdir(parents=True, exist_ok=True)
            _run_sfm(
                window_paths,
                workspace_dir,
                solve_colmap_dir,
                max_feature_count,
                incremental,
                sfmr_dir,
                random_seed,
                output_sfm_file=None,
                refine_rig=refine_rig,
                camera_model=camera_model,
            )
        else:
            with tempfile.TemporaryDirectory(
                prefix=f"colmap_solve{i:03d}_"
            ) as temp_dir:
                _run_sfm(
                    window_paths,
                    workspace_dir,
                    Path(temp_dir),
                    max_feature_count,
                    incremental,
                    sfmr_dir,
                    random_seed,
                    output_sfm_file=None,
                    refine_rig=refine_rig,
                    camera_model=camera_model,
                )

        click.echo()

    click.echo(f"Completed all {len(windows)} sequential solves")


def _run_sfm(
    image_paths: list[str | Path],
    workspace_dir: str | Path | None,
    colmap_dir: str | Path,
    max_feature_count: int | None,
    incremental: bool,
    sfmr_dir: str | Path | None,
    random_seed: int | None,
    output_sfm_file: str | None = None,
    refine_rig: bool = True,
    camera_model: str | None = None,
    matching_mode: str = "exhaustive",
    flow_preset: str = "default",
    flow_wide_baseline_skip: int = 5,
    matches_file: str | Path | None = None,
):
    """Run SfM with the given colmap_dir."""
    if incremental:
        from .._isfm import run_incremental_sfm

        run_incremental_sfm(
            image_paths,
            workspace_dir,
            colmap_dir,
            max_feature_count=max_feature_count,
            sfmr_dir=sfmr_dir,
            random_seed=random_seed,
            output_sfm_file=output_sfm_file,
            refine_rig=refine_rig,
            camera_model=camera_model,
            matching_mode=matching_mode,
            flow_preset=flow_preset,
            flow_wide_baseline_skip=flow_wide_baseline_skip,
            matches_file=matches_file,
        )
    else:
        from .._gsfm import run_global_sfm

        run_global_sfm(
            image_paths,
            workspace_dir,
            colmap_dir,
            max_feature_count=max_feature_count,
            sfmr_dir=sfmr_dir,
            random_seed=random_seed,
            output_sfm_file=output_sfm_file,
            refine_rig=refine_rig,
            camera_model=camera_model,
            matching_mode=matching_mode,
            flow_preset=flow_preset,
            flow_wide_baseline_skip=flow_wide_baseline_skip,
            matches_file=matches_file,
        )
