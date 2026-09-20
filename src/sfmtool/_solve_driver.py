# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Structure from motion solver orchestration."""

import tempfile
from pathlib import Path

import click

from ._filenames import number_from_filename
from ._path_summary import summarize_paths_by_sequence
from ._sfmtool.reconstruction import RangeExpr


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
    detect_infinity: bool = True,
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
        f"{RangeExpr.from_list(sorted_numbers)}"
    )
    click.echo(f"Will perform {len(windows)} solves")
    click.echo()

    for i, window_numbers in enumerate(windows, 1):
        window_paths = [number_to_path[num] for num in window_numbers]
        window_range = RangeExpr.from_list(window_numbers)

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
                detect_infinity=detect_infinity,
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
                    detect_infinity=detect_infinity,
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
    range_expr: str | None = None,
    detect_infinity: bool = True,
):
    """Run SfM with the given colmap_dir."""
    if incremental:
        from ._incremental_sfm import run_incremental_sfm

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
            range_expr=range_expr,
            detect_infinity=detect_infinity,
        )
    else:
        from ._global_sfm import run_global_sfm

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
            range_expr=range_expr,
            detect_infinity=detect_infinity,
        )
