# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Structure from motion solver orchestration."""

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import click

from ._filenames import group_by_directory
from ._path_summary import summarize_paths_by_sequence
from ._sfmtool.reconstruction import RangeExpr

if TYPE_CHECKING:
    from .colmap.db_setup import MatchesImageFilter


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

    sequences = _numbered_sequences(image_paths)
    # Each sequence gets its own windows: a window never mixes sequences, so
    # each window's images come from one directory.
    windows: list[tuple[str, list[int], list[Path]]] = []
    for label, number_to_path in sequences:
        sorted_numbers = sorted(number_to_path)
        for window_numbers in _overlapping_windows(
            sorted_numbers, window_size, overlap_size
        ):
            windows.append(
                (
                    label,
                    window_numbers,
                    [number_to_path[num] for num in window_numbers],
                )
            )

    click.echo(
        f"\nSequential overlap mode: window_size={window_size}, overlap={overlap_size}"
    )
    if len(sequences) == 1:
        sorted_numbers = sorted(sequences[0][1])
        click.echo(
            f"Detected sequence with {len(sorted_numbers)} images: "
            f"{RangeExpr.from_list(sorted_numbers)}"
        )
    else:
        click.echo(
            f"Detected {len(sequences)} sequences, each solved in its own windows:"
        )
        for label, number_to_path in sequences:
            sorted_numbers = sorted(number_to_path)
            click.echo(
                f"  {label}: {len(sorted_numbers)} images "
                f"({RangeExpr.from_list(sorted_numbers)})"
            )
    click.echo(f"Will perform {len(windows)} solves")
    click.echo()

    for i, (label, window_numbers, window_paths) in enumerate(windows, 1):
        window_range = RangeExpr.from_list(window_numbers)
        where = f"{label} {window_range}" if len(sequences) > 1 else f"{window_range}"

        click.echo(
            f"=== Solve {i}/{len(windows)}: {len(window_numbers)} images ({where}) ==="
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


# How many images an error message lists before it says how many more there are.
_MAX_LISTED_IMAGES = 5


def _numbered_sequences(image_paths: list[Path]) -> list[tuple[str, dict[int, Path]]]:
    """Split the images into numbered sequences, in sorted order.

    A sequence is the images of one directory whose file names follow one
    numbered pattern, as `summarize_paths_by_sequence` finds it, so frames
    with the same number in two directories belong to two sequences. Each
    sequence is returned as its label, the directory relative to the deepest
    directory holding every image followed by the pattern (such as
    `walk1/frame_%06d.jpg`), and a map from frame number to image path.

    Raises `click.ClickException` when no image is in a numbered sequence, or
    when some images are in none, naming them in sorted order of directory
    and then file name, so that no image is left out without a word.
    """
    sequences: list[tuple[str, dict[int, Path]]] = []
    outside: list[str] = []
    for dir_parts, paths in sorted(group_by_directory(image_paths).items()):
        name_to_path = {path.name: path for path in paths}
        used: set[str] = set()
        # Sorting here, rather than relying on the input order, which for a
        # directory is the platform's listing order, fixes the order of the
        # sequences and of the images named in the refusal below.
        summaries = sorted(
            summarize_paths_by_sequence(sorted(name_to_path)), key=lambda s: s.path
        )
        for summary in summaries:
            if not summary.index_set:
                continue
            prefix, padding, suffix = _split_pattern(summary.path)
            number_to_path = {}
            for number in summary.index_set:
                name = f"{prefix}{number:0{padding}d}{suffix}"
                number_to_path[number] = name_to_path[name]
                used.add(name)
            label = "/".join((*dir_parts, summary.path))
            sequences.append((label, number_to_path))
        outside.extend(
            "/".join((*dir_parts, name))
            for name in sorted(name_to_path)
            if name not in used
        )

    if not sequences:
        raise click.ClickException(
            "No numbered sequence detected in the provided images. "
            "Sequential overlap mode requires a numbered sequence of at least "
            "three images in one directory."
        )
    if outside:
        listed = ", ".join(outside[:_MAX_LISTED_IMAGES])
        if len(outside) > _MAX_LISTED_IMAGES:
            listed += f" and {len(outside) - _MAX_LISTED_IMAGES} more"
        raise click.ClickException(
            f"Sequential overlap mode solves numbered sequences of at least three "
            f"images, and {len(outside)} of the provided images "
            f"{'is' if len(outside) == 1 else 'are'} in none: "
            f"{listed}. Leave them out, for example with -r or --named-image-list."
        )
    return sequences


def _split_pattern(pattern: str) -> tuple[str, int, str]:
    """Split a printf pattern from `summarize_paths_by_sequence`, such as
    `frame_%06d.jpg`, into its prefix, zero padding (0 for `%d`) and suffix.

    The number is the last `%d` or `%0Nd` in the pattern; a `%` earlier in it
    is part of the file name.
    """
    match = list(re.finditer(r"%(?:0(\d+))?d", pattern))[-1]
    padding = int(match.group(1)) if match.group(1) else 0
    return pattern[: match.start()], padding, pattern[match.end() :]


def _overlapping_windows(
    sorted_numbers: list[int], window_size: int, overlap_size: int
) -> list[list[int]]:
    """Cut `sorted_numbers` into windows of `window_size` numbers, consecutive
    windows sharing `overlap_size` of them; the last window holds what remains."""
    windows = []
    start_idx = 0
    while start_idx < len(sorted_numbers):
        end_idx = min(start_idx + window_size, len(sorted_numbers))
        windows.append(sorted_numbers[start_idx:end_idx])
        if end_idx == len(sorted_numbers):
            break
        start_idx += window_size - overlap_size
    return windows


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
    image_filter: "MatchesImageFilter | None" = None,
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
            image_filter=image_filter,
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
            image_filter=image_filter,
            detect_infinity=detect_infinity,
        )
