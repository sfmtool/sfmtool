# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Feature matching command — produces .matches files."""

import os
from pathlib import Path

import click

from .._cli_utils import timed_command
from .._filenames import expand_paths
from ..camera.cameras import CAMERA_MODEL_NAMES


# Options that only have an effect under one matching method. Passing one of
# these without its companion method silently did nothing before; we now reject
# it so the mistake surfaces instead of producing wrong-looking results.
_MODE_OPTIONS: dict[str, list[tuple[str, str]]] = {
    "sequential": [("sequential_overlap", "--sequential-overlap")],
    "flow": [
        ("flow_preset", "--flow-preset"),
        ("flow_wide_baseline_skip", "--flow-skip"),
    ],
    "cluster": [
        ("cluster_alpha", "--cluster-alpha"),
        ("cluster_d", "--cluster-d"),
        ("cluster_preset", "--cluster-preset"),
    ],
}


def _reject_stray_mode_options(selected: str) -> None:
    """Error if a mode-specific option was passed for an unselected method."""
    ctx = click.get_current_context()
    for mode, opts in _MODE_OPTIONS.items():
        if mode == selected:
            continue
        stray = [
            flag
            for attr, flag in opts
            if ctx.get_parameter_source(attr) == click.ParameterSource.COMMANDLINE
        ]
        if stray:
            verb = "only applies" if len(stray) == 1 else "only apply"
            raise click.UsageError(
                f"{', '.join(stray)} {verb} to --{mode} matching, "
                f"but --{selected} was selected."
            )


@click.command("match")
@timed_command
@click.help_option("--help", "-h")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--exhaustive",
    "-e",
    "exhaustive",
    is_flag=True,
    help="Run exhaustive pairwise matching.",
)
@click.option(
    "--max-features",
    "max_feature_count",
    type=click.IntRange(min=1),
    help="Maximum number of features to use from each image.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    help="Output .matches file path. If not specified, generates a timestamped filename.",
)
@click.option(
    "--range",
    "-r",
    "range_expr",
    help="A range expression of file numbers to use from the input directories.",
)
@click.option(
    "--sequential",
    "-s",
    "sequential",
    is_flag=True,
    help="Run sequential matching (pairs nearby images in sequence order). "
    "Best for ordered image collections with known capture order.",
)
@click.option(
    "--sequential-overlap",
    "sequential_overlap",
    type=click.IntRange(min=1),
    default=10,
    help="Number of overlapping image pairs for --sequential. Default: 10.",
)
@click.option(
    "--flow",
    "flow_match",
    is_flag=True,
    help="Use optical flow-based matching instead of exhaustive descriptor matching. "
    "Best for sequential video frames with small inter-frame motion.",
)
@click.option(
    "--flow-preset",
    "flow_preset",
    type=click.Choice(["fast", "default", "high_quality"]),
    default="default",
    help="Optical flow quality preset for --flow. Default: default.",
)
@click.option(
    "--flow-skip",
    "flow_wide_baseline_skip",
    type=click.IntRange(min=1),
    default=5,
    help="Sliding window size for --flow. 1 = adjacent pairs only. Default: 5.",
)
@click.option(
    "--cluster",
    "cluster_match",
    is_flag=True,
    help="Use the background-floor track-cluster matcher: cluster all images' "
    "descriptors at once instead of matching image pairs.",
)
@click.option(
    "--cluster-alpha",
    "cluster_alpha",
    type=click.FloatRange(min=0.0, min_open=True),
    default=0.8,
    help="Background-floor radius multiplier for --cluster. Default: 0.8.",
)
@click.option(
    "--cluster-d",
    "cluster_d",
    type=click.IntRange(min=1),
    default=10,
    help="Background rank for --cluster: the d-th-nearest distance sets the "
    "floor. Default: 10.",
)
@click.option(
    "--cluster-preset",
    "cluster_preset",
    type=click.Choice(["accurate", "balanced", "fast"]),
    default="accurate",
    help="Kd-tree forest preset for --cluster. Default: accurate.",
)
@click.option(
    "--camera-model",
    "camera_model",
    type=click.Choice(CAMERA_MODEL_NAMES, case_sensitive=False),
    default=None,
    help="Camera model to use (overrides auto-detection).",
)
@click.option(
    "--merge",
    "merge",
    is_flag=True,
    help="Merge multiple .matches files into one. "
    "PATHS should be .matches files instead of image directories.",
)
def match(
    paths,
    exhaustive,
    sequential,
    sequential_overlap,
    max_feature_count,
    output_path,
    range_expr,
    flow_match,
    flow_preset,
    flow_wide_baseline_skip,
    cluster_match,
    cluster_alpha,
    cluster_d,
    cluster_preset,
    camera_model,
    merge,
):
    """Match features between image pairs and write a .matches file.

    Requires a workspace initialized with 'sfm ws init' and SIFT features
    extracted with 'sfm sift --extract'.

    Examples:
        # Exhaustive matching
        sfm match --exhaustive images/

        # Sequential matching for ordered collections
        sfm match --sequential images/

        # Flow-based matching for sequential video
        sfm match --flow images/

        # Background-floor track-cluster matching
        sfm match --cluster images/

        # With feature count limit
        sfm match --exhaustive --max-features 4096 images/

        # Merge matches from different methods
        sfm match --merge seq.matches exhaustive.matches -o combined.matches
    """
    if merge:
        # --merge runs no matching method, so any method-specific option is a
        # no-op; reject it rather than silently ignoring it (B4).
        _reject_stray_mode_options(selected="merge")

        from ..feature_match._run import _run_merge

        try:
            _run_merge(paths, output_path)
        except Exception as e:
            raise click.ClickException(str(e))
        return

    from ..cli import deduce_workspace

    # Default to the current directory when no paths are given
    if not paths:
        paths = (".",)

    method_count = sum([exhaustive, sequential, flow_match, cluster_match])
    if method_count > 1:
        raise click.UsageError(
            "Cannot specify more than one matching method. "
            "Choose one of: --exhaustive (-e), --sequential (-s), --flow, "
            "or --cluster"
        )
    if method_count == 0:
        raise click.UsageError(
            "Must specify a matching method: "
            "--exhaustive (-e), --sequential (-s), --flow, or --cluster"
        )

    matching_method = (
        "flow"
        if flow_match
        else "cluster"
        if cluster_match
        else "sequential"
        if sequential
        else "exhaustive"
    )
    _reject_stray_mode_options(selected=matching_method)

    numbers = None
    if range_expr:
        from .._sfmtool import RangeExpr

        numbers = RangeExpr(range_expr)

    paths = [Path(p) for p in paths]
    filenames = expand_paths(
        paths, extensions=(".png", ".jpg", ".jpeg"), numbers=numbers
    )
    if not filenames:
        raise click.UsageError("No image files found in the provided paths.")

    absolute_paths = [Path(os.path.normpath(os.path.abspath(p))) for p in filenames]
    workspace_dir = deduce_workspace({p.parent for p in absolute_paths})

    from ..camera.config import CameraConfigResolver
    from ..camera.setup import _check_camera_model_conflict
    from ..feature_match._run import _run_matching

    camera_config_resolver = CameraConfigResolver(workspace_dir)
    _check_camera_model_conflict(absolute_paths, camera_config_resolver, camera_model)

    try:
        _run_matching(
            absolute_paths,
            workspace_dir,
            matching_method=matching_method,
            max_feature_count=max_feature_count,
            output_path=output_path,
            camera_model=camera_model,
            flow_preset=flow_preset,
            flow_wide_baseline_skip=flow_wide_baseline_skip,
            sequential_overlap=sequential_overlap,
            cluster_d=cluster_d,
            cluster_alpha=cluster_alpha,
            cluster_preset=cluster_preset,
        )
    except Exception as e:
        raise click.ClickException(str(e))
