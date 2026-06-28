# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Workspace-related commands (`sfm ws ...`)."""

from pathlib import Path

import click

from .._workspace import find_workspace_for_path, init_workspace


@click.group("ws")
@click.help_option("--help", "-h")
def ws():
    """Workspace-related operations."""


@ws.command("init")
@click.help_option("--help", "-h")
@click.argument("workspace_dir", required=False, type=click.Path())
@click.option(
    "--feature-tool",
    type=click.Choice(["colmap", "opencv", "sfmtool"], case_sensitive=False),
    default="sfmtool",
    help="Feature extraction tool: sfmtool (default), colmap, or opencv",
)
@click.option(
    "--dsp/--no-dsp",
    "domain_size_pooling",
    default=False,
    help="Enable/disable domain size pooling (COLMAP only, default: disabled)",
)
@click.option(
    "--max-features",
    "max_num_features",
    type=int,
    default=None,
    help="Maximum features per image (COLMAP and sfmtool, default: 8192)",
)
@click.option(
    "--gpu/--no-gpu",
    "use_gpu",
    default=True,
    help="Enable/disable GPU acceleration for SIFT extraction (COLMAP only, default: enabled)",
)
@click.option(
    "--affine-shape/--no-affine-shape",
    "estimate_affine_shape",
    default=False,
    help="Enable/disable affine shape estimation for SIFT (COLMAP only, default: disabled). "
    "Incompatible with GPU — use --no-gpu with --affine-shape.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Allow creating a workspace even if it's nested or already exists.",
)
def init(
    workspace_dir,
    feature_tool,
    domain_size_pooling,
    max_num_features,
    use_gpu,
    estimate_affine_shape,
    force,
):
    """Initialize a workspace with configuration.

    Creates a .sfm-workspace.json file in the specified directory (or current
    directory if not specified) with default feature extraction settings.

    Example usage:

        sfm ws init

        sfm ws init my_workspace

        sfm ws init --feature-tool opencv opencv_workspace

        sfm ws init --dsp dsp_workspace
    """
    # Validate tool-specific options (these are all COLMAP-only knobs)
    tool_lower = feature_tool.lower()
    if domain_size_pooling and tool_lower != "colmap":
        raise click.UsageError(
            f"The --dsp/--no-dsp option is only supported for COLMAP, not {feature_tool}"
        )
    if max_num_features is not None and tool_lower not in ("colmap", "sfmtool"):
        raise click.UsageError(
            f"The --max-features option is only supported for COLMAP and sfmtool, "
            f"not {feature_tool}"
        )
    if not use_gpu and tool_lower != "colmap":
        raise click.UsageError(
            f"The --gpu/--no-gpu option is only supported for COLMAP, not {feature_tool}"
        )
    if estimate_affine_shape and tool_lower != "colmap":
        raise click.UsageError(
            f"The --affine-shape option is only supported for COLMAP, not {feature_tool}"
        )
    if use_gpu and estimate_affine_shape and feature_tool.lower() == "colmap":
        raise click.UsageError(
            "COLMAP GPU SIFT does not support affine shape estimation. "
            "Use --no-affine-shape with --gpu, or --no-gpu with --affine-shape."
        )

    if workspace_dir is None:
        workspace_dir = Path(".")
    else:
        workspace_dir = Path(workspace_dir)

    workspace_dir = workspace_dir.resolve()
    config_path = workspace_dir / ".sfm-workspace.json"

    if not force:
        if config_path.exists():
            raise click.ClickException(
                f"A workspace already exists at: {workspace_dir}\n"
                "Use --force to overwrite the configuration."
            )

        existing_workspace = find_workspace_for_path(workspace_dir)
        if existing_workspace:
            raise click.ClickException(
                f"Cannot create nested workspace. Target directory is already inside "
                f"workspace: {existing_workspace}\n"
                "Use --force to create the nested workspace anyway."
            )

    workspace = init_workspace(
        workspace_dir,
        feature_tool=feature_tool,
        domain_size_pooling=domain_size_pooling,
        max_num_features=max_num_features,
        estimate_affine_shape=estimate_affine_shape,
        use_gpu=use_gpu,
    )

    click.echo(f"Initialized workspace: {workspace_dir.resolve()}")
    click.echo(f"Configuration file: {config_path.resolve()}")
    click.echo(f"  feature_tool: {feature_tool.lower()}")

    options_dict = workspace["feature_options"]
    if tool_lower == "colmap":
        dsp_value = options_dict.get("domain_size_pooling", False)
        max_features_value = options_dict.get("max_num_features")
        affine_value = options_dict.get("estimate_affine_shape", False)
        gpu_value = options_dict.get("use_gpu", True)
        click.echo(f"  estimate_affine_shape: {affine_value}")
        click.echo(f"  domain_size_pooling: {dsp_value}")
        click.echo(f"  max_num_features: {max_features_value}")
        click.echo(f"  use_gpu: {gpu_value}")
    elif tool_lower == "sfmtool":
        click.echo(f"  contrast_threshold: {options_dict.get('contrast_threshold')}")
        click.echo(f"  octave_layers: {options_dict.get('octave_layers')}")
        click.echo(f"  max_num_features: {options_dict.get('max_num_features')}")
    else:
        click.echo(f"  nfeatures: {options_dict.get('nfeatures', 0)}")
