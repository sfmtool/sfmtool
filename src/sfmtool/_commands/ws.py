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
    type=click.Choice(["colmap", "opencv"], case_sensitive=False),
    default="colmap",
    help="Feature extraction tool: colmap (default) or opencv",
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
    help="Maximum features per image (COLMAP only, default: 8192)",
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
    # Validate tool-specific options
    if domain_size_pooling and feature_tool.lower() == "opencv":
        raise click.UsageError(
            "The --dsp/--no-dsp option is only supported for COLMAP, not OpenCV"
        )
    if max_num_features is not None and feature_tool.lower() == "opencv":
        raise click.UsageError(
            "The --max-features option is only supported for COLMAP, not OpenCV"
        )
    if not use_gpu and feature_tool.lower() == "opencv":
        raise click.UsageError(
            "The --gpu/--no-gpu option is only supported for COLMAP, not OpenCV"
        )
    if estimate_affine_shape and feature_tool.lower() == "opencv":
        raise click.UsageError(
            "The --affine-shape option is only supported for COLMAP, not OpenCV"
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
    )

    click.echo(f"Initialized workspace: {workspace_dir.resolve()}")
    click.echo(f"Configuration file: {config_path.resolve()}")
    click.echo(f"  feature_tool: {feature_tool.lower()}")

    options_dict = workspace["feature_options"]
    if feature_tool.lower() == "colmap":
        dsp_value = options_dict.get("domain_size_pooling", False)
        max_features_value = options_dict.get("max_num_features")
        affine_value = options_dict.get("estimate_affine_shape", False)
        click.echo(f"  estimate_affine_shape: {affine_value}")
        click.echo(f"  domain_size_pooling: {dsp_value}")
        click.echo(f"  max_num_features: {max_features_value}")
    else:
        click.echo(f"  nfeatures: {options_dict.get('nfeatures', 0)}")
