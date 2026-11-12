# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Workspace-related commands (`sfm ws ...`)."""

from pathlib import Path

import click

from .._workspace import WorkspaceUpgradeError, upgrade_workspace
from .._workspace_init_options import (
    check_workspace_init_options,
    create_workspace,
    describe_image_to_gray,
    workspace_init_options,
)
from ..workspace import find_workspace_for_path


@click.group("ws")
@click.help_option("--help", "-h")
def ws():
    """Workspace-related operations."""


@ws.command("init")
@click.help_option("--help", "-h")
@click.argument("workspace_dir", required=False, type=click.Path())
@workspace_init_options()
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
    image_to_gray_formula,
    quantize_bits,
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

        sfm ws init --image-to-gray G --quantize-bits 8 green_workspace
    """
    check_workspace_init_options(
        feature_tool=feature_tool,
        domain_size_pooling=domain_size_pooling,
        max_num_features=max_num_features,
        use_gpu=use_gpu,
        estimate_affine_shape=estimate_affine_shape,
        image_to_gray_formula=image_to_gray_formula,
        quantize_bits=quantize_bits,
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

    create_workspace(
        workspace_dir,
        feature_tool=feature_tool,
        domain_size_pooling=domain_size_pooling,
        max_num_features=max_num_features,
        estimate_affine_shape=estimate_affine_shape,
        use_gpu=use_gpu,
        image_to_gray_formula=image_to_gray_formula,
        quantize_bits=quantize_bits,
    )


@ws.command("upgrade")
@click.help_option("--help", "-h")
@click.argument("path", required=False, type=click.Path(exists=True))
def upgrade(path):
    """Move a version 1 workspace to version 2 and a new features folder.

    Finds the workspace from PATH (or the current directory) by the upward
    search for .sfm-workspace.json, and rewrites a version 1 file as version 2,
    keeping every option, including hand-edited ones. The sfmtool tool's
    gray_formula becomes the image_to_gray object, and a colmap or opencv
    workspace gains its tool's conversion. A new feature_prefix_dir is named
    from the new options, so .sift files are extracted again into a new folder;
    the old folder, its .sift files and the named image lists file stay on
    disk, and .sfmr files that name the old folder still find their .sift
    files.

    Example usage:

        sfm ws upgrade

        sfm ws upgrade my_workspace
    """
    try:
        config_path, old, new = upgrade_workspace(Path(path) if path else Path("."))
    except WorkspaceUpgradeError as e:
        raise click.ClickException(str(e))
    if new is None:
        click.echo(f"{config_path}: nothing to do: already version 2")
        return
    click.echo(f"Upgraded workspace file: {config_path}")
    click.echo("  version: 1 -> 2")
    click.echo(f"  feature_tool: {new['feature_tool']}")
    click.echo(f"  image_to_gray: {describe_image_to_gray(new['feature_options'])}")
    click.echo(f"  old features folder: {old.get('feature_prefix_dir')}")
    click.echo(f"  new features folder: {new['feature_prefix_dir']}")
    click.echo(
        "The old folder and its .sift files are left on disk; .sift files are "
        "extracted again into the new folder."
    )
