# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Workspace-related commands (`sfm ws ...`)."""

from pathlib import Path

import click

from .._workspace import (
    WorkspaceUpgradeError,
    init_workspace,
    upgrade_workspace,
)
from ..workspace import find_workspace_for_path


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
    "--image-to-gray",
    "image_to_gray_formula",
    metavar="FORMULA",
    default=None,
    help="The formula over R, G and B that converts an image to grey (sfmtool only, "
    "default: 0.2126*R + 0.7152*G + 0.0722*B).",
)
@click.option(
    "--quantize-bits",
    type=int,
    metavar="B",
    default=None,
    help="Round the grey conversion's value to B bits, 1 to 16 (sfmtool only, "
    "default: no rounding).",
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
    if image_to_gray_formula is not None and tool_lower != "sfmtool":
        raise click.UsageError(
            f"The --image-to-gray option is only supported for sfmtool, not "
            f"{feature_tool}, whose grey conversion is its own"
        )
    if quantize_bits is not None and tool_lower != "sfmtool":
        raise click.UsageError(
            f"The --quantize-bits option is only supported for sfmtool, not "
            f"{feature_tool}, whose grey conversion is its own"
        )
    if quantize_bits is not None and not 1 <= quantize_bits <= 16:
        raise click.UsageError(
            f"--quantize-bits must be an integer from 1 to 16, got {quantize_bits}"
        )
    if image_to_gray_formula is not None:
        from ..sift import ImageToGray

        try:
            ImageToGray(image_to_gray_formula)
        except ValueError as e:
            raise click.UsageError(f"--image-to-gray: {e}")
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
        image_to_gray_formula=image_to_gray_formula,
        quantize_bits=quantize_bits,
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
    click.echo(f"  image_to_gray: {_describe_image_to_gray(options_dict)}")
    click.echo(f"  feature_prefix_dir: {workspace['feature_prefix_dir']}")


def _describe_image_to_gray(feature_options: dict) -> str:
    """The ``image_to_gray`` object as one line: the formula, and the bit
    depth when there is one."""
    image_to_gray = feature_options["image_to_gray"]
    bits = image_to_gray.get("quantize_bits")
    if bits is None:
        return image_to_gray["formula"]
    return f"{image_to_gray['formula']}, rounded to {bits} bits"


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
    click.echo(f"  image_to_gray: {_describe_image_to_gray(new['feature_options'])}")
    click.echo(f"  old features folder: {old.get('feature_prefix_dir')}")
    click.echo(f"  new features folder: {new['feature_prefix_dir']}")
    click.echo(
        "The old folder and its .sift files are left on disk; .sift files are "
        "extracted again into the new folder."
    )
