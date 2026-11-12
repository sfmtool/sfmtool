# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The workspace options of `sfm ws init`, which `sfm pick-images
--new-workspace` also takes, their checks, and the creation of a workspace
with the lines `sfm ws init` prints.

Both commands declare the options with `workspace_init_options`, check them
with `check_workspace_init_options` and create the workspace with
`create_workspace`, so the two cannot drift apart. The module imports only
click and the standard library at import time, so neither command's `--help`
loads more.
"""

from collections.abc import Callable
from pathlib import Path

import click

# Each option's declarations and keyword arguments, in the order `sfm ws init`
# lists them.
_OPTIONS = (
    (
        ("--feature-tool",),
        {
            "type": click.Choice(["colmap", "opencv", "sfmtool"], case_sensitive=False),
            "default": "sfmtool",
            "help": "Feature extraction tool: sfmtool (default), colmap, or opencv.",
        },
    ),
    (
        ("--dsp/--no-dsp", "domain_size_pooling"),
        {
            "default": False,
            "help": "Enable/disable domain size pooling (COLMAP only, default: "
            "disabled).",
        },
    ),
    (
        ("--max-features", "max_num_features"),
        {
            "type": int,
            "default": None,
            "help": "Maximum features per image (COLMAP and sfmtool, default: 8192).",
        },
    ),
    (
        ("--gpu/--no-gpu", "use_gpu"),
        {
            "default": True,
            "help": "Enable/disable GPU acceleration for SIFT extraction (COLMAP "
            "only, default: enabled).",
        },
    ),
    (
        ("--affine-shape/--no-affine-shape", "estimate_affine_shape"),
        {
            "default": False,
            "help": "Enable/disable affine shape estimation for SIFT (COLMAP only, "
            "default: disabled). Incompatible with GPU — use --no-gpu with "
            "--affine-shape.",
        },
    ),
    (
        ("--image-to-gray", "image_to_gray_formula"),
        {
            "metavar": "FORMULA",
            "default": None,
            "help": "The formula over R, G and B that converts an image to grey "
            "(sfmtool only, default: 0.2126*R + 0.7152*G + 0.0722*B).",
        },
    ),
    (
        ("--quantize-bits",),
        {
            "type": int,
            "metavar": "B",
            "default": None,
            "help": "Round the grey conversion's value to B bits, 1 to 16 (sfmtool "
            "only, default: no rounding).",
        },
    ),
)

# The parameter names the options give the command, and the option each is
# spelled as on the command line, in the order `sfm ws init` lists them.
WORKSPACE_INIT_PARAMETERS = {
    "feature_tool": "--feature-tool",
    "domain_size_pooling": "--dsp/--no-dsp",
    "max_num_features": "--max-features",
    "use_gpu": "--gpu/--no-gpu",
    "estimate_affine_shape": "--affine-shape/--no-affine-shape",
    "image_to_gray_formula": "--image-to-gray",
    "quantize_bits": "--quantize-bits",
}


def workspace_init_options(help_suffix: str = "") -> Callable[[Callable], Callable]:
    """Attach the workspace options to a command, each help text followed by
    `help_suffix`. The command receives the parameters named in
    `WORKSPACE_INIT_PARAMETERS`."""

    def attach(f: Callable) -> Callable:
        for declarations, keywords in reversed(_OPTIONS):
            keywords = dict(keywords, help=keywords["help"] + help_suffix)
            f = click.option(*declarations, **keywords)(f)
        return f

    return attach


def check_workspace_init_options(
    *,
    feature_tool: str,
    domain_size_pooling: bool,
    max_num_features: int | None,
    use_gpu: bool,
    estimate_affine_shape: bool,
    image_to_gray_formula: str | None,
    quantize_bits: int | None,
) -> None:
    """Raise a `click.UsageError` for options the feature tool does not take
    or values out of range, as `sfm ws init` checks them."""
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
        from .sift import ImageToGray

        try:
            ImageToGray(image_to_gray_formula)
        except ValueError as e:
            raise click.UsageError(f"--image-to-gray: {e}")
    if use_gpu and estimate_affine_shape and tool_lower == "colmap":
        raise click.UsageError(
            "COLMAP GPU SIFT does not support affine shape estimation. "
            "Use --no-affine-shape with --gpu, or --no-gpu with --affine-shape."
        )


def create_workspace(workspace_dir: Path, **options) -> None:
    """Write the workspace file in `workspace_dir` with `init_workspace` and
    the checked workspace options, and print the lines `sfm ws init` prints."""
    from ._workspace import init_workspace

    workspace = init_workspace(workspace_dir, **options)
    config_path = workspace_dir / ".sfm-workspace.json"
    tool_lower = options["feature_tool"].lower()
    click.echo(f"Initialized workspace: {workspace_dir.resolve()}")
    click.echo(f"Configuration file: {config_path.resolve()}")
    click.echo(f"  feature_tool: {tool_lower}")

    options_dict = workspace["feature_options"]
    if tool_lower == "colmap":
        click.echo(
            f"  estimate_affine_shape: "
            f"{options_dict.get('estimate_affine_shape', False)}"
        )
        click.echo(
            f"  domain_size_pooling: {options_dict.get('domain_size_pooling', False)}"
        )
        click.echo(f"  max_num_features: {options_dict.get('max_num_features')}")
        click.echo(f"  use_gpu: {options_dict.get('use_gpu', True)}")
    elif tool_lower == "sfmtool":
        click.echo(f"  contrast_threshold: {options_dict.get('contrast_threshold')}")
        click.echo(f"  octave_layers: {options_dict.get('octave_layers')}")
        click.echo(f"  max_num_features: {options_dict.get('max_num_features')}")
    else:
        click.echo(f"  nfeatures: {options_dict.get('nfeatures', 0)}")
    click.echo(f"  image_to_gray: {describe_image_to_gray(options_dict)}")
    click.echo(f"  feature_prefix_dir: {workspace['feature_prefix_dir']}")


def describe_image_to_gray(feature_options: dict) -> str:
    """The ``image_to_gray`` object as one line: the formula, and the bit
    depth when there is one."""
    image_to_gray = feature_options["image_to_gray"]
    bits = image_to_gray.get("quantize_bits")
    if bits is None:
        return image_to_gray["formula"]
    return f"{image_to_gray['formula']}, rounded to {bits} bits"
