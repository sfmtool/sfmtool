# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Epipolar geometry visualization command."""

from pathlib import Path

import click

from .._cli_utils import timed_command
from .._epipolar_viz import draw_epipolar_visualization
from .._filenames import number_from_filename
from .._sfm_reconstruction import get_image_hint_message
from .._sfmtool import SfmrReconstruction
from .._workspace import load_workspace_config


def resolve_image_name(image_input: str, recon) -> str:
    """Resolve an image name that could be either a filename or a file number.

    Args:
        image_input: Either a filename (e.g., "image_0003.jpg") or an integer string (e.g., "3")
        recon: The SfmrReconstruction to search in

    Returns:
        The resolved image filename from the reconstruction

    Raises:
        click.ClickException: If the input is invalid or ambiguous
    """
    try:
        file_number = int(image_input)
    except ValueError:
        return image_input

    image_names = recon.image_names

    matching_images = []
    for image_name in image_names:
        img_file_number = number_from_filename(image_name)
        if img_file_number == file_number:
            matching_images.append(image_name)

    if len(matching_images) == 1:
        return matching_images[0]
    elif len(matching_images) > 1:
        error_msg = (
            f"File number {file_number} is ambiguous - matches multiple images:\n"
        )
        for img in matching_images:
            error_msg += f"  - {img}\n"
        error_msg += "\nPlease specify the full filename instead of the file number."
        raise click.ClickException(error_msg)
    else:
        from openjd.model import IntRangeExpr

        available_numbers = []
        for image_name in image_names:
            img_file_number = number_from_filename(image_name)
            if img_file_number is not None:
                available_numbers.append(img_file_number)

        available_numbers.sort()

        if available_numbers:
            range_expr = IntRangeExpr.from_list(available_numbers)
            error_msg = (
                f"File number {file_number} not found in reconstruction.\n"
                f"Available file numbers: {range_expr}"
            )
        else:
            error_msg = f"File number {file_number} not found in reconstruction."

        raise click.ClickException(error_msg)


@click.command("epipolar")
@timed_command
@click.help_option("--help", "-h")
@click.argument("reconstruction_path", type=click.Path(exists=True))
@click.argument("image1", required=False)
@click.argument("image2", required=False)
@click.option(
    "--draw",
    "-d",
    "output_path",
    type=click.Path(),
    help="Draw epipolar visualization and save to this path.",
)
@click.option(
    "--max-features",
    "max_features",
    type=click.IntRange(min=1),
    help="Maximum number of shared features to visualize. Default: all shared features.",
)
@click.option(
    "--line-thickness",
    "line_thickness",
    type=click.IntRange(min=1),
    default=1,
    help="Thickness of epipolar lines in pixels. Default: 1.",
)
@click.option(
    "--feature-size",
    "feature_size",
    type=click.IntRange(min=1),
    default=3,
    help="Size of feature point markers in pixels. Default: 3.",
)
@click.option(
    "--rectify/--no-rectify",
    default=False,
    help="Whether to rectify images.",
)
@click.option(
    "--undistort",
    is_flag=True,
    default=False,
    help="Remove lens distortion before drawing (mutually exclusive with --rectify).",
)
@click.option(
    "--draw-lines/--no-lines",
    "draw_lines",
    default=True,
    help="Whether to draw epipolar lines or horizontal scanlines.",
)
@click.option(
    "--side-by-side/--separate",
    "side_by_side",
    default=False,
    help="Whether to output a single side-by-side image or two separate images (_A and _B).",
)
@click.option(
    "--sweep-with-max-features",
    "sweep_max_features",
    type=click.IntRange(min=1),
    help="Run sort-and-sweep matching with this many features from the original SIFT features. Requires --draw.",
)
@click.option(
    "--sweep-window-size",
    "sweep_window_size",
    type=click.IntRange(min=1),
    default=None,
    help="Window size for sort-and-sweep matching. Default: 30. (Only with --sweep-with-max-features)",
)
@click.option(
    "--pairs-dir",
    "pairs_dir",
    type=click.Path(),
    help="Process all adjacent pairs in sequence and save to this directory.",
)
def epipolar(
    reconstruction_path,
    image1,
    image2,
    output_path,
    max_features,
    line_thickness,
    feature_size,
    rectify,
    undistort,
    draw_lines,
    side_by_side,
    sweep_max_features,
    sweep_window_size,
    pairs_dir,
):
    """Visualize epipolar geometry between two images from a reconstruction.

    RECONSTRUCTION_PATH must be a .sfmr file.

    Two modes are available:

    1. Single pair mode (default): Specify IMAGE1 and IMAGE2 with --draw

    2. Adjacent pairs mode: Use --pairs-dir to process all adjacent pairs

    Example with filenames:

        sfm epipolar reconstruction.sfmr IMG_001.jpg IMG_005.jpg --draw output.png

    Example with file numbers:

        sfm epipolar reconstruction.sfmr 1 5 --draw output.png

    With rectification:

        sfm epipolar reconstruction.sfmr 1 5 --rectify --draw rectified.png

    Adjacent pairs mode:

        sfm epipolar reconstruction.sfmr --pairs-dir output_dir/
    """
    reconstruction_path = Path(reconstruction_path)

    if reconstruction_path.suffix.lower() != ".sfmr":
        raise click.UsageError(
            f"Reconstruction path must be a .sfmr file, got: {reconstruction_path}"
        )

    pairs_mode = pairs_dir is not None
    single_mode = image1 is not None or image2 is not None or output_path is not None

    if pairs_mode and single_mode:
        raise click.UsageError(
            "--pairs-dir is mutually exclusive with IMAGE1, IMAGE2, and --draw."
        )

    if not pairs_mode and not single_mode:
        raise click.UsageError(
            "Must specify either (IMAGE1, IMAGE2, --draw) for single pair mode "
            "or --pairs-dir for adjacent pairs mode."
        )

    if rectify and undistort:
        raise click.UsageError("--rectify and --undistort are mutually exclusive.")

    if sweep_window_size is not None and sweep_max_features is None:
        raise click.UsageError(
            "--sweep-window-size can only be used with --sweep-with-max-features"
        )

    if sweep_max_features is not None and sweep_window_size is None:
        sweep_window_size = 30

    recon = SfmrReconstruction.load(reconstruction_path)

    if not pairs_mode:
        if image1 is None:
            raise click.UsageError(
                f"Missing argument 'IMAGE1'.\n\n{get_image_hint_message(recon)}"
            )
        if image2 is None:
            raise click.UsageError(
                f"Missing argument 'IMAGE2'.\n\n{get_image_hint_message(recon)}"
            )
        if output_path is None:
            raise click.UsageError("Must specify --draw to enable visualization mode.")

    from ..cli import deduce_workspace

    workspace_dir = deduce_workspace({reconstruction_path})
    try:
        workspace_config = load_workspace_config(workspace_dir)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    tool = workspace_config["feature_tool"]
    feature_options = workspace_config["feature_options"]

    if pairs_mode:
        pairs_dir = Path(pairs_dir)
        pairs_dir.mkdir(parents=True, exist_ok=True)

        image_names = recon.image_names

        images_with_numbers = []
        for img_name in image_names:
            file_num = number_from_filename(img_name)
            if file_num is not None:
                images_with_numbers.append((file_num, img_name))

        if not images_with_numbers:
            raise click.ClickException(
                "No images with file numbers found in reconstruction."
            )

        images_with_numbers.sort(key=lambda x: x[0])
        sorted_images = [img_name for _, img_name in images_with_numbers]

        if len(sorted_images) < 2:
            raise click.ClickException(
                f"Need at least 2 images with file numbers for pairs mode, found {len(sorted_images)}."
            )

        click.echo(f"Processing {len(sorted_images) - 1} adjacent pairs...")
        click.echo(f"Output directory: {pairs_dir}")

        try:
            for i in range(len(sorted_images) - 1):
                left_image = sorted_images[i]
                right_image = sorted_images[i + 1]

                left_basename = Path(left_image).stem
                left_output = pairs_dir / f"{left_basename}.png"

                click.echo(
                    f"\nProcessing pair {i + 1}/{len(sorted_images) - 1}: {left_basename}"
                )

                draw_epipolar_visualization(
                    recon=recon,
                    image1_name=left_image,
                    image2_name=right_image,
                    output_path=str(left_output),
                    max_features=max_features,
                    line_thickness=line_thickness,
                    feature_size=feature_size,
                    rectify=rectify,
                    undistort=undistort,
                    draw_lines=draw_lines,
                    side_by_side=side_by_side,
                    save_which="first" if not side_by_side else "both",
                    feature_tool=tool,
                    feature_options=feature_options,
                    sweep_max_features=sweep_max_features,
                    sweep_window_size=sweep_window_size,
                )

            last_right = sorted_images[-1]
            last_left = sorted_images[-2]
            right_basename = Path(last_right).stem
            right_output = pairs_dir / f"{right_basename}.png"

            click.echo(f"\nProcessing last image: {right_basename}")

            draw_epipolar_visualization(
                recon=recon,
                image1_name=last_right,
                image2_name=last_left,
                output_path=str(right_output),
                max_features=max_features,
                line_thickness=line_thickness,
                feature_size=feature_size,
                rectify=rectify,
                undistort=undistort,
                draw_lines=draw_lines,
                side_by_side=side_by_side,
                save_which="first" if not side_by_side else "both",
                feature_tool=tool,
                feature_options=feature_options,
                sweep_max_features=sweep_max_features,
                sweep_window_size=sweep_window_size,
            )

            click.echo(f"\nCompleted processing {len(sorted_images)} images")

        except Exception as e:
            raise click.ClickException(str(e))

    else:
        output_path = Path(output_path)

        image1 = resolve_image_name(image1, recon)
        image2 = resolve_image_name(image2, recon)

        try:
            draw_epipolar_visualization(
                recon=recon,
                image1_name=image1,
                image2_name=image2,
                output_path=str(output_path),
                max_features=max_features,
                line_thickness=line_thickness,
                feature_size=feature_size,
                rectify=rectify,
                undistort=undistort,
                draw_lines=draw_lines,
                side_by_side=side_by_side,
                feature_tool=tool,
                feature_options=feature_options,
                sweep_max_features=sweep_max_features,
                sweep_window_size=sweep_window_size,
            )
        except Exception as e:
            raise click.ClickException(str(e))
