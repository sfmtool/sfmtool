# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Import COLMAP binary reconstruction to .sfmr format."""

from pathlib import Path

import click

from .._cli_utils import timed_command


@click.command("from-colmap-bin")
@timed_command
@click.help_option("--help", "-h")
@click.argument("colmap_reconstruction_path", type=click.Path(exists=True))
@click.option(
    "--image-dir",
    "image_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing the images (used to find workspace and .sift files).",
)
@click.option(
    "--output",
    "-o",
    "output_sfmr_file",
    required=True,
    type=click.Path(),
    help="Output .sfmr file path.",
)
@click.option(
    "--tool-name",
    "tool_name",
    default="unknown",
    help="Name of the tool that created the reconstruction (e.g., 'colmap', 'glomap'). Default: 'unknown'.",
)
def from_colmap_bin(
    colmap_reconstruction_path,
    image_dir,
    output_sfmr_file,
    tool_name,
):
    """Convert COLMAP .bin files to a .sfmr file.

    Reads a COLMAP reconstruction directory containing cameras.bin, images.bin,
    and points3D.bin and converts to the .sfmr format used by sfmtool.

    The --image-dir option is required to resolve workspace paths and find
    .sift files for each image.

    COLMAP_RECONSTRUCTION_PATH is the path to the COLMAP reconstruction
    directory (e.g., colmap_workspace/reconstruction/0/).

    Example usage:

    \b
        # Import a COLMAP reconstruction
        sfm from-colmap-bin colmap_output/0/ \\
            --image-dir my_workspace/images/ \\
            -o reconstruction.sfmr

        # Specify tool provenance
        sfm from-colmap-bin colmap_output/0/ \\
            --image-dir my_workspace/images/ \\
            -o reconstruction.sfmr \\
            --tool-name glomap
    """
    from .._colmap_io import build_metadata, colmap_binary_to_rust_sfmr
    from .._workspace import find_workspace_for_path, load_workspace_config

    colmap_reconstruction_path = Path(colmap_reconstruction_path)
    image_dir = Path(image_dir)
    output_sfmr_file = Path(output_sfmr_file)

    if output_sfmr_file.suffix.lower() != ".sfmr":
        raise click.UsageError(
            f"Output file must have .sfmr extension, got: {output_sfmr_file}"
        )

    try:
        click.echo(f"Loading COLMAP reconstruction from: {colmap_reconstruction_path}")

        # Resolve workspace for metadata
        workspace_dir = find_workspace_for_path(image_dir)
        if workspace_dir is None:
            raise RuntimeError(
                f"No workspace found at or above {image_dir}. "
                "Initialize one with 'sfm ws init'."
            )
        workspace_config = load_workspace_config(workspace_dir)

        # We need to do a two-pass: first read to get counts, then build metadata.
        # Use a placeholder metadata, then update after loading.
        from .._sfmtool import read_colmap_binary

        data = read_colmap_binary(colmap_reconstruction_path)
        num_images = len(data["image_names"])
        num_points = len(data["positions_xyz"])
        num_cameras = len(data["cameras"])
        # Count observations from track arrays
        num_observations = len(data["track_image_indexes"])

        metadata = build_metadata(
            workspace_dir=workspace_dir,
            output_path=output_sfmr_file.absolute(),
            workspace_config=workspace_config,
            operation="import",
            tool_name=tool_name,
            tool_options={},
            inputs={
                "source_file": {
                    "path": str(colmap_reconstruction_path),
                    "format": "colmap_binary",
                }
            },
            image_count=num_images,
            points3d_count=num_points,
            observation_count=num_observations,
            camera_count=num_cameras,
        )

        recon = colmap_binary_to_rust_sfmr(
            colmap_reconstruction_path, image_dir, metadata
        )

        click.echo(
            f"Loaded: {num_cameras} cameras, {num_images} images, {num_points} points"
        )

        recon.save(output_sfmr_file)

        click.echo("\nSuccessfully converted reconstruction")
        click.echo(f"Saved to: {output_sfmr_file}")

    except Exception as e:
        raise click.ClickException(str(e))
