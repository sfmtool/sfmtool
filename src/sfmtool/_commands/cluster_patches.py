# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Refine SIFT clusters into patch clusters (`sfm cluster-patches`)."""

from pathlib import Path

import click

from .._cli_utils import timed_command


@click.command("cluster-patches")
@timed_command
@click.help_option("--help", "-h")
@click.option(
    "-i",
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Cluster-bearing .matches file (from sfm match --cluster).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output .matches path (default: the input with a -patches suffix).",
)
@click.option(
    "--patch-size",
    "patch_size",
    type=click.FloatRange(min=0.0, min_open=True),
    default=12.0,
    show_default=True,
    help=(
        "Template size — the full patch edge length (in keypoint-frame units), "
        "halved to the kernel's template half-width and passed to "
        "refine_cluster_patches. The default sits at SIFT's ~12x descriptor "
        "window; the larger template vets members against more of the texture "
        "the detector deemed characteristic."
    ),
)
@click.option(
    "--resolution",
    type=click.IntRange(min=3),
    default=25,
    show_default=True,
    help="Template samples per axis.",
)
@click.option(
    "--min-zncc",
    "min_zncc",
    type=click.FloatRange(-1.0, 1.0),
    default=0.85,
    show_default=True,
    help="Member acceptance threshold on the achieved windowed ZNCC.",
)
@click.option(
    "--max-shift",
    "max_shift",
    type=click.FloatRange(min=0.0),
    default=3.0,
    show_default=True,
    help="Max translation drift from the SIFT seed, px.",
)
@click.option(
    "--max-keypoint-uncertainty",
    "max_keypoint_uncertainty",
    type=click.FloatRange(min=0.0),
    default=0.35,
    show_default=True,
    help=(
        "Exclude cluster members whose own patch scores a predicted keypoint "
        "position uncertainty (patch localizability, template-grid px) above "
        "this, before reference selection and refinement — the flat/edge "
        "aperture cases that cannot pin a 2D position. Same default value as "
        "embed-patches' cull (scored here on the template grid with the "
        "refinement window); `0` disables the gate. See "
        "specs/core/patch/patch-localizability.md."
    ),
)
def cluster_patches(
    input_path,
    output_path,
    patch_size,
    resolution,
    min_zncc,
    max_shift,
    max_keypoint_uncertainty,
):
    """Refine a cluster-bearing .matches file into patch clusters.

    Per cluster: exclude members whose patch fails the localizability gate,
    pick a reference member (largest SIFT scale), refine a
    Gaussian-windowed-ZNCC affine warp from the reference's patch to every
    other member (seeded from the SIFT affine shapes), vet members by
    achieved ZNCC and translation drift, and keep at most one member per
    image. Each member is stored fully absolute — its affine SHAPE (the
    refined warp composed onto the reference feature's detector shape) plus
    its refined keypoint position — so a consumer reads a member's extent and
    position with no .sift lookup. Writes a NEW .matches file that copies the
    input's images and clusters sections and adds the cluster_patches
    enrichment (write-once workflow, like adding two-view geometries).

    \b
    Example:
        sfm cluster-patches -i matches/clusters.matches
    """
    from .._cluster_patches import _run_cluster_patches

    try:
        _run_cluster_patches(
            Path(input_path),
            output_path,
            patch_size,
            resolution,
            min_zncc,
            max_shift,
            max_keypoint_uncertainty,
        )
    except click.UsageError:
        raise
    except Exception as e:
        raise click.ClickException(str(e))
