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
        "Full edge length of the template each member is compared over, in "
        "keypoint-frame units. A larger template compares members over more "
        "of the texture around the feature."
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
    "--max-member-zncc-self-similarity-radius",
    "max_member_zncc_self_similarity_radius",
    type=click.FloatRange(min=0.0),
    default=2.5,
    show_default=True,
    help=(
        "Exclude cluster members whose own patch does not pin a 2D position, "
        "before reference selection and refinement: the member's ZNCC "
        "self-similarity radius, how far its template-grid patch can slide "
        "over itself and still match itself as well as a true match between "
        "two views would, is above this bar (template-grid px). Flat and "
        "edge-only patches read 3, the largest radius, so a bar of 3 or more "
        "turns nothing out; `0` disables the gate. See "
        "specs/core/patch/zncc-self-similarity-radius.md."
    ),
)
@click.option(
    "--regate-at-refined-shape/--no-regate-at-refined-shape",
    "regate_at_refined_shape",
    default=True,
    show_default=True,
    help=(
        "Read the member gate again, with the same bar, on each member's own "
        "patch at its refined shape and position (after --piecewise, where "
        "that moved it), for every member that passes the ZNCC and shift "
        "gates, and mark a member over the bar there "
        "rejected_unlocalizable_refined. See "
        "specs/core/patch/cluster-patch-refinement.md."
    ),
)
@click.option(
    "--max-capped-cells",
    "max_capped_cells",
    type=click.IntRange(0, 9),
    default=8,
    show_default=True,
    help=(
        "The most of the nine cells of a member's own patch at its refined "
        "shape that may read the largest ZNCC self-similarity radius (3, "
        "'3 or further'); a member with more is marked "
        "rejected_unlocalizable_cells. 9 turns nothing out."
    ),
)
@click.option(
    "--piecewise/--no-piecewise",
    "piecewise",
    default=False,
    show_default=True,
    help=(
        "After the affine fit, register each of the reference's nine cells "
        "separately against each kept member's image, starting at the "
        "member's fitted shape, and store each cell's displacement, ZNCC and "
        "status in the output (format version 8 per-cell columns). An affine "
        "map fitted to the cells' shifts updates the member's shape and "
        "position wherever the whole-patch ZNCC does not fall. See "
        "specs/core/patch/cluster-patch-refinement.md."
    ),
)
def cluster_patches(
    input_path,
    output_path,
    patch_size,
    resolution,
    min_zncc,
    max_shift,
    max_member_zncc_self_similarity_radius,
    regate_at_refined_shape,
    max_capped_cells,
    piecewise,
):
    """Refine a cluster-bearing .matches file into patch clusters.

    Per cluster: exclude members whose own patch does not pin a position,
    pick a reference member (largest SIFT scale), refine a
    Gaussian-windowed-ZNCC affine warp from the reference's patch to every
    other member (seeded from the SIFT affine shapes), vet members by
    achieved ZNCC and translation drift, read each surviving member's own
    patch again at its refined shape and refuse those that do not pin a
    position there as a whole or have too many cells that do not, and keep
    at most one member per image. With --piecewise, every kept member's
    patch is then cut into
    nine cells that are registered separately against the member's image,
    an affine map fitted to their shifts updates the member's shape where the
    whole-patch ZNCC does not fall, and each cell's displacement, ZNCC and
    status are stored beside the member. Each member is stored fully absolute — its affine SHAPE (the
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
            max_member_zncc_self_similarity_radius,
            piecewise,
            regate_at_refined_shape,
            max_capped_cells,
        )
    except click.UsageError:
        raise
    except Exception as e:
        raise click.ClickException(str(e))
