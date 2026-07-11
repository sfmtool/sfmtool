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
    "--radius",
    type=click.FloatRange(min=0.0, min_open=True),
    default=4.0,
    show_default=True,
    help="Template half-width, keypoint-frame units.",
)
@click.option(
    "--resolution",
    type=click.IntRange(min=3),
    default=15,
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
        "specs/core/patch-localizability.md."
    ),
)
def cluster_patches(
    input_path,
    output_path,
    radius,
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
    image. Writes a NEW .matches file that copies the input's images and
    clusters sections and adds the cluster_patches enrichment (write-once
    workflow, like adding two-view geometries).

    \b
    Example:
        sfm cluster-patches -i matches/clusters.matches
    """
    try:
        _run_cluster_patches(
            Path(input_path),
            output_path,
            radius,
            resolution,
            min_zncc,
            max_shift,
            max_keypoint_uncertainty,
        )
    except click.UsageError:
        raise
    except Exception as e:
        raise click.ClickException(str(e))


def _resolve_workspace(matches_file: Path, ws_meta: dict) -> Path:
    """Resolve the workspace directory a .matches file references (the
    relative-path candidate first, then the absolute path, then an ancestor
    search from the file's directory)."""
    from .._workspace import find_workspace_for_path

    matches_dir = matches_file.parent.absolute()
    rel_path = ws_meta.get("relative_path", "")
    if rel_path:
        candidate = (matches_dir / rel_path).resolve()
        if (candidate / ".sfm-workspace.json").exists():
            return candidate
    abs_path = ws_meta.get("absolute_path", "")
    if abs_path:
        candidate = Path(abs_path)
        if (candidate / ".sfm-workspace.json").exists():
            return candidate
    workspace_dir = find_workspace_for_path(matches_dir)
    if workspace_dir is None:
        raise RuntimeError(
            f"Cannot resolve workspace for {matches_file}. "
            "Ensure the workspace exists and contains .sfm-workspace.json."
        )
    return workspace_dir


def _run_cluster_patches(
    in_path: Path,
    output_path: str | None,
    radius: float,
    resolution: int,
    min_zncc: float,
    max_shift: float,
    max_keypoint_uncertainty: float,
):
    import os
    from datetime import datetime

    import cv2
    import numpy as np

    from .._embed_patches import _poll_progress
    from .._sfmtool.io import read_matches, read_sift, read_sift_metadata, write_matches
    from .._sfmtool.matching import refine_cluster_patches as _refine

    data = read_matches(in_path)
    if not data["has_clusters"]:
        raise click.UsageError(
            f"{in_path} has no clusters section; run `sfm match --cluster` to "
            "produce a cluster-bearing .matches file"
        )
    if data["has_cluster_patches"]:
        raise click.UsageError(
            f"{in_path} already carries a cluster_patches section; .matches "
            "files are write-once — rerun from the original clusters file"
        )

    out = (
        Path(output_path)
        if output_path
        else in_path.with_name(f"{in_path.stem}-patches{in_path.suffix}")
    )
    if out.exists():
        raise click.UsageError(f"{out} already exists; pass -o to choose another path")

    metadata = data["metadata"]
    ws_meta = metadata["workspace"]
    image_names = list(data["image_names"])
    workspace_dir = _resolve_workspace(in_path, ws_meta)
    feature_prefix_dir = ws_meta.get("contents", {}).get("feature_prefix_dir", "")
    cluster_count = int(metadata["cluster_count"])
    member_count = int(metadata["cluster_member_count"])
    click.echo(f"Workspace: {workspace_dir}")
    click.echo(
        f"Images: {len(image_names)}, clusters: {cluster_count}, "
        f"members: {member_count}"
    )

    # Locate each image's .sift via the images section + workspace reference,
    # verify content hashes, and read the feature geometry (capped at the
    # feature count used during matching, so member indices line up).
    feature_counts = data["feature_counts"]
    sift_hashes = data["sift_content_hashes"]
    images, positions, affine_shapes = [], [], []
    click.echo("Reading images and .sift features...")
    for i, name in enumerate(image_names):
        img_path = workspace_dir / name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        images.append(np.ascontiguousarray(img))

        rel = Path(name)
        sift_path = workspace_dir / rel.parent / feature_prefix_dir / f"{rel.name}.sift"
        if not sift_path.exists():
            raise FileNotFoundError(f"SIFT file not found: {sift_path}")
        sift_meta = read_sift_metadata(sift_path)
        actual = bytes.fromhex(sift_meta["content_hash"]["content_xxh128"])
        if actual != bytes(sift_hashes[i]):
            raise RuntimeError(
                f"{sift_path}: content hash differs from the .matches images "
                "section (features re-extracted since matching?)"
            )
        sift = read_sift(sift_path)
        count = int(feature_counts[i])
        positions.append(
            np.ascontiguousarray(sift["positions_xy"][:count], dtype=np.float32)
        )
        affine_shapes.append(
            np.ascontiguousarray(sift["affine_shapes"][:count], dtype=np.float32)
        )

    click.echo(f"Refining {cluster_count} clusters...")
    with _poll_progress(click.echo, cluster_count) as counter:
        result = _refine(
            images,
            positions,
            affine_shapes,
            data["cluster_starts"],
            data["member_images"],
            data["member_features"],
            radius=radius,
            resolution=resolution,
            min_zncc=min_zncc,
            max_shift_px=max_shift,
            max_keypoint_uncertainty=max_keypoint_uncertainty,
            progress=counter,
        )

    statuses = result["member_status"]
    n_ref = int((statuses == 0).sum())
    n_kept = int((statuses == 1).sum())
    n_rejected = int(((statuses == 2) | (statuses == 3)).sum())
    n_dup = int((statuses == 4).sum())
    n_skip = int((statuses == 5).sum())
    n_unloc = int((statuses == 6).sum())

    # New file: images + clusters sections copied verbatim, cluster_patches
    # from the kernel output, metadata updated.
    out_meta = dict(metadata)
    out_meta["has_cluster_patches"] = True
    out_meta["timestamp"] = datetime.now().astimezone().isoformat()
    out_abs = Path(os.path.abspath(out))
    out_meta["workspace"] = dict(ws_meta)
    out_meta["workspace"]["relative_path"] = os.path.relpath(
        workspace_dir, out_abs.parent
    ).replace("\\", "/")

    out_data = {
        "metadata": out_meta,
        "image_names": image_names,
        "feature_tool_hashes": data["feature_tool_hashes"],
        "sift_content_hashes": data["sift_content_hashes"],
        "feature_counts": data["feature_counts"],
        "has_clusters": True,
        "cluster_starts": data["cluster_starts"],
        "member_images": data["member_images"],
        "member_features": data["member_features"],
        "matcher_options": data["matcher_options"],
        "has_cluster_patches": True,
        "reference_members": result["reference_members"],
        "member_status": result["member_status"],
        "member_affines": result["member_affines"],
        "member_zncc": result["member_zncc"],
        "member_shift_px": result["member_shift_px"],
        "member_consistency_residual": result["member_consistency_residual"],
        "refine_options": {
            "radius": radius,
            "resolution": resolution,
            "min_zncc": min_zncc,
            "max_shift_px": max_shift,
            "max_keypoint_uncertainty": max_keypoint_uncertainty,
        },
        "has_two_view_geometries": False,
    }
    click.echo(f"Writing {out}...")
    write_matches(out, out_data)
    consistency = result["member_consistency_residual"]
    finite = consistency[np.isfinite(consistency)]
    if len(finite):
        click.echo(
            f"Warp consistency (stored signal, lower = better): median "
            f"{np.median(finite):.3f}, p90 {np.percentile(finite, 90):.3f} "
            f"over {len(finite)} fitted members"
        )
    click.echo(
        f"Done: {n_ref} references, {n_kept} kept, {n_rejected} rejected, "
        f"{n_unloc} unlocalizable, {n_dup} duplicate-image, "
        f"{n_skip} not evaluated"
    )
