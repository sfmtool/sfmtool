# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster-patch refinement for `sfm cluster-patches`.

Extracted from `_commands/cluster_patches.py` so the command module stays a
thin Click wrapper, matching its `_embed_patches.py` and `_patch_compaction.py`
siblings. See `specs/core/patch/cluster-patch-refinement.md`.
"""

from pathlib import Path

import click


def _resolve_workspace(matches_file: Path, ws_meta: dict) -> Path:
    """Resolve the workspace directory a .matches file references (the
    relative-path candidate first, then the absolute path, then an ancestor
    search from the file's directory)."""
    from ._workspace import find_workspace_for_path

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
    patch_size: float,
    resolution: int,
    min_zncc: float,
    max_shift: float,
    max_keypoint_uncertainty: float,
):
    import os
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime

    import cv2
    import numpy as np

    from ._progress import _poll_progress
    from ._sfmtool.io import read_matches, write_matches
    from ._sfmtool.matching import refine_cluster_patches as _refine

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
    cluster_count = int(metadata["cluster_count"])
    member_count = int(metadata["cluster_member_count"])
    click.echo(f"Workspace: {workspace_dir}")
    click.echo(
        f"Images: {len(image_names)}, clusters: {cluster_count}, "
        f"members: {member_count}"
    )

    feature_counts = data["feature_counts"]
    member_images = np.asarray(data["member_images"])
    member_features = np.asarray(data["member_features"])
    image_dims = np.ascontiguousarray(data["image_dims"], dtype=np.uint32)
    # The input is a matcher output (an already-enriched file is refused
    # above), so its backbone geometry is the members' detections: the seeds
    # the cascade starts from, already in the file. No `.sift` file is opened
    # here at all.
    detected_positions = np.asarray(data["member_positions"], dtype=np.float32)
    detected_shapes = np.asarray(data["member_affine_shapes"], dtype=np.float32)

    def _read_one(name: str):
        img_path = workspace_dir / name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        return np.ascontiguousarray(img)

    click.echo("Reading images...")
    # Decode in a thread pool (cv2 releases the GIL), collecting results in
    # submission order so the list stays parallel to `image_names` (the
    # embed-patches pattern).
    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(_read_one, name) for name in image_names]
        images = []
        try:
            for future in futures:
                images.append(future.result())
        except BaseException:
            # Fail fast: without this, the pool's __exit__ would finish
            # decoding every queued image before the error surfaces.
            for f in futures:
                f.cancel()
            raise

    # The kernel takes per-image feature arrays and reads only the rows its
    # members name, so scattering the file's member geometry back to those
    # rows presents it exactly as a `.sift` read would — same values, same row
    # count, same out-of-range behaviour. Untouched rows are never read.
    positions, affine_shapes = [], []
    for i in range(len(image_names)):
        count = int(feature_counts[i])
        pos = np.zeros((count, 2), dtype=np.float32)
        aff = np.zeros((count, 2, 2), dtype=np.float32)
        on_image = member_images == i
        feats = member_features[on_image]
        pos[feats] = detected_positions[on_image]
        aff[feats] = detected_shapes[on_image]
        positions.append(pos)
        affine_shapes.append(aff)

    click.echo(f"Refining {cluster_count} clusters...")
    with _poll_progress(click.echo, cluster_count) as counter:
        result = _refine(
            images,
            positions,
            affine_shapes,
            data["cluster_starts"],
            data["member_images"],
            data["member_features"],
            # Sole conversion site: --patch-size is the full template edge
            # length (embed-patches' convention); the kernel takes the
            # half-width, so halve it here.
            radius=patch_size / 2.0,
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

    # The output's stage is refinement, so the backbone's geometry becomes the
    # refinement's — for the members the cascade measured. Those are exactly
    # the reference, the kept, and the two rejected-with-a-measurement
    # statuses; the members it never fitted (duplicate_image, not_evaluated,
    # rejected_unlocalizable) keep the detection they came in with, so every
    # row holds a real position and shape and member_status alone says which
    # reading it is.
    measured = np.isin(statuses, (0, 1, 2, 3))
    out_positions = detected_positions.copy()
    out_shapes = detected_shapes.copy()
    out_positions[measured] = result["member_positions"][measured].astype(np.float32)
    out_shapes[measured] = result["member_affine_shapes"][measured].astype(np.float32)

    # New file: images + clusters sections carried over, cluster_patches
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
        "image_dims": image_dims,
        "has_clusters": True,
        "cluster_starts": data["cluster_starts"],
        "member_images": data["member_images"],
        "member_features": data["member_features"],
        "member_positions": out_positions,
        "member_affine_shapes": out_shapes,
        "matcher_options": data["matcher_options"],
        "has_cluster_patches": True,
        "reference_members": result["reference_members"],
        "member_status": result["member_status"],
        "member_zncc": result["member_zncc"],
        "member_shift_px": result["member_shift_px"],
        "member_consistency_residual": result["member_consistency_residual"],
        "refine_options": {
            "patch_size": patch_size,
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
