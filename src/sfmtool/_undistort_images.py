# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""
Undistort images from an SfM reconstruction using camera parameters.

For each image in a reconstruction, this module:
1. Undistorts the image via WarpMap
2. Transforms .sift feature positions and affine shapes
3. Remaps tracks (dropping features outside the pinhole frame)
4. Assembles a new .sfmr reconstruction with pinhole cameras
5. Writes a workspace config (.sfm-workspace.json)
"""

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ._sfmtool import SfmrReconstruction, WarpMap


def _print_camera_params(label: str, cam) -> None:
    """Print camera model and parameters in a compact table."""
    from ._cameras import _CAMERA_PARAM_NAMES

    params = cam.parameters
    canonical_order = _CAMERA_PARAM_NAMES.get(cam.model)
    if canonical_order is not None:
        keys = list(canonical_order)
        extra = sorted(set(params.keys()) - set(keys))
        keys.extend(extra)
    else:
        keys = list(params.keys())

    print(f"  {label}: {cam.model} {cam.width}x{cam.height}")
    if keys:
        name_width = max(len(k) for k in keys)
        print(f"    {'Parameter':<{name_width}}  {'Value':>14}")
        print(f"    {'-' * name_width}  {'-' * 14}")
        for key in keys:
            val = params.get(key)
            if val is not None:
                print(f"    {key:<{name_width}}  {val:>14.6f}")


def _compute_jacobians(
    distorted_cam,
    pinhole,
    positions: np.ndarray,
    eps: float = 0.5,
) -> np.ndarray:
    """Compute 2x2 Jacobians of the undistortion mapping at each keypoint.

    Uses central differences: unproject through distorted camera, project
    through pinhole camera.

    Args:
        distorted_cam: Source camera with distortion
        pinhole: Destination pinhole camera
        positions: Nx2 float64 array of (x, y) pixel coordinates
        eps: Half-pixel step for finite differences

    Returns:
        Nx2x2 float64 array of Jacobians
    """
    n = len(positions)
    if n == 0:
        return np.empty((0, 2, 2), dtype=np.float64)

    pos = positions.astype(np.float64)

    # Build offset arrays: x+eps, x-eps, y+eps, y-eps
    dx_plus = pos.copy()
    dx_plus[:, 0] += eps
    dx_minus = pos.copy()
    dx_minus[:, 0] -= eps
    dy_plus = pos.copy()
    dy_plus[:, 1] += eps
    dy_minus = pos.copy()
    dy_minus[:, 1] -= eps

    # Map through unproject+project
    def _map(pts):
        normalized = distorted_cam.unproject_batch(pts)
        return pinhole.project_batch(normalized)

    mapped_dx_plus = _map(dx_plus)
    mapped_dx_minus = _map(dx_minus)
    mapped_dy_plus = _map(dy_plus)
    mapped_dy_minus = _map(dy_minus)

    inv_2eps = 1.0 / (2.0 * eps)

    J = np.empty((n, 2, 2), dtype=np.float64)
    # J[:, row, col]: row = output dimension, col = input dimension
    J[:, 0, 0] = (mapped_dx_plus[:, 0] - mapped_dx_minus[:, 0]) * inv_2eps
    J[:, 0, 1] = (mapped_dy_plus[:, 0] - mapped_dy_minus[:, 0]) * inv_2eps
    J[:, 1, 0] = (mapped_dx_plus[:, 1] - mapped_dx_minus[:, 1]) * inv_2eps
    J[:, 1, 1] = (mapped_dy_plus[:, 1] - mapped_dy_minus[:, 1]) * inv_2eps

    return J


def _remap_tracks(
    recon: SfmrReconstruction,
    per_image_remap: list[dict[int, int]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Remap tracks after dropping features, removing orphaned 3D points.

    Returns:
        (positions_xyz, colors_rgb, reprojection_errors,
         track_image_indexes, track_feature_indexes, track_point3d_indexes,
         observation_counts, surviving_point_mask)
    """
    src_img_idxs = np.asarray(recon.track_image_indexes)
    src_feat_idxs = np.asarray(recon.track_feature_indexes)
    src_pt3d_idxs = np.asarray(recon.track_point_ids)
    num_points = recon.point_count

    # Remap feature indexes, dropping observations whose features were dropped
    new_img_idxs = []
    new_feat_idxs = []
    new_pt3d_idxs = []

    for obs_idx in range(len(src_img_idxs)):
        img_idx = int(src_img_idxs[obs_idx])
        feat_idx = int(src_feat_idxs[obs_idx])
        pt3d_idx = int(src_pt3d_idxs[obs_idx])

        remap = per_image_remap[img_idx]
        new_feat = remap.get(feat_idx)
        if new_feat is not None:
            new_img_idxs.append(img_idx)
            new_feat_idxs.append(new_feat)
            new_pt3d_idxs.append(pt3d_idx)

    new_img_idxs = np.array(new_img_idxs, dtype=np.uint32)
    new_feat_idxs = np.array(new_feat_idxs, dtype=np.uint32)
    new_pt3d_idxs = np.array(new_pt3d_idxs, dtype=np.uint32)

    # Count observations per point, drop points with 0 observations
    obs_per_point = np.bincount(new_pt3d_idxs, minlength=num_points).astype(np.uint32)
    surviving_mask = obs_per_point > 0
    surviving_indices = np.where(surviving_mask)[0]

    # Renumber surviving points
    old_to_new = np.full(num_points, -1, dtype=np.int64)
    old_to_new[surviving_indices] = np.arange(len(surviving_indices), dtype=np.int64)

    final_pt3d_idxs = old_to_new[new_pt3d_idxs].astype(np.uint32)

    # Recompute observation counts for surviving points
    new_obs_counts = obs_per_point[surviving_mask]

    # Filter point data
    positions_xyz = np.asarray(recon.positions)[surviving_mask]
    colors_rgb = np.asarray(recon.colors)[surviving_mask]
    reprojection_errors = np.asarray(recon.errors)[surviving_mask]

    # Sort tracks by (point3d_index, image_index)
    sort_order = np.lexsort((new_img_idxs, final_pt3d_idxs))
    new_img_idxs = new_img_idxs[sort_order]
    new_feat_idxs = new_feat_idxs[sort_order]
    final_pt3d_idxs = final_pt3d_idxs[sort_order]

    return (
        positions_xyz,
        colors_rgb,
        reprojection_errors,
        new_img_idxs,
        new_feat_idxs,
        final_pt3d_idxs,
        new_obs_counts,
        surviving_mask,
    )


def undistort_reconstruction_images(
    recon: SfmrReconstruction,
    output_dir: Path,
    fit: str = "inside",
    resampling_filter: str = "aniso",
    progress_callback=None,
    source_workspace_config: dict | None = None,
    source_sfmr_path: str | None = None,
) -> tuple[int, str, str | None]:
    """
    Undistort all images in a reconstruction, producing a full workspace.

    For each image: undistorts the image, transforms .sift features, and
    builds a new .sfmr reconstruction with pinhole cameras.

    Args:
        recon: The SfmrReconstruction containing camera parameters and image metadata
        output_dir: Directory where the new workspace will be created
        fit: "inside" (no black borders) or "outside" (no cropping)
        resampling_filter: "aniso" (anisotropic, higher quality) or "bilinear"
        progress_callback: Optional callback(current, total, image_name) for progress
        source_workspace_config: Workspace config from the source workspace
        source_sfmr_path: Relative path of the source .sfmr within its workspace

    Returns:
        Tuple of (number_of_images_processed, output_directory_path, sfmr_path_or_None)
    """
    from ._colmap_io import _build_sfmr_data_dict
    from ._sift_file import (
        SiftReader,
        get_feature_tool_xxh128,
        write_sift,
        xxh128_of_file,
    )
    from ._workspace import find_workspace_for_path, load_workspace_config

    workspace_dir = Path(recon.workspace_dir)
    image_names = recon.image_names
    cameras_meta = recon.cameras
    camera_indexes = recon.camera_indexes

    # Read source workspace feature config
    if source_workspace_config is None:
        ws_dir = find_workspace_for_path(workspace_dir)
        if ws_dir is not None:
            source_workspace_config = load_workspace_config(ws_dir)
    if source_workspace_config is None:
        source_feature_tool = "colmap"
        source_feature_options = {}
    else:
        source_feature_tool = source_workspace_config.get("feature_tool", "colmap")
        source_feature_options = source_workspace_config.get("feature_options", {})

    image_count = len(image_names)
    print(
        f"Undistorting {image_count} images (fit={fit}, filter={resampling_filter})..."
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature tool config for the undistorted .sift files
    undistort_feature_tool = "sfmtool-undistort"
    undistort_feature_options = {
        "source_feature_tool": source_feature_tool,
        "source_feature_options": source_feature_options,
        "fit": fit,
        "filter": resampling_filter,
    }
    undistort_feature_type = f"sift-{undistort_feature_tool}"
    undistort_feature_tool_hash = get_feature_tool_xxh128(
        undistort_feature_tool, undistort_feature_type, undistort_feature_options
    )
    feature_prefix_dir = (
        f"features/{undistort_feature_type}-{undistort_feature_tool_hash}"
    )

    feature_tool_metadata = {
        "feature_tool": undistort_feature_tool,
        "feature_type": undistort_feature_type,
        "feature_options": undistort_feature_options,
    }

    # Build pinhole cameras and warp maps per unique camera (not per image).
    pinhole_cameras = {}
    warp_maps = {}

    # Per-image outputs for .sfmr assembly
    per_image_remap = []  # list of {old_feat_idx: new_feat_idx} per image
    thumbnails = []
    feature_tool_hashes = []
    sift_content_hashes = []

    # Track feature drop stats
    total_features_original = 0
    total_features_kept = 0
    worst_drop_image = None
    worst_drop_rate = 0.0

    # Process each image
    for i, image_name in enumerate(image_names):
        if progress_callback:
            progress_callback(i, image_count, image_name)

        # Construct path to original image file
        image_path = workspace_dir / image_name

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Get camera for this image
        cam_idx = camera_indexes[i]
        cam_meta = cameras_meta[cam_idx]

        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        height, width = image.shape[:2]

        # Build (or reuse) the pinhole camera and warp map for this camera index.
        if cam_idx not in pinhole_cameras:
            if fit == "outside":
                pinhole = cam_meta.best_fit_outside_pinhole(width, height)
            else:
                pinhole = cam_meta.best_fit_inside_pinhole(width, height)
            pinhole_cameras[cam_idx] = pinhole
            warp_maps[cam_idx] = WarpMap.from_cameras(src=cam_meta, dst=pinhole)

            print(f"\nCamera {cam_idx}:")
            _print_camera_params("Original", cam_meta)
            _print_camera_params("Undistorted", pinhole)

        pinhole = pinhole_cameras[cam_idx]
        warp = warp_maps[cam_idx]

        # Step 1: Warp the image
        if resampling_filter == "aniso":
            undistorted = warp.remap_aniso(image)
        else:
            undistorted = warp.remap_bilinear(image)

        # Step 2: Save undistorted image
        output_image_path = output_dir / image_name
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_image_path), undistorted)
        if not success:
            raise RuntimeError(f"Failed to save undistorted image: {output_image_path}")

        # Step 3: Generate thumbnail from undistorted image (BGR -> RGB, 128x128)
        thumb_bgr = cv2.resize(undistorted, (128, 128), interpolation=cv2.INTER_AREA)
        thumbnail_rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
        thumbnails.append(thumbnail_rgb)

        # Step 4: Read source .sift file and transform features
        source_sift_reader = SiftReader.for_image(
            image_path,
            feature_tool=source_feature_tool,
            feature_options=source_feature_options if source_feature_options else None,
        )
        with source_sift_reader as reader:
            src_positions = reader.read_positions()  # Nx2 float32
            src_affine_shapes = reader.read_affine_shapes()  # Nx2x2 float32
            src_descriptors = reader.read_descriptors()  # Nx128 uint8

        # Transform positions: unproject through distorted, project through pinhole
        n_features = len(src_positions)
        total_features_original += n_features

        if n_features > 0:
            pos_f64 = src_positions.astype(np.float64)
            normalized = cam_meta.unproject_batch(pos_f64)
            new_positions_f64 = pinhole.project_batch(normalized)

            # Filter: keep only features within pinhole bounds
            ph_w = float(pinhole.width)
            ph_h = float(pinhole.height)
            keep = (
                (new_positions_f64[:, 0] >= 0)
                & (new_positions_f64[:, 0] < ph_w)
                & (new_positions_f64[:, 1] >= 0)
                & (new_positions_f64[:, 1] < ph_h)
            )

            keep_indices = np.where(keep)[0]
            remap = {int(old): new for new, old in enumerate(keep_indices)}

            new_positions = new_positions_f64[keep].astype(np.float32)
            new_descriptors = src_descriptors[keep]

            # Transform affine shapes via Jacobian
            if len(keep_indices) > 0:
                J = _compute_jacobians(
                    cam_meta, pinhole, src_positions[keep].astype(np.float64)
                )
                # A' = J @ A
                new_affine = np.einsum(
                    "nij,njk->nik", J, src_affine_shapes[keep].astype(np.float64)
                ).astype(np.float32)
            else:
                new_affine = np.empty((0, 2, 2), dtype=np.float32)
        else:
            remap = {}
            new_positions = np.empty((0, 2), dtype=np.float32)
            new_affine = np.empty((0, 2, 2), dtype=np.float32)
            new_descriptors = np.empty((0, 128), dtype=np.uint8)

        per_image_remap.append(remap)
        n_kept = len(new_positions)
        total_features_kept += n_kept

        # Track worst drop rate
        if n_features > 0:
            drop_rate = 1.0 - n_kept / n_features
            if drop_rate > worst_drop_rate:
                worst_drop_rate = drop_rate
                worst_drop_image = image_name

        # Step 5: Write .sift file
        image_file_hash = xxh128_of_file(output_image_path)
        image_file_size = output_image_path.stat().st_size

        sift_metadata = {
            "version": 1,
            "image_name": Path(image_name).name,
            "image_file_xxh128": image_file_hash,
            "image_file_size": image_file_size,
            "image_width": int(pinhole.width),
            "image_height": int(pinhole.height),
            "feature_count": n_kept,
        }

        sift_dir = output_image_path.parent / feature_prefix_dir
        sift_dir.mkdir(parents=True, exist_ok=True)
        sift_path = sift_dir / (Path(image_name).name + ".sift")

        write_sift(
            sift_path,
            feature_tool_metadata,
            sift_metadata,
            new_positions,
            new_affine,
            new_descriptors,
            thumbnail_rgb,
        )

        # Step 6: Read back hashes from written .sift
        with SiftReader(sift_path) as reader:
            ft_hash = reader.content_hash["feature_tool_xxh128"]
            sc_hash = reader.content_hash["content_xxh128"]
            feature_tool_hashes.append(bytes.fromhex(ft_hash))
            sift_content_hashes.append(bytes.fromhex(sc_hash))

        if i < 3 or (i + 1) % 10 == 0:
            print(
                f"  [{i + 1}/{image_count}] {image_name}: "
                f"{width}x{height} -> {pinhole.width}x{pinhole.height}"
                f" ({n_kept}/{n_features} features)"
            )

    if progress_callback:
        progress_callback(image_count, image_count, "")

    # Print feature drop summary
    total_dropped = total_features_original - total_features_kept
    if total_features_original > 0:
        drop_pct = 100.0 * total_dropped / total_features_original
        print(
            f"\nFeature summary: {total_features_kept}/{total_features_original} kept "
            f"({total_dropped} dropped, {drop_pct:.1f}%)"
        )
        if worst_drop_image is not None and worst_drop_rate > 0:
            print(
                f"  Highest drop rate: {worst_drop_image} "
                f"({worst_drop_rate * 100:.1f}%)"
            )

    # === Track remapping ===
    (
        positions_xyz,
        colors_rgb,
        reprojection_errors,
        track_image_indexes,
        track_feature_indexes,
        track_point3d_indexes,
        observation_counts,
        _surviving_mask,
    ) = _remap_tracks(recon, per_image_remap)

    # === Build contiguous pinhole camera list ===
    sorted_cam_idxs = sorted(pinhole_cameras)
    cam_idx_to_new_idx = {idx: j for j, idx in enumerate(sorted_cam_idxs)}
    cameras_list = [pinhole_cameras[idx] for idx in sorted_cam_idxs]
    new_camera_indexes = np.array(
        [cam_idx_to_new_idx[ci] for ci in camera_indexes], dtype=np.uint32
    )

    # === Rig frame data ===
    rig_frame_data = None
    if recon.rig_frame_data is not None:
        rig_frame_data = recon.rig_frame_data
        # Update sensor_camera_indexes to point to new contiguous camera indices
        old_sensor_cam_idxs = rig_frame_data["sensor_camera_indexes"]
        new_sensor_cam_idxs = np.array(
            [cam_idx_to_new_idx[int(ci)] for ci in old_sensor_cam_idxs],
            dtype=np.uint32,
        )
        rig_frame_data = dict(rig_frame_data)
        rig_frame_data["sensor_camera_indexes"] = new_sensor_cam_idxs

    # === .sfmr metadata ===
    try:
        from importlib.metadata import version as get_version

        tool_version = get_version("sfmtool")
    except Exception:
        tool_version = "unknown"

    workspace_contents = {
        "feature_tool": undistort_feature_tool,
        "feature_type": undistort_feature_type,
        "feature_options": undistort_feature_options,
        "feature_prefix_dir": feature_prefix_dir,
    }

    metadata = {
        "version": 1,
        "operation": "undistort",
        "tool": "sfmtool",
        "tool_version": tool_version,
        "workspace": {
            "absolute_path": str(output_dir.resolve()),
            "relative_path": ".",
            "contents": workspace_contents,
        },
        "timestamp": datetime.now().astimezone().isoformat(),
        "image_count": image_count,
        "points3d_count": len(positions_xyz),
        "observation_count": len(track_image_indexes),
        "camera_count": len(cameras_list),
        "tool_options": {
            "fit": fit,
            "filter": resampling_filter,
            "source_sfmr": source_sfmr_path or "unknown",
        },
    }

    # === Assemble and save .sfmr ===
    sfmr_dict = _build_sfmr_data_dict(
        cameras=cameras_list,
        image_names=image_names,
        camera_indexes=new_camera_indexes,
        quaternions_wxyz=np.asarray(recon.quaternions_wxyz),
        translations_xyz=np.asarray(recon.translations),
        positions_xyz=positions_xyz,
        colors_rgb=colors_rgb,
        reprojection_errors=reprojection_errors,
        track_image_indexes=track_image_indexes,
        track_feature_indexes=track_feature_indexes,
        track_point3d_indexes=track_point3d_indexes,
        observation_counts=observation_counts,
        feature_tool_hashes=feature_tool_hashes,
        sift_content_hashes=sift_content_hashes,
        thumbnails=thumbnails,
        metadata=metadata,
        rig_frame_data=rig_frame_data,
    )

    sfmr_out_dir = output_dir / "sfmr"
    sfmr_out_dir.mkdir(parents=True, exist_ok=True)
    sfmr_path = sfmr_out_dir / "undistorted.sfmr"

    new_recon = SfmrReconstruction.from_data(output_dir.resolve(), sfmr_dict)
    new_recon.save(sfmr_path)

    # === Write workspace config ===
    workspace_config = {
        "version": 1,
        "feature_tool": undistort_feature_tool,
        "feature_type": undistort_feature_type,
        "feature_options": undistort_feature_options,
        "feature_prefix_dir": feature_prefix_dir,
    }

    config_path = output_dir / ".sfm-workspace.json"
    with open(config_path, "w") as f:
        json.dump(workspace_config, f, indent=2)

    print(f"\nUndistorted {image_count} images")
    print(f"Output directory: {output_dir}")
    print(f"Reconstruction: {sfmr_path}")
    print(
        f"  {len(cameras_list)} cameras, {len(positions_xyz)} 3D points, "
        f"{len(track_image_indexes)} observations"
    )

    return image_count, str(output_dir), str(sfmr_path)
