# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Remove large features filter."""

from pathlib import Path

import numpy as np

from .._sfmtool import SfmrReconstruction, read_sift_partial


class RemoveLargeFeaturesFilter:
    """Remove 3D points where the largest SIFT feature in the track exceeds a size threshold.

    Feature size is the average radius in pixels, computed as the mean of
    the two column norms of the affine shape matrix.
    """

    def __init__(self, max_size: float):
        if max_size <= 0:
            raise ValueError(f"Feature size threshold must be positive, got {max_size}")
        self.max_size = max_size

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        track_image_indexes = np.asarray(recon.track_image_indexes)
        track_feature_indexes = np.asarray(recon.track_feature_indexes)
        track_point_ids = np.asarray(recon.track_point_ids)
        point_count = recon.point_count

        # Load feature sizes per image (lazily, once per unique image)
        image_feature_sizes: dict[int, np.ndarray] = {}
        unique_images = np.unique(track_image_indexes)

        metadata = recon.metadata()
        feature_prefix_dir = metadata["workspace"]["contents"]["feature_prefix_dir"]
        workspace_dir = Path(recon.workspace_dir)
        image_names = recon.image_names

        for img_idx in unique_images:
            img_idx = int(img_idx)
            image_name = image_names[img_idx]
            image_rel = Path(image_name)
            image_parent = image_rel.parent
            sift_filename = f"{image_rel.name}.sift"
            sift_path = (
                workspace_dir / image_parent / feature_prefix_dir / sift_filename
            )

            if not sift_path.exists():
                raise FileNotFoundError(
                    f"SIFT file not found: {sift_path} (image {img_idx}: {image_name})"
                )

            # Read only enough features (up to the max feature index needed)
            mask = track_image_indexes == img_idx
            max_feat_idx = int(track_feature_indexes[mask].max()) + 1
            sift_data = read_sift_partial(sift_path, max_feat_idx)
            affine_shapes = sift_data["affine_shapes"]  # (N, 2, 2)

            # Compute feature sizes: average of column norms
            col0_norms = np.linalg.norm(affine_shapes[:, :, 0], axis=1)
            col1_norms = np.linalg.norm(affine_shapes[:, :, 1], axis=1)
            image_feature_sizes[img_idx] = 0.5 * (col0_norms + col1_norms)

        # Build a flat array of per-observation feature sizes, then scatter-max
        obs_sizes = np.zeros(len(track_image_indexes), dtype=np.float32)
        for img_idx, sizes in image_feature_sizes.items():
            mask = track_image_indexes == img_idx
            feat_idxs = track_feature_indexes[mask]
            valid = feat_idxs < len(sizes)
            obs_positions = np.flatnonzero(mask)
            obs_sizes[obs_positions[valid]] = sizes[feat_idxs[valid]]

        max_feature_size_per_point = np.zeros(point_count, dtype=np.float32)
        np.maximum.at(max_feature_size_per_point, track_point_ids, obs_sizes)

        points_to_keep_mask = max_feature_size_per_point <= self.max_size

        if not np.any(points_to_keep_mask):
            raise ValueError(
                f"No points remain after removing tracks with max feature size > {self.max_size}"
            )

        removed_count = point_count - int(np.sum(points_to_keep_mask))
        print(
            f"  Removed {removed_count} points with max feature size > {self.max_size:.1f}px "
            f"({point_count - removed_count} remaining)"
        )

        return recon.filter_points_by_mask(points_to_keep_mask)

    def description(self) -> str:
        return f"Remove tracks with max feature size > {self.max_size:.1f}px"
