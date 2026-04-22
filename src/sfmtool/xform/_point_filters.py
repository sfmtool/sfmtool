# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Filters that remove 3D points based on per-point criteria."""

from pathlib import Path

import numpy as np

from .._sfmtool import SfmrReconstruction, read_sift_partial


class RemoveShortTracksFilter:
    """Remove 3D points with track length <= size.

    `max_size=1` is the natural minimum — it keeps only points triangulated
    from at least two views, which is what's required for a valid 3D point.
    Operations like `--include-range` or other image filters commonly strand
    a point with a single surviving observation, so removing length-1
    tracks is a routine clean-up step.
    """

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError(f"Track size must be >= 1, got {max_size}")
        self.max_size = max_size

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        points_to_keep_mask = recon.observation_counts > self.max_size

        if not np.any(points_to_keep_mask):
            raise ValueError(
                f"No points remain after removing tracks with length <= {self.max_size}"
            )

        removed_count = recon.point_count - int(np.sum(points_to_keep_mask))
        print(
            f"  Removed {removed_count} points with track length <= {self.max_size} ({recon.point_count - removed_count} remaining)"
        )

        return recon.filter_points_by_mask(points_to_keep_mask)

    def description(self) -> str:
        return f"Remove tracks with length <= {self.max_size}"


class RemoveNarrowTracksFilter:
    """Remove 3D points with viewing angle span less than threshold."""

    def __init__(self, min_angle_rad: float):
        if min_angle_rad <= 0:
            raise ValueError(f"Minimum angle must be positive, got {min_angle_rad}")
        self.min_angle_rad = min_angle_rad

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        print(
            f"  Computing viewing angles for {recon.point_count} points across {recon.image_count} images..."
        )

        from .._sfmtool import compute_narrow_track_mask

        points_to_keep_mask = compute_narrow_track_mask(
            recon.quaternions_wxyz,
            recon.translations,
            recon.positions,
            recon.track_point_ids,
            recon.track_image_indexes,
            self.min_angle_rad,
        )

        if not np.any(points_to_keep_mask):
            raise ValueError(
                f"No points remain after removing tracks with viewing angle < {np.degrees(self.min_angle_rad):.2f}\u00b0"
            )

        removed_count = recon.point_count - int(np.sum(points_to_keep_mask))
        print(
            f"  Removed {removed_count} points with viewing angle < {np.degrees(self.min_angle_rad):.2f}\u00b0 ({recon.point_count - removed_count} remaining)"
        )

        return recon.filter_points_by_mask(points_to_keep_mask)

    def description(self) -> str:
        return f"Remove tracks with viewing angle < {np.degrees(self.min_angle_rad):.2f}\u00b0"


class RemoveIsolatedPointsFilter:
    """Remove 3D points whose nearest neighbor distance exceeds a threshold."""

    def __init__(self, factor: float, value_spec: str):
        if factor <= 0:
            raise ValueError(f"Factor must be positive, got {factor}")
        self.factor = factor
        self.value_spec = value_spec

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        if recon.point_count < 2:
            raise ValueError("Need at least 2 points to compute nearest neighbors")

        print(
            f"  Computing nearest neighbor distances for {recon.point_count} points..."
        )

        from .._sfmtool import KdTree3d

        nn_distances = KdTree3d(recon.positions).nearest_neighbor_distances()

        if self.value_spec == "median":
            reference_value = np.median(nn_distances)
            ref_desc = "median"
        elif self.value_spec.endswith("percent") or self.value_spec.endswith(
            "percentile"
        ):
            if self.value_spec.endswith("percentile"):
                percentile_str = self.value_spec[: -len("percentile")]
            else:
                percentile_str = self.value_spec[: -len("percent")]

            try:
                percentile = float(percentile_str)
            except ValueError:
                raise ValueError(
                    f"Invalid percentile specification: '{self.value_spec}'"
                )

            if not 0 <= percentile <= 100:
                raise ValueError(f"Percentile must be in [0, 100], got {percentile}")

            reference_value = np.percentile(nn_distances, percentile)
            ref_desc = f"{percentile}th percentile"
        else:
            raise ValueError(
                f"Invalid value_spec: '{self.value_spec}'. "
                f"Expected 'median', '<N>percent', or '<N>percentile'"
            )

        threshold = self.factor * reference_value

        print(f"    Reference value ({ref_desc}): {reference_value:.6f}")
        print(f"    Threshold ({self.factor}\u00d7 reference): {threshold:.6f}")

        points_to_keep_mask = nn_distances <= threshold

        if not np.any(points_to_keep_mask):
            raise ValueError(
                f"No points remain after removing isolated points (threshold: {threshold:.6f})"
            )

        removed_count = recon.point_count - int(np.sum(points_to_keep_mask))
        print(
            f"  Removed {removed_count} isolated points ({recon.point_count - removed_count} remaining)"
        )

        return recon.filter_points_by_mask(points_to_keep_mask)

    def description(self) -> str:
        return f"Remove isolated points (NN > {self.factor}\u00d7 {self.value_spec})"


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
