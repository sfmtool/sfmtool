# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter images by file number range or filename glob."""

import fnmatch

import numpy as np
from openjd.model import IntRangeExpr

from .._filenames import number_from_filename
from .._sfmtool import SfmrReconstruction


class IncludeRangeFilter:
    """Filter to keep only images whose file number is in the specified range."""

    def __init__(self, range_expr: IntRangeExpr):
        self.range_expr = range_expr
        self.range_numbers = set(range_expr)

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        return _filter_images_by_range(
            recon, self.range_numbers, include=True, range_expr_str=str(self.range_expr)
        )

    def description(self) -> str:
        return f"Include images in range {self.range_expr}"


class ExcludeRangeFilter:
    """Filter to exclude images whose file number is in the specified range."""

    def __init__(self, range_expr: IntRangeExpr):
        self.range_expr = range_expr
        self.range_numbers = set(range_expr)

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        return _filter_images_by_range(
            recon,
            self.range_numbers,
            include=False,
            range_expr_str=str(self.range_expr),
        )

    def description(self) -> str:
        return f"Exclude images in range {self.range_expr}"


class IncludeGlobFilter:
    """Filter to keep only images whose name matches a glob pattern."""

    def __init__(self, pattern: str):
        self.pattern = pattern

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        images_to_keep = [
            i
            for i, name in enumerate(recon.image_names)
            if fnmatch.fnmatch(name, self.pattern)
        ]
        if not images_to_keep:
            raise ValueError(
                f"No images match include glob pattern '{self.pattern}'. "
                f"Example image names: {recon.image_names[:5]}"
            )
        images_to_keep = np.array(images_to_keep, dtype=np.uint32)
        kept = len(images_to_keep)
        total = len(recon.image_names)
        print(
            f"  Applied include glob '{self.pattern}': keeping {kept} of {total} images"
        )
        return _filter_images(recon, images_to_keep)

    def description(self) -> str:
        return f"Include images matching '{self.pattern}'"


class ExcludeGlobFilter:
    """Filter to exclude images whose name matches a glob pattern."""

    def __init__(self, pattern: str):
        self.pattern = pattern

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        images_to_keep = [
            i
            for i, name in enumerate(recon.image_names)
            if not fnmatch.fnmatch(name, self.pattern)
        ]
        if not images_to_keep:
            raise ValueError(
                f"No images remain after excluding glob pattern '{self.pattern}'. "
                f"All {len(recon.image_names)} images matched."
            )
        images_to_keep = np.array(images_to_keep, dtype=np.uint32)
        kept = len(images_to_keep)
        total = len(recon.image_names)
        print(
            f"  Applied exclude glob '{self.pattern}': keeping {kept} of {total} images"
        )
        return _filter_images(recon, images_to_keep)

    def description(self) -> str:
        return f"Exclude images matching '{self.pattern}'"


def _filter_rig_frame_data(
    rig_frame_data: dict | None,
    images_to_keep: np.ndarray,
) -> dict | None:
    """Update rig_frame_data after image filtering."""
    if rig_frame_data is None:
        return None

    new_image_sensor_indexes = rig_frame_data["image_sensor_indexes"][images_to_keep]
    new_image_frame_indexes = rig_frame_data["image_frame_indexes"][images_to_keep]

    remaining_frames = np.unique(new_image_frame_indexes)
    old_frame_count = rig_frame_data["frames_metadata"]["frame_count"]
    new_frame_count = len(remaining_frames)

    if new_frame_count < old_frame_count:
        frame_id_mapping = np.full(old_frame_count, -1, dtype=np.int32)
        frame_id_mapping[remaining_frames] = np.arange(new_frame_count, dtype=np.int32)
        new_image_frame_indexes = frame_id_mapping[new_image_frame_indexes].astype(
            np.uint32
        )
        new_rig_indexes = rig_frame_data["rig_indexes"][remaining_frames]
    else:
        new_rig_indexes = rig_frame_data["rig_indexes"]

    return {
        "rigs_metadata": rig_frame_data["rigs_metadata"],
        "sensor_camera_indexes": rig_frame_data["sensor_camera_indexes"],
        "sensor_quaternions_wxyz": rig_frame_data["sensor_quaternions_wxyz"],
        "sensor_translations_xyz": rig_frame_data["sensor_translations_xyz"],
        "frames_metadata": {"frame_count": new_frame_count},
        "rig_indexes": new_rig_indexes,
        "image_sensor_indexes": new_image_sensor_indexes,
        "image_frame_indexes": new_image_frame_indexes,
    }


def _filter_images_by_range(
    recon: SfmrReconstruction,
    range_numbers: set[int],
    include: bool,
    range_expr_str: str,
) -> SfmrReconstruction:
    """Filter images by file number range."""
    images_to_keep = []
    for i, image_name in enumerate(recon.image_names):
        file_number = number_from_filename(image_name)
        if include:
            if file_number is not None and file_number in range_numbers:
                images_to_keep.append(i)
        else:
            if file_number is None or file_number not in range_numbers:
                images_to_keep.append(i)

    if not images_to_keep:
        mode = "include" if include else "exclude"
        available_numbers = sorted(
            {
                number_from_filename(name)
                for name in recon.image_names
                if number_from_filename(name) is not None
            }
        )
        raise ValueError(
            f"No images remain after applying {mode} range filter '{range_expr_str}'. "
            f"Available file numbers: {available_numbers}"
        )

    images_to_keep = np.array(images_to_keep, dtype=np.uint32)

    mode = "include" if include else "exclude"
    print(
        f"  Applied {mode} range filter '{range_expr_str}': "
        f"keeping {len(images_to_keep)} of {len(recon.image_names)} images"
    )

    return _filter_images(recon, images_to_keep)


def _filter_images(
    recon: SfmrReconstruction,
    images_to_keep: np.ndarray,
) -> SfmrReconstruction:
    """Filter a reconstruction to keep only the specified images."""
    # Filter image data
    new_image_names = [recon.image_names[i] for i in images_to_keep]
    new_camera_indexes = recon.camera_indexes[images_to_keep]
    new_quaternions_wxyz = recon.quaternions_wxyz[images_to_keep]
    new_translations = recon.translations[images_to_keep]
    new_feature_tool_hashes = [recon.feature_tool_hashes[i] for i in images_to_keep]
    new_sift_content_hashes = [recon.sift_content_hashes[i] for i in images_to_keep]
    new_thumbnails_y_x_rgb = recon.thumbnails_y_x_rgb[images_to_keep]

    # Filter tracks
    images_to_keep_set = set(images_to_keep)
    track_mask = np.array(
        [idx in images_to_keep_set for idx in recon.track_image_indexes], dtype=bool
    )
    filtered_track_image_indexes = recon.track_image_indexes[track_mask]
    filtered_track_feature_indexes = recon.track_feature_indexes[track_mask]
    filtered_track_point_ids = recon.track_point_ids[track_mask]

    # Remap image indexes
    image_id_mapping = np.full(len(recon.image_names), -1, dtype=np.int32)
    image_id_mapping[images_to_keep] = np.arange(len(images_to_keep), dtype=np.int32)
    new_track_image_indexes = image_id_mapping[filtered_track_image_indexes].astype(
        np.uint32
    )

    # Recompute observation counts and filter points
    points_to_keep, new_observation_counts = np.unique(
        filtered_track_point_ids, return_counts=True
    )

    print(
        f"  Keeping {len(points_to_keep)} of {len(recon.positions)} points "
        f"(removed {len(recon.positions) - len(points_to_keep)} points with no remaining observations)"
    )

    new_positions = recon.positions[points_to_keep]
    new_colors = recon.colors[points_to_keep]
    new_errors = recon.errors[points_to_keep]

    # Remap point IDs
    point_id_mapping = np.full(len(recon.positions), -1, dtype=np.int32)
    point_id_mapping[points_to_keep] = np.arange(len(points_to_keep), dtype=np.int32)
    new_track_point_ids = point_id_mapping[filtered_track_point_ids].astype(np.uint32)

    new_observation_counts = new_observation_counts.astype(np.uint32)

    new_rig_frame_data = _filter_rig_frame_data(recon.rig_frame_data, images_to_keep)

    return recon.clone_with_changes(
        image_names=new_image_names,
        camera_indexes=new_camera_indexes,
        quaternions_wxyz=new_quaternions_wxyz,
        translations=new_translations,
        positions=new_positions,
        colors=new_colors,
        errors=new_errors,
        track_image_indexes=new_track_image_indexes,
        track_feature_indexes=filtered_track_feature_indexes,
        track_point_ids=new_track_point_ids,
        observation_counts=new_observation_counts,
        feature_tool_hashes=new_feature_tool_hashes,
        sift_content_hashes=new_sift_content_hashes,
        thumbnails_y_x_rgb=new_thumbnails_y_x_rgb,
        rig_frame_data=new_rig_frame_data,
    )
