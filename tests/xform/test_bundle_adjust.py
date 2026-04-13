# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for BundleAdjustTransform."""

import numpy as np

from sfmtool._sfmtool import RotQuaternion
from sfmtool.xform import (
    BundleAdjustTransform,
    RemoveShortTracksFilter,
)

from .conftest import apply_transforms_to_file, load_reconstruction_data


def test_bundle_adjust_transform(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that bundle adjustment works."""
    output_path = tmp_path / "bundle_adjusted.sfmr"

    transforms = [BundleAdjustTransform()]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    adjusted = load_reconstruction_data(output_path)

    assert adjusted["point_count"] == original["point_count"]
    assert len(adjusted["positions"]) > 0
    assert len(adjusted["quaternions_wxyz"]) > 0


def test_bundle_adjust_with_filter(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test bundle adjustment combined with filtering."""
    output_path = tmp_path / "filtered_and_adjusted.sfmr"

    transforms = [
        RemoveShortTracksFilter(2),
        BundleAdjustTransform(),
    ]

    result = apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    assert result == output_path
    assert output_path.exists()

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    result_data = load_reconstruction_data(output_path)

    assert result_data["point_count"] < original["point_count"]
    assert np.all(result_data["observation_counts"] > 2)


def test_bundle_adjust_preserves_image_count(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that BA preserves the number of images."""
    output_path = tmp_path / "ba_images.sfmr"

    transforms = [BundleAdjustTransform()]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    adjusted = load_reconstruction_data(output_path)

    assert adjusted["image_count"] == original["image_count"]


def test_bundle_adjust_preserves_observation_count(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that BA preserves the observation count."""
    output_path = tmp_path / "ba_observations.sfmr"

    transforms = [BundleAdjustTransform()]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    adjusted = load_reconstruction_data(output_path)

    assert adjusted["observation_count"] == original["observation_count"]


def test_bundle_adjust_quaternion_consistency(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that BA preserves quaternion ordering (xyzw->wxyz conversion).

    Regression test: pycolmap returns quaternions in xyzw order, but our
    storage format uses wxyz. If the conversion is wrong, camera centers
    computed from the BA result will be wildly different from the original.
    """
    output_path = tmp_path / "ba_quat_check.sfmr"
    transforms = [BundleAdjustTransform()]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    original = load_reconstruction_data(sfmrfile_reconstruction_with_17_images)
    adjusted = load_reconstruction_data(output_path)

    # Compute camera centers from quaternions and translations
    orig_centers = []
    for i in range(original["image_count"]):
        q = RotQuaternion.from_wxyz_array(original["quaternions_wxyz"][i])
        r = q.to_rotation_matrix()
        t = original["translations"][i]
        orig_centers.append(-r.T @ t)

    adj_centers = []
    for i in range(adjusted["image_count"]):
        q = RotQuaternion.from_wxyz_array(adjusted["quaternions_wxyz"][i])
        r = q.to_rotation_matrix()
        t = adjusted["translations"][i]
        adj_centers.append(-r.T @ t)

    orig_centers = np.array(orig_centers)
    adj_centers = np.array(adj_centers)

    scene_extent = np.ptp(orig_centers, axis=0).max()
    center_diffs = np.linalg.norm(adj_centers - orig_centers, axis=1)
    max_drift = center_diffs.max()

    assert max_drift < 0.1 * scene_extent, (
        f"BA moved cameras too far: max drift {max_drift:.4f} vs "
        f"scene extent {scene_extent:.4f} (ratio {max_drift / scene_extent:.2f}). "
        f"This likely indicates a quaternion ordering bug (xyzw vs wxyz)."
    )


def test_bundle_adjust_no_rig_data(sfmrfile_reconstruction_with_17_images, tmp_path):
    """Test that BA on a non-rig reconstruction doesn't add spurious rig data."""
    from sfmtool._sfmtool import SfmrReconstruction

    output_path = tmp_path / "ba_no_rig.sfmr"
    transforms = [BundleAdjustTransform()]

    apply_transforms_to_file(
        sfmrfile_reconstruction_with_17_images, output_path, transforms
    )

    adjusted = SfmrReconstruction.load(output_path)
    assert adjusted.rig_frame_data is None


def test_bundle_adjust_preserves_rig_data(
    sfmrfile_reconstruction_with_17_images, tmp_path
):
    """Test that BA preserves rig_frame_data through the round-trip.

    Regression test: before the fix, _reconstruction_to_data did not call
    _extract_rig_frame_data, so rig data was silently dropped.
    """
    from sfmtool._sfmtool import SfmrReconstruction

    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    image_count = recon.image_count

    # Create a 2-sensor rig by duplicating the camera (each sensor needs a
    # distinct camera ID in COLMAP). Even-indexed images → sensor 0 (camera 0),
    # odd-indexed images → sensor 1 (camera 1). Use 16 images (8 frames of 2).
    use_count = (image_count // 2) * 2  # 16
    num_frames = use_count // 2  # 8

    cam0 = recon.cameras[0]
    cameras = [cam0, cam0]  # Two cameras with identical intrinsics

    camera_indexes = np.array([i % 2 for i in range(image_count)], dtype=np.uint32)

    rig_frame_data = {
        "rigs_metadata": {
            "rig_count": 1,
            "sensor_count": 2,
            "rigs": [
                {
                    "name": "rig0",
                    "sensor_count": 2,
                    "sensor_offset": 0,
                    "ref_sensor_name": "sensor0",
                    "sensor_names": ["sensor0", "sensor1"],
                }
            ],
        },
        "sensor_camera_indexes": np.array([0, 1], dtype=np.uint32),
        "sensor_quaternions_wxyz": np.array(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64
        ),
        "sensor_translations_xyz": np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64
        ),
        "frames_metadata": {"frame_count": num_frames},
        "rig_indexes": np.zeros(num_frames, dtype=np.uint32),
        # Even images → sensor 0, odd images → sensor 1
        "image_sensor_indexes": np.array(
            [i % 2 for i in range(image_count)], dtype=np.uint32
        ),
        # Pairs of images share a frame: images 0,1 → frame 0; 2,3 → frame 1; ...
        # Last image (16) gets frame 8 which doesn't exist for 16-image pairs,
        # so assign it to the last valid frame.
        "image_frame_indexes": np.array(
            [min(i // 2, num_frames - 1) for i in range(image_count)],
            dtype=np.uint32,
        ),
    }

    rig_recon = recon.clone_with_changes(
        cameras=cameras,
        camera_indexes=camera_indexes,
        rig_frame_data=rig_frame_data,
    )
    rig_path = tmp_path / "rig_input.sfmr"
    rig_recon.save(rig_path, operation="test_rig_inject")

    # Now run BA on the rig reconstruction
    output_path = tmp_path / "ba_rig_output.sfmr"
    transforms = [BundleAdjustTransform()]

    apply_transforms_to_file(rig_path, output_path, transforms)

    adjusted = SfmrReconstruction.load(output_path)
    assert adjusted.rig_frame_data is not None, (
        "Rig frame data was lost during bundle adjustment"
    )

    rfd = adjusted.rig_frame_data
    assert rfd["rigs_metadata"]["rig_count"] == 1
    assert rfd["rigs_metadata"]["sensor_count"] == 2
    assert rfd["frames_metadata"]["frame_count"] == num_frames
    assert len(rfd["image_sensor_indexes"]) == image_count


def test_bundle_adjust_description():
    """Test the description method."""
    ba = BundleAdjustTransform()
    desc = ba.description()

    assert "bundle" in desc.lower() or "adjust" in desc.lower() or "BA" in desc
