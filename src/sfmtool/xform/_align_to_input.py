# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Align to input reconstruction transformation."""

from .._align_by_points import estimate_alignment_from_points
from .._sfmtool import SfmrReconstruction


class AlignToInputTransform:
    """Align reconstruction back to original input reconstruction."""

    _original_input: SfmrReconstruction | None = None

    def __init__(self):
        pass

    @classmethod
    def set_original_input(cls, recon: SfmrReconstruction):
        cls._original_input = recon.replace()

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        if self._original_input is None:
            raise RuntimeError(
                "Original input reconstruction not set. "
                "This transform must be used within apply_transforms()."
            )

        print("  Aligning to original input reconstruction using 3D points...")

        source_name_to_idx = {name: idx for idx, name in enumerate(recon.image_names)}
        target_name_to_idx = {
            name: idx for idx, name in enumerate(self._original_input.image_names)
        }

        common_names = set(source_name_to_idx.keys()) & set(target_name_to_idx.keys())

        if len(common_names) < 2:
            raise ValueError(
                f"Insufficient overlap: only {len(common_names)} common images "
                f"(need at least 2 for point-based alignment)"
            )

        print(f"    Found {len(common_names)} common images")

        shared_images = []
        for name in common_names:
            source_idx = source_name_to_idx[name]
            target_idx = target_name_to_idx[name]
            shared_images.append((source_idx, target_idx))

        alignment = estimate_alignment_from_points(
            source_recon=recon,
            target_recon=self._original_input,
            shared_images=shared_images,
            min_points=10,
            use_ransac=True,
            ransac_iterations=1000,
            ransac_percentile=95.0,
        )

        print(f"    Found {alignment.n_point_correspondences} point correspondences")
        if hasattr(alignment, "n_inliers"):
            print(f"    RANSAC inliers: {alignment.n_inliers}")
        print(f"    Point RMS error: {alignment.point_rms_error:.6f}")
        print(f"    Transform scale: {alignment.transform.scale:.6f}")
        print(f"    Transform translation: {alignment.transform.translation}")

        return alignment.transform @ recon

    def description(self) -> str:
        return "Align to original input"
