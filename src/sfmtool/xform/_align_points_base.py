# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared point-based alignment for the `align-to` family of transforms.

Aligning a reconstruction to another one by 3D point correspondences is the
same computation whatever supplies the target: what differs is only where the
target reconstruction comes from. This module holds that computation once;
subclasses supply the target and the name to announce it under.
"""

from ..align.by_points import estimate_alignment_from_points
from .._sfmtool.reconstruction import SfmrReconstruction


class AlignToPointsTransform:
    """Align a reconstruction to a target one using 3D point correspondences.

    Subclasses implement `_target_recon` (the reconstruction to align onto)
    and `_target_label` (how that target is named in the progress line).
    """

    def _target_recon(self) -> SfmrReconstruction:
        """Return the reconstruction to align onto."""
        raise NotImplementedError

    def _target_label(self) -> str:
        """Return a human-readable name for the target, for the progress line."""
        raise NotImplementedError

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        target_recon = self._target_recon()

        print(f"  Aligning to {self._target_label()} using 3D points...")

        source_name_to_idx = {name: idx for idx, name in enumerate(recon.image_names)}
        target_name_to_idx = {
            name: idx for idx, name in enumerate(target_recon.image_names)
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
            target_recon=target_recon,
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
