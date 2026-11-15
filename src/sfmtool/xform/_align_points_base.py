# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared point-based alignment for the `align-to` family of transforms.

Aligning a reconstruction to another one by 3D point correspondences is the
same computation whatever supplies the target: what differs is only where the
target reconstruction comes from. This module holds that computation once;
subclasses supply the target and the name to announce it under.
"""

from ..align.by_points import estimate_alignment_from_points
from .._sfmtool.analysis import pair_images
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

        # Images are paired by file hash, then by name.
        pairing = pair_images(
            recon.image_names,
            recon.find_image_file_hashes(),
            target_recon.image_names,
            target_recon.find_image_file_hashes(),
        )
        shared_images = [(int(s), int(t)) for s, t in pairing["pairs"]]

        if len(shared_images) < 2:
            raise ValueError(
                f"Insufficient overlap: only {len(shared_images)} common images "
                f"(need at least 2 for point-based alignment; {pairing['summary']})"
            )

        print(f"    Found {len(shared_images)} common images ({pairing['summary']})")

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
