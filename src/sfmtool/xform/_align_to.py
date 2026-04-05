# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Align to another reconstruction transformation."""

from pathlib import Path

from .._align_by_points import estimate_alignment_from_points
from .._sfmtool import SfmrReconstruction


class AlignToTransform:
    """Align reconstruction to match another reconstruction using 3D point correspondences."""

    def __init__(self, reference_path: Path):
        self.reference_path = Path(reference_path).absolute()
        if not self.reference_path.exists():
            raise FileNotFoundError(f"Reference file not found: {self.reference_path}")
        if self.reference_path.suffix.lower() != ".sfmr":
            raise ValueError(
                f"Reference must be a .sfmr file, got: {self.reference_path}"
            )
        self._reference_recon = SfmrReconstruction.load(str(self.reference_path))

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        print(f"  Aligning to {self.reference_path.name} using 3D points...")

        source_name_to_idx = {name: idx for idx, name in enumerate(recon.image_names)}
        target_name_to_idx = {
            name: idx for idx, name in enumerate(self._reference_recon.image_names)
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
            target_recon=self._reference_recon,
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
        return f"Align to {self.reference_path.name}"
