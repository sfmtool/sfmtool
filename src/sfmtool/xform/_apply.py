# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply transformation pipeline to reconstructions."""

from .._sfmtool import SfmrReconstruction
from ._interface import Transform


def apply_transforms(
    recon: SfmrReconstruction,
    transforms: list[Transform],
) -> SfmrReconstruction:
    """Apply a sequence of transformations to a reconstruction."""
    from ._align_to_input import AlignToInputTransform

    if any(isinstance(t, AlignToInputTransform) for t in transforms):
        AlignToInputTransform.set_original_input(recon)

    print(f"\nApplying {len(transforms)} transformation(s):")
    for i, transform in enumerate(transforms, 1):
        print(f"  [{i}/{len(transforms)}] {transform.description()}")
        recon = transform.apply(recon)

    print("\nTransformation complete!")
    print(f"  Images: {recon.image_count}")
    print(f"  Points: {recon.point_count}")
    print(f"  Observations: {recon.observation_count}")
    print(f"  Cameras: {recon.camera_count}")

    return recon
