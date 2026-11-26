# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply transformation pipeline to reconstructions."""

from pathlib import Path

from .._feature_source import require_embedded_patches
from .._sfmtool.reconstruction import SfmrReconstruction
from ._interface import Transform


def apply_transforms(
    recon: SfmrReconstruction,
    transforms: list[Transform],
) -> SfmrReconstruction:
    """Apply a sequence of transformations to a reconstruction.

    The steps that read photographs (those with ``reads_photographs``) read
    them through one photograph cache, made for this run, so a photograph's
    tiles and mipmap read by one step are there for the next. After the
    steps, the run prints the cache's whole decodes and mipmaps made.
    """
    from ._align_to_input import AlignToInputTransform

    if any(isinstance(t, AlignToInputTransform) for t in transforms):
        AlignToInputTransform.set_original_input(recon)

    cache = None
    photograph_paths = set()
    print(f"\nApplying {len(transforms)} transformation(s):")
    for i, transform in enumerate(transforms, 1):
        # Per-step precondition, checked against the *current* recon so a chain
        # like `--to-embedded-patches --refine-normals` passes (the conversion
        # runs before the surfel op that requires it).
        if getattr(transform, "required_feature_source", None) == "embedded_patches":
            require_embedded_patches(recon, transform.description())
        print(f"  [{i}/{len(transforms)}] {transform.description()}")
        if getattr(transform, "reads_photographs", False):
            if cache is None:
                from ..photographs import PhotographCache

                cache = PhotographCache()
            workspace_dir = recon.workspace_dir
            photograph_paths.update(Path(workspace_dir) / n for n in recon.image_names)
            recon = transform.apply(recon, cache=cache)
        else:
            recon = transform.apply(recon)

    print("\nTransformation complete!")
    if cache is not None:
        from .._photograph_cache_report import photograph_cache_lines

        for line in photograph_cache_lines(cache, photograph_paths):
            print(f"  {line}")
    print(f"  Images: {recon.image_count}")
    print(f"  Points: {recon.point_count}")
    print(f"  Observations: {recon.observation_count}")
    print(f"  Cameras: {recon.camera_count}")

    return recon
