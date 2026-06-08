# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""High-level helpers built on top of :class:`sfmtool._sfmtool.SphericalTileRig`.

The Rust binding owns the per-tile warps, atlas packing, KD-tree NN, and the
atlas → equirect resampler. This module wires those primitives into the
``resample_atlas_to_equirect`` convenience that builds a full-sphere
equirectangular destination camera and validates the inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sfmtool._sfmtool import (
    CameraIntrinsics,
    RotQuaternion,
    SphericalTileRig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _equirect_camera(width: int, height: int) -> CameraIntrinsics:
    """Build a full-sphere equirectangular camera at the given resolution."""
    return CameraIntrinsics(
        "EQUIRECTANGULAR",
        width,
        height,
        {
            "focal_length_x": width / (2.0 * np.pi),
            "focal_length_y": height / np.pi,
            "principal_point_x": width / 2.0,
            "principal_point_y": height / 2.0,
        },
    )


def resample_atlas_to_equirect(
    rig: SphericalTileRig,
    atlas_image: "NDArray[np.float32]",
    width: int,
    height: int,
    *,
    rotation: RotQuaternion | None = None,
    k: int = 1,
) -> "NDArray[np.float32]":
    """Sample an assembled atlas into an equirectangular image.

    Thin wrapper around :meth:`SphericalTileRig.resample_atlas` that builds
    the equirectangular destination camera at ``(width, height)`` and
    validates the input atlas shape and dtype.

    Args:
        rig: A constructed :class:`SphericalTileRig`.
        atlas_image: ``(H_atlas, W_atlas)`` or ``(H_atlas, W_atlas, C)``
            float32 atlas image whose ``(H, W)`` matches ``rig.atlas_size``.
        width: Output equirectangular image width in pixels.
        height: Output equirectangular image height in pixels.
        rotation: Optional ``rot_world_from_dst`` rotation. Defaults to
            identity (so the output is aligned with the rig's world frame).
        k: Number of nearest tiles to blend (``>= 1``). ``k = 1`` is
            closest-tile sampling; ``k > 1`` blends across Voronoi seams.

    Returns:
        ``(height, width)`` (or ``(height, width, C)``) float32 array.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    aw, ah = rig.atlas_size
    if atlas_image.shape[:2] != (ah, aw):
        raise ValueError(
            f"atlas_image shape {atlas_image.shape[:2]} does not match "
            f"rig atlas_size (H, W) = {(ah, aw)}"
        )
    if atlas_image.dtype != np.float32:
        raise ValueError(f"atlas_image must be float32, got {atlas_image.dtype}")

    rot = rotation if rotation is not None else RotQuaternion.identity()
    equirect = _equirect_camera(width, height)
    return rig.resample_atlas(atlas_image, equirect, rot, k)
