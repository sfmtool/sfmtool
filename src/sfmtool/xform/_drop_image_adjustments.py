# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop per-image parameters from cameras of a reconstruction.

Each chosen camera stops varying each parameter named, every image of the
camera is read through the shared value again, and the images' mean is folded
into the camera, by ``SfmrReconstruction.drop_image_adjustments`` in the Rust
core (see ``specs/core/reconstruction/drop-image-adjustments.md``). Poses and
points are unchanged.
"""

import click

from .._sfmtool.reconstruction import SfmrReconstruction
from ..camera.cameras import _camera_adjustments

# The per-image parameters, by the names a camera's `per_image_adjustments`
# lists, in that order.
IMAGE_ADJUSTMENT_PARAMETERS = (
    "focal",
    "principal_point",
    "radial_bspline",
    "rolling_shutter",
)


class DropImageAdjustmentsTransform:
    """Drop per-image parameters from each chosen camera, folding their mean
    into the camera.

    Args:
        parameters: The names to drop, each one of ``"focal"``,
            ``"principal_point"``, ``"radial_bspline"`` and
            ``"rolling_shutter"``.
        cameras: Camera indexes to drop them from (default: every camera).

    A parameter a camera does not vary is left as it is, so the transform
    drops from each camera only what it varies, and a camera that varies none
    of them is unchanged.

    Raises:
        ValueError: For an empty ``parameters`` or a name that is not one of
            the four.
    """

    def __init__(self, parameters: list[str], cameras: list[int] | None = None):
        unknown = [p for p in parameters if p not in IMAGE_ADJUSTMENT_PARAMETERS]
        if unknown:
            raise ValueError(
                f"{unknown[0]!r} is not a per-image parameter; the parameters "
                f"are {', '.join(IMAGE_ADJUSTMENT_PARAMETERS)}"
            )
        if not parameters:
            raise ValueError("parameters= names no per-image parameter to drop")
        self.parameters = [p for p in IMAGE_ADJUSTMENT_PARAMETERS if p in parameters]
        self.cameras = None if cameras is None else sorted(set(cameras))

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        count = len(recon.cameras)
        cameras = list(range(count)) if self.cameras is None else self.cameras
        bad = [c for c in cameras if not 0 <= c < count]
        if bad:
            raise click.UsageError(
                f"--drop-image-adjustments cameras= names camera(s) {bad}, and "
                f"this reconstruction has {count} camera(s)"
            )
        flags = recon.per_image_adjustments
        nonzero = _camera_adjustments(recon)
        for camera in cameras:
            varied = set(flags[camera]) | set(nonzero.get(camera, []))
            dropped = [p for p in self.parameters if p in varied]
            if dropped:
                recon = recon.drop_image_adjustments(camera, dropped)
                print(f"    Camera {camera}: dropped {', '.join(dropped)} per image")
            else:
                print(f"    Camera {camera}: varies none of them per image")
        return recon

    def description(self) -> str:
        params = "parameters=" + "+".join(self.parameters)
        if self.cameras is not None:
            params += ", cameras=" + "+".join(str(c) for c in self.cameras)
        return f"Drop per-image adjustments ({params})"
