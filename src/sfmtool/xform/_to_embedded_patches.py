# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline ``sift_files`` → ``embedded_patches`` conversion as an ``sfm xform``
operation.

``ToEmbeddedPatchesTransform`` changes a reconstruction's observation
representation **without any photometric adaptation**: it gives each point a
``(u, v)`` patch frame from the mean viewing direction (no normal refinement),
copies each observation's 2D keypoint straight from its ``.sift`` feature, and
copies each image's identity hash from the ``.sift`` metadata. The result is a
valid ``embedded_patches`` reconstruction whose keypoints are exactly the
original SIFT detections, with no photometric sift→patch pipeline (normal
refinement + view selection + keypoint localization) involved.

Because it reads the ``.sift`` files (for keypoints, image hashes, and the
default ``feature_size`` patch sizing), they must still be present where the
reconstruction was created. After this op the reconstruction is
``embedded_patches``; any later ``.sift``-dependent operation in the same chain
will fail.

See ``specs/cli/xform-command.md`` and ``specs/core/sift-to-patch-reconstruction.md``.
"""

from .._sfmtool import SfmrReconstruction

_NORMALS = ("mean_viewing", "stored", "geometric")
# CLI extent policy names mirror the library's ``PatchExtent`` policies, except
# the CLI speaks in **full** patch size (the whole edge length) while the library
# API stores a half-extent (halved in ``apply``), and ``pixel_size`` is the CLI
# spelling of the library's ``pixel_radius``. Mirrors ``_refine_normals``.
_EXTENT_TO_BINDING = {
    "feature_size": "feature_size",
    "fixed": "fixed",
    "relative_spacing": "relative_spacing",
    "pixel_size": "pixel_radius",
}
_EXTENTS = tuple(_EXTENT_TO_BINDING)
_REDUCES = ("min", "max", "median", "mean")


class ToEmbeddedPatchesTransform:
    """Convert ``sift_files`` → ``embedded_patches`` with no photometric adaptation.

    Knobs default to the ``SfmrReconstruction.to_embedded_patches`` binding
    defaults (mean-viewing normals, ``feature_size`` extent), so the two layers
    cannot drift. Point count, positions, poses, and cameras are unchanged; only
    the observation representation (per-observation keypoints + per-image hashes)
    and the per-point patch frame change.
    """

    def __init__(
        self,
        *,
        normal: str = "mean_viewing",
        k_neighbors: int = 12,
        extent: str = "feature_size",
        # Full patch size (whole edge length); halved to the library half-extent.
        extent_value: float = 10.0,
        feature_reduce: str = "median",
        pixel_reduce: str = "min",
    ):
        if normal not in _NORMALS:
            raise ValueError(f"normal must be one of {_NORMALS}, got {normal!r}")
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if extent not in _EXTENTS:
            raise ValueError(f"extent must be one of {_EXTENTS}, got {extent!r}")
        if extent_value <= 0:
            raise ValueError(f"extent_value must be positive, got {extent_value}")
        if feature_reduce not in _REDUCES:
            raise ValueError(
                f"feature_reduce must be one of {_REDUCES}, got {feature_reduce!r}"
            )
        if pixel_reduce not in _REDUCES:
            raise ValueError(
                f"pixel_reduce must be one of {_REDUCES}, got {pixel_reduce!r}"
            )
        self.normal = normal
        self.k_neighbors = k_neighbors
        self.extent = extent
        self.extent_value = extent_value
        self.feature_reduce = feature_reduce
        self.pixel_reduce = pixel_reduce

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        out = recon.to_embedded_patches(
            normal=self.normal,
            k_neighbors=self.k_neighbors,
            extent=_EXTENT_TO_BINDING[self.extent],
            # CLI extent_value is the full patch size; the binding takes a
            # half-extent (or half-size multiplier / radius).
            extent_value=self.extent_value / 2.0,
            feature_reduce=self.feature_reduce,
            pixel_reduce=self.pixel_reduce,
        )
        print(
            f"  Converted to embedded_patches: {out.point_count} points, "
            f"keypoints + image hashes copied from .sift (no photometric adaptation)"
        )
        return out

    def description(self) -> str:
        return (
            f"To embedded_patches (normal={self.normal}, extent={self.extent}, "
            f"no photometric adaptation)"
        )
