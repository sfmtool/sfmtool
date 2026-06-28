# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter images by file number range or filename glob."""

import fnmatch

import numpy as np

from .._filenames import number_from_filename
from .._sfmtool import RangeExpr, SfmrReconstruction


class IncludeRangeFilter:
    """Filter to keep only images whose file number is in the specified range."""

    def __init__(self, range_expr: RangeExpr):
        self.range_expr = range_expr
        self.range_numbers = set(range_expr)

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        return _filter_images_by_range(
            recon, self.range_numbers, include=True, range_expr_str=str(self.range_expr)
        )

    def description(self) -> str:
        return f"Include images in range {self.range_expr}"


class ExcludeRangeFilter:
    """Filter to exclude images whose file number is in the specified range."""

    def __init__(self, range_expr: RangeExpr):
        self.range_expr = range_expr
        self.range_numbers = set(range_expr)

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        return _filter_images_by_range(
            recon,
            self.range_numbers,
            include=False,
            range_expr_str=str(self.range_expr),
        )

    def description(self) -> str:
        return f"Exclude images in range {self.range_expr}"


class IncludeGlobFilter:
    """Filter to keep only images whose name matches a glob pattern."""

    def __init__(self, pattern: str):
        self.pattern = pattern

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        images_to_keep = [
            i
            for i, name in enumerate(recon.image_names)
            if fnmatch.fnmatch(name, self.pattern)
        ]
        if not images_to_keep:
            raise ValueError(
                f"No images match include glob pattern '{self.pattern}'. "
                f"Example image names: {recon.image_names[:5]}"
            )
        images_to_keep = np.array(images_to_keep, dtype=np.uint32)
        kept = len(images_to_keep)
        total = len(recon.image_names)
        print(
            f"  Applied include glob '{self.pattern}': keeping {kept} of {total} images"
        )
        return _filter_images(recon, images_to_keep)

    def description(self) -> str:
        return f"Include images matching '{self.pattern}'"


class ExcludeGlobFilter:
    """Filter to exclude images whose name matches a glob pattern."""

    def __init__(self, pattern: str):
        self.pattern = pattern

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        images_to_keep = [
            i
            for i, name in enumerate(recon.image_names)
            if not fnmatch.fnmatch(name, self.pattern)
        ]
        if not images_to_keep:
            raise ValueError(
                f"No images remain after excluding glob pattern '{self.pattern}'. "
                f"All {len(recon.image_names)} images matched."
            )
        images_to_keep = np.array(images_to_keep, dtype=np.uint32)
        kept = len(images_to_keep)
        total = len(recon.image_names)
        print(
            f"  Applied exclude glob '{self.pattern}': keeping {kept} of {total} images"
        )
        return _filter_images(recon, images_to_keep)

    def description(self) -> str:
        return f"Exclude images matching '{self.pattern}'"


def _filter_images_by_range(
    recon: SfmrReconstruction,
    range_numbers: set[int],
    include: bool,
    range_expr_str: str,
) -> SfmrReconstruction:
    """Filter images by file number range."""
    images_to_keep = []
    for i, image_name in enumerate(recon.image_names):
        file_number = number_from_filename(image_name)
        if include:
            if file_number is not None and file_number in range_numbers:
                images_to_keep.append(i)
        else:
            if file_number is None or file_number not in range_numbers:
                images_to_keep.append(i)

    if not images_to_keep:
        mode = "include" if include else "exclude"
        available_numbers = sorted(
            {
                number_from_filename(name)
                for name in recon.image_names
                if number_from_filename(name) is not None
            }
        )
        raise ValueError(
            f"No images remain after applying {mode} range filter '{range_expr_str}'. "
            f"Available file numbers: {available_numbers}"
        )

    images_to_keep = np.array(images_to_keep, dtype=np.uint32)

    mode = "include" if include else "exclude"
    print(
        f"  Applied {mode} range filter '{range_expr_str}': "
        f"keeping {len(images_to_keep)} of {len(recon.image_names)} images"
    )

    return _filter_images(recon, images_to_keep)


def _filter_images(
    recon: SfmrReconstruction,
    images_to_keep: np.ndarray,
) -> SfmrReconstruction:
    """Filter a reconstruction to keep only the specified images.

    Delegates to the Rust ``subset_by_image_indices`` primitive, which drops
    points orphaned by the removed images, remaps image and point indexes,
    recomputes observation counts, and carries rig/frame data. Crucially — and
    unlike a manual rebuild from the ``(N, 3)`` Euclidean positions, which would
    force ``w = 1`` — it preserves points at infinity (``w = 0`` directions),
    rather than silently materialising them.
    """
    before = recon.point_count
    result = recon.subset_by_image_indices(
        np.ascontiguousarray(images_to_keep, dtype=np.uint32),
        drop_orphaned_points=True,
    )
    removed = before - result.point_count
    print(
        f"  Keeping {result.point_count} of {before} points "
        f"(removed {removed} points with no remaining observations)"
    )
    return result
