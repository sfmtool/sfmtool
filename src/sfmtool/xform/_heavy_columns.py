# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``sfm xform`` steps that drop and add the two heavy optional columns.

A reconstruction's image thumbnails and patch bitmaps are conveniences for a
viewer: neither says where a camera or a point is, and together they are nearly
all of a file's bytes. ``--drop-thumbnails`` and ``--drop-patch-bitmaps``
discard a column and keep every row of everything; ``--add-thumbnails`` and
``--add-patch-bitmaps`` fill an absent column back in (the thumbnails from
the ``.sift`` files first and the photographs second, the bitmaps from the
photographs); ``--minimal`` drops both and marks the output to be saved with
minimal metadata. None of them moves a point, renumbers a row or changes a
keypoint.

See ``specs/cli/reconstruction/xform/xform-command.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._sfmtool import THUMBNAIL_SIZE
from .._sfmtool.reconstruction import SfmrReconstruction
from ._images import load_workspace_images
from ._patch_params import validate_patch_params


class DropThumbnailsTransform:
    """Discard the per-image thumbnail column, keeping every row."""

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        if recon.thumbnails_y_x_rgb is None:
            print("  No thumbnails to drop")
            return recon
        print(f"  Dropping thumbnails of {recon.image_count} images")
        return recon.clone_with_changes(thumbnails_y_x_rgb=None)

    def description(self) -> str:
        return "Drop thumbnails"


class DropPatchBitmapsTransform:
    """Discard the per-point patch bitmap column, keeping the patch frames."""

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        if recon.patch_bitmap_resolution is None:
            print("  No patch bitmaps to drop")
            return recon
        # The frames and normals stay, so a later step can render onto them.
        print(f"  Dropping patch bitmaps of {recon.point_count} points")
        return recon.clone_with_changes(patch_bitmaps=None)

    def description(self) -> str:
        return "Drop patch bitmaps"


class MinimalTransform:
    """``--minimal``: drop both heavy columns here, and save minimal metadata.

    The column part runs at this step's position in the chain. The metadata
    part (an empty ``workspace.absolute_path``, no ``lineage``, ``tool_options``
    holding only this invocation's ``transforms``) is a property of the save,
    which the command applies when any step in the chain is this one.

    ``workspace_path``, from ``wspath=<path>``, is carried to that save as well:
    it is the ``workspace.relative_path`` the output records, stated rather than
    measured from where the output is written. ``None`` measures it.
    """

    def __init__(self, workspace_path: str | None = None) -> None:
        self.workspace_path = workspace_path

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        recon = DropPatchBitmapsTransform().apply(recon)
        return DropThumbnailsTransform().apply(recon)

    def description(self) -> str:
        # The description is what lands in the output's tool_options, so a
        # stated workspace path is named there rather than left implicit.
        stated = (
            f"; workspace path '{self.workspace_path}'"
            if self.workspace_path is not None
            else ""
        )
        return f"Minimal (drop patch bitmaps and thumbnails; minimal metadata{stated})"


def _restore_note(step: str, column: str) -> None:
    print(f"  {step} after --minimal: the output keeps its {column}")


class AddThumbnailsTransform:
    """Build the thumbnail column, from each image's ``.sift`` first and its
    photograph second.

    A ``.sift`` thumbnail is the row an extractor already reduced from the
    photograph, so it is read first, when that ``.sift`` verifiably belongs to
    the image: its content hash is the image's ``sift_content_hashes`` entry in
    a ``sift_files`` file, and its recorded ``image_file_xxh128`` is the
    image's ``image_file_hashes`` entry in an ``embedded_patches`` file.
    Otherwise the row is the photograph at ``workspace_dir / name`` decoded the
    way the SIFT extractors decode it (EXIF orientation ignored) and resized
    the way they resize it, so it is byte-identical to the ``.sift`` thumbnail
    an extractor writes for the same photograph. In an ``embedded_patches``
    file that photograph is first checked against its recorded
    ``image_file_hashes`` entry. An image neither source can supply fails the
    step, naming every image it could not build, since the column is whole or
    absent.
    """

    def __init__(self) -> None:
        # Set by the parser when an earlier --minimal dropped this column.
        self.restores_minimal = False

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        if recon.thumbnails_y_x_rgb is not None:
            print("  Thumbnails already present; nothing to add")
            return recon
        if self.restores_minimal:
            _restore_note("--add-thumbnails", "thumbnails")

        from ..sift.extract_sfmtool import read_image_bgr, thumbnail_of_bgr
        from ..sift.file import xxh128_of_file

        workspace = Path(recon.workspace_dir)
        embedded = recon.feature_source == "embedded_patches"
        image_hashes = recon.image_file_hashes if embedded else None
        rows: list[np.ndarray] = []
        failures: list[str] = []
        from_sift = 0
        for index, name in enumerate(recon.image_names):
            row = _verified_sift_thumbnail(recon, index, name)
            if row is not None:
                from_sift += 1
                rows.append(np.ascontiguousarray(row, dtype=np.uint8))
                continue
            path = workspace / name
            if not path.is_file():
                failures.append(f"{name} (no verified .sift, and no photograph)")
                continue
            if embedded and bytes.fromhex(xxh128_of_file(path)) != bytes(
                image_hashes[index]
            ):
                failures.append(
                    f"{name} (no verified .sift, and not the photograph the "
                    "reconstruction was built from: its image_file_hashes entry "
                    "does not match)"
                )
                continue
            image = read_image_bgr(path)
            if image is None:
                failures.append(
                    f"{name} (no verified .sift, and the photograph is unreadable)"
                )
                continue
            rows.append(np.ascontiguousarray(thumbnail_of_bgr(image), dtype=np.uint8))

        if failures:
            listed = "\n    ".join(failures)
            raise ValueError(
                f"--add-thumbnails could not build a thumbnail for "
                f"{len(failures)} of {recon.image_count} images, and the column "
                f"is whole or absent:\n    {listed}"
            )
        from_photographs = recon.image_count - from_sift
        print(
            f"  Built thumbnails of {recon.image_count} images: {from_sift} from "
            f"verified .sift copies, {from_photographs} from photographs"
        )
        thumbnails = (
            np.stack(rows)
            if rows
            else np.zeros((0, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3), dtype=np.uint8)
        )
        return recon.clone_with_changes(thumbnails_y_x_rgb=thumbnails)

    def description(self) -> str:
        return "Add thumbnails"


def _verified_sift_thumbnail(
    recon: SfmrReconstruction, index: int, name: str
) -> np.ndarray | None:
    """The image's ``.sift`` thumbnail when the ``.sift`` verifiably belongs to it.

    In a ``sift_files`` file the ``.sift`` content hash must equal the image's
    ``sift_content_hashes`` entry; in an ``embedded_patches`` file the
    ``.sift``'s recorded ``image_file_xxh128`` must equal the image's
    ``image_file_hashes`` entry. ``None`` when there is no such ``.sift``.
    """
    from ..sift.file import SiftReader, get_sift_path_from_recon

    try:
        sift_path = get_sift_path_from_recon(recon, name)
    except (KeyError, TypeError):
        return None
    if not sift_path.is_file():
        return None
    try:
        reader = SiftReader(sift_path)
    except OSError:
        return None
    if recon.feature_source == "embedded_patches":
        recorded = reader.metadata.get("image_file_xxh128")
        expected = bytes(recon.image_file_hashes[index]).hex()
    else:
        recorded = reader.content_hash.get("content_xxh128")
        expected = bytes(recon.sift_content_hashes[index]).hex()
    if recorded != expected:
        return None
    return np.asarray(reader.read_thumbnail(), dtype=np.uint8)


class AddPatchBitmapsTransform:
    """Render the patch bitmap column at the stored frames and keypoints.

    Moves nothing: positions, normals, frames, keypoints and tracks come out
    exactly as they went in. The render is the sub-pixel refiner's zero-step
    fuse (``PatchCloud.render_bitmaps``). A point with fewer than two
    observations that render in frame gets a zero row.
    """

    # Precondition checked per-step by `apply_transforms` (see `_apply.py`).
    required_feature_source = "embedded_patches"

    def __init__(self, *, resolution: int = 24, sampler: str = "bilinear_mip"):
        if resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {resolution}")
        validate_patch_params(window="gaussian_disk", window_sigma=1.0, sampler=sampler)
        self.resolution = resolution
        self.sampler = sampler
        # Set by the parser when an earlier --minimal dropped this column.
        self.restores_minimal = False

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        if recon.patch_bitmap_resolution is not None:
            print("  Patch bitmaps already present; nothing to add")
            return recon
        if self.restores_minimal:
            _restore_note("--add-patch-bitmaps", "patch bitmaps")
        cloud = recon.patches
        if cloud is None:
            raise ValueError(
                "reconstruction has no patch frames to render bitmaps onto; "
                "expected embedded_patches (run `sfm xform --to-embedded-patches` "
                "first)"
            )
        images = load_workspace_images(recon)
        print(
            f"  Rendering {self.resolution}x{self.resolution} patch bitmaps for "
            f"{recon.point_count} points"
        )
        bitmaps = cloud.render_bitmaps(
            recon, images, resolution=self.resolution, sampler=self.sampler
        )
        return recon.clone_with_changes(patch_bitmaps=bitmaps)

    def description(self) -> str:
        return (
            f"Add patch bitmaps (resolution={self.resolution}, sampler={self.sampler})"
        )
