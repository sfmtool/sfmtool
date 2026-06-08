# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Equirectangular panorama rendering from an SfM reconstruction.

Wires the spherical-tile pipeline (rig → per-tile consensus atlas → equirect
resample) into a single reusable entry point, :func:`render_equirect_panorama`.
The heavy lifting lives in the Rust core; this module only builds the rig,
loads the source images, and threads the parameters through.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from sfmtool._filenames import number_from_filename
from sfmtool._sfmtool import (
    RangeExpr,
    RotQuaternion,
    SphericalTileRig,
    render_consensus_atlas,
)
from sfmtool.rig.spherical_tile import resample_atlas_to_equirect

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from sfmtool._sfmtool import SfmrReconstruction


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


# Mirrors `sfmtool_core::spherical_tile_rig::MIN_PATCH_SIZE`.
_MIN_PATCH_SIZE = 5


def _patch_size_for_width(half_fov_rad: float, equirect_width: int) -> int:
    """Power-of-two patch size that samples a tile at the equirect pixel rate.

    Replicates the ``SphericalTileRig`` constructor's ``arc_per_pixel``-driven
    formula — ``max(MIN_PATCH_SIZE, ceil(2·half_fov_rad / arc_per_pixel))`` with
    ``arc_per_pixel = 2π / equirect_width`` — then rounds up to the next power
    of two, which the atlas packer requires.
    """
    arc_per_pixel = 2.0 * np.pi / equirect_width
    raw = int(np.ceil(2.0 * half_fov_rad / arc_per_pixel))
    return _next_pow2(max(_MIN_PATCH_SIZE, raw))


def _camera_centers(quaternions, translations) -> NDArray[np.float64]:
    """World-space camera centers ``C = -R(q)^T t`` for each image."""
    n = len(quaternions)
    centers = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        r_cam_from_world = RotQuaternion.from_wxyz_array(
            quaternions[i]
        ).to_rotation_matrix()
        centers[i] = -r_cam_from_world.T @ translations[i]
    return centers


def _resolve_reference_index(image_names: list[str], near_image: str) -> int:
    """Find the single image index matching ``near_image``.

    Matches an exact image name first, then a trailing-path / basename match.
    Raises ``ValueError`` if there is no match or the match is ambiguous (e.g.
    a bare ``frame_000001.jpg`` that exists under two rig sensors).
    """
    exact = [i for i, n in enumerate(image_names) if n == near_image]
    if len(exact) == 1:
        return exact[0]

    target = near_image.replace("\\", "/")
    matches = [
        i
        for i, n in enumerate(image_names)
        if n.replace("\\", "/").endswith("/" + target) or Path(n).name == near_image
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"--near-image {near_image!r} did not match any image in the "
            "reconstruction."
        )
    raise ValueError(
        f"--near-image {near_image!r} is ambiguous; it matches "
        f"{len(matches)} images (e.g. {[image_names[i] for i in matches[:4]]}). "
        "Pass a fuller path such as 'fisheye_left/frame_000001.jpg'."
    )


def select_source_indices(
    recon: SfmrReconstruction,
    *,
    range_expr: str | None = None,
    near_image: str | None = None,
    near_count: int | None = None,
    near_radius: float | None = None,
) -> NDArray[np.uint32] | None:
    """Choose which images to composite into the panorama.

    Filters are applied in order — first ``range_expr`` (by file number), then
    spatial proximity to ``near_image``'s camera center (``near_radius`` keeps
    everything within that world-space distance; ``near_count`` keeps the
    nearest N, the reference always included).

    Returns the kept image indices as ``uint32``, or ``None`` when no filter is
    requested (render every image). Raises ``ValueError`` if a filter removes
    every image or the reference is excluded by ``range_expr``.
    """
    if range_expr is None and near_image is None:
        return None

    image_names = list(recon.image_names)
    n = len(image_names)
    mask = np.ones(n, dtype=bool)

    if range_expr is not None:
        wanted = set(RangeExpr(range_expr))
        for i, name in enumerate(image_names):
            num = number_from_filename(name)
            mask[i] = num is not None and num in wanted
        if not mask.any():
            raise ValueError(f"No images match --range {range_expr!r}.")

    if near_image is not None:
        ref = _resolve_reference_index(image_names, near_image)
        if not mask[ref]:
            raise ValueError(
                f"--near-image {near_image!r} is excluded by --range; widen the "
                "range or choose a reference image inside it."
            )
        centers = _camera_centers(recon.quaternions_wxyz, recon.translations)
        dist = np.linalg.norm(centers - centers[ref], axis=1)
        if near_radius is not None:
            mask &= dist <= near_radius
        if near_count is not None:
            eligible = np.where(mask)[0]
            nearest = eligible[np.argsort(dist[eligible], kind="stable")][:near_count]
            mask = np.zeros(n, dtype=bool)
            mask[nearest] = True
        mask[ref] = True

    indices = np.where(mask)[0].astype(np.uint32)
    if indices.size == 0:
        raise ValueError("No images remain after filtering.")
    return indices


def build_panorama_rig(
    equirect_width: int, n_tiles: int, *, seed: int = 1234
) -> SphericalTileRig:
    """Build a spherical-tile rig sized for a target equirect width.

    The rig's angular resolution is ``2*pi / equirect_width`` so one atlas
    sample maps to roughly one output pixel along the equator. ``patch_size``
    is rounded up to the next power of two, which the atlas packer requires.
    """
    arc_per_pixel = 2.0 * np.pi / equirect_width
    rig = SphericalTileRig(n=n_tiles, arc_per_pixel=arc_per_pixel, seed=seed)
    rig.set_patch_size(_patch_size_for_width(rig.half_fov_rad, equirect_width))
    return rig


def load_panorama_rig(camrig_path: str | Path, equirect_width: int) -> SphericalTileRig:
    """Load a pre-built spherical-tile rig from a ``.camrig`` file.

    The tile *layout* — count, look directions, and half-FOV — comes from the
    saved rig, decoupling tile density from output size. The per-tile
    ``patch_size`` is **not** inherited from the file; it is re-derived from the
    target ``equirect_width`` (sampling one tile pixel per output pixel at the
    equator, then rounded up to the next power of two) exactly as
    :func:`build_panorama_rig` does for a synthesized rig. So a rig saved at a
    high resolution renders a small panorama with a correspondingly small patch
    size rather than wastefully over-sampling. The tile half-FOV is preserved,
    so coverage is unchanged.

    Raises:
        ValueError: If the file is not a ``spherical_tiles`` rig.
    """
    rig = SphericalTileRig.read_camrig(str(camrig_path))
    rig.set_patch_size(_patch_size_for_width(rig.half_fov_rad, equirect_width))
    return rig


def resolve_panorama_rig(
    *,
    equirect_width: int,
    n_tiles: int,
    camrig_path: str | Path | None = None,
    seed: int = 1234,
) -> SphericalTileRig:
    """Load or synthesize the rig used to render the panorama.

    When ``camrig_path`` is given the rig is loaded from disk
    (:func:`load_panorama_rig`) and ``n_tiles`` is ignored; otherwise a fresh
    rig is synthesized (:func:`build_panorama_rig`). Either way the per-tile
    patch resolution is sized to ``equirect_width``.
    """
    if camrig_path is not None:
        return load_panorama_rig(camrig_path, equirect_width)
    return build_panorama_rig(equirect_width, n_tiles, seed=seed)


def load_sources(
    recon: SfmrReconstruction, image_dir: str | Path
) -> list[tuple[object, RotQuaternion, NDArray[np.uint8]]]:
    """Load ``(camera, rotation, rgb_image)`` tuples for every image in ``recon``.

    Images are read from ``image_dir / image_name`` as RGB. Raises
    ``FileNotFoundError`` for the first image that cannot be read.
    """
    image_dir = Path(image_dir)
    cameras = recon.cameras
    camera_indexes = recon.camera_indexes
    quats = recon.quaternions_wxyz
    image_names = recon.image_names

    sources = []
    for i, name in enumerate(image_names):
        cam = cameras[camera_indexes[i]]
        q = RotQuaternion(quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3])
        path = image_dir / name
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read source image: {path}")
        sources.append((cam, q, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return sources


def render_equirect_panorama(
    recon: SfmrReconstruction,
    image_dir: str | Path,
    *,
    rig: SphericalTileRig | None = None,
    equirect_width: int = 2160,
    n_tiles: int = 320,
    camrig_path: str | Path | None = None,
    batch_size: int = 32,
    dtype: str = "float32",
    k: int = 1,
    seed: int = 1234,
    inlier_threshold: float = 8.0,
    gamma: float = 1.0,
    ransac_seed: int = 0,
) -> NDArray[np.float32]:
    """Render an equirectangular panorama from a posed reconstruction.

    Pipeline: build a :class:`SphericalTileRig`, composite a per-tile consensus
    atlas over the source images (photometric RANSAC selects the agreeing
    cluster per tile), then resample the atlas through a full-sphere
    equirectangular camera.

    Args:
        recon: A loaded reconstruction with per-image poses and intrinsics.
        image_dir: Directory the reconstruction's image names are relative to.
        rig: A pre-built rig to render with. When ``None`` one is resolved from
            ``camrig_path`` / ``equirect_width`` / ``n_tiles``.
        equirect_width: Output width in pixels; height is ``width // 2``.
        n_tiles: Number of spherical tiles in a synthesized rig (ignored when
            ``rig`` or ``camrig_path`` is given).
        camrig_path: Load the rig from this ``.camrig`` file instead of
            synthesizing one (ignored when ``rig`` is given).
        batch_size: Tiles composited per batch (bounds peak memory).
        dtype: Per-batch stack storage, ``"float32"`` or ``"float16"``.
        k: Nearest tiles blended during resampling (``k = 1`` is closest-tile).
        seed: Rig relaxer seed (synthesized rig only).
        inlier_threshold: Photometric RANSAC inlier threshold (luma units).
        gamma: Photometric RANSAC tone exponent.
        ransac_seed: Photometric RANSAC seed.

    Returns:
        ``(height, width, channels)`` float32 RGB panorama. Uncovered regions
        are left as ``NaN`` so callers can choose how to fill them.
    """
    if rig is None:
        rig = resolve_panorama_rig(
            equirect_width=equirect_width,
            n_tiles=n_tiles,
            camrig_path=camrig_path,
            seed=seed,
        )
    sources = load_sources(recon, image_dir)
    atlas, *_ = render_consensus_atlas(
        rig,
        sources,
        batch_size=batch_size,
        dtype=dtype,
        inlier_threshold=inlier_threshold,
        gamma=gamma,
        seed=ransac_seed,
    )
    return resample_atlas_to_equirect(
        rig, atlas, equirect_width, equirect_width // 2, k=k
    )
