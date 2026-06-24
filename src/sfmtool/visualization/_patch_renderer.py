# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Render oriented patches from a reconstruction on top of its source images.

Each finite 3D point of a reconstruction can carry an oriented planar patch (the
quad spanned by the in-plane half-extent vectors ``u``, ``v`` centred on the
point, with the point's surface normal). This module projects those quads into
every registered image and composites them onto the source frame, for visually
inspecting the reconstruction's geometry and patches.

Fill modes:

* ``texture`` -- warp the per-point RGBA patch bitmap onto the projected quad
  (requires patch bitmaps, e.g. ``sfm xform ... --refine-normals bitmaps=true``).
* ``normal``  -- flat-shade each quad by its world normal (``(n + 1) / 2``).
* ``flat``    -- flat-shade each quad by the point's reconstruction colour.
* ``wire``    -- outline only, no fill.

Patches are painted back-to-front (painter's algorithm); there is no true
occlusion buffer, so a distant patch can still show through a nearer one.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np

from .._sfmtool import SfmrReconstruction
from .._sfmtool.geometry import RotQuaternion

MODES = ("texture", "normal", "flat", "wire")


class PatchRenderError(ValueError):
    """Raised when a reconstruction can't be rendered (e.g. no patch cloud)."""


def collect_patches(
    recon: SfmrReconstruction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull per-patch geometry into parallel numpy arrays.

    Returns ``(centers, u_vec, v_vec, normals, point_ids, w)`` where ``u_vec`` and
    ``v_vec`` are the world-space half-extent vectors (axis * half_extent), so a
    patch covers ``center + s*u_vec + t*v_vec`` for ``(s, t)`` in ``[-1, 1]^2``
    -- the same parametrisation used to render the patch bitmaps. ``w`` is the
    homogeneous weight per patch (``1.0`` finite; ``0.0`` for a point at infinity,
    whose ``center`` is a direction and corners are directions).

    Raises:
        PatchRenderError: the reconstruction carries no patch cloud.
    """
    cloud = recon.patches
    if cloud is None:
        raise PatchRenderError(
            "reconstruction has no patch cloud; produce one with "
            "`sfm xform in.sfmr out.sfmr --refine-normals save_patches=true`"
        )
    n = len(cloud)
    centers = np.empty((n, 3), np.float64)
    u_vec = np.empty((n, 3), np.float64)
    v_vec = np.empty((n, 3), np.float64)
    normals = np.empty((n, 3), np.float64)
    w = np.empty(n, np.float64)
    for i in range(n):
        p = cloud[i]
        hu, hv = p.half_extent
        centers[i] = p.center
        u_vec[i] = np.asarray(p.u_axis) * hu
        v_vec[i] = np.asarray(p.v_axis) * hv
        normals[i] = p.normal
        w[i] = p.w
    point_ids = np.asarray(cloud.point_ids, dtype=np.int64)
    return centers, u_vec, v_vec, normals, point_ids, w


def _quad_corners(
    centers: np.ndarray, u_vec: np.ndarray, v_vec: np.ndarray
) -> np.ndarray:
    """World-space quad corners, shape ``(P, 4, 3)``.

    Corner order matches the patch bitmap layout (bitmap ``[row=y, col=x]`` maps
    to ``s`` along ``u`` and ``t`` along ``v``, both in ``[-1, 1]``):
    ``(-u,-v), (+u,-v), (+u,+v), (-u,+v)``.
    """
    return np.stack(
        [
            centers - u_vec - v_vec,
            centers + u_vec - v_vec,
            centers + u_vec + v_vec,
            centers - u_vec + v_vec,
        ],
        axis=1,
    )


def _normal_to_bgr(normals: np.ndarray) -> np.ndarray:
    """Map unit normals to BGR colours via ``(n + 1) / 2 -> [0, 255]``."""
    rgb = (np.clip((normals + 1.0) * 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb[:, ::-1]  # RGB -> BGR


def _render_image(
    recon: SfmrReconstruction,
    img_idx: int,
    quads_world: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
    normal_bgr: np.ndarray,
    point_ids: np.ndarray,
    colors_bgr: np.ndarray,
    bitmaps: np.ndarray | None,
    patch_w: np.ndarray,
    *,
    mode: str,
    border: bool,
    border_color: tuple[int, int, int],  # (R, G, B)
    border_thickness: int,
    alpha: float,
    opaque_threshold: float | None,
    upscale: float,
    backface_cull: bool,
) -> tuple[np.ndarray, int]:
    """Composite all visible patches onto one source image; return (canvas, n).

    ``quads_world`` and ``normal_bgr`` are precomputed once by the caller (they
    do not depend on the image), so this only does the per-image projection.
    """
    cam = recon.cameras[int(recon.camera_indexes[img_idx])]
    rot = np.asarray(
        RotQuaternion.from_wxyz_array(
            recon.quaternions_wxyz[img_idx]
        ).to_rotation_matrix()
    )
    trans = np.asarray(recon.translations[img_idx])

    img_path = Path(recon.workspace_dir) / recon.image_names[img_idx]
    canvas = cv2.imread(str(img_path))
    if canvas is None:
        raise FileNotFoundError(f"could not read source image: {img_path}")
    if upscale != 1.0:
        canvas = cv2.resize(
            canvas, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )
    h, w = canvas.shape[:2]

    p = quads_world.shape[0]
    # Homogeneous projection: a finite corner (w == 1) translates and projects as
    # a point; a point at infinity (w == 0) is a direction, so it rotates without
    # translating and projects as a ray. The cheirality (depth > 0) and backface
    # tests then hold for both — an infinity patch is front-facing iff its ray
    # R·d points forward.
    cam_pts = quads_world @ rot.T + trans * patch_w[:, None, None]  # (P, 4, 3)
    px = (
        np.asarray(cam.ray_to_pixel_batch(cam_pts.reshape(-1, 3))).reshape(p, 4, 2)
        * upscale
    )
    depth = cam_pts[:, :, 2].mean(axis=1)

    visible = np.isfinite(px).all(axis=(1, 2)) & (depth > 0)
    if backface_cull:
        normal_cam = normals @ rot.T
        center_cam = centers @ rot.T + trans * patch_w[:, None]
        visible &= np.einsum("ij,ij->i", normal_cam, center_cam) < 0

    xs, ys = px[:, :, 0], px[:, :, 1]
    if visible.any():
        # Fully-invisible patches are all-NaN rows; nanmin/nanmax warn on those
        # (harmless -- they're masked out below).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xmin, xmax = np.nanmin(xs, axis=1), np.nanmax(xs, axis=1)
            ymin, ymax = np.nanmin(ys, axis=1), np.nanmax(ys, axis=1)
        # Drop patches whose projected bbox lies entirely off-frame.
        visible &= (xmax >= 0) & (xmin < w) & (ymax >= 0) & (ymin < h)

    idxs = np.nonzero(visible)[0]
    idxs = idxs[np.argsort(-depth[idxs])]  # painter's: far first

    n_drawn = 0
    for i in idxs:
        quad = px[i]
        x0, y0 = int(np.floor(quad[:, 0].min())), int(np.floor(quad[:, 1].min()))
        x1, y1 = int(np.ceil(quad[:, 0].max())), int(np.ceil(quad[:, 1].max()))
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if cx1 <= cx0 or cy1 <= cy0:
            continue

        if mode == "texture":
            bmp = bitmaps[int(point_ids[i])]  # (R, R, 4); colour channels are BGR
            r = bmp.shape[0]
            src = np.array([[0, 0], [r, 0], [r, r], [0, r]], np.float32)
            dst = (quad - [cx0, cy0]).astype(np.float32)
            m = cv2.getPerspectiveTransform(src, dst)
            tw, th = cx1 - cx0, cy1 - cy0
            # Patch bitmaps are rendered from cv2-loaded (BGR) source images, so
            # the colour channels are already BGR -- use them directly.
            warped = cv2.warpPerspective(
                np.ascontiguousarray(bmp[:, :, :3]), m, (tw, th)
            )
            warped_a = cv2.warpPerspective(bmp[:, :, 3], m, (tw, th))
            norm_a = warped_a.astype(np.float32) / 255.0
            if opaque_threshold is not None:
                # Paint texels whose confidence exceeds the threshold at full
                # opacity; drop the rest (no confidence bleed-through).
                a = (norm_a > opaque_threshold).astype(np.float32) * alpha
            else:
                a = norm_a * alpha
            a = a[:, :, None]
            region = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
            canvas[cy0:cy1, cx0:cx1] = (
                region * (1 - a) + warped.astype(np.float32) * a
            ).astype(np.uint8)
        elif mode in ("normal", "flat"):
            color = (
                tuple(int(c) for c in normal_bgr[i])
                if mode == "normal"
                else tuple(int(c) for c in colors_bgr[i])
            )
            mask = np.zeros((cy1 - cy0, cx1 - cx0), np.uint8)
            cv2.fillConvexPoly(mask, np.round(quad - [cx0, cy0]).astype(np.int32), 255)
            region = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
            fill = np.empty_like(region)
            fill[:] = color
            a = (mask.astype(np.float32) / 255.0 * alpha)[:, :, None]
            canvas[cy0:cy1, cx0:cx1] = (region * (1 - a) + fill * a).astype(np.uint8)

        if border or mode == "wire":
            cv2.polylines(
                canvas,
                [np.round(quad).astype(np.int32)],
                isClosed=True,
                color=border_color[::-1],  # RGB -> BGR for cv2
                thickness=border_thickness,
                lineType=cv2.LINE_AA,
            )
        n_drawn += 1

    return canvas, n_drawn


def render_patches(
    recon: SfmrReconstruction,
    output_dir: Path,
    *,
    mode: str = "texture",
    border: bool = False,
    border_color: tuple[int, int, int] = (0, 255, 0),  # (R, G, B)
    border_thickness: int = 1,
    alpha: float = 1.0,
    opaque_threshold: float | None = None,
    scale: float = 1.0,
    upscale: float = 1.0,
    backface_cull: bool = True,
    image_filter: Iterable[str] | None = None,
    progress: Callable[[str, int, Path], None] | None = None,
) -> list[tuple[str, int, Path]]:
    """Render patch overlays for every (matching) registered image.

    Args:
        recon: reconstruction carrying a patch cloud (and, for ``texture`` mode,
            per-point patch bitmaps).
        output_dir: directory for the ``<image>_<mode>.png`` outputs (created).
        mode: one of :data:`MODES`.
        opaque_threshold: texture mode only. ``None`` (default) blends each texel
            by its confidence alpha (the source image bleeds through where
            confidence is low). A value in ``[0, 1]`` instead paints texels whose
            normalised confidence exceeds the threshold at full opacity and drops
            the rest -- an honest alignment check without bleed-through.
        image_filter: if given, only render images whose name contains one of
            these substrings.
        progress: optional callback ``(image_name, n_patches, output_path)``
            invoked after each image is written.

    Returns:
        ``(image_name, n_patches_drawn, output_path)`` for each rendered image.

    Raises:
        PatchRenderError: no patch cloud, or ``texture`` mode without bitmaps.
    """
    if mode not in MODES:
        raise PatchRenderError(
            f"unknown mode {mode!r} (expected one of {', '.join(MODES)})"
        )

    centers, u_vec, v_vec, normals, point_ids, w = collect_patches(recon)
    colors_bgr = np.asarray(recon.colors)[point_ids][:, ::-1]
    bitmaps = recon.patch_bitmaps
    if bitmaps is not None:
        bitmaps = np.asarray(bitmaps)
    if mode == "texture" and bitmaps is None:
        raise PatchRenderError(
            "texture mode needs per-point patch bitmaps; rerun refine with "
            "`bitmaps=true`, or pick --mode normal/flat/wire"
        )

    # Precompute everything that doesn't depend on the image: the scaled quad
    # corners and the per-patch normal colours.
    quads_world = _quad_corners(centers, u_vec * scale, v_vec * scale)
    normal_bgr = _normal_to_bgr(normals)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, int, Path]] = []
    for idx, name in enumerate(recon.image_names):
        if image_filter and not any(s in name for s in image_filter):
            continue
        canvas, n_drawn = _render_image(
            recon,
            idx,
            quads_world,
            centers,
            normals,
            normal_bgr,
            point_ids,
            colors_bgr,
            bitmaps,
            w,
            mode=mode,
            border=border,
            border_color=border_color,
            border_thickness=border_thickness,
            alpha=alpha,
            opaque_threshold=opaque_threshold,
            upscale=upscale,
            backface_cull=backface_cull,
        )
        # Preserve the full relative path (sensor subdirs collide on stem alone,
        # e.g. fisheye_left/frame_05 vs fisheye_right/frame_05).
        stem = Path(name).with_suffix("").as_posix().replace("/", "__")
        out_path = output_dir / f"{stem}_{mode}.png"
        cv2.imwrite(str(out_path), canvas)
        results.append((name, n_drawn, out_path))
        if progress is not None:
            progress(name, n_drawn, out_path)
    return results
