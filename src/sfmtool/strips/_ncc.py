# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Patch-strip rendering and weighted-NCC photoconsistency scoring.

Pure helpers for the ``compare --strips`` engine: a Gaussian patch window,
weighted normalized cross-correlation between two patches, and rendering a
point's per-view patches into a single horizontal strip annotated with each
view's image index. This is a Python, per-pair scorer distinct from the Rust
``PatchCloud.refine_normals`` path (which scores a single reconstruction's
surfels, not cross-reconstruction pairs).
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


def gauss_window(patch: int) -> np.ndarray:
    """Flattened Gaussian weighting (sigma = patch/4) over a ``patch``x``patch`` tile."""
    u = np.arange(patch) - patch / 2 + 0.5
    gx, gy = np.meshgrid(u, u)
    sig = patch / 4.0
    return np.exp(-(gx**2 + gy**2) / (2 * sig**2)).ravel()


def _wncc(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    sw = w.sum()
    da = a - (w * a).sum() / sw
    db = b - (w * b).sum() / sw
    den = np.sqrt((w * da * da).sum() * (w * db * db).sum())
    return float((w * da * db).sum() / den) if den > 1e-9 else 0.0


def _wncc_color(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    if a.ndim == 2:
        return _wncc(a, b, w)
    return float(np.mean([_wncc(a[..., c], b[..., c], w) for c in range(a.shape[-1])]))


def render_track_strip(
    obs_imgs: list[int],
    patch_of: Callable[[int], np.ndarray],
    w: np.ndarray,
    *,
    tile: int,
    inner: tuple[int, int] | None = None,
    sep: int = 2,
    reproj_errs: list[float] | None = None,
    per_view_scores: bool = False,
    normal_offsets: list[tuple[float, float] | None] | None = None,
) -> tuple[np.ndarray, float, int]:
    """Render a point's observations as a horizontal BGR patch strip, returning
    ``(strip, mean_pairwise_ncc, n_views)``. Tiles are labeled by image index;
    ``obs_imgs`` is expected pre-sorted by the caller.

    With ``inner = (offset, size)`` (context mode) each patch is rendered larger
    than the validated extent; NCC is scored on the central ``size`` sub-patch
    and a 1px box is drawn around it, so the surrounding scene context is visible
    while the score still reflects the point's own surfel.

    With ``per_view_scores`` each tile is additionally labeled (bottom-left) with
    that observation's NCC against the other views (the mean of its pairwise
    scores), and, when ``reproj_errs`` is given (parallel to ``obs_imgs``), its
    reprojection error in pixels.

    With ``normal_offsets`` (parallel to ``obs_imgs``, each an ``(s, t)`` in the
    patch tangent frame or ``None``) each tile gets an obliquity marker: a line
    from the patch-extent centre (fronto-parallel, i.e. viewing straight down the
    surface normal) to a dot at ``(s, t)`` mapped onto the patch-extent box. The
    dot reaches the box edge as the view grazes the surface (90 deg off the
    normal), so its displacement shows how oblique — and thus how distortion-prone
    — that view of the surfel is, and in which in-plane direction the camera lies.
    """
    obs = obs_imgs
    patches = [patch_of(i) for i in obs]
    if inner is not None:
        off, sz = inner
        cores = [p[off : off + sz, off : off + sz] for p in patches]
    else:
        cores = patches
    n = len(cores)
    # Symmetric pairwise NCC matrix; the mean of all pairs is the strip score, and
    # each view's mean over the others is its per-view (leave-one-out) score.
    pair = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(a + 1, n):
            pair[a, b] = pair[b, a] = _wncc_color(cores[a], cores[b], w)
    sims = pair[np.triu_indices(n, k=1)]
    mean_ncc = float(np.nanmean(sims)) if sims.size else float("nan")
    per_view = (
        [float(np.nanmean(pair[k])) if n > 1 else float("nan") for k in range(n)]
        if per_view_scores
        else None
    )

    tiles = []
    for k, (img_idx, pf) in enumerate(zip(obs, patches)):
        p8 = np.clip(pf, 0, 255).astype(np.uint8)
        src_sz = p8.shape[0]
        p8 = cv2.resize(p8, (tile, tile), interpolation=cv2.INTER_NEAREST)
        # Patches arrive RGB (the source images are loaded RGB); this is the cv2
        # boundary where the tile is annotated and handed to the BGR montage, so
        # convert RGB→BGR here.
        bgr = (
            cv2.cvtColor(p8, cv2.COLOR_RGB2BGR)
            if p8.ndim == 3
            else cv2.cvtColor(p8, cv2.COLOR_GRAY2BGR)
        )
        if inner is not None:
            off, sz = inner
            scale = tile / src_sz
            x0, x1 = round(off * scale), round((off + sz) * scale)
            cv2.rectangle(bgr, (x0, x0), (x1 - 1, x1 - 1), (0, 255, 0), 1)
        if normal_offsets is not None and normal_offsets[k] is not None:
            # Obliquity marker: box centre = fronto-parallel; the dot at (s, t)
            # (patch tangent frame, |(s,t)| <= 1) drifts to the box edge as the
            # view grazes. Box maps [-1, 1] to the patch-extent box (the green box
            # in context mode, else the whole tile).
            s, t = normal_offsets[k]
            if inner is not None:
                off, sz = inner
                scale = tile / src_sz
                bx0, bx1 = round(off * scale), round((off + sz) * scale)
            else:
                bx0, bx1 = 0, tile
            bc = 0.5 * (bx0 + bx1)
            bh = 0.5 * (bx1 - bx0)
            dx, dy = round(bc + s * bh), round(bc + t * bh)
            c0 = round(bc)
            cv2.line(bgr, (c0, c0), (dx, dy), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(
                bgr, (dx, dy), max(2, tile // 36), (255, 0, 255), -1, cv2.LINE_AA
            )
            cv2.circle(bgr, (c0, c0), 1, (255, 255, 255), -1)
        cv2.putText(
            bgr,
            str(img_idx),
            (2, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if per_view is not None:
            cv2.putText(
                bgr,
                f"n{per_view[k]:+.2f}",
                (2, tile - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if reproj_errs is not None:
                cv2.putText(
                    bgr,
                    f"e{reproj_errs[k]:.1f}",
                    (2, tile - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        tiles.append(bgr)

    sep_col = np.full((tile, sep, 3), 40, np.uint8)
    row: list[np.ndarray] = []
    for t in tiles:
        row.extend((t, sep_col))
    return np.hstack(row[:-1]), mean_ncc, len(obs)
