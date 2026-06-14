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
) -> tuple[np.ndarray, float, int]:
    """Render a point's observations as a horizontal BGR patch strip, returning
    ``(strip, mean_pairwise_ncc, n_views)``. Tiles are labeled by image index;
    ``obs_imgs`` is expected pre-sorted by the caller.

    With ``inner = (offset, size)`` (context mode) each patch is rendered larger
    than the validated extent; NCC is scored on the central ``size`` sub-patch
    and a 1px box is drawn around it, so the surrounding scene context is visible
    while the score still reflects the point's own surfel.
    """
    obs = obs_imgs
    patches = [patch_of(i) for i in obs]
    if inner is not None:
        off, sz = inner
        cores = [p[off : off + sz, off : off + sz] for p in patches]
    else:
        cores = patches
    sims = [
        _wncc_color(cores[a], cores[b], w)
        for a in range(len(cores))
        for b in range(a + 1, len(cores))
    ]
    mean_ncc = float(np.mean(sims)) if sims else float("nan")

    tiles = []
    for img_idx, pf in zip(obs, patches):
        p8 = np.clip(pf, 0, 255).astype(np.uint8)
        src_sz = p8.shape[0]
        p8 = cv2.resize(p8, (tile, tile), interpolation=cv2.INTER_NEAREST)
        bgr = p8 if p8.ndim == 3 else cv2.cvtColor(p8, cv2.COLOR_GRAY2BGR)
        if inner is not None:
            off, sz = inner
            scale = tile / src_sz
            x0, x1 = round(off * scale), round((off + sz) * scale)
            cv2.rectangle(bgr, (x0, x0), (x1 - 1, x1 - 1), (0, 255, 0), 1)
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
        tiles.append(bgr)

    sep_col = np.full((tile, sep, 3), 40, np.uint8)
    row: list[np.ndarray] = []
    for t in tiles:
        row.extend((t, sep_col))
    return np.hstack(row[:-1]), mean_ncc, len(obs)
