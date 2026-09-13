# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the patch visualization scripts in this directory.

``viz_keypoint_localization.py``, ``viz_keypoint_localization_strips.py`` and
``viz_view_selection_strips.py`` each render a montage of patch tiles for a
reconstruction, and all three need the same reconstruction accessors (images,
rotations, tracks), the same windowed-ZNCC primitives, the same patch-plane
unprojection and the same montage drawing idioms.

The scripts import this module by plain name (``from _viz_common import ...``),
which resolves because Python puts a script's own directory at the head of
``sys.path``; they are run as ``pixi run python scripts/viz_*.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


# ===== Reconstruction accessors =====


def load_images(recon) -> list[np.ndarray]:
    ws = recon.workspace_dir
    out = []
    for name in recon.image_names:
        bgr = cv2.imread(os.path.join(ws, name), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image {name!r} under {ws}")
        out.append(np.ascontiguousarray(bgr))
    return out


def rotation_matrices(recon) -> np.ndarray:
    """Per-image world->camera rotation matrices from the wxyz quaternions."""
    q = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = w * w + x * x + y * y + z * z
    s = np.where(n > 0, 2.0 / n, 0.0)
    rot = np.empty((len(q), 3, 3))
    rot[:, 0, 0] = 1 - s * (y * y + z * z)
    rot[:, 0, 1] = s * (x * y - z * w)
    rot[:, 0, 2] = s * (x * z + y * w)
    rot[:, 1, 0] = s * (x * y + z * w)
    rot[:, 1, 1] = 1 - s * (x * x + z * z)
    rot[:, 1, 2] = s * (y * z - x * w)
    rot[:, 2, 0] = s * (x * z - y * w)
    rot[:, 2, 1] = s * (y * z + x * w)
    rot[:, 2, 2] = 1 - s * (x * x + y * y)
    return rot


def track_views(recon) -> dict[int, set[int]]:
    pids = np.asarray(recon.track_point_indexes)
    imgs = np.asarray(recon.track_image_indexes)
    tracks: dict[int, set[int]] = {}
    for pid, im in zip(pids.tolist(), imgs.tolist()):
        tracks.setdefault(int(pid), set()).add(int(im))
    return tracks


def infinity_first_sample(recon, ids, sample_size, rng, *, interleave: bool = True):
    """A point-id sample that keeps ALL points at infinity and fills the rest with
    random finite points, capped at ``sample_size`` — so a prioritized montage shows
    BOTH kinds even when infinity is a tiny fraction of the cloud.

    With ``interleave`` (the default) the two kinds alternate, infinity leading each
    pair, which is what a montage that renders rows in sample order wants. With
    ``interleave=False`` all the infinity ids come first, for a caller that balances
    the row mix itself downstream.
    """
    from itertools import zip_longest

    is_inf = np.asarray(recon.point_is_at_infinity)
    ids = [int(i) for i in np.asarray(ids).tolist()]
    inf_ids = [i for i in ids if is_inf[i]]
    fin_ids = [i for i in ids if not is_inf[i]]
    k = max(0, min(sample_size, len(ids)) - len(inf_ids))
    fin = (
        sorted(
            int(x)
            for x in rng.choice(fin_ids, size=min(k, len(fin_ids)), replace=False)
        )
        if fin_ids and k
        else []
    )
    if not interleave:
        return (inf_ids + fin)[:sample_size]
    merged = [x for pair in zip_longest(inf_ids, fin) for x in pair if x is not None]
    return merged[:sample_size]


def label_for(path: Path, recon) -> str:
    """A short dataset label: the workspace dir name minus a trailing '_ws',
    else the .sfmr stem."""
    ws = Path(recon.workspace_dir).name
    if ws:
        return ws[:-3] if ws.endswith("_ws") else ws
    return path.stem


# ===== Photometric primitives =====


def gauss_window(n: int) -> np.ndarray:
    u = np.arange(n) - n / 2 + 0.5
    gx, gy = np.meshgrid(u, u)
    return np.exp(-(gx**2 + gy**2) / (2 * (n / 4.0) ** 2)).ravel()


def znorm(tile: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-channel z-normalized (C, P) vector, sqrt(window) folded in so a dot of
    two such vectors is a windowed ZNCC (mirrors the Rust convention)."""
    flat = tile.reshape(-1, tile.shape[-1]) if tile.ndim == 3 else tile.reshape(-1, 1)
    a = flat.astype(np.float64)
    g = np.sqrt(w)
    chans = []
    for c in range(a.shape[1]):
        x = a[:, c]
        x = x - (w * x).sum() / w.sum()
        nrm = np.sqrt((w * x * x).sum())
        chans.append(g * (x / nrm if nrm > 1e-9 else np.zeros_like(x)))
    return np.stack(chans, 0)


def sharpness(img) -> float:
    """Gradient energy of an image — higher = sharper (less registration blur)."""
    g = img.astype(np.float32)
    g = g.mean(2) if g.ndim == 3 else g
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float((gx * gx + gy * gy).mean())


def plane_hit(cam, rot, t, kpt, center, normal, w=1.0):
    """Unproject keypoint `kpt` to a re-anchored patch center.

    Finite patch (`w == 1`): the world point where the view's ray meets the
    patch plane (point `center`, `normal`). Point at infinity (`w == 0`): every
    ray to it is parallel to its direction, so the re-anchored "center" is simply
    the world ray direction the keypoint points along."""
    ray_cam = np.asarray(cam.pixel_to_ray(float(kpt[0]), float(kpt[1])))
    dir_world = rot.T @ ray_cam
    if w == 0.0:
        n = float(np.linalg.norm(dir_world))
        return dir_world / n if n > 0.0 else None
    cam_center = -rot.T @ t
    denom = float(dir_world @ normal)
    if abs(denom) < 1e-12:
        return None
    s = float((center - cam_center) @ normal) / denom
    return cam_center + s * dir_world


# ===== Montage drawing =====

#: Background grey every montage canvas is filled with.
CANVAS_BG = 28


def new_canvas(width: int, height: int) -> np.ndarray:
    """A montage canvas: a ``height x width`` BGR image of the dark background grey."""
    return np.full((height, width, 3), CANVAS_BG, np.uint8)


def draw_text(img, text: str, org, scale: float, color) -> None:
    """The montage text idiom: Hershey simplex, hairline, antialiased."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def chip(img, text: str, org, color, scale: float) -> None:
    """A text label on a dark plate, for drawing over a rendered tile."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + 2), (15, 15, 15), -1)
    draw_text(img, text, (x, y), scale, color)
