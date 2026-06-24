# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Visualize ``PatchCloud.localize_keypoints`` (congealing) as context-tile strips
with the before/after reference patch.

For each point, a row of:

- the **reference patch** consensus before vs after congealing (left) — the
  (plain, unweighted) mean of the kept views' cores rendered at the raw projection
  vs at the congealed keypoints. The "after" panel is sharper when the views
  co-register; the ``x..`` label is the gradient-energy sharpness ratio.
- a strip of per-view **context tiles** (right). Each tile is the larger context
  the search slides within (the scored ``R×R`` core extended by ``±⌈search⌉`` px),
  with two boxes: **white** = core at the projection (``acc = 0``), **cyan** = core
  at the congealed keypoint; their offset is the recovered in-plane shift.

Tile borders: yellow (``trk`` track view), green (``add`` view-selection-added),
red (``drop`` — dropped for out-of-frame / beyond ``max_shift_px`` / low
leave-one-out ZNCC, projection box only). Kept tiles carry offset (source px) and
leave-one-out ZNCC. Points are ordered by largest median shift.

The before/after geometry faithfully reproduces the kernel, but the reference
panel's mean/sharpness is an independent simpler check, not the kernel's
IRLS-weighted consensus. A row with ``k 2/..`` is the kernel's two-view floor
(each view is the other's reference) — read its agreement with care.

This mirrors the prototype context-tile + reference montage; it is a
dev/inspection tool, not a test (coverage lives in
``tests/test_patch_keypoint_localization.py``). See
``specs/core/patch-keypoint-localization.md``.

Example::

    pixi run python scripts/viz_keypoint_localization_strips.py \\
        seoul_bull_ws/sfmr/*.sfmr kerry_park_ws/sfmr/*.sfmr --out-dir /tmp/strips
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np

from sfmtool._sfmtool import (
    OrientedPatch,
    PatchCloud,
    RigidTransform,
    SfmrReconstruction,
)
from sfmtool._sfmtool.flow import WarpMap


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
    pids = np.asarray(recon.track_point_ids)
    imgs = np.asarray(recon.track_image_indexes)
    tracks: dict[int, set[int]] = {}
    for pid, im in zip(pids.tolist(), imgs.tolist()):
        tracks.setdefault(int(pid), set()).add(int(im))
    return tracks


def plane_hit(cam, rot, t, kpt, center, normal, w=1.0):
    """Re-anchored patch center for the view's ray through `kpt`.

    Finite patch (`w == 1`): the world point where the ray meets the patch plane.
    Point at infinity (`w == 0`): the (unit) world ray direction itself — every
    ray to the point is parallel to its direction."""
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


def render_patch(image, cam, t_i, quat_i, center, normal, up, half_extent, res, w=1.0):
    """Render one view's `res × res` tile for a surfel centred at `center`. For a
    point at infinity (`w == 0`), `center` is a direction and the patch is tangent
    to the unit sphere."""
    if w == 0.0:
        patch = OrientedPatch.from_infinity_direction(
            list(center), list(up), list(half_extent)
        )
    else:
        patch = OrientedPatch.from_center_normal(
            list(center), list(normal), list(up), list(half_extent)
        )
    pose = RigidTransform.from_wxyz_translation(quat_i.tolist(), t_i.tolist())
    wm = WarpMap.from_patch(patch, cam, pose, res)
    return np.asarray(wm.remap_bilinear(image), dtype=np.float32)


def consensus(cores):
    """Plain mean consensus image over a stack of `res × res` tiles."""
    return np.stack([np.asarray(c, np.float64) for c in cores]).mean(0)


def sharpness(img) -> float:
    """Gradient energy of an image — higher = sharper (less registration blur)."""
    g = img.astype(np.float32)
    g = g.mean(2) if g.ndim == 3 else g
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float((gx * gx + gy * gy).mean())


def _chip(img, text, org, color, scale=0.34):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + 2), (15, 15, 15), -1)
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


YELLOW = (0, 220, 220)
GREEN = (0, 200, 0)
RED = (0, 0, 230)
WHITE = (245, 245, 245)
CYAN = (60, 200, 255)
GREY = (200, 200, 200)


def _to_bgr(img, disp):
    p8 = np.clip(img, 0, 255).astype(np.uint8)
    bgr = p8 if p8.ndim == 3 else cv2.cvtColor(p8, cv2.COLOR_GRAY2BGR)
    return cv2.resize(bgr, (disp, disp), interpolation=cv2.INTER_NEAREST)


def _ref_panel(img, disp, label, accent, bot_label=None):
    bgr = _to_bgr(img, disp)
    cv2.rectangle(bgr, (0, 0), (disp - 1, disp - 1), accent, 2)
    _chip(bgr, label, (4, 14), accent)
    if bot_label is not None:
        _chip(bgr, bot_label, (4, disp - 7), accent, 0.34)
    return bgr


def _box(img, col0, row0, size, scale, color):
    x0, y0 = round(col0 * scale), round(row0 * scale)
    x1, y1 = round((col0 + size) * scale), round((row0 + size) * scale)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), color, 1)


def _ctx_tile(ctx_img, disp, margin, res, border, top_label, acc=None, bot_label=None):
    bgr = _to_bgr(ctx_img, disp)
    scale = disp / ctx_img.shape[0]
    _box(bgr, margin, margin, res, scale, WHITE)  # core @ projection
    if acc is not None:
        _box(
            bgr, margin + acc[0], margin + acc[1], res, scale, CYAN
        )  # core @ congealed
    cv2.rectangle(bgr, (0, 0), (disp - 1, disp - 1), border, 3)
    _chip(bgr, top_label, (4, 14), border)
    if bot_label is not None:
        _chip(bgr, bot_label, (4, disp - 7), CYAN, 0.32)
    return bgr


def _infinity_first_sample(recon, ids, sample_size, rng):
    """A point-id sample that interleaves ALL points at infinity with random
    finite points (infinity leading each pair), capped at ``sample_size`` — so a
    prioritized montage shows BOTH kinds even when infinity is a tiny fraction of
    the cloud."""
    from itertools import zip_longest

    is_inf = np.asarray(recon.point_is_at_infinity)
    ids = [int(i) for i in np.asarray(ids).tolist()]
    inf_ids = [i for i in ids if is_inf[i]]
    fin_ids = [i for i in ids if not is_inf[i]]
    k = max(0, min(sample_size, len(ids)) - len(inf_ids))
    fin = (
        sorted(int(x) for x in rng.choice(fin_ids, size=min(k, len(fin_ids)), replace=False))
        if fin_ids and k
        else []
    )
    merged = [x for pair in zip_longest(inf_ids, fin) for x in pair if x is not None]
    return merged[:sample_size]


def gather(recon, cloud, images, args):
    """Run select_views + localize for the sample; return lightweight per-point
    metadata (no rendering yet, so only the chosen rows pay for tiles)."""
    tracks = track_views(recon)
    ids = np.asarray(cloud.point_ids)
    rng = np.random.default_rng(args.seed)
    if args.prioritize_infinity:
        sample = _infinity_first_sample(recon, ids, args.sample, rng)
    else:
        sample = np.sort(
            rng.choice(ids, size=min(args.sample, len(ids)), replace=False)
        ).tolist()

    sel = cloud.select_views(
        recon, images, point_ids=sample, resolution=args.resolution
    )
    view_sets = {int(r["point_id"]): np.asarray(r["admitted"]).tolist() for r in sel}
    loc = cloud.localize_keypoints(
        recon,
        images,
        view_sets=view_sets,
        point_ids=sample,
        resolution=args.resolution,
        search=args.search,
        max_shift_px=args.max_shift_px,
    )
    out = []
    for r in loc:
        pid = int(r["point_id"])
        kept = np.asarray(r["views"], dtype=np.int64).tolist()
        in_set = view_sets.get(pid, sorted(tracks.get(pid, set())))
        if len(kept) < 2 or len(in_set) < 2:
            continue
        offs = np.asarray(r["offsets_px"], dtype=np.float64)
        out.append(
            dict(
                pid=pid,
                kept=kept,
                kpts=np.asarray(r["keypoints"], dtype=np.float64),
                offs=offs,
                loo=np.asarray(r["loo_zncc"], dtype=np.float64),
                in_set=in_set,
                shift=float(np.median(offs)) if len(offs) else 0.0,
            )
        )
    return out, tracks


def render_row(meta, trk, recon, cloud, images, geom, args):
    rot, t, quats, cams, cam_idx, pid_to_patch = geom
    res, margin = args.resolution, max(1, math.ceil(args.search))
    ctx = res + 2 * margin
    pid = meta["pid"]
    kept_set = set(meta["kept"])
    kept_kpt = {v: meta["kpts"][i] for i, v in enumerate(meta["kept"])}
    kept_off = {v: float(meta["offs"][i]) for i, v in enumerate(meta["kept"])}
    kept_loo = {v: float(meta["loo"][i]) for i, v in enumerate(meta["kept"])}

    patch_obj = cloud[pid_to_patch[pid]]
    center = np.asarray(patch_obj.center, dtype=np.float64)
    normal = np.asarray(patch_obj.normal, dtype=np.float64)
    u_axis = np.asarray(patch_obj.u_axis, dtype=np.float64)
    v_axis = np.asarray(patch_obj.v_axis, dtype=np.float64)
    he = list(patch_obj.half_extent)
    he_ctx = [he[0] * ctx / res, he[1] * ctx / res]
    wpp_u, wpp_v = 2.0 * he[0] / res, 2.0 * he[1] / res
    w = float(patch_obj.w)

    before_cores, after_cores = [], []
    tiles = []
    shown = [v for v in meta["in_set"] if v in kept_set] + [
        v for v in meta["in_set"] if v not in kept_set
    ]
    for v in shown[: args.max_views]:
        cam = cams[int(cam_idx[v])]
        ctx_tile = render_patch(
            images[v], cam, t[v], quats[v], center, normal, u_axis, he_ctx, ctx, w
        )
        if v in kept_set:
            hit = plane_hit(cam, rot[v], t[v], kept_kpt[v], center, normal, w)
            c = center if hit is None else hit
            off = (c - center) if hit is not None else np.zeros(3)
            acc = (float(off @ u_axis) / wpp_u, float(off @ v_axis) / wpp_v)
            border = YELLOW if v in trk else GREEN
            kind = "trk" if v in trk else "add"
            bot = f"{kept_off[v]:.1f}px"
            if np.isfinite(kept_loo[v]):
                bot += f" z{kept_loo[v]:.2f}"
            tiles.append(
                _ctx_tile(
                    ctx_tile, args.tile, margin, res, border, f"{kind}{v}", acc, bot
                )
            )
        else:
            tiles.append(_ctx_tile(ctx_tile, args.tile, margin, res, RED, f"drop{v}"))

    # Reference consensus over the kept views: cores at the projection (before) vs
    # at the congealed keypoints (after).
    for v in meta["kept"]:
        cam = cams[int(cam_idx[v])]
        before_cores.append(
            render_patch(
                images[v], cam, t[v], quats[v], center, normal, u_axis, he, res, w
            )
        )
        hit = plane_hit(cam, rot[v], t[v], kept_kpt[v], center, normal, w)
        c = center if hit is None else hit
        after_cores.append(
            render_patch(images[v], cam, t[v], quats[v], c, normal, u_axis, he, res, w)
        )
    ref_before = consensus(before_cores)
    ref_after = consensus(after_cores)
    sharp = sharpness(ref_after) / max(sharpness(ref_before), 1e-9)
    return dict(
        pid=pid,
        nkept=len(meta["kept"]),
        nset=len(meta["in_set"]),
        shift=meta["shift"],
        ref_before=ref_before,
        ref_after=ref_after,
        sharp=sharp,
        tiles=tiles,
    )


def _compose(rows, args):
    header_w, gap, tile = 120, 4, args.tile
    sep = 14  # gap between the reference pair and the context strip
    max_cols = max(len(x["tiles"]) for x in rows)
    ref_w = 2 * (tile + gap) + sep
    width = header_w + ref_w + max_cols * (tile + gap)
    total_h = (tile + gap) * len(rows) + 56
    canvas = np.full((total_h, width, 3), 28, np.uint8)

    cv2.putText(
        canvas,
        f"keypoint localization: {args.label}  "
        f"(sample={args.sample}, RES={args.resolution}, search={args.search:g})",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "left: reference patch before|after (x = sharpness ratio).  right: per-view "
        "context tiles, white=core@projection cyan=core@congealed.",
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        GREY,
        1,
        cv2.LINE_AA,
    )

    y = 50
    for x in rows:
        cv2.putText(
            canvas,
            f"pt {x['pid']}",
            (8, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        floor = "  FLOOR" if x["nkept"] == 2 and x["nset"] > 2 else ""
        cv2.putText(
            canvas,
            f"k {x['nkept']}/{x['nset']}{floor}",
            (8, y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{x['shift']:.2f}px",
            (8, y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        # Reference before / after.
        xoff = header_w
        canvas[y : y + tile, xoff : xoff + tile] = _ref_panel(
            x["ref_before"], tile, "ref before", GREY
        )
        xoff += tile + gap
        canvas[y : y + tile, xoff : xoff + tile] = _ref_panel(
            x["ref_after"], tile, "ref after", CYAN, f"x{x['sharp']:.2f}"
        )
        # Context strip.
        xoff = header_w + ref_w
        for t in x["tiles"]:
            canvas[y : y + tile, xoff : xoff + tile] = t
            xoff += tile + gap
        y += tile + gap
    return canvas


def _label_for(path: Path, recon) -> str:
    ws = Path(recon.workspace_dir).name
    if ws:
        return ws[:-3] if ws.endswith("_ws") else ws
    return path.stem


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sfmr", nargs="+", type=Path, help="one or more solved .sfmr files")
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--rows", type=int, default=10, help="points (rows) per montage")
    p.add_argument("--sample", type=int, default=300, help="random points per recon")
    p.add_argument(
        "--prioritize-infinity",
        action="store_true",
        help="order the sample so points at infinity (w=0) lead the montage",
    )
    p.add_argument("--max-views", type=int, default=9, help="max view tiles per point")
    p.add_argument(
        "--resolution", type=int, default=24, help="scored core grid (R x R)"
    )
    p.add_argument(
        "--search", type=float, default=6.0, help="search margin (patch-grid px)"
    )
    p.add_argument("--max-shift-px", type=float, default=3.0)
    p.add_argument("--tile", type=int, default=96, help="display tile size in px")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.sfmr:
        recon = SfmrReconstruction.load(str(path))
        args.label = _label_for(path, recon)
        images = load_images(recon)
        cloud = PatchCloud.from_reconstruction(
            recon, normal="mean_viewing", extent_value=5.0
        )
        metas, tracks = gather(recon, cloud, images, args)
        metas.sort(key=lambda m: m["shift"], reverse=True)
        chosen = metas[: args.rows]
        if not chosen:
            print(f"{args.label}: nothing to render", flush=True)
            continue
        geom = (
            rotation_matrices(recon),
            np.asarray(recon.translations, dtype=np.float64),
            np.asarray(recon.quaternions_wxyz, dtype=np.float64),
            recon.cameras,
            np.asarray(recon.camera_indexes),
            {int(p): i for i, p in enumerate(np.asarray(cloud.point_ids))},
        )
        rows = [
            render_row(m, tracks.get(m["pid"], set()), recon, cloud, images, geom, args)
            for m in chosen
        ]
        canvas = _compose(rows, args)
        out = args.out_dir / f"keypoint_localization_strips_{args.label}.jpg"
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{args.label}: wrote {out}  (rows={len(rows)})", flush=True)


if __name__ == "__main__":
    main()
