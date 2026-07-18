# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Visualize ``PatchCloud.localize_keypoints`` (congealing) as before/after
consensus patches.

For each sampled 3D point the production kernel selects a view set and congeals
the per-view keypoints. This tool renders, per point, the consensus of the surfel
**before** congealing (every kept view's patch rendered at the point's raw
projection) next to the consensus **after** (each view's patch re-anchored to its
refined keypoint). Sharper edges and a higher leave-one-out (LOO) ZNCC after mean
the views co-register better — the whole point of the algorithm.

Per row it reports the kept-view count, the median keypoint shift (source px), the
mean LOO ZNCC before -> after, and the consensus sharpness ratio (gradient energy).
Rows prefer points whose keypoints actually moved, so the effect is visible.

The before/after *geometry* faithfully reproduces the kernel (keypoints unprojected
back onto the patch plane), but the metrics here are an **independent, deliberately
simpler check**: a plain (unweighted) mean consensus and a square-Gaussian window,
not the kernel's IRLS-weighted consensus over the inscribed-disk window. They show
the registration direction honestly but should not be read as the kernel's own LOO.
A row kept at exactly two views is the kernel's two-view floor — its LOO is high by
construction (each view is the other's reference), so treat 2-view rows with care.

This is a dev/inspection tool, not a test — the automated coverage lives in
``tests/test_patch_keypoint_localization.py``. See
``specs/core/patch-keypoint-localization.md``.

Example::

    pixi run python scripts/viz_keypoint_localization.py \\
        seoul_bull_ws/sfmr/*.sfmr kerry_park_ws/sfmr/*.sfmr --out-dir /tmp/congeal
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.patches import OrientedPatch, PatchCloud
from sfmtool._sfmtool.geometry import RigidTransform
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


def consensus_and_loo(tiles, w):
    """Mean consensus *image* over the stack and the mean leave-one-out ZNCC (each
    view vs the unit-mean of the others) — the honest "did they register" signal."""
    if len(tiles) < 2:
        return tiles[0] if tiles else None, float("nan")
    vecs = [znorm(t, w) for t in tiles]
    raw = np.stack([np.asarray(t, np.float64) for t in tiles])
    img = raw.mean(0)
    loo = []
    for k in range(len(vecs)):
        others = [vecs[j] for j in range(len(vecs)) if j != k]
        tmpl = np.mean(others, axis=0)
        tn = np.sqrt((tmpl**2).sum(1, keepdims=True))
        tmpl = tmpl / np.where(tn > 1e-9, tn, 1.0)
        loo.append(float((vecs[k] * tmpl).sum(1).mean()))
    return img, float(np.mean(loo))


def sharpness(img) -> float:
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


def render_at(images, cams, cam_idx, t, quats, i, center, normal, up, half_extent, res, w=1.0):
    """Render view `i`'s patch tile for a surfel centred at `center` (with `normal`,
    in-plane up `up`, and per-axis `half_extent`). For a point at infinity
    (`w == 0`), `center` is a direction and the patch is tangent to the sphere."""
    if w == 0.0:
        patch = OrientedPatch.from_infinity_direction(
            list(center), list(up), list(half_extent)
        )
    else:
        patch = OrientedPatch.from_center_normal(
            list(center), list(normal), list(up), list(half_extent)
        )
    pose = RigidTransform.from_wxyz_translation(quats[i].tolist(), t[i].tolist())
    wm = WarpMap.from_patch(patch, cams[int(cam_idx[i])], pose, res)
    return np.asarray(wm.remap_bilinear(images[i]), dtype=np.float32)


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


def render_rows(recon, cloud, images, args):
    rot = rotation_matrices(recon)
    t = np.asarray(recon.translations, dtype=np.float64)
    quats = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    cams = recon.cameras
    cam_idx = np.asarray(recon.camera_indexes)
    pid_to_patch = {int(p): i for i, p in enumerate(np.asarray(cloud.point_indexes))}

    ids = np.asarray(cloud.point_indexes)
    rng = np.random.default_rng(args.seed)
    if args.prioritize_infinity:
        sample = _infinity_first_sample(recon, ids, args.sample, rng)
    else:
        sample = np.sort(
            rng.choice(ids, size=min(args.sample, len(ids)), replace=False)
        ).tolist()

    sel = cloud.select_views(
        recon, images, point_indexes=sample, resolution=args.resolution
    )
    view_sets = {int(r["point_index"]): np.asarray(r["admitted"]).tolist() for r in sel}
    loc = cloud.localize_keypoints(
        recon,
        images,
        view_sets=view_sets,
        point_indexes=sample,
        resolution=args.resolution,
        max_shift_px=args.max_shift_px,
    )
    loc_by_pid = {int(r["point_index"]): r for r in loc}
    w = gauss_window(args.resolution)

    rows = []
    for pid in sample:
        pid = int(pid)
        r = loc_by_pid.get(pid)
        if r is None:
            continue
        views = np.asarray(r["views"], dtype=np.int64).tolist()
        kpts = np.asarray(r["keypoints"], dtype=np.float64)
        offs = np.asarray(r["offsets_px"], dtype=np.float64)
        if len(views) < 2:
            continue
        patch_obj = cloud[pid_to_patch[pid]]
        center = np.asarray(patch_obj.center, dtype=np.float64)
        normal = np.asarray(patch_obj.normal, dtype=np.float64)
        up = np.asarray(patch_obj.u_axis, dtype=np.float64)
        he = list(patch_obj.half_extent)
        pw = float(patch_obj.w)  # patch homogeneous weight (w shadows gauss window)

        before, after = [], []
        for k, i in enumerate(views):
            before.append(
                render_at(
                    images,
                    cams,
                    cam_idx,
                    t,
                    quats,
                    i,
                    center,
                    normal,
                    up,
                    he,
                    args.resolution,
                    pw,
                )
            )
            hit = plane_hit(
                cams[int(cam_idx[i])], rot[i], t[i], kpts[k], center, normal, pw
            )
            c = center if hit is None else hit
            after.append(
                render_at(
                    images,
                    cams,
                    cam_idx,
                    t,
                    quats,
                    i,
                    c,
                    normal,
                    up,
                    he,
                    args.resolution,
                    pw,
                )
            )

        img0, loo0 = consensus_and_loo(before, w)
        img1, loo1 = consensus_and_loo(after, w)
        rows.append(
            dict(
                pid=pid,
                nviews=len(views),
                shift=float(np.median(offs)),
                loo0=loo0,
                loo1=loo1,
                sharp=sharpness(img1) / max(sharpness(img0), 1e-9),
                img0=img0,
                img1=img1,
            )
        )

    # Prefer the points that actually moved (largest median shift).
    rows.sort(key=lambda x: x["shift"], reverse=True)
    chosen = rows[: args.rows]
    stats = dict(
        sample=len(rows),
        d_loo_median=float(np.median([x["loo1"] - x["loo0"] for x in rows]))
        if rows
        else None,
        sharp_median=float(np.median([x["sharp"] for x in rows])) if rows else None,
    )
    if not chosen:
        return None, stats
    return _compose(chosen, args), stats


def _panel(img, label, tile):
    p8 = np.clip(img, 0, 255).astype(np.uint8)
    big = cv2.resize(p8, (tile, tile), interpolation=cv2.INTER_NEAREST)
    if big.ndim == 2:
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(big, (0, 0), (tile - 1, 15), (15, 15, 15), -1)
    cv2.putText(
        big,
        label,
        (3, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return big


def _compose(rows, args):
    header_w, gap, tile = 330, 6, args.tile
    width = header_w + 2 * tile + gap
    total_h = (tile + gap) * len(rows) + 56
    canvas = np.full((total_h, width, 3), 28, np.uint8)
    cv2.putText(
        canvas,
        f"keypoint localization (congealing): {args.label}  "
        f"(sample={args.sample}, RES={args.resolution})",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "consensus patch: before (raw projection) vs after (congealed)",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    y = 52
    for x in rows:
        for j, line in enumerate(
            [
                f"pt {x['pid']}  ({x['nviews']} views)"
                + ("  FLOOR" if x["nviews"] == 2 else ""),
                f"shift {x['shift']:.2f}px",
                f"LOO {x['loo0']:.3f}->{x['loo1']:.3f}",
                f"sharp x{x['sharp']:.2f}",
            ]
        ):
            cv2.putText(
                canvas,
                line,
                (8, y + 16 + 17 * j),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
        xoff = header_w
        canvas[y : y + tile, xoff : xoff + tile] = _panel(x["img0"], "before", tile)
        xoff += tile + gap
        canvas[y : y + tile, xoff : xoff + tile] = _panel(x["img1"], "after", tile)
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
    p.add_argument("--rows", type=int, default=8, help="points (rows) per montage")
    p.add_argument(
        "--prioritize-infinity",
        action="store_true",
        help="order the sample so points at infinity (w=0) lead the montage",
    )
    p.add_argument("--sample", type=int, default=300, help="random points per recon")
    p.add_argument("--resolution", type=int, default=24, help="patch grid (R x R)")
    p.add_argument("--max-shift-px", type=float, default=3.0)
    p.add_argument("--tile", type=int, default=120, help="display tile size in px")
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
        canvas, stats = render_rows(recon, cloud, images, args)
        if canvas is None:
            print(f"{args.label}: nothing to render ({stats})", flush=True)
            continue
        out = args.out_dir / f"keypoint_localization_{args.label}.jpg"
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{args.label}: wrote {out}  {stats}", flush=True)


if __name__ == "__main__":
    main()
