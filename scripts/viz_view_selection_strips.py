# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Visualize ``PatchCloud.select_views`` as patch strips.

A *patch strip* renders one 3D point's surfel into each relevant view and lays
the tiles in a row, so the photometric view selection can be eyeballed:

- **yellow** ``trk`` — the point's track views
- **green**  ``add`` — candidates the selector *admitted* (with their
  ``select_views`` ZNCC score)
- **red**    ``rej`` — geometrically-visible candidates (cheirality +
  ``is_front_facing`` + in-frame) that were *not* admitted, scored as windowed
  ZNCC against the track-consensus template for contrast

Rows are chosen to prefer points that have rejected candidates (so the
admitted-vs-rejected contrast is visible), then expanded points, then the rest.
One montage image is written per input reconstruction.

This is a dev/inspection tool, not a test — the automated coverage lives in
``tests/test_patch_view_selection.py``. See
``specs/core/patch-view-selection.md``.

Example::

    pixi run python scripts/viz_view_selection_strips.py \\
        seoul_bull_ws/sfmr/*.sfmr kerry_park_ws/sfmr/*.sfmr --out-dir /tmp/strips
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from sfmtool._sfmtool import (
    OrientedPatch,
    PatchCloud,
    SfmrReconstruction,
)
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


def track_views(recon) -> dict[int, set[int]]:
    pids = np.asarray(recon.track_point_ids)
    imgs = np.asarray(recon.track_image_indexes)
    tracks: dict[int, set[int]] = {}
    for pid, im in zip(pids.tolist(), imgs.tolist()):
        tracks.setdefault(int(pid), set()).add(int(im))
    return tracks


def geometric_candidates(
    recon, patch, point_xyz, rot, t, quats, cams, cam_idx, w=1.0
) -> set[int]:
    """Image indices that geometrically see the surfel: point in front of the
    camera, patch front-facing, and the projection inside the frame — the same
    gate the keypoint pipeline uses (see tests/test_patch_view_selection.py).

    For a point at infinity (`w == 0`) `point_xyz` is a direction `d`: the
    camera-frame point is `R·d` with no translation (every ray to it is parallel
    to `d`), and cheirality is `(R·d).z > 0`."""
    out = set()
    for i in range(recon.image_count):
        x_cam = rot[i] @ point_xyz + (t[i] if w != 0.0 else 0.0)
        if x_cam[2] <= 0:
            continue
        pose = RigidTransform.from_wxyz_translation(quats[i].tolist(), t[i].tolist())
        if not patch.is_front_facing(pose):
            continue
        cam = cams[int(cam_idx[i])]
        u, v = cam.project(x_cam[0] / x_cam[2], x_cam[1] / x_cam[2])
        if 0 <= u < cam.width and 0 <= v < cam.height:
            out.add(i)
    return out


def gauss_window(n: int) -> np.ndarray:
    u = np.arange(n) - n / 2 + 0.5
    gx, gy = np.meshgrid(u, u)
    return np.exp(-(gx**2 + gy**2) / (2 * (n / 4.0) ** 2)).ravel()


def znorm(tile: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-channel z-normalized vector with sqrt(window) folded in -> (C, P), so
    a dot of two such vectors is a windowed ZNCC (mirrors the Rust convention)."""
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


def _chip(img, text, org, color, scale=0.3):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + 2), (15, 15, 15), -1)
    cv2.putText(
        img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def _draw_tile(tile, idx, score, color, kind, tile_px):
    p8 = np.clip(tile, 0, 255).astype(np.uint8)
    bgr = p8 if p8.ndim == 3 else cv2.cvtColor(p8, cv2.COLOR_GRAY2BGR)
    bgr = cv2.resize(bgr, (tile_px, tile_px), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(bgr, (0, 0), (tile_px - 1, tile_px - 1), color, 3)
    _chip(bgr, f"{kind}{idx}", (3, 12), color)
    if score is not None:
        _chip(bgr, f"{score:+.2f}", (3, tile_px - 5), color)
    return bgr


def _infinity_first_sample(recon, ids, sample_size, rng):
    """A point-id sample that includes ALL points at infinity plus random finite
    points up to ``sample_size`` — guaranteeing infinity points reach row
    selection even when they are a tiny fraction of the cloud. (The montage row
    mix is balanced separately in ``render_strips``.)"""
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
    return (inf_ids + fin)[:sample_size]


def render_strips(recon, cloud, images, args) -> tuple[np.ndarray | None, dict]:
    """Build the montage canvas (or None if nothing to show) and a stats dict."""
    positions = np.asarray(recon.positions, dtype=np.float64)
    rot = rotation_matrices(recon)
    t = np.asarray(recon.translations, dtype=np.float64)
    quats = np.asarray(recon.quaternions_wxyz, dtype=np.float64)
    cams = recon.cameras
    cam_idx = np.asarray(recon.camera_indexes)
    tracks = track_views(recon)
    pid_to_patch = {int(p): i for i, p in enumerate(np.asarray(cloud.point_ids))}

    ids = np.asarray(cloud.point_ids)
    rng = np.random.default_rng(args.seed)
    if args.prioritize_infinity:
        sample = _infinity_first_sample(recon, ids, args.sample, rng)
    else:
        sample = np.sort(
            rng.choice(ids, size=min(args.sample, len(ids)), replace=False)
        ).tolist()
    results = cloud.select_views(
        recon,
        images,
        point_ids=sample,
        resolution=args.resolution,
        min_relative_zncc=args.min_relative_zncc,
    )
    res_by_pid = {int(r["point_id"]): r for r in results}
    w = gauss_window(args.resolution)

    def patch_geo(patch_obj):
        # Preserve the homogeneous weight: a point at infinity (w == 0) rebuilds
        # as a tangent-sphere patch so WarpMap renders its direction corners.
        if float(patch_obj.w) == 0.0:
            return OrientedPatch.from_infinity_direction(
                list(patch_obj.center),
                list(patch_obj.u_axis),
                list(patch_obj.half_extent),
            )
        return OrientedPatch.from_center_normal(
            list(patch_obj.center),
            list(patch_obj.normal),
            list(patch_obj.u_axis),
            list(patch_obj.half_extent),
        )

    # Classify each sampled point's visible candidates into track/admitted/rejected.
    rows_info = []
    for pid in sample:
        pid = int(pid)
        r = res_by_pid[pid]
        admitted = {int(a) for a in r["admitted"]}
        scores = {int(a): float(s) for a, s in zip(r["admitted"], r["scores"])}
        patch_obj = cloud[pid_to_patch[pid]]
        cand = geometric_candidates(
            recon,
            patch_geo(patch_obj),
            positions[pid],
            rot,
            t,
            quats,
            cams,
            cam_idx,
            float(patch_obj.w),
        )
        track = tracks[pid]
        rows_info.append(
            dict(
                pid=pid,
                track=track,
                admitted=admitted,
                scores=scores,
                sa=float(r["self_agreement"]),
                rejected=sorted(cand - admitted),
                expanded=len(admitted) > len(track),
                patch=patch_obj,
            )
        )

    # Prefer rows with rejected candidates, then expanded, then the rest.
    with_rej = [x for x in rows_info if x["rejected"]]
    expanded_only = [x for x in rows_info if not x["rejected"] and x["expanded"]]
    rest = [x for x in rows_info if not x["rejected"] and not x["expanded"]]
    for group in (with_rej, expanded_only, rest):
        rng.shuffle(group)
    ordered = with_rej + expanded_only + rest
    if args.prioritize_infinity:
        # Balance the montage: interleave infinity and finite rows (infinity
        # leading each pair) so both kinds show even though the grouping above
        # would otherwise bury the rare infinity points.
        from itertools import zip_longest

        is_inf = np.asarray(recon.point_is_at_infinity)
        inf_rows = [x for x in ordered if is_inf[int(x["pid"])]]
        fin_rows = [x for x in ordered if not is_inf[int(x["pid"])]]
        ordered = [
            x for pair in zip_longest(inf_rows, fin_rows) for x in pair if x is not None
        ]
    chosen = ordered[: args.rows]

    rej_scores_all = []
    rendered = []
    for x in chosen:
        patch = patch_geo(x["patch"])

        def render(i, _patch=patch):
            pose = RigidTransform.from_wxyz_translation(
                quats[i].tolist(), t[i].tolist()
            )
            wm = WarpMap.from_patch(
                _patch, cams[int(cam_idx[i])], pose, args.resolution
            )
            return np.asarray(wm.remap_bilinear(images[i]), dtype=np.float32)

        track_full = sorted(x["track"])
        track_vecs = [znorm(render(i), w) for i in track_full]
        tmpl_unit = None
        if track_vecs:
            tmpl = np.mean(track_vecs, axis=0)
            tn = np.sqrt((tmpl**2).sum(1, keepdims=True))
            tmpl_unit = tmpl / np.where(tn > 1e-9, tn, 1.0)

        def to_template(i, _tmpl=tmpl_unit):
            if _tmpl is None:
                return None
            return float((znorm(render(i), w) * _tmpl).sum(1).mean())

        tiles = []
        for i in track_full[: args.max_track]:
            tiles.append(
                _draw_tile(
                    render(i), i, to_template(i), (0, 220, 220), "trk", args.tile
                )
            )
        vetted = sorted(a for a in x["admitted"] if a not in x["track"])
        for i in vetted[: args.max_add]:
            tiles.append(
                _draw_tile(
                    render(i), i, x["scores"].get(i), (0, 200, 0), "add", args.tile
                )
            )
        for i in x["rejected"][: args.max_rej]:
            s = to_template(i)
            rej_scores_all.append(s)
            tiles.append(_draw_tile(render(i), i, s, (0, 0, 230), "rej", args.tile))
        rendered.append((x, tiles))

    stats = _summarize(rows_info, rej_scores_all, args)
    if not rendered:
        return None, stats

    canvas = _compose(rendered, rows_info, args)
    return canvas, stats


def _summarize(rows_info, rej_scores_all, args) -> dict:
    adm = [
        s
        for x in rows_info
        for k, s in x["scores"].items()
        if k not in x["track"] and s is not None and np.isfinite(s)
    ]
    rej = [s for s in rej_scores_all if s is not None and np.isfinite(s)]
    sa = [x["sa"] for x in rows_info if np.isfinite(x["sa"])]
    return dict(
        sample_size=len(rows_info),
        n_expanded=sum(1 for x in rows_info if x["expanded"]),
        n_with_rej=sum(1 for x in rows_info if x["rejected"]),
        adm_range=(min(adm), max(adm)) if adm else None,
        rej_range=(min(rej), max(rej)) if rej else None,
        sa_range=(min(sa), max(sa)) if sa else None,
    )


def _compose(rendered, rows_info, args) -> np.ndarray:
    max_tiles = max(len(tiles) for _, tiles in rendered)
    header_w, gap = 360, 4
    width = header_w + max_tiles * (args.tile + gap)
    total_h = (args.tile + gap) * len(rendered) + 60
    canvas = np.full((total_h, width, 3), 28, np.uint8)

    title = (
        f"select_views patch strips: {args.label}  "
        f"(sample={len(rows_info)}, RES={args.resolution}, min_rel={args.min_relative_zncc})"
    )
    cv2.putText(
        canvas,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    legend = "yellow=track  green=admitted(+add)  red=rejected(visible, not admitted)"
    cv2.putText(
        canvas,
        legend,
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    y = 56
    for x, tiles in rendered:
        sa_str = f"{x['sa']:.2f}" if np.isfinite(x["sa"]) else "nan"
        n_add = len([a for a in x["admitted"] if a not in x["track"]])
        cv2.putText(
            canvas,
            f"pt {x['pid']}  trk={len(x['track'])} +add={n_add}",
            (8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"rej={len(x['rejected'])}  self_agr={sa_str}",
            (8, y + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        xoff = header_w
        for tile in tiles:
            canvas[y : y + args.tile, xoff : xoff + args.tile] = tile
            xoff += args.tile + gap
        y += args.tile + gap
    return canvas


def _label_for(path: Path, recon) -> str:
    """A short dataset label: the workspace dir name minus a trailing '_ws',
    else the .sfmr stem."""
    ws = Path(recon.workspace_dir).name
    if ws:
        return ws[:-3] if ws.endswith("_ws") else ws
    return path.stem


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sfmr", nargs="+", type=Path, help="one or more solved .sfmr files")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="directory for the montage JPEGs",
    )
    p.add_argument(
        "--rows", type=int, default=9, help="points (rows) to show per montage"
    )
    p.add_argument(
        "--prioritize-infinity",
        action="store_true",
        help="order rows so points at infinity (w=0) lead the montage",
    )
    p.add_argument(
        "--sample", type=int, default=400, help="random points to evaluate per recon"
    )
    p.add_argument("--min-relative-zncc", type=float, default=0.7)
    p.add_argument(
        "--resolution", type=int, default=24, help="patch render/score grid (R x R)"
    )
    p.add_argument("--tile", type=int, default=56, help="display tile size in px")
    p.add_argument(
        "--max-track", type=int, default=4, help="max track tiles shown per row"
    )
    p.add_argument(
        "--max-add", type=int, default=4, help="max admitted tiles shown per row"
    )
    p.add_argument(
        "--max-rej", type=int, default=3, help="max rejected tiles shown per row"
    )
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
        canvas, stats = render_strips(recon, cloud, images, args)
        if canvas is None:
            print(f"{args.label}: nothing to render ({stats})", flush=True)
            continue
        out = args.out_dir / f"view_selection_strips_{args.label}.jpg"
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{args.label}: wrote {out}  {stats}", flush=True)


if __name__ == "__main__":
    main()
