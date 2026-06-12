#!/usr/bin/env python3
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Cross-validate feature matches by comparing normalized patches.

For each descriptor match in a ``.matches`` file, render a scale/orientation-
normalized patch around each matched keypoint by warping the source image
through the keypoint's affine frame (via the Rust ``WarpMap`` / ``ImagePyramid``
remap path), then score the two patches with a (optionally windowed) normalized
cross-correlation (NCC).

This is an appearance check that is independent of the descriptor and of
two-view geometry. To gauge how informative the score is, the script reports the
NCC of three groups per run:

  - INLIERS  — matches consistent with a RANSAC fundamental matrix
  - OUTLIERS — matches inconsistent with it
  - RANDOM   — a control pairing each patch with a random keypoint patch in the
               other image

and the rank-AUC separating inliers from random and from outliers. A high
inlier-vs-random AUC means patch NCC reliably tells a real correspondence from a
non-correspondence; the inlier-vs-outlier AUC measures agreement with geometry
(which is itself imperfect, e.g. a single fundamental matrix mislabels matches on
wide-baseline orbits).

Run inside the project environment, e.g.:

    pixi run -e test python scripts/patch_crossval.py WORKSPACE [MATCHES]
    pixi run -e test python scripts/patch_crossval.py WORKSPACE --sweep
    pixi run -e test python scripts/patch_crossval.py WORKSPACE --strips out.png
    pixi run -e test python scripts/patch_crossval.py WORKSPACE \
        --strips out.png --sfmr sfmr/recon.sfmr   # 3D-patch tracks from a recon

Track strips (``--strips``) come from one of two sources: the match graph
(union-find over the ``.matches``, normalized per keypoint by its 2D affine
shape), or a reconstruction (``--sfmr``), where each track is a triangulated 3D
point rendered as one oriented surfel projected into every observing camera via
``WarpMap.from_patch`` — so the tiles are the same world patch seen from each view.

WORKSPACE is an sfm workspace with extracted ``.sift`` features and the source
images; MATCHES defaults to the first ``*.matches`` found under it.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

import sfmtool._sfmtool as s
from sfmtool._sfmtool import (
    ImagePyramid,
    OrientedPatch,
    PatchCloud,
    RigidTransform,
    WarpMap,
)


def find_sift_paths(workspace: str, names: list[str]) -> dict[str, str]:
    # Scope each lookup to the image's own directory so identically-named frames
    # in different rig sensor dirs (fisheye_left/right) don't collide.
    out = {}
    for n in names:
        base = os.path.basename(n) + ".sift"
        hits = glob.glob(
            os.path.join(workspace, os.path.dirname(n), "features", "*", base)
        )
        if hits:
            out[n] = hits[0]
    return out


def warp_map_for(pos, A, patch: int, radius: float):
    """(map_x, map_y) sampling a normalized ``patch`` x ``patch`` window around a
    keypoint, spanning +-``radius`` keypoint-frame units (affine-shape columns)."""
    u = (np.arange(patch) - patch / 2 + 0.5) * (2.0 * radius / patch)
    nx, ny = np.meshgrid(u, u)
    map_x = pos[0] + A[0, 0] * nx + A[0, 1] * ny
    map_y = pos[1] + A[1, 0] * nx + A[1, 1] * ny
    return map_x.astype(np.float32), map_y.astype(np.float32)


def gauss_window(patch: int) -> np.ndarray:
    u = np.arange(patch) - patch / 2 + 0.5
    gx, gy = np.meshgrid(u, u)
    sig = patch / 4.0
    return np.exp(-(gx**2 + gy**2) / (2 * sig**2)).ravel()


def wncc(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted normalized cross-correlation of two patches (w uniform => NCC)."""
    a, b = a.ravel(), b.ravel()
    sw = w.sum()
    da = a - (w * a).sum() / sw
    db = b - (w * b).sum() / sw
    den = np.sqrt((w * da * da).sum() * (w * db * db).sum())
    return float((w * da * db).sum() / den) if den > 1e-9 else 0.0


def wncc_color(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Per-channel weighted ZNCC averaged over channels.

    Each channel is normalized independently, so the score is invariant to a
    per-channel affine (gain/offset) — i.e. robust to per-camera white balance
    and exposure differences. ``(H, W)`` inputs fall through to luminance NCC.
    """
    if a.ndim == 2:
        return wncc(a, b, w)
    return float(np.mean([wncc(a[..., c], b[..., c], w) for c in range(a.shape[-1])]))


def auc(pos, neg) -> float:
    """Rank AUC: probability a random positive scores above a random negative."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.concatenate([pos, neg]).argsort().argsort().astype(np.float64)
    return float(
        (ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2)
        / (len(pos) * len(neg))
    )


def make_renderer(
    workspace, names, sift_paths, *, patch, radius, sampler, aniso=16, color=False
):
    """Return ``(get, patch_of)``: a cached image+keypoint loader and a patch
    renderer that warps the normalized patch around a keypoint via the Rust
    remap path (reusing one ImagePyramid per image for the aniso sampler).

    ``color`` renders 3-channel BGR patches; otherwise single-channel luminance.
    """
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    images, kpts, cache, pyrs = {}, {}, {}, {}

    def get(idx):
        if idx not in images:
            img = cv2.imread(os.path.join(workspace, names[idx]), flag)
            if img is None:
                raise FileNotFoundError(names[idx])
            images[idx] = np.ascontiguousarray(img)
            d = s.read_sift(sift_paths[names[idx]])
            kpts[idx] = (
                np.asarray(d["positions_xy"], np.float64),
                np.asarray(d["affine_shapes"], np.float64),
            )
        return images[idx], kpts[idx]

    def patch_of(idx, fidx):
        key = (idx, fidx)
        if key not in cache:
            src, (pos, aff) = get(idx)
            mx, my = warp_map_for(pos[fidx], aff[fidx], patch, radius)
            wm = WarpMap.from_numpy(mx, my)
            if sampler == "aniso":
                pyr = pyrs.get(idx) or pyrs.setdefault(idx, ImagePyramid(src))
                img = pyr.remap_aniso(wm, aniso)  # one pyramid per image, reused
            else:
                img = wm.remap_bilinear(src)
            cache[key] = np.asarray(img, np.float32)
        return cache[key]

    return get, patch_of


def evaluate(
    workspace,
    matches_path,
    *,
    patch=32,
    radius=5.0,
    sampler="bilinear",
    window="gauss",
    aniso=16,
    max_pairs=None,
    seed=0,
):
    """Score every match and return inlier/outlier/random NCC arrays."""
    m = s.read_matches(matches_path)
    names = list(m["image_names"])
    pairs = np.asarray(m["image_index_pairs"])
    counts = np.asarray(m["match_counts"])
    feat = np.asarray(m["match_feature_indexes"])
    sift_paths = find_sift_paths(workspace, names)

    get, patch_of = make_renderer(
        workspace,
        names,
        sift_paths,
        patch=patch,
        radius=radius,
        sampler=sampler,
        aniso=aniso,
    )
    w = np.ones(patch * patch) if window == "uniform" else gauss_window(patch)

    inl, outl, rand = [], [], []
    rng = np.random.default_rng(seed)
    off = 0
    npairs = 0
    for p in range(len(pairs)):
        n = int(counts[p])
        block = feat[off : off + n]
        off += n
        if n < 12 or (max_pairs and npairs >= max_pairs):
            continue
        npairs += 1
        i, j = int(pairs[p, 0]), int(pairs[p, 1])
        _, (pos_i, _) = get(i)
        _, (pos_j, _) = get(j)
        _, mask = cv2.findFundamentalMat(
            pos_i[block[:, 0]], pos_j[block[:, 1]], cv2.FM_RANSAC, 1.5, 0.999
        )
        if mask is None:
            continue
        mask = mask.ravel().astype(bool)
        nj = pos_j.shape[0]
        for k in range(n):
            fi, fj = int(block[k, 0]), int(block[k, 1])
            sc = wncc(patch_of(i, fi), patch_of(j, fj), w)
            (inl if mask[k] else outl).append(sc)
            rand.append(wncc(patch_of(i, fi), patch_of(j, int(rng.integers(nj))), w))
    return dict(inl=np.asarray(inl), outl=np.asarray(outl), rand=np.asarray(rand))


def fmt_row(label, r) -> str:
    def med(v):
        return float(np.median(v)) if len(v) else float("nan")

    return (
        f"{label:34s} in={med(r['inl']):+.3f} out={med(r['outl']):+.3f} "
        f"rand={med(r['rand']):+.3f}  AUC(in/rand)={auc(r['inl'], r['rand']):.3f} "
        f"AUC(in/out)={auc(r['inl'], r['outl']):.3f}  n={len(r['inl']) + len(r['outl'])}"
    )


def build_tracks(pairs, counts, feat):
    """Union-find over (image, feature) nodes linked by matches -> tracks."""
    parent: dict = {}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    off = 0
    for p in range(len(pairs)):
        n = int(counts[p])
        block = feat[off : off + n]
        off += n
        i, j = int(pairs[p, 0]), int(pairs[p, 1])
        for k in range(n):
            union((i, int(block[k, 0])), (j, int(block[k, 1])))

    comps: dict = {}
    for node in parent:
        comps.setdefault(find(node), []).append(node)
    return list(comps.values())


def _quat_to_rotation(wxyz):
    """Rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def strips_mode_3d(
    workspace,
    out,
    sfmr_path,
    *,
    patch,
    radius,
    window,
    min_track,
    num_tracks,
    disp,
    context=None,
    normal="mean",
    k_neighbors=12,
    refine=False,
    refine_range=25.0,
    refine_steps=7,
):
    """Render reconstruction tracks as patch strips by projecting one 3D oriented
    patch per track into each observing camera (`WarpMap.from_patch`).

    Each track is a triangulated 3D point; its surfel (center = position, normal
    per the chosen policy, world half-size = the median over views of each
    keypoint's scale back-projected to world) is rendered into every observing
    view, so the strip tiles are the same world patch seen from each camera.

    With ``refine`` the per-patch normals are optimized for cross-view
    photoconsistency by the Rust ``PatchCloud.refine_normals`` routine
    (``refine_range`` / ``refine_steps`` set its search cone and grid resolution),
    and each track is shown as a before/after pair.
    """
    recon = s.SfmrReconstruction.load(sfmr_path)
    names = list(recon.image_names)
    positions = np.asarray(recon.positions, np.float64)
    cam_idx = np.asarray(recon.camera_indexes)
    cameras = list(recon.cameras)
    quats = np.asarray(recon.quaternions_wxyz, np.float64)
    trans = np.asarray(recon.translations, np.float64)
    # Per-point normal (chosen policy) and world half-size (feature_size = each
    # observing keypoint's scale back-projected to world, median over views) from
    # the library. "geometric" is a local PCA plane fit; "mean" is the mean
    # viewing direction; "stored" is whatever estimated normal is in the .sfmr
    # (not necessarily the mean viewing direction).
    policy = {
        "mean": "mean_viewing",
        "stored": "stored",
        "geometric": "geometric",
    }[normal]
    # extent defaults to "feature_size" (factor = extent_value, default 5,
    # median over views).
    cloud = PatchCloud.from_reconstruction(
        recon, normal=policy, k_neighbors=k_neighbors, extent_value=radius
    )
    pid_normal = {
        int(p): np.asarray(cloud[i].normal, np.float64)
        for i, p in enumerate(cloud.point_ids)
    }
    pid_half = {
        int(p): float(cloud[i].half_extent[0]) for i, p in enumerate(cloud.point_ids)
    }

    cam_of = [cameras[int(cam_idx[i])] for i in range(len(names))]
    pose_of = [
        RigidTransform.from_wxyz_translation(quats[i].tolist(), trans[i].tolist())
        for i in range(len(names))
    ]
    rot_of = [_quat_to_rotation(quats[i]) for i in range(len(names))]
    images = {}

    def image(i):
        if i not in images:
            img = cv2.imread(os.path.join(workspace, names[i]), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(names[i])
            images[i] = np.ascontiguousarray(img)
        return images[i]

    # Group observations by 3D point id.
    pids = np.asarray(recon.track_point_ids).tolist()
    timg = np.asarray(recon.track_image_indexes).tolist()
    tfeat = np.asarray(recon.track_feature_indexes).tolist()
    by_point: dict = defaultdict(list)
    for pid, im, ft in zip(pids, timg, tfeat):
        by_point[pid].append((im, ft))

    def n_images(obs):
        return len(set(i for i, _ in obs))

    tracks = [(pid, obs) for pid, obs in by_point.items() if n_images(obs) >= min_track]
    tracks.sort(key=lambda kv: (n_images(kv[1]), len(kv[1])), reverse=True)
    tracks = tracks[:num_tracks]
    if not tracks:
        print(f"error: no tracks span >= {min_track} images", file=sys.stderr)
        return 1

    # Context handling mirrors strips_mode: render a larger patch, score the
    # central validated sub-patch, draw a 1px box.
    if context:
        render_patch, inner, tile = context, ((context - patch) // 2, patch), context
        extent_scale = context / patch
    else:
        render_patch, inner, tile = patch, None, disp
        extent_scale = 1.0
    w = np.ones(patch * patch) if window == "uniform" else gauss_window(patch)

    def patch_frame(pid, obs):
        """(center, up, validated_half) world-space frame for a track. The
        half-size is the library's FeatureSize (radius x median feature size)."""
        center = positions[pid]
        # Every cloud point has a half-size; the default only guards a pid that
        # isn't in the cloud (e.g. a point at infinity), which shouldn't reach here.
        half = pid_half.get(int(pid), 1e-2)
        up = rot_of[obs[0][0]].T @ np.array([0.0, -1.0, 0.0])  # first camera's up
        return center, up, half

    def make_patch(center, up, ext, n):
        return OrientedPatch.from_center_normal(
            center.tolist(), n.tolist(), up.tolist(), [ext, ext]
        )

    def render_into(p, i, res):
        wm = WarpMap.from_patch(p, cam_of[i], pose_of[i], res)
        return np.asarray(wm.remap_bilinear(image(i)), np.float32)

    # Optionally refine the rendered tracks' patch normals with the Rust routine
    # (PatchCloud.refine_normals): photometric consensus over all observing views,
    # in place and in parallel. Restricting to the displayed tracks' point ids
    # keeps it cheap on large clouds. Base normals are already in `pid_normal`;
    # read the refined normals and the Rust consensus Φ (init / refined, scored on
    # the same frozen support, so Δ >= 0) back per point id.
    pid_refined = pid_normal
    pid_phi: dict = {}
    pid_init_phi: dict = {}
    if refine:
        all_images = [image(i) for i in range(len(names))]
        res = cloud.refine_normals(
            recon,
            all_images,
            resolution=patch,
            angular_range_deg=refine_range,
            init_steps=refine_steps,
            point_ids=[int(pid) for pid, _ in tracks],
        )
        pid_refined = {
            int(p): np.asarray(cloud[i].normal, np.float64)
            for i, p in enumerate(cloud.point_ids)
        }
        pid_phi = {
            int(p): float(res["photoconsistency"][i])
            for i, p in enumerate(cloud.point_ids)
        }
        pid_init_phi = {
            int(p): float(res["init_photoconsistency"][i])
            for i, p in enumerate(cloud.point_ids)
        }

    rows = []
    deltas = []
    print(f"{len(tracks)} tracks (>= {min_track} images), labeled by image index:")
    for ti, (pid, obs) in enumerate(tracks):
        base_n = pid_normal.get(int(pid))
        if base_n is None or np.linalg.norm(base_n) < 0.5:
            base_n = np.array([0.0, 0.0, 1.0])
        center, up, half = patch_frame(pid, obs)

        if refine:
            best_n = pid_refined.get(int(pid), base_n)
            base_phi = pid_init_phi.get(int(pid), float("nan"))
            ref_phi = pid_phi.get(int(pid), float("nan"))
            if np.isfinite(base_phi) and np.isfinite(ref_phi):
                deltas.append(ref_phi - base_phi)
        else:
            best_n, base_phi, ref_phi = base_n, None, None

        ext = half * extent_scale

        def patch_of(i, _f, _p=make_patch(center, up, ext, best_n)):
            return render_into(_p, i, render_patch)

        def base_patch_of(i, _f, _p=make_patch(center, up, ext, base_n)):
            return render_into(_p, i, render_patch)

        if refine:
            # Before (base normal) row, then after (refined) row.
            b_strip, _, _ = render_track_strip(
                obs, base_patch_of, w, tile=tile, inner=inner
            )
            r_strip, _, nobs = render_track_strip(
                obs, patch_of, w, tile=tile, inner=inner
            )
            rows.extend((b_strip, r_strip))
            print(
                f"  track {ti:3d}: {nobs:2d} obs  consensus Φ base={base_phi:+.3f} "
                f"refined={ref_phi:+.3f}  Δ={ref_phi - base_phi:+.3f}"
            )
        else:
            strip, mean_ncc, nobs = render_track_strip(
                obs, patch_of, w, tile=tile, inner=inner
            )
            rows.append(strip)
            print(f"  track {ti:3d}: {nobs:2d} obs  mean pairwise NCC {mean_ncc:+.3f}")

    if refine and deltas:
        print(
            f"normal refinement: mean ΔΦ = {np.mean(deltas):+.3f} over {len(deltas)} tracks"
        )

    width = max(r.shape[1] for r in rows)
    sep_row = np.full((2, width, 3), 40, np.uint8)
    padded = []
    for r in rows:
        if r.shape[1] < width:
            r = np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)])
        padded.extend((r, sep_row))
    montage = np.vstack(padded[:-1])
    cv2.imwrite(out, montage)
    print(f"wrote {out}  ({montage.shape[1]}x{montage.shape[0]})")
    return 0


def render_track_strip(track, patch_of, w, *, tile, inner=None, sep=2):
    """Render one track's observations as a horizontal BGR patch strip.

    Patches are rendered at the renderer's size and displayed at ``tile`` px. If
    ``inner = (offset, size)`` (context mode), NCC is scored on that central
    validated sub-patch and a 1px box is drawn around it; otherwise the whole
    patch is scored. Observations are ordered and labeled by image index.
    """
    obs = sorted(track, key=lambda nd: (nd[0], nd[1]))
    patches = [patch_of(i, f) for (i, f) in obs]

    # Score (per-channel color) NCC on the validated sub-patch (context) or the
    # whole patch; keep channels so chrominance counts.
    if inner is not None:
        off, sz = inner
        cores = [p[off : off + sz, off : off + sz] for p in patches]
    else:
        cores = patches
    sims = [
        wncc_color(cores[a], cores[b], w)
        for a in range(len(cores))
        for b in range(a + 1, len(cores))
    ]
    mean_ncc = float(np.mean(sims)) if sims else float("nan")

    tiles = []
    for (img_idx, _), pf in zip(obs, patches):
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
    row = []
    for t in tiles:
        row.extend((t, sep_col))
    strip = np.hstack(row[:-1])
    return strip, mean_ncc, len(obs)


def strips_mode(
    workspace,
    out,
    names,
    tracks,
    *,
    patch,
    radius,
    sampler,
    window,
    min_track,
    num_tracks,
    disp,
    context=None,
):
    """Render the longest tracks as stacked patch strips to an image file."""
    sift_paths = find_sift_paths(workspace, names)
    # Context mode: render a larger patch at the same sampling density and mark
    # the validated extent with a 1px box; score NCC on that inner sub-patch.
    if context:
        render_patch, render_radius = context, radius * context / patch
        inner, tile = ((context - patch) // 2, patch), context
    else:
        render_patch, render_radius = patch, radius
        inner, tile = None, disp
    _, patch_of = make_renderer(
        workspace,
        names,
        sift_paths,
        patch=render_patch,
        radius=render_radius,
        sampler=sampler,
        color=True,
    )
    w = np.ones(patch * patch) if window == "uniform" else gauss_window(patch)

    def n_images(t):
        return len(set(img for img, _ in t))

    tracks = [t for t in tracks if n_images(t) >= min_track]
    tracks.sort(key=lambda t: (n_images(t), len(t)), reverse=True)
    tracks = tracks[:num_tracks]
    if not tracks:
        print(f"error: no tracks span >= {min_track} images", file=sys.stderr)
        return 1

    rows = []
    print(f"{len(tracks)} tracks (>= {min_track} images), labeled by image index:")
    for ti, t in enumerate(tracks):
        strip, mean_ncc, nobs = render_track_strip(
            t, patch_of, w, tile=tile, inner=inner
        )
        rows.append(strip)
        print(f"  track {ti:3d}: {nobs:2d} obs  mean pairwise NCC {mean_ncc:+.3f}")

    width = max(r.shape[1] for r in rows)
    sep_row = np.full((2, width, 3), 40, np.uint8)
    padded = []
    for r in rows:
        if r.shape[1] < width:
            r = np.hstack([r, np.zeros((r.shape[0], width - r.shape[1], 3), np.uint8)])
        padded.extend((r, sep_row))
    montage = np.vstack(padded[:-1])
    cv2.imwrite(out, montage)
    print(f"wrote {out}  ({montage.shape[1]}x{montage.shape[0]})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "workspace", help="sfm workspace dir (with .sift features + images)"
    )
    ap.add_argument("matches", nargs="?", help="a .matches file (default: first found)")
    ap.add_argument("--patch", type=int, default=32, help="patch side in pixels")
    ap.add_argument(
        "--radius",
        type=float,
        default=5.0,
        help="half-extent in keypoint units for 2D warps; the feature-size "
        "multiplier for 3D --sfmr patches (default 5)",
    )
    ap.add_argument("--sampler", choices=["bilinear", "aniso"], default="bilinear")
    ap.add_argument("--window", choices=["uniform", "gauss"], default="gauss")
    ap.add_argument(
        "--max-pairs", type=int, default=None, help="cap image pairs scored"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="run a parameter sweep")
    ap.add_argument(
        "--strips", metavar="OUT.png", help="render track patch strips to an image"
    )
    ap.add_argument(
        "--sfmr",
        metavar="RECON.sfmr",
        help="build --strips tracks from a reconstruction's 3D points "
        "instead of the match graph",
    )
    ap.add_argument(
        "--min-track", type=int, default=5, help="min images per track (--strips)"
    )
    ap.add_argument(
        "--num-tracks", type=int, default=24, help="tracks to render (--strips)"
    )
    ap.add_argument(
        "--disp", type=int, default=64, help="display px per patch (--strips)"
    )
    ap.add_argument(
        "--normal",
        choices=["mean", "stored", "geometric"],
        default="mean",
        help="patch normal policy for --sfmr (geometric = local PCA plane fit)",
    )
    ap.add_argument(
        "--k-neighbors", type=int, default=12, help="neighbors for --normal geometric"
    )
    ap.add_argument(
        "--refine-normal",
        action="store_true",
        help="optimize each --sfmr patch normal for cross-view photoconsistency",
    )
    ap.add_argument(
        "--context",
        type=int,
        default=None,
        metavar="N",
        help="render NxN context patches with a 1px box on the validated extent "
        "(--strips); scored on the inner --patch region",
    )
    args = ap.parse_args(argv)

    if args.context is not None and args.context < args.patch:
        print(
            f"error: --context ({args.context}) must be >= --patch ({args.patch})",
            file=sys.stderr,
        )
        return 1

    matches = args.matches
    if matches is None and not (args.strips and args.sfmr):
        hits = sorted(
            glob.glob(os.path.join(args.workspace, "**", "*.matches"), recursive=True)
        )
        if not hits:
            print(f"error: no .matches under {args.workspace}", file=sys.stderr)
            return 1
        matches = hits[0]

    if args.strips:
        if args.sfmr:
            return strips_mode_3d(
                args.workspace,
                args.strips,
                args.sfmr,
                patch=args.patch,
                radius=args.radius,
                window=args.window,
                min_track=args.min_track,
                num_tracks=args.num_tracks,
                disp=args.disp,
                context=args.context,
                normal=args.normal,
                k_neighbors=args.k_neighbors,
                refine=args.refine_normal,
            )
        m = s.read_matches(matches)
        names = list(m["image_names"])
        tracks = build_tracks(
            np.asarray(m["image_index_pairs"]),
            np.asarray(m["match_counts"]),
            np.asarray(m["match_feature_indexes"]),
        )
        return strips_mode(
            args.workspace,
            args.strips,
            names,
            tracks,
            patch=args.patch,
            radius=args.radius,
            sampler=args.sampler,
            window=args.window,
            min_track=args.min_track,
            num_tracks=args.num_tracks,
            disp=args.disp,
            context=args.context,
        )

    common = dict(max_pairs=args.max_pairs, seed=args.seed)
    if args.sweep:
        print(f"=== sweep on {matches} ===")
        for label, kw in [
            ("patch=16 radius=5 bilinear gauss", dict(patch=16)),
            ("patch=32 radius=3 bilinear gauss", dict(radius=3.0)),
            ("patch=32 radius=5 bilinear gauss", dict()),
            ("patch=32 radius=8 bilinear gauss", dict(radius=8.0)),
            ("patch=32 radius=5 bilinear uniform", dict(window="uniform")),
            ("patch=32 radius=5 aniso gauss", dict(sampler="aniso")),
        ]:
            print(fmt_row(label, evaluate(args.workspace, matches, **kw, **common)))
    else:
        r = evaluate(
            args.workspace,
            matches,
            patch=args.patch,
            radius=args.radius,
            sampler=args.sampler,
            window=args.window,
            **common,
        )
        print(
            fmt_row(
                f"patch={args.patch} radius={args.radius} {args.sampler} {args.window}",
                r,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
