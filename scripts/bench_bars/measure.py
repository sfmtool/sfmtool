# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Read the bench's track-stage scores on planted wrong views, near misses and
blurred members, for measuring the track stage's ZNCC bars.

    pixi run python scripts/bench_bars/measure.py <dataset> --out <dir> [--tracks 150]

Samples ``--tracks`` points of at least four observations from the dataset
(``datasets.py``), puts each on the bench and evaluates it twice with the
bitmap rendered (``bench.evaluate(..., render_bitmap=True)``), then plants rows
on it in groups and evaluates again:

- ``near_miss``: the true pixels with the keypoint moved 2 to 6 px, in an image
  the track observes and in one it does not;
- ``true_unobs``: the unmodified pixels at the point's projection in an image
  the track does not observe;
- ``similar_other`` / ``random_other``: a substitution, a square of another
  place in the same photograph (the most similar one found, or a random other
  point's keypoint) pasted unwarped over the row's keypoint (observed image) or
  over the point's projection (unobserved image);
- ``blurred_member``: a true member whose square is blurred with a Gaussian of
  sigma 1.5 or 3 source px.

Every planted row is pinned ``out`` with ``bench.set_verdict``, so it never
joins the ``in`` set and leaves the other rows' readings unchanged. Writes one
JSON line per row, members and reference included, to ``<out>/<dataset>.jsonl``
with the row's readings: the plain and blur-matched scores against the stored
bitmap (whole, middle, and the least ninth of the grid), the bitmap's blur and
the geometry readings the other bars judge. ``analyze.py`` reads the files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import DATASETS_DIR, REPO, resolve, read_image  # noqa: E402

SCALARS = (
    "plain_zncc",
    "plain_zncc_middle",
    "blur_matched_zncc",
    "blur_matched_zncc_middle",
    "bitmap_blur_sigma",
    "seed_shift_px",
    "zncc_self_similarity_radius",
    "reprojection_error",
    "projection_offset_px",
)
GRIDS = ("plain_zncc_grid", "blur_matched_zncc_grid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True, help="directory the .jsonl is written to")
    ap.add_argument("--tracks", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument(
        "--sfmr",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="read dataset NAME from PATH instead of datasets.py's path",
    )
    args = ap.parse_args()
    out_dir = Path(args.out).resolve()
    for forbidden in (REPO / "test-data", DATASETS_DIR):
        if out_dir.is_relative_to(forbidden.resolve()):
            raise SystemExit(f"refusing to write under {forbidden}")
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = dict(s.split("=", 1) for s in args.sfmr)
    run(args.dataset, resolve(args.dataset, overrides), args.tracks, args.seed, out_dir)


def run(name: str, sfmr: Path, n_tracks: int, seed: int, out_dir: Path) -> None:
    from sfmtool import bench as B
    from sfmtool.patches import ImagePyramidSet
    from sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction

    r = SfmrReconstruction.load(str(sfmr)).clone_with_changes(patch_bitmaps=None)
    t0 = time.time()
    imgs = [read_image(r.workspace_dir, n) for n in r.image_names]
    pyr0 = ImagePyramidSet(r, imgs)
    print(name, "images loaded", round(time.time() - t0, 1), "s", flush=True)
    ed = EditedReconstruction(r)
    n_images = len(imgs)
    rot = [_quat_matrix(q) for q in np.asarray(r.quaternions_wxyz)]
    trans = np.asarray(r.translations)
    cams = [r.cameras[int(c)] for c in np.asarray(r.camera_indexes)]
    # Points in front of a pinhole camera sit at z < 0 in this convention
    # (checked against keypoints on every dataset); a fisheye accepts both.
    fisheye = ["FISHEYE" in str(c.model) for c in cams]
    counts = np.asarray(r.observation_counts)
    tii = np.asarray(r.track_image_indexes)
    tpi = np.asarray(r.track_point_indexes)
    kxy = np.asarray(r.keypoints_xy, float)
    by_image: dict[int, list[int]] = {}
    for i in range(len(tii)):
        by_image.setdefault(int(tii[i]), []).append(i)
    by_image_arr = {k: np.asarray(v) for k, v in by_image.items()}
    grey_cache: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(seed)

    def grey(im):
        g = grey_cache.get(im)
        if g is None:
            g = imgs[im].astype(np.float32).mean(axis=2)
            if len(grey_cache) > 2:
                grey_cache.clear()
            grey_cache[im] = g
        return g

    def project(im, x):
        x = np.asarray(x, float)
        xyz, w = (x[:3], x[3]) if len(x) == 4 else (x, 1.0)
        xc = rot[im] @ xyz + w * trans[im]
        if not fisheye[im] and xc[2] >= -1e-9:
            return None
        px = cams[im].ray_to_pixel_batch((xc / np.linalg.norm(xc))[None])[0]
        return px if np.all(np.isfinite(px)) else None

    def footprint(im, pl):
        w = float(pl.get("w", 1.0))
        c = project(im, np.r_[pl["center"], w])
        if c is None:
            return None
        d = 0.0
        for su in (-1, 1):
            for sv in (-1, 1):
                corner = pl["center"] + su * pl["u_halfvec"] + sv * pl["v_halfvec"]
                q = project(im, np.r_[corner, w])
                if q is None:
                    return None
                d = max(d, float(np.linalg.norm(q - c)))
        return d

    def inside(im, c, h):
        hh, ww = imgs[im].shape[:2]
        x, y = int(round(c[0] - 0.5)), int(round(c[1] - 0.5))
        return h <= x < ww - h and h <= y < hh - h

    def pick_sources(im, c, h, f, p):
        """Source squares in image ``im``: keypoints of other points plus
        random positions, each square inside the image and clear of the
        target's. Similarity is the ZNCC of the footprint-sized grey square
        against the target's own. Returns (best_xy, best_sim, best_is_kp,
        random_kp_xy, random_sim) or None."""
        idx = by_image_arr.get(im, np.zeros(0, int))
        idx = idx[tpi[idx] != p]
        pts = [kxy[j] for j in idx]
        is_kp = [True] * len(pts)
        hh, ww = imgs[im].shape[:2]
        for _ in range(1500):
            pts.append(
                np.array(
                    [rng.uniform(h + 1, ww - h - 1), rng.uniform(h + 1, hh - h - 1)]
                )
            )
            is_kp.append(False)
        pts = np.asarray(pts, float)
        is_kp = np.asarray(is_kp)
        ok = (np.max(np.abs(pts - c), axis=1) > 2 * h + 2) & np.array(
            [inside(im, q, h) for q in pts], bool
        )
        pts, is_kp = pts[ok], is_kp[ok]
        if len(pts) == 0 or not is_kp.any():
            return None
        if len(pts) > 2000:
            sel = rng.choice(len(pts), 2000, replace=False)
            pts, is_kp = pts[sel], is_kp[sel]
        g = grey(im)
        fh = max(4, int(math.ceil(f)))
        tgt = _square(g, c, fh)
        sims = np.array([_zncc(_square(g, q, fh), tgt) for q in pts])
        b = int(np.argmax(sims))
        kps = np.flatnonzero(is_kp)
        if len(kps) == 0:
            return None
        rr = int(rng.choice(kps))
        return pts[b], float(sims[b]), bool(is_kp[b]), pts[rr], float(sims[rr])

    def evaluate(t, pyr):
        t1, _ = B.evaluate(t, ed, pyr, render_bitmap=True)
        t1, _ = B.evaluate(t1, ed, pyr, render_bitmap=True)
        return t1

    cand = np.flatnonzero(counts >= 4)
    picked = rng.choice(cand, size=min(n_tracks, len(cand)), replace=False)
    out = open(out_dir / f"{name}.jsonl", "w")

    def write(d):
        out.write(json.dumps(d) + "\n")

    t0 = time.time()
    n_done = 0
    for p in picked:
        p = int(p)
        try:
            _, t = B.create_track(B.Bench(), ed, p)
            t1 = evaluate(t, pyr0)
        except Exception as e:
            write(dict(ds=name, point=p, kind="error", reason=str(e)[:200]))
            continue
        ref = t1.reference_observation
        obs = t1.observations
        n_obs = len(obs)
        base = dict(ds=name, point=p, n_obs=n_obs, ref=ref)
        for i, o in enumerate(obs):
            kind = "reference" if i == ref else "member"
            write(dict(base, kind=kind, row=i, image=o["image"], **_readings(o)))
        if ref is None:
            continue
        pl = t1.placement
        x = np.r_[np.asarray(pl["center"], float), float(pl.get("w", 1.0))]
        observed = {o["image"] for o in obs}
        obs_rows = [
            i
            for i, o in enumerate(obs)
            if i != ref and o["track"].get("keypoint") is not None
        ]
        rng.shuffle(obs_rows)
        unobs = [im for im in range(n_images) if im not in observed]
        rng.shuffle(unobs)
        # (group, kind, image, row or None, pixel, modification, info)
        plans = []
        # Group 0 changes no pixel.
        for row in obs_rows[:2]:
            kp = np.asarray(obs[row]["track"]["keypoint"], float)
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(2, 6)
            px = kp + rad * np.array([math.cos(ang), math.sin(ang)])
            plans.append(
                (
                    0,
                    "near_miss",
                    obs[row]["image"],
                    None,
                    px,
                    None,
                    dict(offset_px=rad, obs_image=True),
                )
            )
        # Unobserved images where the point projects in frame with room.
        unobs_ok = []
        for im in unobs:
            if len(unobs_ok) >= 4:
                break
            c = project(im, x)
            if c is None:
                continue
            f = footprint(im, pl)
            if f is None or f > 250:
                continue
            h = int(math.ceil(f)) + 10
            if inside(im, c, h + 2):
                unobs_ok.append((im, c, f, h))
        g0_unobs, g1_unobs = unobs_ok[:2], unobs_ok[2:]
        if len(g0_unobs) >= 1:
            im, c, f, h = g0_unobs[0]
            plans.append((0, "true_unobs", im, None, c, None, dict(foot_px=f)))
        if len(g0_unobs) >= 2:
            im, c, f, h = g0_unobs[1]
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(2, 6)
            px = c + rad * np.array([math.cos(ang), math.sin(ang)])
            plans.append(
                (
                    0,
                    "near_miss",
                    im,
                    None,
                    px,
                    None,
                    dict(offset_px=rad, obs_image=False),
                )
            )
        # Substitutions and blur on observed rows.
        m = len(obs_rows)
        for k, kind in enumerate(["similar_other", "random_other", "blur1.5", "blur3"]):
            if m == 0:
                break
            row = obs_rows[k % m]
            grp = 1 + k // m
            im = obs[row]["image"]
            kp = np.asarray(obs[row]["track"]["keypoint"], float)
            f = footprint(im, pl)
            if f is None or f > 250:
                continue
            h = int(math.ceil(f)) + 10
            if not inside(im, kp, h + (11 if kind.startswith("blur") else 2)):
                continue
            if kind.startswith("blur"):
                sigma = float(kind[4:])
                info = dict(sigma=sigma, foot_px=f, obs_image=True)
                plans.append(
                    (grp, "blurred_member", im, row, kp, ("blur", sigma, h), info)
                )
            else:
                s = pick_sources(im, kp, h, f, p)
                if s is None:
                    continue
                src, sim, skp = (
                    (s[0], s[1], s[2])
                    if kind == "similar_other"
                    else (s[3], s[4], True)
                )
                info = dict(src_sim=sim, src_is_keypoint=skp, foot_px=f, obs_image=True)
                plans.append((grp, kind, im, row, kp, ("paste", src, h), info))
        for k, (im, c, f, h) in enumerate(g1_unobs):
            kind = ["similar_other", "random_other"][k]
            s = pick_sources(im, c, h, f, p)
            if s is None:
                continue
            src, sim, skp = (
                (s[0], s[1], s[2]) if kind == "similar_other" else (s[3], s[4], True)
            )
            info = dict(src_sim=sim, src_is_keypoint=skp, foot_px=f, obs_image=False)
            plans.append((1, kind, im, None, c, ("paste", src, h), info))
        member_z = [o["track"].get("plain_zncc") for o in obs]
        for g in sorted({g for g, *_ in plans}):
            gp = [pl_ for pl_ in plans if pl_[0] == g]
            try:
                if any(pl_[5] is not None for pl_ in gp):
                    gi = list(imgs)
                    for _, kind, im, row, px, mod, info in gp:
                        if mod is not None:
                            gi[im] = _modified(
                                gi[im] if gi[im] is not imgs[im] else imgs[im].copy(),
                                imgs[im],
                                px,
                                mod,
                            )
                    pyr = ImagePyramidSet(r, gi)
                    gi = None
                else:
                    pyr = pyr0
                t2 = t1
                rows = []
                for _, kind, im, row, px, mod, info in gp:
                    if row is None:
                        t2, rep = B.add_observation(
                            t2, im, [float(px[0]), float(px[1])]
                        )
                        row = rep["observation"]
                    t2, _ = B.set_verdict(t2, row, "out")
                    rows.append((row, kind, im, info))
                t3 = evaluate(t2, pyr)
            except Exception as e:
                write(dict(base, kind="plant_error", group=g, reason=str(e)[:200]))
                continue
            del pyr
            obs3 = t3.observations
            same_ref = t3.reference_observation == ref
            for row, kind, im, info in rows:
                write(
                    dict(
                        base,
                        kind=kind,
                        group=g,
                        row=row,
                        image=im,
                        same_ref=same_ref,
                        **info,
                        **_readings(obs3[row]),
                    )
                )
            planted = {row for row, *_ in rows}
            dz = [
                abs((obs3[i]["track"].get("plain_zncc") or 0) - (member_z[i] or 0))
                for i in range(n_obs)
                if i not in planted
            ]
            if dz and max(dz) > 1e-6:
                write(
                    dict(
                        base,
                        kind="note",
                        group=g,
                        reason=f"untouched member plain_zncc moved {max(dz):.4f}",
                    )
                )
        n_done += 1
        if n_done % 50 == 0:
            print(name, n_done, "tracks", round(time.time() - t0, 1), "s", flush=True)
    out.close()
    print(
        name,
        "done",
        n_done,
        "of",
        len(picked),
        round(time.time() - t0, 1),
        "s",
        flush=True,
    )


def _quat_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _square(a, c, h):
    x, y = int(round(c[0] - 0.5)), int(round(c[1] - 0.5))
    return a[y - h : y + h + 1, x - h : x + h + 1]


def _zncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else -1.0


def _gaussian_blur(a, s):
    k = int(math.ceil(3 * s))
    x = np.arange(-k, k + 1)
    w = np.exp(-x * x / (2 * s * s))
    w /= w.sum()
    out = a.astype(np.float32)
    for ax in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[ax] = (k, k)
        padded = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for i, wi in enumerate(w):
            sl = [slice(None)] * out.ndim
            sl[ax] = slice(i, i + out.shape[ax])
            acc += wi * padded[tuple(sl)]
        out = acc
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _modified(a, original, px, mod):
    """``a`` (a copy of the image) with the square at ``px`` blurred or
    replaced by the square at the source in ``original``."""
    x0, y0 = int(round(px[0] - 0.5)), int(round(px[1] - 0.5))
    h = mod[2]
    if mod[0] == "blur":
        s = mod[1]
        e = h + int(math.ceil(3 * s)) + 1
        blk = _gaussian_blur(a[y0 - e : y0 + e + 1, x0 - e : x0 + e + 1], s)
        a[y0 - h : y0 + h + 1, x0 - h : x0 + h + 1] = blk[
            e - h : e + h + 1, e - h : e + h + 1
        ]
    else:
        sx, sy = int(round(mod[1][0] - 0.5)), int(round(mod[1][1] - 0.5))
        a[y0 - h : y0 + h + 1, x0 - h : x0 + h + 1] = original[
            sy - h : sy + h + 1, sx - h : sx + h + 1
        ]
    return a


def _readings(o):
    tr = o.get("track") or {}
    d = {k: (float(tr[k]) if tr.get(k) is not None else None) for k in SCALARS}
    for k in GRIDS:
        g = tr.get(k)
        d[k + "_min"] = (
            float(np.nanmin(np.asarray(g, float))) if g is not None else None
        )
    d["sharper_than_bitmap"] = tr.get("sharper_than_bitmap")
    d["reason"] = o.get("reason") or tr.get("reason")
    return d


if __name__ == "__main__":
    main()
