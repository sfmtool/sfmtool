# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the track stage's ZNCC bars on the rows ``measure.py`` wrote, on
the plain score and on the blur-matched score side by side.

    pixi run python scripts/bench_bars/analyze.py <data dir> [--out <file>]

Reads ``<data dir>/<dataset>.jsonl`` for the eight datasets and prints the
tables (and writes them to ``--out`` when given).

A row is judged when it carries both a plain and a blur-matched score, so the
two scores are compared on the same rows. A row fails a geometry bar when its
projection error (reprojection error, or projection offset where the point was
not refitted) is over 3 px, its seed shift over 6 px, or its ZNCC
self-similarity radius over 2.5. A row passes the ZNCC bars of a score at
(whole, middle) when its whole score is at least ``whole`` and, with a middle
bar on (``middle > 0``), its middle score is at least ``middle``.

The objective is the ZNCC bars' own measure: the mean over tracks of the
average of two shares, over rows that clear every geometry bar:

- members kept: of the track's judged members (blurred members included, since
  a blurred view of the right place is a view to keep), the share that clears
  the ZNCC bars;
- substitutions turned out: of the track's judged substitutions
  (``similar_other``, ``random_other``) planted in an image the track observes,
  the share that fails a ZNCC bar.

A row that fails a geometry bar is out whatever the ZNCC bars are, so it says
nothing about them. Substitutions in images the track does not observe are
reported apart: there the unmodified true pixels at the projection already fail
the ZNCC bars often (``true_unobs``). The sensitivity sections count other sets
of rows.

The grid is the whole bar from 0.50 to 0.85 in steps of 0.05 and the middle bar
off or from 0.30 up to the whole bar in steps of 0.05. The pick is held out by
reconstruction: the grid's best on seven reconstructions is scored on the
eighth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import DATASETS, NAMES  # noqa: E402

SCORES = {
    "plain": ("plain_zncc", "plain_zncc_middle"),
    "blur": ("blur_matched_zncc", "blur_matched_zncc_middle"),
}
SCORE_LABEL = {"plain": "plain", "blur": "blur-matched"}
WRONG = ("similar_other", "random_other")
KEEP = ("member", "blurred_member")
WHOLES = [round(w, 2) for w in np.arange(0.50, 0.851, 0.05)]
GRID = [
    (w, m)
    for w in WHOLES
    for m in [0.0] + [round(x, 2) for x in np.arange(0.30, w + 1e-9, 0.05)]
]
GI = {g: i for i, g in enumerate(GRID)}
BANDS = (
    (-1, 0.5, "below 0.5"),
    (0.5, 0.7, "0.5 to 0.7"),
    (0.7, 0.85, "0.7 to 0.85"),
    (0.85, 1.01, "0.85 and above"),
)
# The columns of the per-band and per-reconstruction tables.
COLS = [
    ("plain 0.60", "plain", 0.60, 0.0),
    ("plain 0.65*", "plain", 0.65, 0.0),
    ("plain 0.70", "plain", 0.70, 0.0),
    ("blur 0.60", "blur", 0.60, 0.0),
    ("blur 0.65", "blur", 0.65, 0.0),
    ("blur 0.70", "blur", 0.70, 0.0),
    ("blur 0.75", "blur", 0.75, 0.0),
    ("blur .70/.30", "blur", 0.70, 0.30),
    ("blur .70/.50", "blur", 0.70, 0.50),
]
# The fixed (whole, middle) bars the held-out tables score beside the pick.
FIXED = [(0.6, 0.0), (0.65, 0.0), (0.7, 0.0), (0.75, 0.0), (0.7, 0.3), (0.7, 0.5)]
LINES: list[str] = []


def P(*a):
    LINES.append(" ".join(str(x) for x in a))


def judged(r):
    return r.get("plain_zncc") is not None and r.get("blur_matched_zncc") is not None


def geo_fail(r):
    p = r.get("reprojection_error")
    if p is None:
        p = r.get("projection_offset_px")
    return {
        "proj": p is not None and p > 3.0,
        "shift": r.get("seed_shift_px") is not None and r["seed_shift_px"] > 6.0,
        "selfsim": r.get("zncc_self_similarity_radius") is not None
        and r["zncc_self_similarity_radius"] > 2.5,
    }


def zncc_pass(r, whole, middle, score):
    zk, mk = SCORES[score]
    z, m = r.get(zk), r.get(mk)
    if z is None or z < whole:
        return False
    if middle > 0 and m is not None and m < middle:
        return False
    return True


def kept(r, whole, middle, score):
    return (not r["_geo"]) and zncc_pass(r, whole, middle, score)


def pct(rs, f):
    return 100 * np.mean([f(r) for r in rs]) if rs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data_dir")
    ap.add_argument("--out", help="also write the tables to this file")
    args = ap.parse_args()
    data = Path(args.data_dir)

    rows = []
    for d in NAMES:
        rows += [json.loads(line) for line in open(data / f"{d}.jsonl")]
    for r in rows:
        if r["kind"] in KEEP + WRONG + ("near_miss", "true_unobs"):
            r["_g"] = geo_fail(r)
            r["_geo"] = any(r["_g"].values())
        r["_track"] = (r["ds"], r["point"])

    geok = lambda r: not r["_geo"]  # noqa: E731
    anyk = lambda r: True  # noqa: E731
    sel = {
        "objective: geometry-passing members; geometry-passing substitutions in observed images": (
            geok,
            lambda r: r.get("obs_image") and not r["_geo"],
        ),
        "A: every member, every bar; geometry-passing substitutions in observed images": (
            anyk,
            lambda r: r.get("obs_image") and not r["_geo"],
        ),
        "B: geometry-passing members; geometry-passing substitutions in all images": (
            geok,
            lambda r: not r["_geo"],
        ),
        "C: every member and every substitution, every bar": (anyk, lambda r: True),
        "D: similar substitutions only, observed images, geometry-passing": (
            geok,
            lambda r: (
                r["kind"] == "similar_other" and r.get("obs_image") and not r["_geo"]
            ),
        ),
    }
    obj = next(iter(sel))
    allk = "C: every member and every substitution, every bar"

    def track_table(s, score):
        ksel, wsel = s
        by = defaultdict(lambda: ([], []))
        for r in rows:
            if not judged(r):
                continue
            if r["kind"] in KEEP and ksel(r):
                by[r["_track"]][0].append(r)
            elif r["kind"] in WRONG and wsel(r):
                by[r["_track"]][1].append(r)
        tab = {}
        for t, (ks, ws) in by.items():
            k = np.array([[kept(r, w, m, score) for (w, m) in GRID] for r in ks], float)
            o = np.array(
                [[not kept(r, w, m, score) for (w, m) in GRID] for r in ws], float
            )
            tab[t] = (k.mean(0) if len(k) else None, o.mean(0) if len(o) else None)
        return tab

    def halves(tab, dss):
        ks = [v[0] for t, v in tab.items() if t[0] in dss and v[0] is not None]
        os_ = [v[1] for t, v in tab.items() if t[0] in dss and v[1] is not None]
        return 100 * np.mean(ks, 0), 100 * np.mean(os_, 0), len(ks), len(os_)

    def bal(tab, dss):
        k, o, _, _ = halves(tab, dss)
        return (k + o) / 2

    tabs = {(k, sc): track_table(s, sc) for k, s in sel.items() for sc in SCORES}

    # ------------------------------------------------------------ header
    P("=" * 78)
    P(
        "Track-stage ZNCC bars, plain and blur-matched scores (scripts/bench_bars/analyze.py)"
    )
    P("Data:")
    for d in NAMES:
        f = data / f"{d}.jsonl"
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        P(
            f"  {d + '.jsonl':15s} sha256 {h}  lines {sum(1 for _ in open(f))}  ({DATASETS[d].label})"
        )

    # ------------------------------------------------------------ 0
    P("")
    P("== 0. Counts per dataset (tracks / tracks with plants; rows all/judged) ==")
    kinds = [
        "member",
        "blurred_member",
        "similar_other",
        "random_other",
        "near_miss",
        "true_unobs",
    ]
    P(
        f"{'dataset':10s} {'tracks':>6s} {'w/plant':>7s} "
        + " ".join(f"{k[:14]:>15s}" for k in kinds)
    )
    for d in NAMES + ["ALL"]:
        rs = [r for r in rows if d == "ALL" or r["ds"] == d]
        tr = {r["_track"] for r in rs if r["kind"] in ("member", "reference")}
        trp = {
            r["_track"]
            for r in rs
            if r["kind"] in WRONG + ("blurred_member", "near_miss")
        }
        cells = [
            f"{sum(r['kind'] == k for r in rs)}/{sum(r['kind'] == k and judged(r) for r in rs)}"
            for k in kinds
        ]
        P(
            f"{d:10s} {len(tr):>6d} {len(trp):>7d} "
            + " ".join(f"{c:>15s}" for c in cells)
        )
    P(
        "error / plant_error / note rows:",
        {
            k: sum(r["kind"] == k for r in rows)
            for k in ("error", "plant_error", "note")
        },
    )
    P("Judged rows read against a blurred bitmap (bitmap_blur_sigma > 0), % (n):")
    for k in kinds:
        rk = [r for r in rows if r["kind"] == k and judged(r)]
        blurred = [r for r in rk if (r.get("bitmap_blur_sigma") or 0) > 0]
        sig = [r["bitmap_blur_sigma"] for r in blurred]
        med = f", median sigma {np.median(sig):.2f} grid px" if sig else ""
        P(
            f"  {k:16s} {pct(rk, lambda r: (r.get('bitmap_blur_sigma') or 0) > 0):5.1f} (n={len(rk)}){med}"
        )
    for s in (1.5, 3.0):
        rk = [
            r
            for r in rows
            if r["kind"] == "blurred_member" and r.get("sigma") == s and judged(r)
        ]
        P(
            f"  blurred s={s:<4}     {pct(rk, lambda r: (r.get('bitmap_blur_sigma') or 0) > 0):5.1f} (n={len(rk)})"
        )

    # ------------------------------------------------------------ 1
    P("")
    P(
        "== 1. Geometry bars on judged rows: % failing projection>3px / shift>6px / self-sim>2.5 / any =="
    )
    for k in kinds:
        for oi in (
            (None,)
            if k in ("member", "blurred_member", "true_unobs")
            else (True, False)
        ):
            rk = [
                r
                for r in rows
                if r["kind"] == k
                and judged(r)
                and (oi is None or bool(r.get("obs_image")) == oi)
            ]
            lab = k + (
                "" if oi is None else (" (observed img)" if oi else " (unobserved img)")
            )
            f = lambda key: pct(rk, lambda r: r["_g"][key])  # noqa: E731
            P(
                f"  {lab:32s} n={len(rk):5d}  proj {f('proj'):5.1f}  shift {f('shift'):5.1f}  selfsim {f('selfsim'):5.1f}  any {pct(rk, lambda r: r['_geo']):5.1f}"
            )
    nm = [r for r in rows if r["kind"] == "near_miss" and judged(r)]
    for lo, hi in ((2, 3), (3, 4), (4, 6.01)):
        rk = [r for r in nm if lo <= r["offset_px"] < hi]
        P(
            f"  near_miss offset [{lo},{hi:.0f}) px n={len(rk):4d}: projection bar {pct(rk, lambda r: r['_g']['proj']):5.1f}  any geometry bar {pct(rk, lambda r: r['_geo']):5.1f}"
        )
    fp = np.array(
        [
            r["foot_px"]
            for r in rows
            if r["kind"] in WRONG and r.get("foot_px") is not None
        ]
    )
    if len(fp):
        P(
            f"  substitutions' footprint radius, source px: median {np.median(fp):.1f}, 10th pct {np.percentile(fp, 10):.1f}, 90th pct {np.percentile(fp, 90):.1f}"
        )

    # ------------------------------------------------------------ 2
    P("")
    P(
        "== 2. ZNCC bars alone on judged geometry-passing rows: % turned out (substitutions) or lost (true rows) =="
    )
    P("   middle bar off in every column; * = the current default (plain 0.65)")
    hdr = f"  {'':26s} {'n':>5s} " + " ".join(f"{c[0]:>11s}" for c in COLS)
    for title, osel in (
        ("observed images (the objective's)", lambda r: r.get("obs_image")),
        ("unobserved images (reported apart)", lambda r: not r.get("obs_image")),
        ("all images", lambda r: True),
    ):
        P(f" -- substitutions, {title}, similarity band --")
        P(hdr)
        for lo, hi, lab in BANDS + ((-1, 1.01, "all bands"),):
            rk = [
                r
                for r in rows
                if r["kind"] in WRONG
                and lo <= r["src_sim"] < hi
                and not r["_geo"]
                and osel(r)
                and judged(r)
            ]
            P(
                f"  {lab:26s} {len(rk):5d} "
                + " ".join(
                    f"{pct(rk, lambda r: not zncc_pass(r, w, m, sc)):11.1f}"
                    for _, sc, w, m in COLS
                )
            )
    P(" -- true rows, lost --")
    P(hdr)
    for lab, f in (
        ("member", lambda r: r["kind"] == "member"),
        (
            "  member read blurred",
            lambda r: r["kind"] == "member" and (r.get("bitmap_blur_sigma") or 0) > 0,
        ),
        ("blurred_member", lambda r: r["kind"] == "blurred_member"),
        (
            "  blurred s=1.5",
            lambda r: r["kind"] == "blurred_member" and r.get("sigma") == 1.5,
        ),
        (
            "  blurred s=3",
            lambda r: r["kind"] == "blurred_member" and r.get("sigma") == 3.0,
        ),
        ("true_unobs", lambda r: r["kind"] == "true_unobs"),
    ):
        rk = [r for r in rows if f(r) and not r["_geo"] and judged(r)]
        P(
            f"  {lab:26s} {len(rk):5d} "
            + " ".join(
                f"{pct(rk, lambda r: not zncc_pass(r, w, m, sc)):11.1f}"
                for _, sc, w, m in COLS
            )
        )
    P(
        "  Middle score alone (whole bar off), % of judged geometry-passing rows below the middle bar:"
    )
    for mb in (0.3, 0.4, 0.5):
        cells = []
        for lab, f in (
            ("member", lambda r: r["kind"] == "member"),
            ("blurred", lambda r: r["kind"] == "blurred_member"),
            ("subst obs", lambda r: r["kind"] in WRONG and r.get("obs_image")),
        ):
            rk = [r for r in rows if f(r) and not r["_geo"] and judged(r)]
            cells.append(
                f"{lab} plain {pct(rk, lambda r: not zncc_pass(r, -9, mb, 'plain')):5.1f} blur {pct(rk, lambda r: not zncc_pass(r, -9, mb, 'blur')):5.1f}"
            )
        P(f"    middle {mb:.2f}: " + "; ".join(cells))

    # ------------------------------------------------------------ 3
    P("")
    P(
        "== 3. Per reconstruction: per-track mean of members kept / substitutions turned out =="
    )
    P(
        "   on the objective's rows (geometry-passing members; geometry-passing substitutions in observed images)"
    )
    P(
        f"  {'dataset':10s} "
        + " ".join(f"{c[0]:>13s}" for c in COLS)
        + f" {'tracks keep/out':>16s}"
    )
    for d in NAMES + ["ALL"]:
        dss = NAMES if d == "ALL" else [d]
        cells = []
        for _, sc, w, m in COLS:
            k, o, nk, no = halves(tabs[(obj, sc)], dss)
            cells.append(f"{k[GI[(w, m)]]:5.1f}/{o[GI[(w, m)]]:5.1f}")
        P(f"  {d:10s} " + " ".join(f"{c:>13s}" for c in cells) + f" {nk:>8d}/{no:<7d}")
    P(
        "  objective (average of the two, mean over tracks): "
        + ", ".join(
            f"{lab} {bal(tabs[(obj, sc)], NAMES)[GI[(w, m)]]:.2f}"
            for lab, sc, w, m in COLS
        )
    )
    P("  every bar, every member and every substitution in every image (C), keep/out:")
    for d in NAMES + ["ALL"]:
        dss = NAMES if d == "ALL" else [d]
        cells = []
        for _, sc, w, m in COLS:
            k, o, _, _ = halves(tabs[(allk, sc)], dss)
            cells.append(f"{k[GI[(w, m)]]:5.1f}/{o[GI[(w, m)]]:5.1f}")
        P(f"  {d:10s} " + " ".join(f"{c:>13s}" for c in cells))

    # ------------------------------------------------------------ 4
    def lodo(key, title):
        P("")
        P(title)
        for sc in SCORES:
            tab = tabs[(key, sc)]
            P(f" -- {SCORE_LABEL[sc]} score --")
            fixed = FIXED
            P(
                f"  {'held out':10s} {'pick':>10s} {'pick':>6s} "
                + " ".join(f"{w:.2f}/{m:.2f}"[:9].rjust(9) for w, m in fixed)
                + f" {'pick, mid off':>14s}"
            )
            res = []
            wo = [i for i, (w, m) in enumerate(GRID) if m == 0]
            for d in NAMES:
                gtr = bal(tab, [x for x in NAMES if x != d])
                gte = bal(tab, [d])
                b = int(np.argmax(gtr))
                bw = wo[int(np.argmax(gtr[wo]))]
                res.append(
                    (
                        d,
                        GRID[b],
                        gte[b],
                        *[gte[GI[f]] for f in fixed],
                        GRID[bw][0],
                        gte[bw],
                    )
                )
                P(
                    f"  {d:10s} {GRID[b][0]:.2f}/{GRID[b][1]:.2f} {gte[b]:6.1f} "
                    + " ".join(f"{gte[GI[f]]:9.1f}" for f in fixed)
                    + f" {GRID[bw][0]:7.2f} {gte[bw]:6.1f}"
                )
            nf = len(fixed)
            a = np.array([[x[2], *x[3 : 3 + nf], x[4 + nf]] for x in res])
            mean = a.mean(0)
            P(
                f"  {'mean':10s} {'':10s} {mean[0]:6.2f} "
                + " ".join(f"{v:9.2f}" for v in mean[1 : 1 + nf])
                + f" {'':7s} {mean[1 + nf]:6.2f}"
            )
            picks = defaultdict(int)
            wpicks = defaultdict(int)
            for x in res:
                picks[f"{x[1][0]:.2f}/{x[1][1]:.2f}"] += 1
                wpicks[f"{x[3 + nf]:.2f}"] += 1
            P(f"  picks (whole/middle): {dict(picks)}; whole bar alone: {dict(wpicks)}")
            g = bal(tab, NAMES)
            P(f"  all data: best {GRID[int(np.argmax(g))]} {g.max():.2f}")
            P(
                "  whole bar alone, all data: "
                + ", ".join(f"{w:.2f}:{g[GI[(w, 0.0)]]:.2f}" for w in WHOLES)
            )
            for w in (0.6, 0.65, 0.7):
                P(
                    f"  middle bar at whole {w:.2f}, all data: "
                    + ", ".join(
                        f"{m:.2f}:{g[GI[(w, m)]]:.2f}" for (ww, m) in GRID if ww == w
                    )
                )

    lodo(obj, "== 4. Leave-one-reconstruction-out on the objective ==")
    for k in sel:
        if k != obj:
            lodo(k, f"== 4b. Sensitivity: {k} ==")

    # ------------------------------------------------------------ 5
    P("")
    P(
        "== 5. Least scores of geometry-passing judged members (blurred included), per dataset =="
    )
    for d in NAMES:
        bk = [
            r
            for r in rows
            if r["ds"] == d and r["kind"] in KEEP and judged(r) and not r["_geo"]
        ]
        if bk:
            P(
                f"  {d:10s} n={len(bk):5d} least plain {min(r['plain_zncc'] for r in bk):.3f}  least blur-matched {min(r['blur_matched_zncc'] for r in bk):.3f}"
            )

    text = "\n".join(LINES) + "\n"
    sys.stdout.write(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
