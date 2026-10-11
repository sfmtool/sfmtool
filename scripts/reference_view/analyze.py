# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tune the reference-view rule's parameters against the hand picks.

    pixi run python scripts/reference_view/analyze.py <dir> [--out <file>]

Reads ``measure.py``'s output in ``<dir>`` and prints:

1. The check that ``choose``, the rule written again in Python below, gives
   the Rust rule's pick and costs at the starting values on every track.
2. The seeded split of the hand-picked cases into a tuning and a held-out
   half, stratified by dataset.
3. A sweep of each parameter alone from the starting values, and per
   parameter a candidate: the middle of the run of values tied best on the
   tuning half by the lenient count and then the exact count, or the starting
   value where it lies in such a run. A candidate is kept only where the
   held-out half scores at least as well on both counts and better on one.
4. The counts for the starting values, ``σ_n = 0``, the chosen values where
   they differ and the previous rule of gates, on each half and on all: exact,
   lenient, and lenient within ``--angle-bound`` degrees of the hand pick's
   viewing angle, and the two-sided lenient count.
5. The agreement penalty read on blur-matched pair ZNCC beside the plain one.
6. The pool: picks that change between two rules, and the term of the second
   rule's costs the first rule's pick loses on.
7. The ground truths: for the tracks whose pick changes, the error of the
   other views aligned to each pick's tile, against the projection of the
   point, and the mean error over every track as some parameters are swept.
8. What the pool's readings exercise, and the points with no eligible view.

A hand pick is matched **exactly** when the rule picks the hand-picked row,
and **leniently** when it picks a row at most 0.1 grid px blurrier (semi-major
axis) and at most 0.05 less typical (pair ZNCC) than it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

CASES = Path(__file__).resolve().parent / "cases.json"
TERMS = ("sharpness", "angle", "agreement", "cells", "coverage", "clipped")


@dataclass(frozen=True)
class Options:
    """``ReferenceViewOptions``, field for field."""

    normal_error_deg: float = 3.0
    agreement_band: float = 0.15
    agreement_scale: float = 0.15
    cell_deficit_band: float = 0.3
    cell_deficit_scale: float = 0.3
    coverage_band: float = 0.99
    coverage_scale: float = 0.03
    clipped_band: float = 0.05
    clipped_scale: float = 0.05
    min_coverage: float = 0.9
    unread_radius: float = 1.0


START = Options()

# The values each parameter is swept over, the others held.
SWEEP = {
    "normal_error_deg": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 15, 20],
    "agreement_band": [
        0.0,
        0.025,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
        0.25,
        0.3,
        0.4,
    ],
    "agreement_scale": [
        0.015,
        0.03,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.45,
        0.6,
        1.0,
        1.5,
    ],
    "cell_deficit_band": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6],
    "cell_deficit_scale": [0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.9, 1.5, 3.0],
    "coverage_band": [0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.0],
    "coverage_scale": [
        0.003,
        0.006,
        0.01,
        0.015,
        0.02,
        0.03,
        0.045,
        0.06,
        0.09,
        0.15,
        0.3,
    ],
    "clipped_band": [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15],
    "clipped_scale": [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5],
    "unread_radius": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
}


def _finite(v):
    return v if v is not None and math.isfinite(v) else None


def _angle(v):
    a = v["viewing_angle_deg"]
    return None if a is None or math.isnan(a) else a


def _hinge(x, band, scale):
    return 1.0 + max(x - band, 0.0) / scale


def choose(rows, resolution, o: Options, pair_key="pair_zncc"):
    """``choose_reference_view`` over the rule rows of one track: the index
    of the pick (or ``None``), and per row its cost dict or exclusion name."""
    exclusions = []
    for v in rows:
        c = v["coverage"]
        if c is None or math.isnan(c):
            exclusions.append("no_tile")
        elif c < o.min_coverage:
            exclusions.append("coverage")
        elif _angle(v) is not None and _angle(v) >= 90.0:
            exclusions.append("back_facing")
        else:
            exclusions.append(None)
    pick_from = [i for i, e in enumerate(exclusions) if e is None]
    c = resolution / math.sqrt(12.0) * math.radians(o.normal_error_deg)

    def angle_term(a):
        return 0.0 if c == 0.0 else c * math.tan(math.radians(max(a, 0.0)))

    def largest(values):
        values = [x for x in values if x is not None]
        return max(values) if values else None

    unread_radius = largest(_finite(rows[i]["semi_major"]) for i in pick_from)
    if unread_radius is None:
        unread_radius = o.unread_radius
    best_pair = largest(_finite(rows[i].get(pair_key)) for i in pick_from)

    def shortfall(v):
        p = _finite(v.get(pair_key))
        return None if best_pair is None or p is None else best_pair - p

    unread_shortfall = largest(shortfall(rows[i]) for i in pick_from)
    unread_angle = largest(
        angle_term(_angle(rows[i])) for i in pick_from if _angle(rows[i]) is not None
    )
    if unread_angle is None:
        unread_angle = 0.0

    costs = []
    for v, exclusion in zip(rows, exclusions):
        if exclusion is not None:
            costs.append(exclusion)
            continue
        own = _finite(v["semi_major"])
        radius = own if own is not None else unread_radius
        at = angle_term(_angle(v)) if _angle(v) is not None else unread_angle
        s = shortfall(v)
        if s is None:
            s = unread_shortfall
        agreement = 1.0 if s is None else _hinge(s, o.agreement_band, o.agreement_scale)
        d = _finite(v["cell_deficit"])
        cells = (
            1.0 if d is None else _hinge(d, o.cell_deficit_band, o.cell_deficit_scale)
        )
        coverage = (
            1.0
            if v["coverage"] is None
            else _hinge(o.coverage_band - v["coverage"], 0.0, o.coverage_scale)
        )
        k = _finite(v["clipped_share"])
        clipped = 1.0 if k is None else _hinge(k, o.clipped_band, o.clipped_scale)
        costs.append(
            dict(
                radius=radius,
                angle_term=at,
                agreement=agreement,
                cells=cells,
                coverage=coverage,
                clipped=clipped,
                cost=math.hypot(radius, at) * agreement * cells * coverage * clipped,
            )
        )

    def tie(i):
        m = _finite(rows[i]["semi_minor"])
        return math.inf if m is None else m

    pick = None
    for i in pick_from:
        if pick is None:
            pick = i
            continue
        cost, best = costs[i]["cost"], costs[pick]["cost"]
        if cost < best or (cost == best and tie(i) < tie(pick)):
            pick = i
    return pick, costs


def ln_parts(cost):
    r_eff = math.hypot(cost["radius"], cost["angle_term"])
    return [
        math.log(cost["radius"]),
        math.log(r_eff / cost["radius"]),
        math.log(cost["agreement"]),
        math.log(cost["cells"]),
        math.log(cost["coverage"]),
        math.log(cost["clipped"]),
    ]


def costliest_term(cost, pick_cost):
    parts = [a - b for a, b in zip(ln_parts(cost), ln_parts(pick_cost))]
    best = None
    for term, part in zip(TERMS, parts):
        if math.isnan(part):
            continue
        if best is None or part > best[1]:
            best = (term, part)
    return None if best is None else best[0]


@dataclass
class Track:
    ds: str
    kind: str
    point: int
    case: str | None
    resolution: int
    rows: list  # the rule rows, in the order the rule read them
    localized: dict | None


def load(directory: Path):
    tracks, all_tracks, eligible = [], [], []
    for path in sorted(directory.glob("*.jsonl")):
        target = all_tracks if path.stem.endswith("-all") else tracks
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "rows" not in r:
                print(
                    f"  {path.name}: point {r['point']} not measured: {r.get('error')}"
                )
                continue
            rows = [x for x in r["rows"] if x["rule"] is not None]
            target.append(
                Track(
                    r["ds"],
                    r["kind"],
                    r["point"],
                    r["case"],
                    r["resolution"],
                    rows,
                    r.get("localized"),
                )
            )
    for path in sorted(directory.glob("*-eligible.json")):
        eligible.append(json.loads(path.read_text()))
    return tracks, all_tracks, eligible


def check_against_rust(tracks):
    """Section 1: the Python rule against the Rust rule's output."""
    picks_differ, costs_differ, worst = [], 0, 0.0
    for t in tracks:
        pick, costs = choose(t.rows, t.resolution, START)
        rust_pick = next(
            (i for i, x in enumerate(t.rows) if x["rule"]["is_reference"]), None
        )
        if pick != rust_pick:
            picks_differ.append((t.ds, t.point))
        for x, c in zip(t.rows, costs):
            rust = x["rule"]
            if isinstance(c, str):
                if rust["excluded_by"] != c:
                    costs_differ += 1
                continue
            if rust["cost"] is None:
                costs_differ += 1
                continue
            rel = abs(c["cost"] - rust["cost"]) / max(abs(rust["cost"]), 1e-300)
            worst = max(worst, rel)
            if rel > 1e-12:
                costs_differ += 1
            for term in (
                "agreement",
                "cells",
                "coverage",
                "clipped",
                "angle_term",
                "radius",
            ):
                if abs(c[term] - rust["terms"][term]) > 1e-12 * max(1.0, abs(c[term])):
                    costs_differ += 1
    n_rows = sum(len(t.rows) for t in tracks)
    return picks_differ, costs_differ, worst, n_rows


@dataclass
class Case:
    id: str
    ds: str
    track: Track
    hand: int  # index into track.rows


def hand_cases(tracks, cases_path):
    cases = json.loads(Path(cases_path).read_text())
    by_case = {t.case: t for t in tracks if t.case}
    out, skipped = [], []
    for c in cases:
        if c.get("pick") is None:
            skipped.append((c["id"], c.get("verdict")))
            continue
        t = by_case.get(c["id"])
        if t is None:
            skipped.append((c["id"], "not measured"))
            continue
        hand = next(
            (i for i, x in enumerate(t.rows) if x["image_name"] == c["pick"]), None
        )
        if hand is None:
            skipped.append((c["id"], "hand pick not among the rule's rows"))
            continue
        out.append(Case(c["id"], c["dataset"], t, hand))
    return out, skipped


def split(cases, seed):
    """A seeded split stratified by dataset: each dataset's cases are shuffled
    and dealt alternately, the first to whichever half is smaller so far."""
    rng = np.random.default_rng(seed)
    tune, held = [], []
    for ds in sorted({c.ds for c in cases}):
        mine = [c for c in cases if c.ds == ds]
        order = rng.permutation(len(mine))
        first, second = (tune, held) if len(tune) <= len(held) else (held, tune)
        for k, i in enumerate(order):
            (first if k % 2 == 0 else second).append(mine[i])
    return tune, held


def score_picks(cases, picks, angle_bound):
    """(exact, lenient, lenient within ``angle_bound``, two-sided lenient)
    counts of ``picks``, one row index (or ``None``) per case.

    Lenient is one-sided, as the previous rule's tuning read it: the pick is
    at most 0.1 grid px blurrier (semi-major axis) and at most 0.05 less
    typical (pair ZNCC) than the hand pick, so a view as sharp and as typical
    as it. Two-sided bounds both differences by those amounts."""
    exact = lenient = bounded = two_sided = 0
    for c, pick in zip(cases, picks):
        if pick is None:
            continue
        if pick == c.hand:
            exact += 1
            lenient += 1
            bounded += 1
            two_sided += 1
            continue
        a, b = c.track.rows[pick], c.track.rows[c.hand]
        if None in (a["semi_major"], b["semi_major"], a["pair_zncc"], b["pair_zncc"]):
            continue
        dr = a["semi_major"] - b["semi_major"]
        dz = a["pair_zncc"] - b["pair_zncc"]
        if abs(dr) <= 0.1 and abs(dz) <= 0.05:
            two_sided += 1
        if dr <= 0.1 and dz >= -0.05:
            lenient += 1
            if (
                _angle(a) is not None
                and _angle(b) is not None
                and abs(_angle(a) - _angle(b)) <= angle_bound
            ):
                bounded += 1
    return exact, lenient, bounded, two_sided


def score(cases, o: Options, angle_bound, pair_key="pair_zncc"):
    """``score_picks`` of the rule's picks at ``o``."""
    picks = [choose(c.track.rows, c.track.resolution, o, pair_key)[0] for c in cases]
    return score_picks(cases, picks, angle_bound)


def previous_rule(rows):
    """The rule of gates the cost replaced, on the same readings, for
    comparison: candidates with coverage at least 0.99, clipped share at most
    0.05, viewing angle at most 65° and cell deficit at most 0.3, the tests
    dropped in turn (the angle, then the cells, then coverage and clipping)
    where no view passes, a view at 90° or more never a candidate; then the
    candidates within 0.15 of the best candidate's pair ZNCC; then the
    smallest semi-major axis, a tie to the smaller semi-minor axis and then
    the earlier view."""
    usable = [
        i
        for i, v in enumerate(rows)
        if v["coverage"] is not None
        and not (_angle(v) is not None and _angle(v) >= 90.0)
    ]

    def passes(v, dropped):
        tests = []
        if dropped < 3:
            tests += [
                v["coverage"] >= 0.99,
                v["clipped_share"] is None or v["clipped_share"] <= 0.05,
            ]
        if dropped < 2:
            tests.append(v["cell_deficit"] is None or v["cell_deficit"] <= 0.3)
        if dropped < 1:
            tests.append(_angle(v) is None or _angle(v) <= 65.0)
        return all(tests)

    candidates = []
    for dropped in range(4):
        candidates = [i for i in usable if passes(rows[i], dropped)]
        if candidates:
            break
    pairs = [
        rows[i]["pair_zncc"] for i in candidates if rows[i]["pair_zncc"] is not None
    ]
    if pairs:
        best = max(pairs)
        candidates = [
            i
            for i in candidates
            if rows[i]["pair_zncc"] is None or rows[i]["pair_zncc"] >= best - 0.15
        ]
    candidates = [i for i in candidates if rows[i]["semi_major"] is not None]
    if not candidates:
        return None

    def key(i):
        minor = rows[i]["semi_minor"]
        return (rows[i]["semi_major"], math.inf if minor is None else minor, i)

    return min(candidates, key=key)


def sweep_one(cases, o, field, angle_bound):
    """The counts at each swept value of ``field``, the others held at ``o``."""
    return [
        (v, score(cases, replace(o, **{field: v}), angle_bound)) for v in SWEEP[field]
    ]


def best_runs(results):
    """The runs of adjacent swept values tied at the best (lenient, exact),
    and the middle of the longest run, the lower middle of an even run."""
    key = [(s[1], s[0]) for _, s in results]
    best = max(key)
    runs, run = [], []
    for (v, _), k in zip(results, key):
        if k == best:
            run.append(v)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    longest = max(runs, key=len)
    return runs, longest[(len(longest) - 1) // 2]


def tune_options(tune, held, angle_bound, out):
    """Each parameter alone from the starting values: the candidate is the
    middle of the best tied run on the tuning half by (lenient, exact), or
    the starting value where it lies in a best run. A candidate is kept only
    where the held-out half confirms it, scoring at least as well on both
    counts and better on one; the kept candidates together are the chosen
    values."""
    chosen = {}
    for field in SWEEP:
        start = getattr(START, field)
        runs, candidate = best_runs(sweep_one(tune, START, field, angle_bound))
        if any(start in r for r in runs):
            out(f"  {field:20s} {start:g} is in a best run on the tuning half: kept")
            continue
        o = replace(START, **{field: candidate})
        t_start, t_cand = score(tune, START, angle_bound), score(tune, o, angle_bound)
        h_start, h_cand = score(held, START, angle_bound), score(held, o, angle_bound)
        confirmed = (
            h_cand[0] >= h_start[0]
            and h_cand[1] >= h_start[1]
            and h_cand[:2] != h_start[:2]
        )
        verdict = "confirmed" if confirmed else "not confirmed, kept"
        out(
            f"  {field:20s} {start:g} -> {candidate:g}: tuning {fmt(t_start)} -> {fmt(t_cand)},"
            f" held out {fmt(h_start)} -> {fmt(h_cand)}: {verdict}"
        )
        if confirmed:
            chosen[field] = candidate
    return replace(START, **chosen)


def fmt(s):
    return f"{s[0]:2d} / {s[1]:2d} / {s[2]:2d}"


def changes_between(tracks, pick_a, pick_b, costs_b=None):
    """The tracks whose pick differs between two rules as ``(track, a, b)``,
    and a count of the term of ``costs_b`` (the second rule's costs) that
    most separates the first rule's pick from the second's."""
    changed, terms = [], Counter()
    for t in tracks:
        a, b = pick_a(t), pick_b(t)
        if a == b:
            continue
        changed.append((t, a, b))
        if costs_b is not None and a is not None and b is not None:
            costs = costs_b(t)
            if isinstance(costs[a], str):
                terms["excluded: " + costs[a]] += 1
            else:
                terms[costliest_term(costs[a], costs[b])] += 1
        else:
            terms["none"] += 1
    return changed, terms


def localization_error(t, row):
    """The mean error, px, of the views other than ``row`` aligned to its
    tile, or ``None`` where that alignment was not read."""
    image = str(t.rows[row]["image"])
    errors = (t.localized or {}).get(image)
    if errors is None:
        return None
    others = [e for k, e in errors.items() if k != image]
    return float(np.mean(others)) if others else None


def compare_localization(changed):
    """For tracks whose pick changes, the mean error of the views other than
    both picks, aligned to each pick's tile: two arrays, one per pick."""
    ea, eb = [], []
    for t, a, b in changed:
        if a is None or b is None or not t.localized:
            continue
        ia, ib = str(t.rows[a]["image"]), str(t.rows[b]["image"])
        la, lb = t.localized.get(ia), t.localized.get(ib)
        if la is None or lb is None:
            continue
        common = [k for k in la if k in lb and k not in (ia, ib)]
        if common:
            ea.append(float(np.mean([la[k] for k in common])))
            eb.append(float(np.mean([lb[k] for k in common])))
    return np.array(ea), np.array(eb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory")
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--angle-bound", type=float, default=10.0)
    ap.add_argument("--out", help="also write the report here")
    args = ap.parse_args()
    bound = args.angle_bound
    lines: list[str] = []

    def out(s=""):
        print(s)
        lines.append(s)

    tracks, all_tracks, eligible = load(Path(args.directory))
    fill_blur_matched(tracks)
    out("1. The Python rule against the Rust rule at the starting values")
    every = tracks + all_tracks
    differ, costs_differ, worst, n_rows = check_against_rust(every)
    out(
        f"  {len(every)} tracks, {n_rows} rows: picks differ on {len(differ)} {differ[:5]},"
        f" costs or terms differ on {costs_differ} rows,"
        f" largest relative cost difference {worst:.1e}"
    )

    cases, skipped = hand_cases(tracks, args.cases)
    tune, held = split(cases, args.seed)
    out()
    out("2. Hand-picked cases")
    out(f"  {len(cases)} cases scored; left out: {skipped}")
    out(f"  tuning {len(tune)}, held out {len(held)}")
    for ds in sorted({c.ds for c in cases}):
        n_tune = sum(c.ds == ds for c in tune)
        out(f"    {ds:12s} tuning {n_tune}, held out {sum(c.ds == ds for c in held)}")

    out()
    out(
        "3. Each parameter swept alone from the starting values"
        f" (exact / lenient / lenient within {bound:g}°)"
    )
    for field in SWEEP:
        out(f"  {field}")
        for v in SWEEP[field]:
            o = replace(START, **{field: v})
            mark = " *" if v == getattr(START, field) else ""
            out(
                f"    {v:<7g} tuning {fmt(score(tune, o, bound))}"
                f"   held out {fmt(score(held, o, bound))}"
                f"   all {fmt(score(cases, o, bound))}{mark}"
            )
    out()
    out("  Candidates on the tuning half, and the held-out half's check:")
    tuned = tune_options(tune, held, bound, out)
    changed = {
        f.name: (getattr(START, f.name), getattr(tuned, f.name))
        for f in dataclasses.fields(Options)
        if getattr(START, f.name) != getattr(tuned, f.name)
    }
    out(f"  chosen changes: {changed or 'none'}")

    sigma0 = replace(START, normal_error_deg=0.0)
    settings = [("starting values", START), ("σ_n = 0", sigma0)]
    if tuned != START:
        settings.append(("chosen values", tuned))

    def rule(o, pair_key="pair_zncc"):
        return lambda t: choose(t.rows, t.resolution, o, pair_key)[0]

    def previous(t):
        return previous_rule(t.rows)

    pickers = [(label, rule(o)) for label, o in settings]
    pickers.append(("previous rule", previous))
    out()
    out(f"4. Counts, exact / lenient / lenient within {bound:g}°")
    out(
        f"  {'':18s} tuning ({len(tune)})     held out ({len(held)})   all ({len(cases)})"
    )
    table = {}
    for label, pick in pickers:
        table[label] = [
            score_picks(cs, [pick(c.track) for c in cs], bound)
            for cs in (tune, held, cases)
        ]
        t, h, a = table[label]
        out(f"  {label:18s} {fmt(t)}    {fmt(h)}    {fmt(a)}")
    out(
        "  Two-sided lenient (both differences within the bounds), tuning / held out / all:"
    )
    for label, _ in pickers:
        t, h, a = table[label]
        out(f"  {label:18s} {t[3]:2d} / {h[3]:2d} / {a[3]:2d}")

    out()
    out("5. The agreement penalty on blur-matched pair ZNCC")
    with_bm = [
        c for c in cases if any(x.get("bm_pair_zncc") is not None for x in c.track.rows)
    ]
    out(f"  {len(with_bm)} of {len(cases)} cases have blur-matched readings")
    rows = [x for c in cases for x in c.track.rows if x.get("bm_pair_zncc") is not None]
    moved = [x["bm_pair_zncc"] - x["py_pair_zncc"] for x in rows]
    moved = [d for d in moved if abs(d) > 1e-9]
    gap = [
        abs(x["py_pair_zncc"] - x["pair_zncc"])
        for x in rows
        if x["pair_zncc"] is not None
    ]
    out(
        f"  {len(rows)} rows: blur matching moves the pair ZNCC of {len(moved)}"
        f" (mean {np.mean(moved) if moved else 0:+.3f}); the tiles' plain pair ZNCC is"
        f" within {max(gap) if gap else 0:.1e} of the bench's"
    )
    for label, o in settings:
        plain = score(cases, o, bound, "pair_zncc")
        bm = score(cases, o, bound, "bm_or_plain")
        n = sum(rule(o)(c.track) != rule(o, "bm_or_plain")(c.track) for c in cases)
        out(
            f"  {label:18s} plain {fmt(plain)}   blur-matched {fmt(bm)}   picks that differ {n}"
        )
    pool_bm = [
        t for t in tracks if any(x.get("bm_pair_zncc") is not None for x in t.rows)
    ]
    n = sum(rule(tuned)(t) != rule(tuned, "bm_or_plain")(t) for t in pool_bm)
    out(f"  pool: blur matching changes the pick on {n} of {len(pool_bm)} tracks")
    for field in ("agreement_band", "agreement_scale"):
        res = [
            (v, score(tune, replace(tuned, **{field: v}), bound, "bm_or_plain"))
            for v in SWEEP[field]
        ]
        cells = "  ".join(f"{v:g}:{s[0]}/{s[1]}" for v, s in res)
        out(f"    tuning half, blur-matched, {field} (exact/lenient): {cells}")

    out()
    out("6. Pool: picks that change, and the term the old pick loses on")
    out(f"  {len(tracks)} tracks: the review cases and each dataset's sampled tracks")
    comparisons = [
        ("σ_n = 0 -> starting", rule(sigma0), rule(START), START),
        ("previous rule -> starting", previous, rule(START), START),
    ]
    if tuned != START:
        comparisons.insert(0, ("starting -> chosen", rule(START), rule(tuned), tuned))
    for label, pa, pb, ob in comparisons:
        ch, terms = changes_between(
            tracks, pa, pb, lambda t, ob=ob: choose(t.rows, t.resolution, ob)[1]
        )
        by_ds = dict(Counter(t.ds for t, _, _ in ch))
        out(f"  {label}: {len(ch)} change; the old pick's costliest term {dict(terms)}")
        out(f"    by dataset {by_ds}")

    out()
    out(
        "7. Ground truths: the other views aligned to a pick's tile, error from the projection (px)"
    )
    gt = [t for t in (all_tracks or tracks) if t.localized]
    out(f"  {len(gt)} ground-truth tracks of three or more views")
    for label, pa, pb, _ in comparisons:
        ch, _ = changes_between(gt, pa, pb)
        ea, eb = compare_localization(ch)
        if len(ea):
            d = eb - ea
            out(
                f"  {label}: {len(ch)} picks change; mean {ea.mean():.3f} -> {eb.mean():.3f}"
                f" ({d.mean():+.3f}, median {np.median(d):+.3f});"
                f" better on {(d < 0).sum()}, worse on {(d > 0).sum()}"
            )

    def mean_error(pick):
        e = [localization_error(t, pick(t)) for t in gt if pick(t) is not None]
        e = [x for x in e if x is not None]
        return np.mean(e), np.median(e)

    best = [
        min(
            e
            for e in (localization_error(t, i) for i in range(len(t.rows)))
            if e is not None
        )
        for t in gt
        if any(localization_error(t, i) is not None for i in range(len(t.rows)))
    ]
    out(
        f"  Mean over every track of the other views' error (the best view per track {np.mean(best):.3f}):"
    )
    for field in ("normal_error_deg", "agreement_band", "cell_deficit_band"):
        cells = [
            f"{v:g}:{mean_error(rule(replace(tuned, **{field: v})))[0]:.3f}"
            for v in SWEEP[field]
        ]
        out(f"    {field}: " + "  ".join(cells))
    m = mean_error(previous)
    out(f"    previous rule: {m[0]:.3f}")
    above = {}
    for label, pick in pickers:
        above[label] = sum(
            1
            for t in tracks + gt
            if pick(t) is not None and (_angle(t.rows[pick(t)]) or 0.0) > 65.0
        )
    out(
        f"  Picks above 65° over the pool and the ground truths ({len(tracks) + len(gt)} tracks): {above}"
    )

    out()
    out("8. What the readings exercise, over the pool")
    facts = Counter()
    for t in tracks:
        eligible_rows = [x for x in t.rows if x["rule"]["excluded_by"] is None]
        facts["tracks"] += 1
        if any(x["semi_major"] is None for x in eligible_rows):
            facts["an eligible view with no radius"] += 1
        if eligible_rows and all(x["semi_major"] is None for x in eligible_rows):
            facts["no eligible view with a radius"] += 1
        if any(
            START.min_coverage <= x["coverage"] < START.coverage_band
            for x in eligible_rows
        ):
            facts["an eligible view covering 0.9 to 0.99"] += 1
        if any((x["clipped_share"] or 0.0) > START.clipped_band for x in eligible_rows):
            facts["an eligible view clipped past 0.05"] += 1
        if any(x["rule"]["excluded_by"] == "back_facing" for x in t.rows):
            facts["a back-facing view"] += 1
        if any(x["rule"]["excluded_by"] == "coverage" for x in t.rows):
            facts["a view under the coverage floor"] += 1
    out(f"  {dict(facts)}")
    out("  Points with no eligible view (every stored reference cleared):")
    for e in eligible:
        out(f"    {e['ds']:12s} {e['no_eligible_view']:5d} of {e['with_two_views']:6d}")

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def fill_blur_matched(tracks):
    """Give every row ``bm_or_plain``: its blur-matched pair ZNCC where it
    has one and its plain pair ZNCC where not (a patch at infinity, whose
    tiles ``measure.py`` does not render)."""
    for t in tracks:
        for x in t.rows:
            bm = x.get("bm_pair_zncc")
            x["bm_or_plain"] = bm if bm is not None else x["pair_zncc"]


if __name__ == "__main__":
    sys.exit(main())
