# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize a sweep of ``grid_plane_eval.py`` runs as Markdown tables.

    pixi run -e test python scripts/track_at_pixel/normal_sources/grid_plane_sweep.py \\
        DATASET_DIR [--tolerance PX]

``DATASET_DIR`` holds one subdirectory per variant, each with the one
``*.jsonl`` that ``grid_plane_eval.py --out`` wrote for it; the variant's name
is its directory's. Every run is from the same fitted mean-view starts, so the
start and today's 2x2 Grid Plane Normal are read from the first variant.

Errors are the rows' angles to the ground-truth normal, in degrees. Besides
the step as built (``new``: the normal the step left, which differs from the
start only where it turned), two estimators are read from each trace: the
first plane (the normal the first render's turn reaches) and the last
proposal (the normal the last proposed turn reaches). A precision bar is
applied after the fact from the trace's predicted precision.
"""

import argparse
import json
from pathlib import Path

import numpy as np

BASELINE_BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 181)]
PRECISION_BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 1e9)]
PRECISION_BARS = [5.0, 10.0, 15.0, 20.0]
BASELINE_BARS = [5.0, 10.0, 20.0]


def load(directory):
    variants = {}
    for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
        files = sorted(sub.glob("*.jsonl"))
        if not files:
            continue
        rows = [json.loads(line) for line in open(files[0])]
        variants[sub.name] = [r for r in rows if r["status"] == "ok"]
    return variants


def bin_name(lo, hi):
    if lo == 0:
        return f"<{hi:g}"
    if hi > 180 or hi >= 1e9:
        return f">={lo:g}"
    return f"{lo:g}-{hi:g}"


def stats(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return "-", "-", "-", "-"
    p50, p75 = np.percentile(a, [50, 75])
    return (
        f"{p50:.1f}",
        f"{p75:.1f}",
        f"{np.mean(a < 10):.0%}",
        f"{np.mean(a < 20):.0%}",
    )


def first_plane(row):
    for entry in row.get("new_trace", [])[:1]:
        if entry["proposed_deg"] is not None:
            return entry["proposed_deg"], entry["normal_precision_deg"]
    return None, None


def last_proposal(row):
    proposed = [e for e in row.get("new_trace", []) if e["proposed_deg"] is not None]
    if not proposed:
        return None
    return proposed[-1]["proposed_deg"]


def baseline(row):
    return row.get("new_view_baseline_deg", float("nan"))


def in_bin(value, lo, hi):
    return value is not None and np.isfinite(value) and lo <= value < hi


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def by_baseline(variants):
    """Turned points per baseline bin, paired with the start and today's step."""
    print("### Turned points by baseline\n")
    header = [
        "variant",
        "baseline",
        "points",
        "turned",
        "new med",
        "new p75",
        "new <10",
        "new <20",
        "start med",
        "old med",
        "old <10",
        "new closer than start",
    ]
    rows = []
    for name, data in variants.items():
        for lo, hi in BASELINE_BINS:
            pts = [r for r in data if in_bin(baseline(r), lo, hi)]
            turned = [r for r in pts if r["new_outcome"] == "turned"]
            n, p75, u10, u20 = stats(r["new_deg"] for r in turned)
            closer = (
                f"{np.mean([r['new_deg'] < r['start_deg'] for r in turned]):.0%}"
                if turned
                else "-"
            )
            rows.append(
                [
                    name,
                    bin_name(lo, hi),
                    len(pts),
                    len(turned),
                    n,
                    p75,
                    u10,
                    u20,
                    stats(r["start_deg"] for r in turned)[0],
                    stats(r["old_deg"] for r in turned)[0],
                    stats(r["old_deg"] for r in turned)[2],
                    closer,
                ]
            )
    print(table(header, rows) + "\n")


def baselines_alone(variants):
    """The start and today's step on every point, per baseline bin."""
    data = next(iter(variants.values()))
    print("### Start and today's 2x2 on all points, by baseline\n")
    header = ["baseline", "points", "start med", "start <20", "old med", "old <10"]
    header += ["old <20", "old closer than start"]
    rows = []
    for lo, hi in BASELINE_BINS:
        pts = [r for r in data if in_bin(baseline(r), lo, hi)]
        s = stats(r["start_deg"] for r in pts)
        o = stats(r["old_deg"] for r in pts)
        closer = (
            f"{np.mean([r['old_deg'] < r['start_deg'] for r in pts]):.0%}"
            if pts
            else "-"
        )
        rows.append([bin_name(lo, hi), len(pts), s[0], s[3], o[0], o[2], o[3], closer])
    print(table(header, rows) + "\n")


def first_planes(variants):
    """The first render's plane per baseline bin, one row per search bound."""
    print("### The first plane, by baseline\n")
    header = ["variant", "baseline", "points", "proposed", "first med", "first p75"]
    header += ["first <10", "first <20", "start med", "old med", "closer than start"]
    rows = []
    seen = set()
    for name, data in variants.items():
        bound = name.split("_")[0]
        if bound in seen:
            continue  # the first render does not depend on the acceptance rule
        seen.add(bound)
        for lo, hi in BASELINE_BINS:
            pts = [r for r in data if in_bin(baseline(r), lo, hi)]
            prop = [(r, first_plane(r)[0]) for r in pts]
            prop = [(r, e) for r, e in prop if e is not None]
            f = stats(e for _, e in prop)
            closer = (
                f"{np.mean([e < r['start_deg'] for r, e in prop]):.0%}" if prop else "-"
            )
            rows.append(
                [
                    bound,
                    bin_name(lo, hi),
                    len(pts),
                    len(prop),
                    *f,
                    stats(r["start_deg"] for r, _ in prop)[0],
                    stats(r["old_deg"] for r, _ in prop)[0],
                    closer,
                ]
            )
    print(table(header, rows) + "\n")


def convergence(variants, tolerance):
    print("### Convergence\n")
    header = ["variant", "render", "points", "proposed med", "proposed p75"]
    header += ["change from first, med", "move med px", "move p10", "move p90"]
    header += [f"moves <= {tolerance:g}"]
    rows = []
    for name, data in variants.items():
        if not name.endswith("_none"):
            continue
        for k in range(8):
            at = [
                (r, r["new_trace"][k])
                for r in data
                if len(r.get("new_trace", [])) > k
                and r["new_trace"][k]["proposed_deg"] is not None
            ]
            if not at:
                continue
            firsts = [first_plane(r)[0] for r, _ in at]
            change = [
                e["proposed_deg"] - f for (_, e), f in zip(at, firsts) if f is not None
            ]
            moves = np.asarray([e["largest_move_px"] for _, e in at], float)
            rows.append(
                [
                    name,
                    k + 1,
                    len(at),
                    *stats(e["proposed_deg"] for _, e in at)[:2],
                    f"{np.median(change):+.1f}" if change else "-",
                    f"{np.median(moves):.2f}",
                    f"{np.percentile(moves, 10):.2f}",
                    f"{np.percentile(moves, 90):.2f}",
                    f"{np.mean(moves <= tolerance):.0%}",
                ]
            )
    print(table(header, rows) + "\n")

    print("#### Is the first plane the best one proposed?\n")
    header = ["variant", "points with 2+ proposals", "first is best"]
    header += ["last better than first", "last worse than first by 5+ deg"]
    rows = []
    for name, data in variants.items():
        if not name.endswith("_none"):
            continue
        multi = []
        for r in data:
            errs = [
                e["proposed_deg"]
                for e in r.get("new_trace", [])
                if e["proposed_deg"] is not None
            ]
            if len(errs) >= 2:
                multi.append(errs)
        if not multi:
            continue
        rows.append(
            [
                name,
                len(multi),
                f"{np.mean([e[0] <= min(e) for e in multi]):.0%}",
                f"{np.mean([e[-1] < e[0] for e in multi]):.0%}",
                f"{np.mean([e[-1] > e[0] + 5 for e in multi]):.0%}",
            ]
        )
    print(table(header, rows) + "\n")

    print("#### Stop reasons\n")
    stops = ["converged", "measured", "rejected", "oscillation", "cap", "not_run"]
    header = ["variant"] + stops
    rows = []
    for name, data in variants.items():
        rows.append([name] + [sum(r.get("new_stop") == s for r in data) for s in stops])
    print(table(header, rows) + "\n")


def rejections(variants):
    """Turns the ZNCC rule rejected: did they lower the error?"""
    print("### Rejected turns that would have lowered the error\n")
    header = ["variant", "baseline", "rejected", "would lower the error"]
    header += ["by 5+ deg", "would raise it by 5+ deg"]
    rows = []
    for name, data in variants.items():
        if name.endswith("_none"):
            continue
        for lo, hi in BASELINE_BINS:
            deltas = []
            for r in data:
                if r.get("new_stop") != "rejected" or not in_bin(baseline(r), lo, hi):
                    continue
                trace = r["new_trace"]
                k = max(i for i, e in enumerate(trace) if e["proposed_deg"] is not None)
                before = r["start_deg"] if k == 0 else trace[k - 1]["proposed_deg"]
                deltas.append(trace[k]["proposed_deg"] - before)
            d = np.asarray(deltas, float)
            rows.append(
                [
                    name,
                    bin_name(lo, hi),
                    len(d),
                    f"{np.mean(d < 0):.0%}" if d.size else "-",
                    f"{np.mean(d < -5):.0%}" if d.size else "-",
                    f"{np.mean(d > 5):.0%}" if d.size else "-",
                ]
            )
    print(table(header, rows) + "\n")


def precision(variants):
    print("### The first plane's error against its predicted precision\n")
    header = ["variant", "precision", "points", "first med", "first p75"]
    header += ["first <10", "first <20", "start med"]
    rows = []
    seen = set()
    for name, data in variants.items():
        bound = name.split("_")[0]
        if bound in seen:
            continue
        seen.add(bound)
        pairs = [(r, *first_plane(r)) for r in data]
        pairs = [(r, e, p) for r, e, p in pairs if e is not None]
        for lo, hi in PRECISION_BINS:
            sel = [(r, e) for r, e, p in pairs if in_bin(p, lo, hi)]
            rows.append(
                [
                    bound,
                    bin_name(lo, hi),
                    len(sel),
                    *stats(e for _, e in sel),
                    stats(r["start_deg"] for r, _ in sel)[0],
                ]
            )
    print(table(header, rows) + "\n")

    print("### Gates on the first plane, by baseline\n")
    header = ["variant", "gate", "baseline", "points", "kept", "first med"]
    header += ["first <10", "first <20", "start med", "old med"]
    rows = []
    seen = set()
    for name, data in variants.items():
        bound = name.split("_")[0]
        if bound in seen:
            continue
        seen.add(bound)
        gates = [(f"precision <= {b:g}", "p", b) for b in PRECISION_BARS]
        gates += [(f"baseline >= {b:g}", "b", b) for b in BASELINE_BARS]
        for label, kind, bar in gates:
            # A baseline gate keeps a whole bin or none of it, so it is
            # read over all points only.
            bins = BASELINE_BINS + [(0, 181)] if kind == "p" else [(0, 181)]
            for lo, hi in bins:
                pts = [r for r in data if in_bin(baseline(r), lo, hi)]
                kept = []
                for r in pts:
                    e, p = first_plane(r)
                    if e is None:
                        continue
                    if kind == "p":
                        ok = p is not None and np.isfinite(p) and p <= bar
                    else:
                        ok = baseline(r) >= bar
                    if ok:
                        kept.append((r, e))
                f = stats(e for _, e in kept)
                rows.append(
                    [
                        bound,
                        label,
                        "all" if hi == 181 and lo == 0 else bin_name(lo, hi),
                        len(pts),
                        len(kept),
                        f[0],
                        f[2],
                        f[3],
                        stats(r["start_deg"] for r, _ in kept)[0],
                        stats(r["old_deg"] for r, _ in kept)[0],
                    ]
                )
    print(table(header, rows) + "\n")

    print("### Gates on the step as built (Turned points)\n")
    header = ["variant", "gate", "turned", "kept", "new med", "new <10", "new <20"]
    header += ["start med", "old med"]
    rows = []
    for name, data in variants.items():
        turned = [r for r in data if r["new_outcome"] == "turned"]
        for bar in PRECISION_BARS:
            kept = [
                r
                for r in turned
                if np.isfinite(r["new_normal_precision_deg"])
                and r["new_normal_precision_deg"] <= bar
            ]
            f = stats(r["new_deg"] for r in kept)
            rows.append(
                [
                    name,
                    f"precision <= {bar:g}",
                    len(turned),
                    len(kept),
                    f[0],
                    f[2],
                    f[3],
                    stats(r["start_deg"] for r in kept)[0],
                    stats(r["old_deg"] for r in kept)[0],
                ]
            )
    print(table(header, rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()
    variants = load(args.directory)
    print(f"## {args.directory.name}: variants {', '.join(variants)}\n")
    baselines_alone(variants)
    by_baseline(variants)
    first_planes(variants)
    convergence(variants, args.tolerance)
    rejections(variants)
    precision(variants)


if __name__ == "__main__":
    main()
