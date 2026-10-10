# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Print markdown tables from the align_vs_congealing harness output.

Usage: python report.py OUT_DIR

OUT_DIR holds `<dataset>_<ref-mode>_align.json` (methods pd and ex, from the
branch build) and `<dataset>_<ref-mode>_congeal.json` (method congeal, from the
83ffb08e build), for ref-mode `displaced` and `stored`.
"""

import json
import statistics
import sys
from pathlib import Path

GOOD_PX = 1.0
BAD_PX = 1.5
RETRI_BLOWUP_PX = 5.0
GATE_DISPS = (0.0, 0.5, 1.0)
ABSOLUTE_FLOORS = (0.4, 0.5, 0.6)
RELATIVE_BARS = (0.6, 0.7, 0.8)


def load(out_dir):
    """Return {(dataset, mode): {"points": {pid: point}, "runs": {...}}}."""
    data = {}
    for path in sorted(Path(out_dir).glob("*_*_*.json")):
        dataset, mode, _kind = path.stem.split("_")
        doc = json.loads(path.read_text())
        entry = data.setdefault((dataset, mode), {"points": {}, "runs": {}})
        for point in doc["points"]:
            entry["points"][point["point"]] = point
            for run in point["runs"]:
                key = (run["disp"], run["method"])
                entry["runs"].setdefault(key, {})[point["point"]] = run["views"]
    return data


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def median(values):
    return statistics.median(values) if values else float("nan")


def accuracy_tables(data):
    lines = []
    for (dataset, mode), entry in sorted(data.items()):
        total_views = sum(p["n"] for p in entry["points"].values())
        lines.append(f"### {dataset}, reference {mode}: {len(entry['points'])} tracks")
        lines.append("")
        lines.append(
            "| disp px | method | views kept | raw mean / median | "
            "retri mean / median | side-peak share | tracks retri mean > 5 px |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        methods = sorted(
            {m for _, m in entry["runs"]}, key=["congeal", "pd", "ex"].index
        )
        disps = sorted({d for d, _ in entry["runs"]})
        for disp in disps:
            for method in methods:
                runs = entry["runs"][(disp, method)]
                views = [v for vs in runs.values() for v in vs]
                raw = [v["err"] for v in views if v["err"] is not None]
                retri = [v["retri"] for v in views if "retri" in v]
                side = sum(e > BAD_PX for e in raw) / len(raw) if raw else float("nan")
                blowups = sum(
                    1
                    for vs in runs.values()
                    if any("retri" in v for v in vs)
                    and mean([v["retri"] for v in vs if "retri" in v]) > RETRI_BLOWUP_PX
                )
                lines.append(
                    f"| {disp:g} | {method} | {len(views)}/{total_views} | "
                    f"{mean(raw):.3f} / {median(raw):.3f} | "
                    f"{mean(retri):.3f} / {median(retri):.3f} | "
                    f"{100 * side:.1f}% | {blowups} |"
                )
        lines.append("")
    return lines


def relative_scores(views, reference):
    """Each non-reference view's score over the median of the run's non-reference scores.

    A view whose score is missing (NaN in the harness, null in its JSON) is left out.
    """
    scored = [v for v in views if v["slot"] != reference and v["zncc"] is not None]
    mid = median([v["zncc"] for v in scored])
    return {v["slot"]: v["zncc"] / mid for v in scored}


def gate_tables(data):
    lines = []
    for (dataset, mode), entry in sorted(data.items()):
        if mode != "stored":
            continue
        for disp in GATE_DISPS:
            rows = {"pd": [], "congeal": []}
            for pid, point in entry["points"].items():
                reference = point["reference"]
                by_method = {}
                for method in rows:
                    views = entry["runs"][(disp, method)].get(pid, [])
                    rel = relative_scores(views, reference)
                    by_method[method] = {
                        v["slot"]: (v["err"], v["zncc"], rel[v["slot"]])
                        for v in views
                        if v["slot"] in rel and v["err"] is not None
                    }
                shared = by_method["pd"].keys() & by_method["congeal"].keys()
                for method in rows:
                    rows[method].extend(by_method[method][s] for s in sorted(shared))
            n = len(rows["pd"])
            lines.append(
                f"### {dataset}, reference stored, disp {disp:g} px: {n} views"
            )
            lines.append("")
            counts = []
            for method, label in (
                ("pd", "plain (pd)"),
                ("congeal", "leave-one-out (congeal)"),
            ):
                good = sum(e < GOOD_PX for e, _, _ in rows[method])
                bad = sum(e > BAD_PX for e, _, _ in rows[method])
                counts.append(f"{label}: n {n}, good {good}, bad {bad}")
            lines.append("; ".join(counts) + ".")
            lines.append("")
            lines.append(
                "| gate | plain good dropped | plain bad dropped | LOO good dropped | LOO bad dropped |"
            )
            lines.append("|---|---|---|---|---|")
            gates = [("absolute", f, 1) for f in ABSOLUTE_FLOORS]
            gates += [("relative", b, 2) for b in RELATIVE_BARS]
            for kind, bar, col in gates:
                cells = []
                for method in ("pd", "congeal"):
                    good = [r for r in rows[method] if r[0] < GOOD_PX]
                    bad = [r for r in rows[method] if r[0] > BAD_PX]
                    for group in (good, bad):
                        dropped = sum(r[col] < bar for r in group)
                        share = 100 * dropped / len(group) if group else float("nan")
                        cells.append(f"{share:.1f}%")
                lines.append(f"| {kind} {bar:g} | " + " | ".join(cells) + " |")
            lines.append("")
    return lines


def main():
    data = load(sys.argv[1])
    print("## Accuracy\n")
    print("\n".join(accuracy_tables(data)))
    print("## Gates (reference stored)\n")
    print("\n".join(gate_tables(data)))


if __name__ == "__main__":
    main()
