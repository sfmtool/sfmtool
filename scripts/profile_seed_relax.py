# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Where the seed relaxation spends its time, on one stored member.

The relaxation runs six stages on a member and its source selection handle
(`scripts/seed_relax/pipeline.py`).  This harness runs that chain once on a
member restored from a study sidecar and reports a wall-clock table: the six
stages, and inside the fill-in the join, the ring assignment, the point
estimation and the held adjustment, each summed over the rings it ran on.

The timing is taken by wrapping the package's own functions where the caller
looks them up, so the chain runs exactly as it ships and nothing in the package
reads a clock.

Usage::

    python scripts/profile_seed_relax.py CANDIDATE_SOLVES_DIR IDX MATCHES_PATH

``CANDIDATE_SOLVES_DIR`` holds the ``member_arrays.npz`` sidecar and its
``manifest.json``; ``IDX`` is the hypothesis index of a rotation-only member in
that sidecar; ``MATCHES_PATH`` is the capture's cluster-patches ``.matches``
file, opened the way the run opens it.  ``--repeat N`` runs the chain N times
and reports the median of each row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

#: The minimum cluster span the loader admits, matching `exp_fast_seed`.
MIN_SPAN = 2


def load_member(cs_dir, idx):
    """One rotation-only member of a study sidecar, as it was committed."""
    import seed_candidate_eval as EV

    d = np.load(Path(cs_dir) / "member_arrays.npz")
    meta = json.loads(zlib.decompress(d["_meta"]).decode("utf-8"))
    key = f"m{int(idx):04d}"
    arr = {k.split("__", 1)[1]: d[k] for k in d.files if k.startswith(key + "__")}
    mm = meta["members"][key]
    return EV.member_from_arrays(
        {
            "idx": int(idx),
            "model": mm["model"],
            "names": meta["names"],
            "camera": mm["camera"],
            "f_eq": mm["f_eq"],
            "rvec": arr["rvec"],
            "tvec": arr["tvec"],
            "posed": arr["posed"],
            "pts": arr.get("pts"),
            "obs_c": arr["obs_c"],
            "obs_i": arr["obs_i"],
            "obs_uv": arr["obs_uv"],
            "obs_f": arr.get("obs_f"),
            "obs_shape": arr.get("obs_shape"),
            "keep": arr.get("keep"),
        }
    )


def open_source(matches_path):
    """The selection handle the run holds, from the capture's matches file."""
    from sfmtool._sfmtool.io import MatchesFile

    return MatchesFile(str(matches_path)).select_clusters(min_span=MIN_SPAN)


class Clock:
    """Wall-clock totals and call counts, by name."""

    def __init__(self):
        self.total = defaultdict(float)
        self.calls = defaultdict(int)
        self.each = defaultdict(list)

    def wrap(self, module, name, label=None):
        """Time every call of ``module.name`` where its callers look it up."""
        if label is None:
            label = f"{module.__name__.rsplit('.', 1)[-1]}.{name}"
        fn = getattr(module, name)

        def timed(*a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                dt = time.perf_counter() - t0
                self.total[label] += dt
                self.calls[label] += 1
                self.each[label].append(dt)

        timed.__name__ = getattr(fn, "__name__", name)
        timed.__doc__ = getattr(fn, "__doc__", None)
        setattr(module, name, timed)
        return fn


#: Rows of the table: the label, and how deep it sits under its parent.
ROWS = [
    ("stage 1 lens on bearings", 0),
    ("stage 2 relaxation", 0),
    ("graph.member_graph", 1),
    ("graph.stage_pairs", 1),
    ("graph.pair_rays", 2),
    ("graph.baseline_direction", 2),
    ("averaging.centres_by_averaging", 1),
    ("orientation.angw_bit", 1),
    ("structure.triangulate_placed", 1),
    ("structure.grow_more", 1),
    ("structure.build_ba_inputs", 1),
    ("structure.stage_adjust", 1),
    ("stage 3 fill-in", 0),
    ("fill.source_clusters", 1),
    ("rings.assign_rings", 1),
    ("fill.extend_member", 1),
    ("fill.ring_rows", 1),
    ("fill.estimate_points", 1),
    ("fill.adjust_held", 1),
    ("stage 4 late lens release", 0),
    ("stage 5 re-estimate", 0),
    ("structure.state_rows", 1),
    ("structure.estimate_points", 1),
    ("structure.reprojection", 1),
    ("stage 6 runaway report", 0),
    ("run_member", 0),
]


def instrument(clock):
    """Wrap every timed name, in the module its caller reads it from."""
    import seed_relax
    from seed_relax import (
        averaging,
        fill,
        graph,
        orientation,
        pipeline,
        rings,
        structure,
    )

    clock.wrap(graph, "member_graph")
    clock.wrap(graph, "stage_pairs")
    clock.wrap(graph, "pair_rays")
    clock.wrap(graph, "baseline_direction")
    clock.wrap(averaging, "centres_by_averaging")
    clock.wrap(orientation, "angw_bit")
    clock.wrap(structure, "triangulate_placed")
    clock.wrap(structure, "grow_more")
    clock.wrap(structure, "build_ba_inputs")
    clock.wrap(structure, "stage_adjust")
    clock.wrap(structure, "estimate_points")
    clock.wrap(structure, "reprojection")
    clock.wrap(structure, "state_rows")
    clock.wrap(fill, "source_clusters")
    clock.wrap(fill, "extend_member")
    clock.wrap(fill, "ring_rows")
    clock.wrap(fill, "adjust_held")
    # `fill` binds the estimator by name at import, so the fill-in's own calls
    # are counted separately from stage 5's.
    clock.wrap(fill, "estimate_points")
    clock.wrap(rings, "assign_rings")
    return seed_relax, pipeline


def stage_clock(pipeline_mod, clock):
    """Time the six stages by the boundaries the pipeline already draws."""
    import seed_relax
    from seed_relax import fill, lens, relaxation, report, structure

    marks = [
        (lens, "rot_lens_ba", "stage 1 lens on bearings"),
        (relaxation, "relax_oriented", "stage 2 relaxation"),
        (fill, "fill_in", "stage 3 fill-in"),
        (lens, "release_at_knots", "stage 4 late lens release"),
        (report, "runaway_report", "stage 6 runaway report"),
    ]
    for mod, name, label in marks:
        clock.wrap(mod, name, label)
    del seed_relax, structure


def run_once(cs_dir, idx, matches_path, source):
    """One whole chain, timed.  Returns the clock and the result."""
    clock = Clock()
    seed_relax, pipeline = instrument(clock)
    stage_clock(pipeline, clock)
    m = load_member(cs_dir, idx)
    opts = seed_relax.Options()
    t0 = time.perf_counter()
    result = pipeline.run_member(m, source, opts)
    clock.total["run_member"] = time.perf_counter() - t0
    clock.calls["run_member"] = 1
    # Stage 5 is the pipeline's own re-estimation, which is what is left of the
    # chain once the four stages that carry their own mark are taken out.
    clock.total["stage 5 re-estimate"] = (
        clock.total["structure.estimate_points"]
        + clock.total["structure.reprojection"]
        + clock.total["structure.state_rows"]
    )
    clock.calls["stage 5 re-estimate"] = 1
    return clock, result


def render(clock, total):
    """The table, one row per timed name."""
    lines = [f"{'stage':<40}{'calls':>8}{'seconds':>12}{'% of run':>10}"]
    lines.append("-" * 70)
    for label, depth in ROWS:
        if label not in clock.total:
            continue
        secs = clock.total[label]
        name = ("  " * depth) + label
        pct = 100.0 * secs / total if total > 0 else 0.0
        lines.append(f"{name:<40}{clock.calls[label]:>8}{secs:>12.3f}{pct:>10.1f}")
    return "\n".join(lines)


def per_ring(clock):
    """The fill-in's per-ring rows, in the order the rings ran."""
    names = ["fill.ring_rows", "fill.estimate_points", "fill.adjust_held"]
    n = max((len(clock.each[k]) for k in names), default=0)
    lines = [f"{'ring':<8}" + "".join(f"{k.split('.')[1]:>22}" for k in names)]
    for r in range(n):
        row = f"{r:<8}"
        for k in names:
            v = clock.each[k]
            row += f"{(v[r] if r < len(v) else float('nan')):>22.3f}"
        lines.append(row)
    for k in ("fill.source_clusters", "rings.assign_rings", "fill.extend_member"):
        lines.append(f"{k:<30}{clock.total[k]:>10.3f} s")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate_solves", help="directory holding member_arrays.npz")
    ap.add_argument("idx", type=int, help="hypothesis index of the member")
    ap.add_argument("matches", help="the capture's cluster-patches .matches file")
    ap.add_argument("--repeat", type=int, default=1, help="runs to take the median of")
    args = ap.parse_args(argv)

    source = open_source(args.matches)
    runs = []
    for r in range(max(1, int(args.repeat))):
        clock, result = run_once(args.candidate_solves, args.idx, args.matches, source)
        runs.append(clock)
        if r == 0:
            state = result.state
            n = 0 if state is None else len(state["at_inf"])
            fin = (
                0 if state is None else int((~np.asarray(state["at_inf"], bool)).sum())
            )
            print(
                f"member h{args.idx}: refused={result.refused} "
                f"points={n} finite={fin} "
                f"placed={0 if state is None else len(state['frames'])}"
            )
            fc = result.census.get("fill", {})
            print(
                f"  fill-in: {fc.get('n_candidates')} candidates in "
                f"{fc.get('n_rings')} rings, {fc.get('n_added')} added"
            )

    merged = Clock()
    for label in {k for c in runs for k in c.total}:
        vals = sorted(c.total.get(label, 0.0) for c in runs)
        merged.total[label] = vals[len(vals) // 2]
        merged.calls[label] = runs[0].calls.get(label, 0)
    print()
    print(render(merged, merged.total.get("run_member", 0.0)))
    print()
    print(per_ring(runs[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
