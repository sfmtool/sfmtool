# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Recompute the relaxation's two fleet constants from a study directory.

Provenance: the settling rule is the stabilisation reading of the study's
`v2/densify/densify_assemble.py` (the vote-spread bar at 80-96 and the
settle-position sweep at 246-340), and the ring rule is
`v2/densify/ring_edges.py` (97-116).  Both are read back out of the tables
those passes wrote, so this script needs no workspace and no member.

    pixi run -e dev python scripts/seed_relax/derive_constants.py --study DIR

``DIR`` is the directory holding `densify_stability.tsv`, `ring_pool.tsv` and
`ring_edges.json`.  What comes out is the two constants with their inputs
checksummed, and a comparison against the values `fleet_constants.py` carries,
so a later fleet re-derives them rather than re-tuning them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seed_relax import fleet_constants as FC  # noqa: E402
from seed_relax.rings import octave_edges  # noqa: E402

#: The knot count the perspective chart's late release adopts.  The bar is read
#: at the knot count it will be applied at: a lens with more freedom settles at
#: a different amount of evidence.
PINHOLE_KNOTS = 2


def md5_of(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def read_tsv(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    cols = lines[0].split("\t")
    return [dict(zip(cols, ln.split("\t"))) for ln in lines[1:] if ln]


def num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def settling_bar(stability_rows, member_keys, knots=PINHOLE_KNOTS):
    """The finite-point count a perspective lens reading needs.

    A member is SETTLED when no position after the pre-fill one moves its
    released equivalent focal by as much as the focal vote's own spread, which
    is what `settle_pos == 0` records.  The bar is the smallest finite count
    among the settled members that no unsettled member reaches: below it the
    population still contains members whose lens is moving, above it none
    does."""
    settled, unsettled = [], []
    for r in stability_rows:
        if num(r.get("knots")) != float(knots):
            continue
        if r.get("key") not in member_keys:
            continue
        n_fin = num(r.get("n_finite_pre"))
        if n_fin is None:
            continue
        pos = r.get("settle_pos")
        if num(pos) == 0.0:
            settled.append(n_fin)
        elif pos not in ("", None):
            unsettled.append(n_fin)
    if not settled or not unsettled:
        return None, {"n_settled": len(settled), "n_unsettled": len(unsettled)}
    worst = max(unsettled)
    above = sorted(x for x in settled if x > worst)
    census = {
        "n_settled": len(settled),
        "n_unsettled": len(unsettled),
        "worst_unsettled_n_finite": int(worst),
    }
    return (int(above[0]) if above else None), census


def ring_ratio_p1(edges_json):
    """The pooled first-percentile candidate radius over the admission floor.

    The pooled percentile is a statistic of the whole pool, and `ring_pool.tsv`
    records only each member's own percentiles, so the value is read from the
    pooled reading the same pass wrote beside that table.  The member census
    below cross-checks that the two describe one pooling run."""
    d = json.loads(Path(edges_json).read_text(encoding="utf-8"))
    return float(d["ratio_p1"]), {
        "n_members_pooled": int(d["n_members_pooled"]),
        "n_candidates_pooled": int(d["n_candidates_pooled"]),
        "octave_edges": [float(x) for x in d["octave_edges"]],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--study",
        required=True,
        help="the densify study directory holding the three tables",
    )
    ap.add_argument("--knots", type=int, default=PINHOLE_KNOTS)
    args = ap.parse_args()

    study = Path(args.study)
    stability = study / "densify_stability.tsv"
    pool = study / "ring_pool.tsv"
    edges_json = study / "ring_edges.json"
    for p in (stability, pool, edges_json):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    pool_rows = read_tsv(pool)
    member_keys = {r["key"] for r in pool_rows if r.get("key")}
    stab_rows = read_tsv(stability)

    bar, bar_census = settling_bar(stab_rows, member_keys, knots=args.knots)
    p1, p1_census = ring_ratio_p1(edges_json)
    edges = octave_edges(round(p1, 4))

    print(f"study directory: {study}")
    print(f"  densify_stability.tsv  md5 {md5_of(stability)}  {len(stab_rows)} rows")
    print(f"  ring_pool.tsv          md5 {md5_of(pool)}  {len(member_keys)} members")
    print(f"  ring_edges.json        md5 {md5_of(edges_json)}")
    print()
    print("SETTLING_FINITE_COUNT")
    print(f"  rule: {FC.SETTLING_FINITE_COUNT_PROVENANCE['rule']}")
    print(
        f"  at knots {args.knots}: {bar_census['n_settled']} settled, "
        f"{bar_census['n_unsettled']} unsettled, worst unsettled finite "
        f"count {bar_census.get('worst_unsettled_n_finite')}"
    )
    print(
        f"  derived {bar}   shipped {FC.SETTLING_FINITE_COUNT}   "
        f"{'MATCH' if bar == FC.SETTLING_FINITE_COUNT else 'DIFFERS'}"
    )
    print()
    print("RING_RATIO_P1")
    print(f"  rule: {FC.RING_RATIO_P1_PROVENANCE['rule']}")
    print(
        f"  pooled over {p1_census['n_members_pooled']} members, "
        f"{p1_census['n_candidates_pooled']} candidate clusters"
    )
    print(
        f"  derived {p1:.6f} -> {round(p1, 4)}   shipped {FC.RING_RATIO_P1}   "
        f"{'MATCH' if round(p1, 4) == FC.RING_RATIO_P1 else 'DIFFERS'}"
    )
    print(f"  octave grid: {len(edges) - 1} rings, edges {edges}")
    same = edges == [float(x) for x in p1_census["octave_edges"]]
    print(f"  against the study's own edges: {'MATCH' if same else 'DIFFERS'}")


if __name__ == "__main__":
    main()
