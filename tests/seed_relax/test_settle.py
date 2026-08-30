# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The two fleet constants, and the rules that produce them."""

import json

from seed_relax import fleet_constants as FC
from seed_relax.derive_constants import ring_ratio_p1, settling_bar
from seed_relax.rings import octave_edges

STABILITY_COLUMNS = ["key", "family", "knots", "settle_pos", "n_finite_pre"]


def _stability(rows):
    """A `densify_stability.tsv` in the columns the settling rule reads."""
    out = ["\t".join(STABILITY_COLUMNS)]
    for r in rows:
        out.append("\t".join(str(r.get(c, "")) for c in STABILITY_COLUMNS))
    return "\n".join(out) + "\n"


def _member(key, settle_pos, n_finite_pre, knots=2, family="pinhole"):
    return {
        "key": key,
        "family": family,
        "knots": knots,
        "settle_pos": "" if settle_pos is None else settle_pos,
        "n_finite_pre": n_finite_pre,
    }


def _rows(text):
    lines = text.splitlines()
    cols = lines[0].split("\t")
    return [dict(zip(cols, ln.split("\t"))) for ln in lines[1:] if ln]


def test_the_settling_rule_reproduces_the_shipped_bar(tmp_path):
    # The decisive members of the study fleet: the worst member whose lens was
    # still moving carried 1662 finite points, and the smallest settled member
    # above it carried 1778.
    rows = [
        _member("a", 0, 1553),
        _member("b", 0, 1631),
        _member("c", 0, 1778),
        _member("d", 0, 2442),
        _member("e", 1, 1204),
        _member("f", 2, 1662),
        _member("g", None, 3100),
        # Another knot count, which the rule does not read.
        _member("h", 0, 12, knots=8),
    ]
    p = tmp_path / "densify_stability.tsv"
    p.write_text(_stability(rows), encoding="utf-8")
    bar, census = settling_bar(
        _rows(p.read_text(encoding="utf-8")), {r["key"] for r in rows}
    )
    assert bar == FC.SETTLING_FINITE_COUNT == 1778
    assert census["worst_unsettled_n_finite"] == 1662
    assert census["n_settled"] == 4
    assert census["n_unsettled"] == 2


def test_a_member_outside_the_pooled_set_does_not_move_the_bar(tmp_path):
    rows = [_member("a", 0, 1778), _member("f", 2, 1662), _member("x", 2, 5000)]
    p = tmp_path / "densify_stability.tsv"
    p.write_text(_stability(rows), encoding="utf-8")
    parsed = _rows(p.read_text(encoding="utf-8"))
    assert settling_bar(parsed, {"a", "f"})[0] == 1778
    # With the outlier pooled there is no settled member above every unsettled
    # one, and the rule states no bar rather than inventing one.
    assert settling_bar(parsed, {"a", "f", "x"})[0] is None


def test_a_population_with_nothing_unsettled_states_no_bar(tmp_path):
    rows = [_member("a", 0, 100), _member("b", 0, 200)]
    p = tmp_path / "densify_stability.tsv"
    p.write_text(_stability(rows), encoding="utf-8")
    bar, census = settling_bar(_rows(p.read_text(encoding="utf-8")), {"a", "b"})
    assert bar is None
    assert census["n_unsettled"] == 0


def test_the_ring_rule_reproduces_five_rings(tmp_path):
    p = tmp_path / "ring_edges.json"
    p.write_text(
        json.dumps(
            {
                "ratio_p1": 0.08870889328692777,
                "n_members_pooled": 39,
                "n_candidates_pooled": 1491855,
                "octave_edges": [
                    float("inf"),
                    1.0,
                    0.5,
                    0.25,
                    0.125,
                    0.0625,
                ],
            }
        ),
        encoding="utf-8",
    )
    p1, census = ring_ratio_p1(p)
    assert round(p1, 4) == FC.RING_RATIO_P1
    assert census["n_members_pooled"] == 39
    edges = octave_edges(round(p1, 4))
    assert len(edges) - 1 == 5
    assert edges == census["octave_edges"]


def test_every_constant_carries_its_provenance():
    for prov in (
        FC.SETTLING_FINITE_COUNT_PROVENANCE,
        FC.RING_RATIO_P1_PROVENANCE,
    ):
        for field in ("fleet", "source_table", "source_table_md5", "rule"):
            assert prov[field]
        assert len(prov["source_table_md5"]) == 32
        assert prov["rule"].endswith(".")
