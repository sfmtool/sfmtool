# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a rule that stops the trailing solve of a scanlines race early.

Reads the JSON the ``scanlines_race`` example of ``sfmtool-core`` writes
(``crates/sfmtool-core/examples/scanlines_race.rs``): both solves of each race
run to the end, with the cost each schedule round's damping ladder compares
before its first iteration and after each. It replays the two trajectories in
lockstep and prints Markdown tables: each solve's rounds, the gap between the
solves as the race runs, the first point after which the gap's sign never
changes, and what each variant of the rule would have done.

The gap after a pair of steps is ``D = (C_trailing - C_leading) / sigma^2``,
with ``sigma^2`` the mean of the two solves' noise levels squared in the round
the pair is in, as each round measures it at its start (the first round's is
the same for both, since they start from one state). The rule stops the trailing solve when, for ``K``
consecutive pairs with the same solve trailing, ``D > T`` and ``D`` exceeds
what the trailing solve would remove in the rest of its budget at its recent
rate: the mean of its last three decreases times the steps the schedule has
left. Two lockstep units are replayed:

- ``iteration``: one LM iteration of each solve per pair, the two held at
  each round's end until both have finished the round, so the costs compared
  are always at one loss scale. A solve that has finished the round keeps its
  last cost. The budget is the iterations left in the round plus ``max_iters``
  for each round after it. The pair at a round's start, before either solve's
  first iteration in it, is skipped: it neither counts toward ``K`` nor breaks
  a run.
- ``round``: one schedule round of each solve per pair, compared at the
  round's end. The rate is the mean decrease over the last three rounds, each
  from its start to its end, and the budget is the rounds left.

The rule the adjustment runs is the ``iteration`` one at ``T = 100`` and
``K = 3``, built as ``EarlyStopRule`` in
``crates/sfmtool-core/src/reconstruction/bundle_adjust/race.rs`` with these
definitions of the pairs, the gap, the rate and the budget, and one more
condition this replay does not apply: no pair holds while the two rounds kept
different counts of observations. Record the trajectories with the example's
early stop off, its default, so that both solves run to the end.

Usage::

    pixi run python scripts/scanlines_race_early_stop.py out1.json [out2.json ...]
"""

import json
import sys

THRESHOLDS = (10.0, 25.0, 100.0)
COUNTS = (1, 3, 5)
RATE_WINDOW = 3


def load_race(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, data["races"]


def winner_of(race):
    """0 for rows, 1 for columns: the lower race cost, a tie to rows."""
    return 1 if race["columns_cost"] < race["rows_cost"] else 0


def rounds_of(race):
    solves = race["solves"]
    return [solves["rows"]["rounds"], solves["columns"]["rounds"]]


def iterations(rounds):
    return [len(r["costs"]) - 1 for r in rounds]


def round_sigma2(rounds, r):
    """The mean of the two solves' noise levels squared in round ``r``."""
    return 0.5 * (rounds[0][r]["noise"] ** 2 + rounds[1][r]["noise"] ** 2)


def iteration_pairs(rounds):
    """Lockstep per LM iteration: ``(round, i, [c_rows, c_cols], ran)`` per
    pair, the costs in units of the round's ``sigma^2``, and how many
    iterations each solve has run by the pair."""
    pairs = []
    n_rounds = min(len(rounds[0]), len(rounds[1]))
    before = [0, 0]
    for r in range(n_rounds):
        sigma2 = round_sigma2(rounds, r)
        costs = [rounds[s][r]["costs"] for s in range(2)]
        n = max(len(c) for c in costs)
        for i in range(n):
            c = [costs[s][min(i, len(costs[s]) - 1)] / sigma2 for s in range(2)]
            ran = [before[s] + min(i, len(costs[s]) - 1) for s in range(2)]
            pairs.append((r, i, c, ran))
        for s in range(2):
            before[s] += len(costs[s]) - 1
    return pairs


def iteration_rate(costs, i, sigma2):
    """Mean of the last ``RATE_WINDOW`` decreases up to step ``i``, or None."""
    i = min(i, len(costs) - 1)
    steps = [(costs[k - 1] - costs[k]) / sigma2 for k in range(1, i + 1)]
    if not steps:
        return None
    window = steps[-RATE_WINDOW:]
    return sum(window) / len(window)


def evaluate_iteration(rounds, max_iters, threshold, count, budget=True):
    """The first pair at which the rule fires per LM iteration, as
    ``(pair, round, i, stopped, d)``, or None."""
    n_rounds = min(len(rounds[0]), len(rounds[1]))
    run, trailing_before = 0, None
    for p, (r, i, c, _) in enumerate(iteration_pairs(rounds)):
        # A round's start, before either solve's first iteration in it, says
        # nothing about the round yet: it is neither counted nor a reset.
        if i == 0:
            continue
        trailing = 0 if c[0] > c[1] else 1
        d = abs(c[0] - c[1])
        rate = iteration_rate(rounds[trailing][r]["costs"], i, round_sigma2(rounds, r))
        left = max(0, max_iters - i) + max_iters * (n_rounds - 1 - r)
        holds = d > threshold and rate is not None
        if holds and budget:
            holds = d > rate * left
        if holds and trailing == trailing_before:
            run += 1
        elif holds:
            run = 1
        else:
            run = 0
        trailing_before = trailing if holds else None
        if run >= count:
            return p, r, i, trailing, d
    return None


def evaluate_round(rounds, threshold, count, budget=True):
    """The first round end at which the rule fires per schedule round, as
    ``(round, stopped, d)``, or None."""
    n_rounds = min(len(rounds[0]), len(rounds[1]))
    run, trailing_before = 0, None
    for r in range(n_rounds):
        sigma2 = round_sigma2(rounds, r)
        c = [rounds[s][r]["costs"][-1] / sigma2 for s in range(2)]
        trailing = 0 if c[0] > c[1] else 1
        d = abs(c[0] - c[1])
        drops = [
            (rounds[trailing][k]["costs"][0] - rounds[trailing][k]["costs"][-1])
            / round_sigma2(rounds, k)
            for k in range(max(0, r + 1 - RATE_WINDOW), r + 1)
        ]
        rate = sum(drops) / len(drops)
        left = n_rounds - 1 - r
        holds = d > threshold and (not budget or d > rate * left)
        if holds and trailing == trailing_before:
            run += 1
        elif holds:
            run = 1
        else:
            run = 0
        trailing_before = trailing if holds else None
        if run >= count:
            return r, trailing, d
    return None


NAMES = ("rows", "columns")


def describe_rounds(rounds):
    lines = [
        "| Solve | Round | Iterations | Kept | Noise, px | Start cost | End cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in range(2):
        for r, rd in enumerate(rounds[s]):
            sigma2 = round_sigma2(rounds, r)
            noise = rd["noise"]
            lines.append(
                f"| {NAMES[s]} | {r + 1} | {len(rd['costs']) - 1} | {rd['kept']:.0f} | "
                f"{noise:.4f} | {rd['costs'][0] / sigma2:.1f} | {rd['costs'][-1] / sigma2:.1f} |"
                if noise is not None
                else f"| {NAMES[s]} | {r + 1} | {len(rd['costs']) - 1} | {rd['kept']:.0f} | - | "
                f"{rd['costs'][0] / sigma2:.1f} | {rd['costs'][-1] / sigma2:.1f} |"
            )
    return "\n".join(lines)


def describe_gap(rounds):
    """The signed gap ``(C_rows - C_columns)/sigma^2`` at a few iterations of
    each round, and at its end."""
    marks = (0, 1, 2, 3, 5, 10, 20, 40)
    pairs = iteration_pairs(rounds)
    lines = ["| Round | " + " | ".join(f"i={m}" for m in marks) + " | end |"]
    lines.append("|---" * (len(marks) + 2) + "|")
    n_rounds = min(len(rounds[0]), len(rounds[1]))
    for r in range(n_rounds):
        mine = [(i, c) for (rr, i, c, _) in pairs if rr == r]
        cells = []
        for m in marks:
            hit = [c for (i, c) in mine if i == m]
            cells.append(f"{hit[0][0] - hit[0][1]:+.1f}" if hit else "")
        end = mine[-1][1]
        lines.append(
            f"| {r + 1} | " + " | ".join(cells) + f" | {end[0] - end[1]:+.1f} |"
        )
    return "\n".join(lines)


def stable_sign(rounds, winner):
    """The first pair after which the gap's sign never changes from the
    winner's, as ``(pair, round, i, pairs)``, or None when the last pair's sign
    is not the winner's."""
    pairs = iteration_pairs(rounds)
    want = -1 if winner == 0 else 1  # rows winning means c_rows < c_cols

    def sign(c):
        diff = c[0] - c[1]
        return (diff > 0) - (diff < 0)

    if sign(pairs[-1][2]) != want:
        return None
    last_bad = -1
    for p, (_, _, c, _) in enumerate(pairs):
        if sign(c) != want:
            last_bad = p
    first = last_bad + 1
    r, i, _, _ = pairs[first]
    return first, r, i, len(pairs)


def report(path):
    data, races = load_race(path)
    max_iters = data["max_iters"]
    out = [f"## {data['input']} ({data['schedule']} schedule)", ""]
    for race in races:
        rounds = rounds_of(race)
        noise = rounds[0][0]["noise"]
        assert noise == rounds[1][0]["noise"], "the two solves share their first round"
        winner = winner_of(race)
        its = [sum(iterations(rounds[s])) for s in range(2)]
        out.append(
            f"Camera {race['camera']}: race cost {race['rows_cost']:.1f} with rows, "
            f"{race['columns_cost']:.1f} with columns (sigma {race['sigma_px']:.4f} px, "
            f"{race['observations']} observations); winner {NAMES[winner]}; "
            f"gap {abs(race['rows_cost'] - race['columns_cost']):.1f}. Iterations: "
            f"rows {its[0]}, columns {its[1]}. Seconds: rows "
            f"{race['solves']['rows']['seconds']:.1f}, columns "
            f"{race['solves']['columns']['seconds']:.1f}. First-round noise {noise:.4f} px."
        )
        out.append("")
        out.append(describe_rounds(rounds))
        out.append("")
        out.append("Signed gap (C_rows - C_columns)/sigma^2:")
        out.append("")
        out.append(describe_gap(rounds))
        out.append("")
        stable = stable_sign(rounds, winner)
        if stable is None:
            out.append("The last pair's sign is not the race winner's.")
        else:
            p, r, i, n = stable
            out.append(
                f"Sign stable from pair {p} of {n} (round {r + 1}, iteration {i})."
            )
        out.append("")
        out.append(
            "| Unit | T | K | Budget | Fires at | Stops | Correct | Iterations saved |"
        )
        out.append("|---|---|---|---|---|---|---|---|")
        variants = [(t, k, True) for t in THRESHOLDS for k in COUNTS] + [
            (25.0, 3, False)
        ]
        pairs = iteration_pairs(rounds)
        for t, k, budget in variants:
            hit = evaluate_iteration(rounds, max_iters, t, k, budget)
            if hit is None:
                cells = "never | | | 0"
            else:
                p, r, i, stopped, d = hit
                ran = pairs[p][3][stopped]
                saved = its[stopped] - ran
                cells = (
                    f"round {r + 1}, i={i} (D {d:.1f}) | {NAMES[stopped]} | "
                    f"{'yes' if stopped != winner else '**no**'} | {saved}"
                )
            out.append(
                f"| iteration | {t:g} | {k} | {'on' if budget else 'off'} | {cells} |"
            )
        for t, k, budget in variants:
            hit = evaluate_round(rounds, t, k, budget)
            if hit is None:
                cells = "never | | | 0"
            else:
                r, stopped, d = hit
                saved = sum(iterations(rounds[stopped])[r + 1 :])
                cells = (
                    f"end of round {r + 1} (D {d:.1f}) | {NAMES[stopped]} | "
                    f"{'yes' if stopped != winner else '**no**'} | {saved}"
                )
            out.append(
                f"| round | {t:g} | {k} | {'on' if budget else 'off'} | {cells} |"
            )
        out.append("")
    return "\n".join(out)


def main(paths):
    for path in paths:
        print(report(path))


if __name__ == "__main__":
    main(sys.argv[1:])
