# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Built, correct and wrong tracks and precision, per harness run directory.

    pixi run python scripts/track_at_pixel/summarize_sweep.py RUN_DIR [RUN_DIR ...]

A built track of the full pass is correct when it passes the harness's good
bar (``metrics.GOOD_BAR``) with the bar's own ZNCC test left out, since the
median gate a sweep moves reads the same score. Precision is correct over
built. ``sweep_median_gate.sh`` writes the run directories this reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import good_failures  # noqa: E402


def summarize(run: Path) -> str:
    rows = [json.loads(line) for line in open(run / "rows.jsonl")]
    rows = [r for r in rows if r.get("pass", "full") == "full"]
    n = len(rows)
    built = [r for r in rows if r["status"] == "ok"]
    correct = [r for r in built if not [f for f in good_failures(r) if f != "zncc"]]
    wrong = len(built) - len(correct)
    precision = len(correct) / len(built) if built else float("nan")
    share = 100 * len(built) / n if n else float("nan")
    return (
        f"| {run.name} | {n} | {len(built)} ({share:.1f}%) | {len(correct)} | {wrong} "
        f"| {precision:.3f} |"
    )


def main() -> None:
    runs = [Path(a) for a in sys.argv[1:] if (Path(a) / "rows.jsonl").exists()]
    if not runs:
        raise SystemExit("usage: summarize_sweep.py RUN_DIR [RUN_DIR ...]")
    print("| run | queries | built | correct | wrong | precision |")
    print("|---|---|---|---|---|---|")
    for run in runs:
        print(summarize(run))


if __name__ == "__main__":
    main()
