# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Put several harness runs side by side.

    pixi run -e test python scripts/track_at_pixel/compare.py RUN_DIR [RUN_DIR ...]

Each run is re-judged against the current ``metrics.GOOD_BAR``, so runs made
before a change to the bar are still compared on one scale. Queries are matched
by ``(pass, point, image)``, and each pass is reported on its own line; a row
from a run made before the harness had passes counts as the ``full`` pass. The
counts over the queries every run holds are printed as well, so runs over
different samples are not mistaken for each other.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import good_failures  # noqa: E402


def load(run: Path) -> dict[tuple[int, int], dict]:
    rows = {}
    for line in open(run / "rows.jsonl"):
        r = json.loads(line)
        if r["status"] == "ok":
            r["good"] = not good_failures(r)
        rows[(r.get("pass", "full"), r["point"], r["image"])] = r
    return rows


def median(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(vals)) if vals else float("nan")


def main(argv=None) -> int:
    runs = [Path(a) for a in (argv or sys.argv[1:])]
    data = {run: load(run) for run in runs}
    common = set.intersection(*(set(d) for d in data.values()))
    print(f"{len(common)} queries held by every run\n")
    head = (
        f"{'run':34s} {'queries':>7s} {'built':>6s} {'good':>6s} {'bad':>5s} "
        f"{'good%':>6s} {'common good':>11s} {'angle':>7s} {'normal':>7s} "
        f"{'recall':>6s} {'zncc':>6s} {'sec':>6s}"
    )
    print(head)
    passes = sorted(
        {k[0] for d in data.values() for k in d}, key=["full", "empty"].index
    )
    for pass_name in passes:
        for run, all_rows in data.items():
            rows = {k: r for k, r in all_rows.items() if k[0] == pass_name}
            if not rows:
                continue
            ok = [r for r in rows.values() if r["status"] == "ok"]
            good = [r for r in ok if r["good"]]
            common_good = sum(
                1 for k in common if k[0] == pass_name and rows[k].get("good")
            )
            label = f"{run.name[:26]} [{pass_name}]"
            print(
                f"{label:34s} {len(rows):7d} {len(ok):6d} {len(good):6d} "
                f"{len(ok) - len(good):5d} {100 * len(good) / max(len(rows), 1):5.1f}% "
                f"{common_good:11d} {median(good, 'position_err_angle_deg'):7.3f} "
                f"{median(good, 'normal_err_deg'):7.2f} {median(good, 'image_recall'):6.2f} "
                f"{median(good, 'zncc_median'):6.3f} "
                f"{median(list(rows.values()), 'seconds'):6.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
