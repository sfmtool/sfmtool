# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Goal score for track-at-pixel runs, normal-weighted, with speed.

    pixi run -e test python scripts/track_at_pixel/goal_score.py RUN_DIR [RUN_DIR ...]

Per (run, pass):
  G  = good / queries                         (the harness's good-track bar)
  N  = sum over good tracks of max(0, 1 - normal_err/30deg) / queries
       (good tracks at infinity get credit 1: they have no normal)
  S  = (G + 2 N) / 3                          (perfect = 1)
Also: N10 = good with normal err <= 10 deg, median normal err of good,
median and p90 seconds per query.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import good_failures  # noqa: E402

NORMAL_SCALE_DEG = 30.0


def credit(r):
    if r.get("gt_at_infinity") or r.get("normal_err_deg") is None:
        return 1.0
    return max(0.0, 1.0 - r["normal_err_deg"] / NORMAL_SCALE_DEG)


def score_rows(rows):
    n = len(rows)
    good = [r for r in rows if r["status"] == "ok" and not good_failures(r)]
    G = len(good) / n
    N = sum(credit(r) for r in good) / n
    errs = [r["normal_err_deg"] for r in good if r.get("normal_err_deg") is not None]
    n10 = sum(1 for e in errs if e <= 10) / n
    secs = np.asarray([r["seconds"] for r in rows])
    return {
        "queries": n,
        "built": sum(r["status"] == "ok" for r in rows),
        "good": len(good),
        "G": G,
        "N": N,
        "S": (G + 2 * N) / 3,
        "N10": n10,
        "normal_med": float(np.median(errs)) if errs else float("nan"),
        "sec_med": float(np.median(secs)),
        "sec_p90": float(np.percentile(secs, 90)),
    }


def load(run: Path):
    rows = [json.loads(line) for line in open(run / "rows.jsonl")]
    return {p: [r for r in rows if r["pass"] == p] for p in ("full", "empty")}


def main():
    print(
        f"{'run':38s} {'pass':5s} {'q':>5s} {'good':>5s} {'G':>6s} {'N':>6s} "
        f"{'S':>6s} {'N10':>6s} {'nrm°':>6s} {'s/q':>6s} {'p90':>6s}"
    )
    for arg in [a for a in sys.argv[1:] if Path(a).is_dir()]:
        run = Path(arg)
        per = load(run)
        ss = []
        for p, rows in per.items():
            if not rows:
                continue
            m = score_rows(rows)
            ss.append(m["S"])
            print(
                f"{run.name[:38]:38s} {p:5s} {m['queries']:5d} {m['good']:5d} "
                f"{m['G']:6.3f} {m['N']:6.3f} {m['S']:6.3f} {m['N10']:6.3f} "
                f"{m['normal_med']:6.1f} {m['sec_med']:6.2f} {m['sec_p90']:6.2f}"
            )
        print(
            f"{'':38s} {'mean':5s} {'':5s} {'':5s} {'':6s} {'':6s} {np.mean(ss):6.3f}"
        )


if __name__ == "__main__":
    main()
