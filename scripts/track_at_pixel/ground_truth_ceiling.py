# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the ground-truth tracks themselves, as if a candidate had returned them.

    pixi run -e test python scripts/track_at_pixel/ground_truth_ceiling.py --dataset DATASET

Every query the harness makes is answered with the held-out point's own track,
read by ``evaluate`` and scored by ``metrics.score`` against itself. A query
whose ground-truth track misses the good-track bar cannot be counted good for
any candidate that returns the same track, so the fraction that pass is the
ceiling of ``G`` the bar allows on that ground truth. The rows are written as a
run directory (``rows.jsonl`` with both passes, identical), so
``goal_score.py`` reads it like any run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import TrackAtPixelResult  # noqa: E402
from context import DatasetContext  # noqa: E402
from dataset import prepare  # noqa: E402
from harness import _jsonable  # noqa: E402
from metrics import ground_truth_reading, score  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="seoul_bull")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ds = DatasetContext(prepare(args.dataset, None, quiet=True))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in range(ds.recon.point_count):
        images = ds.point_images[p]
        if len(images) < 2:
            continue
        gt = ground_truth_reading(ds, p)
        track = gt["track"]
        for j, image in enumerate(images):
            image = int(image)
            pixel = ds.point_keypoints[p][j]
            q = next(
                i for i, o in enumerate(track.observations) if int(o["image"]) == image
            )
            result = TrackAtPixelResult(track=track, query_observation=q)
            row = {"point": p, "image": image, "status": "ok", "seconds": 0.0}
            row.update(score(ds, p, image, pixel, result, gt))
            for pass_name in ("full", "empty"):
                rows.append({**row, "pass": pass_name})
    with open(args.out / "rows.jsonl", "w") as sink:
        for r in rows:
            sink.write(json.dumps(_jsonable(r)) + "\n")
    good = sum(1 for r in rows if r["good"]) // 2
    print(f"{good} of {len(rows) // 2} queries: the ground truth meets the bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
