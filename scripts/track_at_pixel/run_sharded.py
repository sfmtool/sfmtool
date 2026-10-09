# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the harness over every point in parallel shards and merge the rows.

    pixi run -e test python scripts/track_at_pixel/run_sharded.py \\
        --dataset seoul_bull --candidate core_cascade --shards 16 --out RUN_DIR \\
        [--opt key=value ...] [--passes full,empty]

The points ``harness.py`` would choose (every point seen in two or more
images) are dealt round-robin into ``--shards`` groups, and each group runs as
its own ``harness.py --point-ids`` process writing to ``RUN_DIR/shard-NN``.
When every shard has finished, their rows are concatenated into
``RUN_DIR/rows.jsonl`` and summarised into ``RUN_DIR/summary.txt``, so the run
directory reads like a single harness run to ``compare.py``. The per-shard
``tracks-*.sfmr`` files stay in the shard directories.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def point_ids(dataset: str, cache_dir, min_track_length: int) -> list[int]:
    from sfmtool.reconstruction import (
        EditedReconstruction,
        SfmrReconstruction,
    )

    from dataset import prepare

    prepared = prepare(dataset, cache_dir, quiet=True)
    edited = EditedReconstruction(SfmrReconstruction.load(prepared.sfmr))
    return [
        p
        for p in range(edited.point_count)
        if len(edited.point(p)["image_indexes"]) >= min_track_length
    ]


def merge_tracks(out: Path, n_shards: int, pass_name: str, shard_rows) -> None:
    """Merge the shards' ``tracks-<pass>.sfmr`` into one file in the run directory.

    Each shard point is put on a bench, copied (a copy carries no origin, so
    its commit creates a point instead of replacing the one with its index)
    and committed, in shard order. The rows' ``output_point`` is renumbered to
    the merged file's indexes.
    """
    from sfmtool import bench as B
    from sfmtool.reconstruction import (
        EditedReconstruction,
        SfmrReconstruction,
    )

    merged = None
    for i in range(n_shards):
        path = out / f"shard-{i:02d}" / f"tracks-{pass_name}.sfmr"
        if not path.exists():
            continue
        src = SfmrReconstruction.load(path)
        if merged is None:
            base = src.filter_points_by_mask(np.zeros(src.point_count, dtype=bool))
            merged = EditedReconstruction(base.clone_with_changes(patch_bitmaps=None))
        edited = EditedReconstruction(src)
        index = {}
        for p in range(src.point_count):
            bench, _ = B.create_track(B.Bench(), edited, p)
            bench, copied = B.duplicate(bench, bench.labels[0])
            merged, committed = B.commit(
                merged,
                bench.track(copied["label"]),
                node=f"tracks-{pass_name}.sfmr",
            )
            index[p] = committed["point"]
        for r in shard_rows[i]:
            if r["pass"] == pass_name and "output_point" in r:
                r["output_point"] = index[r["output_point"]]
    if merged is None or merged.point_count == 0:
        return
    recon, _, _ = merged.materialize()
    recon.save(
        out / f"tracks-{pass_name}.sfmr",
        operation="track_at_pixel",
        tool_options={"pass": pass_name, "merged_shards": str(n_shards)},
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="seoul_bull")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--mode", choices=["track", "anchors"], default="track")
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--passes", default="full,empty")
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1, help="kernel threads per shard")
    ap.add_argument("--min-track-length", type=int, default=2)
    ap.add_argument("--points", type=int, default=0, help="sample this many (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    points = point_ids(args.dataset, args.cache_dir, args.min_track_length)
    if args.points and args.points < len(points):
        rng = np.random.default_rng(args.seed)
        points = sorted(int(p) for p in rng.choice(points, args.points, replace=False))
    shards = [points[i :: args.shards] for i in range(args.shards)]
    shards = [s for s in shards if s]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    procs = []
    for i, shard in enumerate(shards):
        out = args.out / f"shard-{i:02d}"
        cmd = [
            sys.executable,
            str(HERE / "harness.py"),
            "--dataset",
            args.dataset,
            "--candidate",
            args.candidate,
            "--passes",
            args.passes,
            "--mode",
            args.mode,
            "--point-ids",
            ",".join(map(str, shard)),
            "--out",
            str(out),
        ]
        for opt in args.opt:
            cmd += ["--opt", opt]
        if args.cache_dir:
            cmd += ["--cache-dir", str(args.cache_dir)]
        log = open(args.out / f"shard-{i:02d}.log", "w")
        # One kernel thread per shard: the shards are the parallelism, and a
        # rayon pool per process sized to the machine oversubscribes it.
        env = {**os.environ, "RAYON_NUM_THREADS": str(args.threads)}
        procs.append(
            (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log)
        )
    failed = 0
    for proc, log in procs:
        failed += proc.wait() != 0
        log.close()

    if args.mode == "anchors":
        from anchors import summarize
    else:
        from harness import summarize

    shard_rows = []
    for i in range(len(shards)):
        path = args.out / f"shard-{i:02d}" / "rows.jsonl"
        shard_rows.append(
            [json.loads(line) for line in open(path)] if path.exists() else []
        )
    passes = [p.strip() for p in args.passes.split(",") if p.strip()]
    for pass_name in passes if args.mode == "track" else []:
        merge_tracks(args.out, len(shards), pass_name, shard_rows)
    rows = [r for part in shard_rows for r in part]
    with open(args.out / "rows.jsonl", "w") as sink:
        for r in rows:
            sink.write(json.dumps(r) + "\n")
    summary = "\n\n".join(
        f"== {p} pass ==\n{summarize([r for r in rows if r['pass'] == p])}"
        for p in passes
    )
    (args.out / "summary.txt").write_text(summary + "\n")
    (args.out / "config.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "candidate": args.candidate,
                "options": args.opt,
                "shards": len(shards),
                "points": len(points),
                "wall_seconds": time.perf_counter() - t0,
            },
            indent=2,
        )
    )
    print(summary)
    print(
        f"\n{len(rows)} rows from {len(shards)} shards ({failed} failed) in "
        f"{time.perf_counter() - t0:.0f}s: {args.out}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
