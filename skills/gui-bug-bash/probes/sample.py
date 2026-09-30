# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Call one read tool over many indexes from one process, into JSON lines.

    python probes/sample.py --port P --tool get_point --arg point \
        --random 400 --below 18991 --out points.jsonl
    python probes/sample.py --port P --tool get_camera_image --arg camera_image \
        --range 0:85 --extra '{"reconstruction_label": "kerry"}' --out images.jsonl

Each line is {"index": i, "reply": ...} or {"index": i, "error": "..."}. The
summary counts the replies and groups the errors by message, which is often a
finding in itself: a run of "no live point" errors means the bare index went
to a different reconstruction than intended (bare indexes follow the
selection). Analyse the file afterwards with a few lines of Python.
"""

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp import Client, McpError  # noqa: E402


def indexes(a: argparse.Namespace) -> list[int]:
    if a.range:
        parts = [int(x) for x in a.range.split(":")]
        return list(range(*parts))
    if a.random is not None and a.below is not None:
        rng = random.Random(a.seed)
        return sorted(rng.sample(range(a.below), min(a.random, a.below)))
    raise SystemExit("give --range START:STOP[:STEP], or --random N with --below M")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--tool", required=True)
    p.add_argument("--arg", required=True, help="the argument the index goes in")
    p.add_argument("--extra", default="{}", help="JSON object of fixed arguments")
    p.add_argument("--range")
    p.add_argument("--random", type=int)
    p.add_argument("--below", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    client = Client(a.port)
    extra = json.loads(a.extra)
    ok = 0
    errors: collections.Counter[str] = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as out:
        for i in indexes(a):
            try:
                reply = client.call(a.tool, {**extra, a.arg: i})
                out.write(json.dumps({"index": i, "reply": reply}) + "\n")
                ok += 1
            except McpError as e:
                out.write(json.dumps({"index": i, "error": str(e)}) + "\n")
                errors[re.sub(r"\b\d+\b", "N", str(e))[:160]] += 1
    print(f"{ok} replies, {sum(errors.values())} errors -> {a.out}")
    for message, n in errors.most_common(10):
        print(f"  {n} x {message}")


if __name__ == "__main__":
    main()
