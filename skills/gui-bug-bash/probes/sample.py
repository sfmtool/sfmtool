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

The calls go through one `sfm explorer-call --batch` process, which checks the
tool and every argument object against the extension's catalog first; pass
--validate-tool-names-only for a viewer whose catalog differs from it.
"""

import argparse
import collections
import json
import random
import re
import shutil
import subprocess
import sys


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
    p.add_argument(
        "--validate-tool-names-only",
        action="store_true",
        help="passed on to sfm explorer-call",
    )
    a = p.parse_args()

    sfm = shutil.which("sfm")
    if sfm is None:
        raise SystemExit("`sfm` is not on PATH: run this with `pixi run python`")
    extra = json.loads(a.extra)
    chosen = indexes(a)
    lines = "".join(json.dumps({**extra, a.arg: i}) + "\n" for i in chosen)
    command = [sfm, "explorer-call", "--mcp", str(a.port), a.tool, "--batch", "-"]
    if a.validate_tool_names_only:
        command.append("--validate-tool-names-only")
    done = subprocess.run(
        command, input=lines, capture_output=True, text=True, encoding="utf-8"
    )
    if done.returncode != 0:
        raise SystemExit(done.stderr.strip() or f"sfm explorer-call: {done.returncode}")

    ok = 0
    errors: collections.Counter[str] = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as out:
        for i, line in zip(chosen, done.stdout.splitlines()):
            record = json.loads(line)
            if "reply" in record:
                out.write(json.dumps({"index": i, "reply": record["reply"]}) + "\n")
                ok += 1
            else:
                error = record["error"]
                out.write(json.dumps({"index": i, "error": error}) + "\n")
                errors[re.sub(r"\b\d+\b", "N", error)[:160]] += 1
    print(f"{ok} replies, {sum(errors.values())} errors -> {a.out}")
    for message, n in errors.most_common(10):
        print(f"  {n} x {message}")


if __name__ == "__main__":
    main()
