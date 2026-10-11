# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

r"""The reconstructions the reference-view measurement reads.

The keys are the ones the review cases (``cases.json``) name: the two
checked-in ground truths and eight solves outside the repository. Seven of the
solves are the files ``scripts/bench_bars/datasets.py`` lists, found under
``BENCH_BARS_DATASETS_DIR`` (``C:\DataSets`` when it is not set), with
``omni_temple`` and ``omni_coast`` added here; ``omni_coast`` has no review
cases and is read for the pool alone. ``xmas`` is the 30-frame Christmas-tree
subset that ``bench_bars/datasets.py`` describes, found at ``BENCH_BARS_XMAS``.

Every file is read in place and never written. ``--sfmr NAME=PATH`` on
``measure.py`` points at a file elsewhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# bench_bars' module is also named ``datasets``, so it is loaded by path.
_spec = importlib.util.spec_from_file_location(
    "bench_bars_datasets",
    Path(__file__).resolve().parents[1] / "bench_bars/datasets.py",
)
_bench_bars = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _bench_bars
_spec.loader.exec_module(_bench_bars)
BENCH_BARS = _bench_bars.DATASETS
DATASETS_DIR = _bench_bars.DATASETS_DIR
REPO = _bench_bars.REPO
read_image = _bench_bars.read_image

__all__ = ["DATASETS", "DATASETS_DIR", "REPO", "read_image", "resolve"]

_OMNI = "sfmr/20260823-01-solve-rigcfg-frame_1-{}x12-clean-3-ba-embedded.sfmr"

DATASETS: dict[str, Path | None] = {
    "seoul_bull": BENCH_BARS["seoul_gt"].sfmr,
    "kerry_park": BENCH_BARS["kerry_gt"].sfmr,
    "dino": BENCH_BARS["dino"].sfmr,
    "altona": BENCH_BARS["altona"].sfmr,
    "badland": BENCH_BARS["badland"].sfmr,
    "mossy": BENCH_BARS["mossy"].sfmr,
    "kerry480": BENCH_BARS["kerry480"].sfmr,
    "omni_temple": DATASETS_DIR / "OmniTemple1" / _OMNI.format(469),
    "omni_coast": DATASETS_DIR / "OmniCoast" / _OMNI.format(517),
    "xmas": BENCH_BARS["xmas"].sfmr,
}


def resolve(name: str, overrides: dict[str, str] | None = None) -> Path:
    """The ``.sfmr`` to read for ``name``, with ``--sfmr`` overrides applied."""
    if overrides and name in overrides:
        return Path(overrides[name])
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name!r} (known: {', '.join(DATASETS)})")
    path = DATASETS[name]
    if path is None:
        raise SystemExit(
            f"no path for {name!r}: set BENCH_BARS_XMAS or pass --sfmr {name}=PATH"
        )
    return path
