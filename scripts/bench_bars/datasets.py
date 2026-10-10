# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

r"""The eight reconstructions the bench-bar measurement samples tracks from.

Two are the checked-in ground truths, whose members are correct; six are
solves outside the repository, whose members are assumed correct. Every file is
read in place and never written to: the measurement loads it, drops its stored
patch bitmaps in memory, and writes only its own ``.jsonl`` output.

The paths are data, in ``DATASETS`` below. Five of the solves are found under
``BENCH_BARS_DATASETS_DIR`` (an environment variable, ``C:\DataSets`` when it is
not set), and ``xmas`` at ``BENCH_BARS_XMAS`` (no default). For any one
reconstruction, ``--sfmr <name>=<path>`` on ``measure.py`` overrides both. Each
file's own workspace must hold its images.

``xmas`` is not a file of the dataset as it was solved. The solve of
``ChristmasTreeWithPresents`` is a ``sift_files`` reconstruction of 4054 frames
with no patches; the file measured is a 30-frame subset of it (frames 1501 to
1675, every sixth, image indices 1501 + 6k), kept to the points seen in three
or more of those frames and run through ``embed_patches`` with its defaults
(2026-10-06), saved outside the dataset directory. To build it again, subset the
solve with ``SfmrReconstruction.subset_by_image_indices`` and
``filter_points_by_mask(observation_counts >= 3)``, save it, and run
``sfm embed-patches`` on that file. A rebuild under a later ``embed_patches``
gives different tracks, so its figures are not those of the file measured on
2026-10-10, which was kept outside the repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATASETS_DIR = Path(os.environ.get("BENCH_BARS_DATASETS_DIR", r"C:\DataSets"))
# The derived Christmas-tree file (see the module docstring); no default.
XMAS_DERIVED = (
    Path(os.environ["BENCH_BARS_XMAS"]) if "BENCH_BARS_XMAS" in os.environ else None
)


@dataclass(frozen=True)
class Dataset:
    name: str
    label: str
    sfmr: Path | None
    ground_truth: bool


DATASETS = {
    d.name: d
    for d in [
        Dataset(
            "seoul_gt",
            "seoul_bull ground truth",
            REPO
            / "test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr",
            True,
        ),
        Dataset(
            "kerry_gt",
            "kerry_park ground truth",
            REPO / "test-data/images/kerry_park/kerry_park_ground_truth.sfmr",
            True,
        ),
        Dataset(
            "kerry480",
            "kerry480",
            DATASETS_DIR
            / "KerryPark480/sfmr/20260823-01-solve-frame_1-24-clean-2b-ba-embedded.sfmr",
            False,
        ),
        Dataset(
            "badland",
            "badlands",
            DATASETS_DIR / "BadlandPanorama/sfmr/seed-final.sfmr",
            False,
        ),
        Dataset(
            "mossy",
            "mossy railing",
            DATASETS_DIR / "MossyRailing/sfmr/seed-final.sfmr",
            False,
        ),
        Dataset(
            "altona",
            "gallery sculpture",
            DATASETS_DIR / "AltonaGalleryInTheParkBoyReading/sfmr/seed-final.sfmr",
            False,
        ),
        Dataset(
            "dino",
            "dino toy",
            DATASETS_DIR / "DinoDogToyWS/sfmr/bugbash-20260929/dino-saved2.sfmr",
            False,
        ),
        Dataset("xmas", "Christmas tree", XMAS_DERIVED, False),
    ]
}
NAMES = list(DATASETS)


def resolve(name: str, overrides: dict[str, str] | None = None) -> Path:
    """The ``.sfmr`` to read for ``name``, with ``--sfmr`` overrides applied."""
    if overrides and name in overrides:
        return Path(overrides[name])
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name!r} (known: {', '.join(NAMES)})")
    path = DATASETS[name].sfmr
    if path is None:
        raise SystemExit(
            f"no path for {name!r}: set BENCH_BARS_XMAS or pass --sfmr {name}=PATH"
        )
    return path


def read_image(workspace_dir: str | Path, image_name: str) -> np.ndarray:
    """One workspace image as a contiguous RGB array, as the patch kernels read it.

    sfmtool has no public reader for a workspace image, so this reads it with
    OpenCV the way the package's own loader does: ``workspace_dir /
    image_name``, converted from BGR to RGB.
    """
    import cv2

    path = Path(workspace_dir) / image_name
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"image not found or unreadable: {path}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
