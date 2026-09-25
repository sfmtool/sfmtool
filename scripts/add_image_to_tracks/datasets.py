# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The ground-truth reconstructions the Add Image to Tracks harness runs on.

Each dataset is a ground-truth ``.sfmr`` with embedded patches and a
cluster-patches ``.matches`` file over the same images, which the resection
needs to pose an image that has lost all of its track observations.

- ``seoul_bull``: the checked-in ground truth. It has no ``.sift`` files or
  index files beside it, and its directory is a workspace whose output must not
  land in ``test-data``, so it is copied into a cache directory and the index
  files are built there by the track-at-pixel harness's own preparation
  (``scripts/track_at_pixel/dataset.py``).
- ``kerry_park``: a candidate ground truth outside the repository, read in
  place and never written to. Its index files were built for an earlier
  candidate of the same workspace and images; the resection matches a
  ``.matches`` file's images to a reconstruction's by name, so they serve.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERRY_WS = Path(r"C:\Dev\prod2\sfmtool\kerry_park_demo_ws")


@dataclass
class Dataset:
    name: str
    sfmr: Path
    matches: Path


def prepare(name: str, cache_dir: Path) -> Dataset:
    """The dataset's ground truth and cluster-patches file, built if needed."""
    if name == "seoul_bull":
        sys.path.insert(0, str(REPO / "scripts" / "track_at_pixel"))
        from dataset import prepare as prepare_track_at_pixel

        prepared = prepare_track_at_pixel("seoul_bull", cache_dir, quiet=False)
        return Dataset(name, prepared.sfmr, prepared.matches)
    if name == "kerry_park":
        sfmr_dir = KERRY_WS / "sfmr"
        return Dataset(
            name,
            sfmr_dir / "kerry_park_ground_truth_candidate_tk106.sfmr",
            sfmr_dir
            / "kerry_park_ground_truth_candidate_tk105-cluster-patches.matches",
        )
    raise SystemExit(f"unknown dataset {name!r} (known: seoul_bull, kerry_park)")
