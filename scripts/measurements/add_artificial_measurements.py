# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Write a copy of a reconstruction with artificial distance measurements.

No checked-in reconstruction stores measurements, and SfM Explorer's
Measurement View cannot be judged or tested without them. This script reads a
reconstruction, gives it a set of distance measurements whose readings are
known by construction, and writes the result to a path it is given. It never
writes into ``test-data/``.

Usage::

    pixi run python scripts/measurements/add_artificial_measurements.py \
        SOURCE.sfmr DESTINATION.sfmr [--no-unit-copy OTHER.sfmr]

Each endpoint the set places on a point is an observation's keypoint, so it
resolves to texel ``(0, 0)`` of that point's own patch unless another square
holds the keypoint too. Each stated distance is the distance the
reconstruction gives the two endpoints times a known factor, so the expected
error is known. The set covers, in order:

- both endpoints in one image, stated at ``--factor`` times the
  reconstructed distance;
- endpoints in two images, at the same factor;
- a second measurement in one image, at the same factor;
- one bad measurement among the good ones, at ``--bad-factor``, so
  Measurement View's median line flags it;
- an endpoint that is refused: a pixel no observation's square holds;
- an endpoint several squares hold, where such a pixel can be found among the
  first observations of the reconstruction's images.

The copy keeps the source's ``world_space_unit``, so the readings come out in
real units for the ground truths, which are in metres. ``--no-unit-copy`` also
writes the same set to a second copy with ``world_space_unit`` cleared, which
reads in scene units.

The copy records the source's workspace, so the viewer finds the photographs
wherever the copy is written.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sfmtool.fileio import LENGTH_UNIT_NAMES
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.workspace import find_workspace_for_path

ROOT = Path(__file__).resolve().parents[2]
"""The repository root, which holds ``test-data/``."""

TEST_DATA = ROOT / "test-data"

OPERATION = "add_artificial_measurements"
"""The operation the copy's metadata records."""

UNIT_CM = LENGTH_UNIT_NAMES.index("cm")
UNIT_M = LENGTH_UNIT_NAMES.index("m")

METRES = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}
"""The length of each unit a file may name, in metres."""

SEARCH_LIMIT = 400
"""How many observations the searches for a pair or a pixel look through, at
most, so the script costs a bounded number of resolver calls."""


@dataclass
class Endpoint:
    """One end of a measurement: an image, a pixel, and the 3D position it
    resolves to, or ``None`` where it does not."""

    image: int
    pixel: tuple[float, float]
    position: np.ndarray | None


@dataclass
class Measurement:
    """One artificial distance measurement and what it is for."""

    case: str
    label: str
    a: Endpoint
    b: Endpoint
    stated: float
    unit: int


def refuse_test_data(path: Path) -> None:
    """Raise ``SystemExit`` when ``path`` lies under ``test-data/``."""
    resolved = path.resolve()
    if resolved == TEST_DATA or TEST_DATA in resolved.parents:
        raise SystemExit(
            f"refusing to write {path}: it is under {TEST_DATA}, which holds the "
            "checked-in datasets; write the copy somewhere else"
        )


def resolve(recon: SfmrReconstruction, image: int, pixel) -> dict | None:
    """The resolution of ``pixel`` in ``image``, or ``None`` where it is refused."""
    try:
        return recon.resolve_measurement_endpoint(
            image, (float(pixel[0]), float(pixel[1]))
        )
    except ValueError:
        return None


def finite_position(resolution: dict | None) -> np.ndarray | None:
    """The chosen candidate's finite position, or ``None``."""
    if resolution is None:
        return None
    position = resolution["candidates"][resolution["chosen"]]["position"]
    return None if position is None else np.asarray(position, dtype=np.float64)


class Observations:
    """The reconstruction's observations, grouped by image, with the ones whose
    keypoint resolves alone to their own point found on demand."""

    def __init__(self, recon: SfmrReconstruction):
        self.recon = recon
        keypoints = recon.keypoints_xy
        if keypoints is None:
            raise SystemExit(
                "the source stores no keypoints inline; convert it with "
                "sfm xform --to-embedded-patches first"
            )
        self.keypoints = np.asarray(keypoints, dtype=np.float64)
        self.images = np.asarray(recon.track_image_indexes)
        self.points = np.asarray(recon.track_point_indexes)
        self.by_image: dict[int, list[int]] = {}
        for row, image in enumerate(self.images):
            self.by_image.setdefault(int(image), []).append(row)

    def endpoint(self, row: int) -> Endpoint | None:
        """Observation ``row``'s keypoint as an endpoint, where it resolves
        alone to the observation's own point at a finite position."""
        image = int(self.images[row])
        pixel = (float(self.keypoints[row, 0]), float(self.keypoints[row, 1]))
        resolution = resolve(self.recon, image, pixel)
        if resolution is None:
            return None
        chosen = resolution["candidates"][resolution["chosen"]]
        if chosen["point"] != int(self.points[row]) or chosen["standing"] != "alone":
            return None
        position = finite_position(resolution)
        if position is None:
            return None
        return Endpoint(image, pixel, position)

    def alone_in(self, image: int, skip: set[int]) -> list[Endpoint]:
        """Endpoints in ``image`` that resolve alone, at most a few, none of
        them on a point in ``skip``."""
        found = []
        for row in self.by_image.get(image, [])[:SEARCH_LIMIT]:
            if int(self.points[row]) in skip:
                continue
            endpoint = self.endpoint(row)
            if endpoint is not None:
                found.append(endpoint)
                skip.add(int(self.points[row]))
            if len(found) >= 4:
                break
        return found


def distance(a: Endpoint, b: Endpoint) -> float:
    return float(np.linalg.norm(a.position - b.position))


def farthest_pair(endpoints: list[Endpoint]) -> tuple[Endpoint, Endpoint]:
    """The two of ``endpoints`` farthest apart in the scene."""
    best = None
    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1 :]:
            if best is None or distance(a, b) > distance(*best):
                best = (a, b)
    return best


def refused_pixel(recon: SfmrReconstruction, image: int) -> tuple[float, float] | None:
    """A pixel of ``image`` that no observation's square holds, from a grid
    over the image, corners first."""
    camera = recon.cameras[int(recon.camera_indexes[image])]
    width, height = camera.width, camera.height
    candidates = [
        (2.5, 2.5),
        (width - 3.5, 2.5),
        (2.5, height - 3.5),
        (width - 3.5, height - 3.5),
    ]
    for fy in np.linspace(0.05, 0.95, 7):
        for fx in np.linspace(0.05, 0.95, 7):
            candidates.append((fx * width, fy * height))
    for pixel in candidates:
        try:
            recon.resolve_measurement_endpoint(image, pixel)
        except ValueError as refusal:
            if "holds pixel" in str(refusal):
                return pixel
    return None


def shared_pixel(recon: SfmrReconstruction, obs: Observations) -> Endpoint | None:
    """A pixel several squares hold, which resolves to a finite position: the
    midpoint of two nearby keypoints of one image, or a keypoint another square
    holds as well."""
    tried = 0
    for image, rows in obs.by_image.items():
        rows = rows[:60]
        for i, ri in enumerate(rows):
            for rj in rows[i + 1 :]:
                gap = obs.keypoints[ri] - obs.keypoints[rj]
                if not 0.5 < math.hypot(gap[0], gap[1]) < 12.0:
                    continue
                for pixel in (
                    (obs.keypoints[ri] + obs.keypoints[rj]) / 2.0,
                    obs.keypoints[ri],
                ):
                    tried += 1
                    resolution = resolve(recon, image, pixel)
                    if resolution is None or len(resolution["candidates"]) < 2:
                        continue
                    position = finite_position(resolution)
                    if position is not None:
                        return Endpoint(
                            image, (float(pixel[0]), float(pixel[1])), position
                        )
                if tried > SEARCH_LIMIT:
                    return None
    return None


def build(
    recon: SfmrReconstruction, factor: float, bad_factor: float
) -> list[Measurement]:
    """The artificial measurements of ``recon``, as the module docstring lists
    them."""
    obs = Observations(recon)
    order = sorted(obs.by_image, key=lambda image: -len(obs.by_image[image]))
    if len(order) < 2:
        raise SystemExit("the source needs observations in at least two images")
    first, second = order[0], order[1]
    used: set[int] = set()
    in_first = obs.alone_in(first, used)
    in_second = obs.alone_in(second, used)
    if len(in_first) < 4 or len(in_second) < 2:
        raise SystemExit("too few observations resolve alone to their own points")
    a0, b0 = farthest_pair(in_first[:2])
    a2, b2 = in_first[2], in_first[3]
    # One scene unit in metres: the file's unit, or a metre where it names none.
    metres = METRES.get(recon.world_space_unit or "m", 1.0)

    def stated(a: Endpoint, b: Endpoint, scale: float, unit: int) -> float:
        value = distance(a, b) * scale * metres
        return value * 100.0 if unit == UNIT_CM else value

    out = [
        Measurement(
            "one image", "one image", a0, b0, stated(a0, b0, factor, UNIT_CM), UNIT_CM
        ),
        Measurement(
            "two images",
            "two images",
            in_first[0],
            in_second[0],
            stated(in_first[0], in_second[0], factor, UNIT_M),
            UNIT_M,
        ),
        Measurement(
            "one image again", "", a2, b2, stated(a2, b2, factor, UNIT_M), UNIT_M
        ),
        Measurement(
            "bad",
            "bad measurement",
            in_second[0],
            in_second[1],
            stated(in_second[0], in_second[1], bad_factor, UNIT_M),
            UNIT_M,
        ),
    ]
    bare = refused_pixel(recon, first)
    if bare is not None:
        out.append(
            Measurement(
                "refused",
                "refused endpoint",
                in_first[1],
                Endpoint(first, bare, None),
                1.0,
                UNIT_M,
            )
        )
    shared = shared_pixel(recon, obs)
    if shared is not None:
        far = in_second[1]
        out.append(
            Measurement(
                "several squares",
                "several squares",
                shared,
                far,
                stated(shared, far, factor, UNIT_M),
                UNIT_M,
            )
        )
    return out


def measurements_dict(measurements: list[Measurement]) -> dict:
    """The measurements in the form ``clone_with_changes(measurements=...)``
    takes."""
    n = len(measurements)
    image_indexes = np.zeros((n, 2), dtype=np.uint32)
    endpoints_xy = np.zeros((n, 2, 2), dtype=np.float32)
    for i, m in enumerate(measurements):
        for k, end in enumerate((m.a, m.b)):
            image_indexes[i, k] = end.image
            endpoints_xy[i, k] = end.pixel
    return {
        "distance": {
            "distances": np.array([m.stated for m in measurements], dtype=np.float32),
            "units": np.array([m.unit for m in measurements], dtype=np.uint8),
            "image_indexes": image_indexes,
            "endpoints_xy": endpoints_xy,
            "labels": [m.label for m in measurements],
        }
    }


def write_copy(
    recon: SfmrReconstruction, destination: Path, workspace: Path | None
) -> None:
    refuse_test_data(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    recon.save(
        destination,
        operation=OPERATION,
        workspace_path=None if workspace is None else str(workspace),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="the reconstruction to copy")
    parser.add_argument("destination", type=Path, help="where to write the copy")
    parser.add_argument(
        "--no-unit-copy",
        type=Path,
        default=None,
        help="also write the same measurements here, with world_space_unit cleared",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=1.02,
        help="stated over reconstructed for the good measurements (default 1.02)",
    )
    parser.add_argument(
        "--bad-factor",
        type=float,
        default=1.2,
        help="stated over reconstructed for the bad measurement (default 1.2)",
    )
    args = parser.parse_args(argv)
    refuse_test_data(args.destination)
    if args.no_unit_copy is not None:
        refuse_test_data(args.no_unit_copy)

    recon = SfmrReconstruction.load(args.source)
    measurements = build(recon, args.factor, args.bad_factor)
    copy = recon.clone_with_changes(measurements=measurements_dict(measurements))
    workspace = find_workspace_for_path(args.source.resolve())
    write_copy(copy, args.destination, workspace)
    print(f"wrote {args.destination} with {len(measurements)} distance measurements:")
    for index, m in enumerate(measurements):
        print(f"  #{index} {m.case}: {LENGTH_UNIT_NAMES[m.unit]} {m.stated:.6g}")
    if args.no_unit_copy is not None:
        write_copy(
            copy.clone_with_changes(world_space_unit=None), args.no_unit_copy, workspace
        )
        print(f"wrote {args.no_unit_copy} with no world_space_unit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
