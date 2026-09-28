# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``bench.distance_range``, ``bench.camera_spread`` and
``bench.classify_range``.

The capture is the 17-image seoul_bull fixture the other bench tests use. A
range reads only the cameras, so the sightings are a reconstruction point's
own keypoints.
"""

import math

import numpy as np
import pytest

from sfmtool._sfmtool import bench
from sfmtool._sfmtool.patches import ImagePyramidSet
from sfmtool._sfmtool.reconstruction import EditedReconstruction

from .test_bench_rust_bindings import embedded, images  # noqa: F401 (fixtures)


@pytest.fixture(scope="module")
def pyramids(embedded, images):  # noqa: F811
    return ImagePyramidSet(embedded, images)


def _centers(recon) -> np.ndarray:
    """Each image's camera centre, ``-R^T t`` of its ``cam_from_world``."""
    out = []
    for (w, x, y, z), t in zip(recon.quaternions_wxyz, recon.translations):
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        out.append(-R.T @ np.asarray(t))
    return np.asarray(out)


@pytest.fixture(scope="module")
def sighted(embedded):  # noqa: F811
    """The reconstruction's longest track: its first image and keypoint, its
    sightings, and its distance from that image's camera centre."""
    edited = EditedReconstruction(embedded)
    best = max(
        range(edited.point_count),
        key=lambda p: len(edited.point(p)["image_indexes"]),
    )
    record = edited.point(best)
    assert record["w"] == 1.0
    seen = [int(i) for i in record["image_indexes"]]
    xy = np.asarray(record["keypoints_xy"], float)
    sightings = [(i, (float(p[0]), float(p[1]))) for i, p in zip(seen, xy)]
    image, pixel = sightings[0]
    center = _centers(embedded)[image]
    distance = float(np.linalg.norm(np.asarray(record["position"]) - center))
    return image, pixel, sightings, distance


def test_a_well_seen_point_has_a_bounded_range_around_it(
    embedded,  # noqa: F811
    pyramids,
    sighted,
):
    image, pixel, sightings, distance = sighted
    edited = EditedReconstruction(embedded)
    near, far = bench.distance_range(
        edited, pyramids, image, pixel, sightings, distance
    )
    assert 0 < near < distance < far < math.inf
    # A looser tolerance gives a wider range.
    wide = bench.distance_range(
        edited, pyramids, image, pixel, sightings, distance, tolerance_px=4.0
    )
    assert wide[0] <= near and far <= wide[1]
    spread = bench.camera_spread(edited, pyramids)
    c = bench.classify_range((near, far), spread, max_span=far / near)
    assert c == {"bounded": True, "far": False, "usable": True}
    assert not bench.classify_range((near, far), spread, max_span=0.99 * far / near)[
        "bounded"
    ]


def test_the_queried_images_sightings_are_not_checked(
    embedded,  # noqa: F811
    images,  # noqa: F811
    pyramids,
    sighted,
):
    image, pixel, sightings, distance = sighted
    edited = EditedReconstruction(embedded)
    want = bench.distance_range(edited, pyramids, image, pixel, sightings, distance)
    moved = sightings + [(image, (1.0, 1.0))]
    assert bench.distance_range(edited, images, image, pixel, moved, distance) == want
    # Nothing else to check allows every distance.
    assert bench.distance_range(
        edited, pyramids, image, pixel, [(image, pixel)], distance
    ) == (0.0, math.inf)


def test_a_point_at_infinity_has_no_far_end(embedded, pyramids, sighted):  # noqa: F811
    image, pixel, sightings, _ = sighted
    edited = EditedReconstruction(embedded)
    # The sightings are of a finite point, so the error at infinity widens the
    # tolerance; whatever it allows, a range at infinity has no far end.
    near, far = bench.distance_range(
        edited, pyramids, image, pixel, sightings, math.inf
    )
    assert far == math.inf and 0 <= near < math.inf


def test_the_camera_spread_is_the_widest_pair(embedded, pyramids):  # noqa: F811
    edited = EditedReconstruction(embedded)
    C = _centers(embedded)
    want = float(np.linalg.norm(C[:, None] - C[None], axis=2).max())
    assert bench.camera_spread(edited, pyramids) == pytest.approx(want, rel=1e-9)


def test_a_range_is_far_when_open_and_distant():
    assert bench.classify_range((10.0, math.inf), 2.0) == {
        "bounded": False,
        "far": True,
        "usable": True,
    }
    assert not bench.classify_range((9.0, math.inf), 2.0)["usable"]
    assert bench.classify_range((9.0, math.inf), 2.0, far_spread=4.0)["far"]
    assert bench.classify_range((1.0, 3.0), 2.0)["bounded"]


def test_an_image_that_is_not_there_is_refused(embedded, pyramids):  # noqa: F811
    edited = EditedReconstruction(embedded)
    with pytest.raises(ValueError, match="not one of"):
        bench.distance_range(edited, pyramids, 999, (10.0, 10.0), [], 1.0)
    with pytest.raises(ValueError, match="not one of"):
        bench.distance_range(
            edited, pyramids, 0, (10.0, 10.0), [(999, (1.0, 1.0))], 1.0
        )
