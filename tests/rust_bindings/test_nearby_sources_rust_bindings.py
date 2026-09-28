# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the matching sources of finding the tracks near a pixel:
``bench.nearby_points``.

The capture is the 17-image seoul_bull fixture the other bench tests use. A
point is deleted from the version and the query is made at one of its pixels,
the way the track-at-pixel harness holds a point out.
"""

import numpy as np
import pytest

from sfmtool._sfmtool import bench
from sfmtool._sfmtool.patches import ImagePyramidSet
from sfmtool._sfmtool.reconstruction import EditedReconstruction

from .test_bench_rust_bindings import (  # noqa: F401 (fixtures)
    embedded,
    images,
    long_track_point,
)

ANCHOR_KEYS = {
    "source",
    "id",
    "position",
    "views",
    "query_pixel",
    "distance_px",
    "n_views",
    "max_reproj_px",
    "max_ray_angle_deg",
    "depth",
}


@pytest.fixture(scope="module")
def pyramids(embedded, images):  # noqa: F811
    return ImagePyramidSet(embedded, images)


@pytest.fixture
def query(embedded, long_track_point):  # noqa: F811
    """The longest track held out: the version without it, and its first image
    and pixel."""
    record = EditedReconstruction(embedded).point(long_track_point)
    image = int(record["image_indexes"][0])
    pixel = tuple(float(v) for v in record["keypoints_xy"][0])
    edited = EditedReconstruction(embedded)
    edited.delete_point(long_track_point)
    return edited, image, pixel


def test_the_points_near_the_pixel_are_found_without_the_held_out_one(
    pyramids,
    query,
    long_track_point,  # noqa: F811
):
    edited, image, pixel = query
    found = bench.nearby_points(edited, pyramids, image, pixel)

    assert 0 < len(found) <= 8
    for a in found:
        assert set(a) == ANCHOR_KEYS
        assert a["source"] == "tracks"
        assert a["id"] != long_track_point
        assert a["views"][0][0] == image
        assert a["query_pixel"] == a["views"][0][1:]
        assert a["n_views"] == len(a["views"]) >= 2
        assert a["max_reproj_px"] <= 2.0
        assert a["distance_px"] <= 40.0
        assert a["depth"] > 0
        # The point is the reconstruction's own.
        record = edited.point(a["id"])
        assert np.allclose(a["position"], record["position"])
    distances = [a["distance_px"] for a in found]
    assert distances == sorted(distances)

    # The options narrow it.
    few = bench.nearby_points(
        edited, pyramids, image, pixel, options={"max_points": 2, "radius_px": 40.0}
    )
    assert len(few) == min(2, len(found))
    with pytest.raises(ValueError, match="unknown option"):
        bench.nearby_points(edited, pyramids, image, pixel, options={"radius": 1.0})
    with pytest.raises(ValueError):
        bench.nearby_points(edited, pyramids, 99, pixel)
