# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``bench.far_field_sweep``.

The capture is the 17-image seoul_bull fixture the other bench tests use. The
sweep reads the photographs only, so a query is made at a reconstruction
keypoint and needs nothing else.
"""

import json
import math

import numpy as np
import pytest

from sfmtool._sfmtool import bench
from sfmtool._sfmtool.patches import ImagePyramidSet
from sfmtool._sfmtool.reconstruction import EditedReconstruction

from .test_bench_rust_bindings import embedded, images  # noqa: F401 (fixtures)

# The keys every reading carries, as the harness's `from_farfield` anchors do.
COMMON = {
    "source",
    "id",
    "position",
    "w",
    "views",
    "query_pixel",
    "distance_px",
    "n_views",
    "max_reproj_px",
    "max_ray_angle_deg",
    "depth",
    "farfield",
}
METRICS = {
    "whole",
    "middle",
    "prominence",
    "peak_rank",
    "peaks",
    "middle_flat",
    "middle_std",
    "images",
    "parallax_px",
    "widest_px",
    "profile_whole",
    "profile_middle",
}


@pytest.fixture(scope="module")
def pyramids(embedded, images):  # noqa: F811
    return ImagePyramidSet(embedded, images)


@pytest.fixture(scope="module")
def query(embedded, pyramids):  # noqa: F811
    """The first reconstruction keypoint, in track order, the sweep finds a
    reading at: an image, a pixel and what the sweep returned there."""
    edited = EditedReconstruction(embedded)
    for point in range(edited.point_count):
        record = edited.point(point)
        for image, xy in zip(record["image_indexes"], record["keypoints_xy"]):
            pixel = (float(xy[0]), float(xy[1]))
            found = bench.far_field_sweep(edited, pyramids, int(image), pixel)
            if found:
                return int(image), pixel, found
    pytest.skip("the sweep found no far-field reading in the fixture")


def test_a_reading_carries_the_harness_anchor_keys(embedded, query):  # noqa: F811
    image, pixel, found = query
    for rank, a in enumerate(found, 1):
        assert COMMON <= set(a)
        assert a["source"] == "farfield" and a["id"] is None
        assert METRICS | {"group_middle", "left_out"} <= set(a["farfield"])
        assert a["farfield"]["peak_rank"] == rank
        assert a["farfield"]["peaks"] == len(found)
        assert {"groups", "query_middle", "refit"} <= set(a)
        assert a["n_views"] == len(a["views"])
        assert all(len(v) == 3 and isinstance(v[0], int) for v in a["views"])
        assert a["views"][0][0] == image
        if a["refit"] == "moved":
            assert "sweep_disparity" in a and "range_override" not in a
            assert "disparity" not in a
            assert a["distance_px"] > 0
        else:
            assert a["refit"] in {"agrees", "grouped", "unsplit", "stands"}
            assert a["query_pixel"] == list(pixel)
            assert a["distance_px"] == 0.0
            near, far = a["range_override"]
            assert 0 < near < far
            if a["disparity"] == 0:
                assert a["w"] == 0.0 and math.isinf(a["depth"]) and math.isinf(far)
                assert np.linalg.norm(a["position"]) == pytest.approx(1.0)
            else:
                assert a["w"] == 1.0 and a["depth"] > 0
        # Every image of the queried group is in the reading's views.
        mine = next(g for g in a["groups"] if image in g)
        if a["refit"] in {"agrees", "grouped"}:
            assert sorted(v[0] for v in a["views"]) == mine


def test_a_list_of_images_reads_as_the_prebuilt_set_does(
    embedded,  # noqa: F811
    images,  # noqa: F811
    query,
):
    image, pixel, found = query
    edited = EditedReconstruction(embedded)
    again = bench.far_field_sweep(edited, images, image, pixel)
    # Through JSON, so a NaN pairwise score compares equal to itself.
    assert json.dumps(again) == json.dumps(found)


def test_the_options_are_overrides(embedded, pyramids, query):  # noqa: F811
    image, pixel, found = query
    edited = EditedReconstruction(embedded)
    plain = bench.far_field_sweep(
        edited, pyramids, image, pixel, options={"refit": False}
    )
    assert plain
    assert all(
        "refit" not in a and "groups" not in a and "left_out" not in a["farfield"]
        for a in plain
    )
    # The refit keeps, moves or drops each peak; it adds none.
    assert {a.get("disparity", a.get("sweep_disparity")) for a in found} <= {
        a["disparity"] for a in plain
    }
    one = bench.far_field_sweep(
        edited, pyramids, image, pixel, options={"max_peaks": 1, "refit": False}
    )
    assert len(one) == 1
    with pytest.raises(ValueError, match="unknown option"):
        bench.far_field_sweep(edited, pyramids, image, pixel, options={"nope": 1})
    with pytest.raises(ValueError, match="wide_among"):
        bench.far_field_sweep(
            edited, pyramids, image, pixel, options={"wide_among": "some"}
        )


def test_a_query_that_names_no_place_is_refused(embedded, pyramids):  # noqa: F811
    edited = EditedReconstruction(embedded)
    with pytest.raises(ValueError, match="not on the"):
        bench.far_field_sweep(edited, pyramids, 0, (-5.0, 10.0))
    with pytest.raises(ValueError, match="not one of"):
        bench.far_field_sweep(edited, pyramids, 999, (10.0, 10.0))
