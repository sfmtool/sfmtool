# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `SfmrReconstruction.distance_measurement_scale` on the demo
scene, whose one stored measurement is the cube's 1 m edge."""

import numpy as np
import pytest

from sfmtool.fileio import LENGTH_UNIT_NAMES
from sfmtool.reconstruction import SfmrReconstruction


@pytest.fixture(scope="module")
def demo(demo_workspace_once):
    return SfmrReconstruction.load(demo_workspace_once)


def test_the_cube_edge_gives_the_scale_in_each_unit(demo):
    (m,) = demo.resolve_distance_measurements()
    stated_m = (
        float(demo.measurements["distance"]["distances"][0])
        * {
            "mm": 0.001,
            "cm": 0.01,
            "m": 1.0,
            "in": 0.0254,
            "ft": 0.3048,
        }[LENGTH_UNIT_NAMES[int(demo.measurements["distance"]["units"][0])]]
    )

    mm = demo.distance_measurement_scale("mm")
    cm = demo.distance_measurement_scale("cm")

    assert mm["unit"] == "mm"
    assert mm["factor"] == pytest.approx(1000.0 * stated_m / m["scene_distance"])
    assert cm["factor"] == pytest.approx(mm["factor"] / 10.0)
    (f,) = mm["factors"]
    assert f["index"] == 0
    assert f["scene_distance"] == pytest.approx(m["scene_distance"])
    assert f["stated"] == pytest.approx(1000.0 * stated_m)
    assert f["deviation_percent"] == 0.0
    assert mm["excluded"] == []
    s = mm["statistics"]
    assert s["count"] == 1
    assert s["median"] == mm["factor"]
    assert s["sample_std"] is None and s["sample_std_percent"] is None
    assert s["disagrees"] is False
    assert s["disagreement_percent"] == 5.0


def test_a_measurement_that_disagrees_is_reported(demo):
    """Three measurements of the same edge, one stated 20% too long: the
    factor is the median, the other two's, and the long one is 20% off it."""
    d = demo.measurements["distance"]
    one = demo.distance_measurement_scale("m")["factor"]
    stated = d["distances"][0]
    three = demo.clone_with_changes(
        measurements={
            "distance": {
                "distances": np.array([stated, stated, 1.2 * stated], dtype=np.float32),
                "units": np.repeat(d["units"][:1], 3),
                "image_indexes": np.repeat(d["image_indexes"][:1], 3, axis=0),
                "endpoints_xy": np.repeat(d["endpoints_xy"][:1], 3, axis=0),
                "labels": ["", "", "too long"],
            }
        }
    )

    scale = three.distance_measurement_scale("m")

    assert scale["factor"] == pytest.approx(one, rel=1e-9)
    assert [f["label"] for f in scale["factors"]] == ["", "", "too long"]
    deviations = [f["deviation_percent"] for f in scale["factors"]]
    assert deviations == pytest.approx([0.0, 0.0, 20.0], abs=1e-4)
    assert scale["statistics"]["mean"] == pytest.approx(one * 3.2 / 3, rel=1e-6)
    assert scale["statistics"]["max_abs_deviation_percent"] == pytest.approx(
        20.0, abs=1e-4
    )
    assert scale["statistics"]["disagrees"] is True
    assert scale["statistics"]["sample_std"] is not None


def test_no_measurement_raises_cores_sentence(demo):
    with pytest.raises(ValueError, match="stores no distance measurement"):
        demo.clone_with_changes(measurements=None).distance_measurement_scale("mm")


def test_no_usable_measurement_raises_with_each_reason(demo):
    d = demo.measurements["distance"]
    bare = demo.clone_with_changes(
        measurements={
            "distance": {
                "distances": d["distances"][:1],
                "units": d["units"][:1],
                "image_indexes": d["image_indexes"][:1],
                "endpoints_xy": np.full((1, 2, 2), 0.5, dtype=np.float32),
                "labels": [""],
            }
        }
    )
    with pytest.raises(
        ValueError,
        match="none of the 1 distance measurements gives a scale; measurement 0:",
    ):
        bare.distance_measurement_scale("mm")


def test_an_unknown_unit_raises(demo):
    with pytest.raises(ValueError, match="unknown unit"):
        demo.distance_measurement_scale("furlong")
