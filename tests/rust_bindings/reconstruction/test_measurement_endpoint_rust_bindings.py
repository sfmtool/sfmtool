# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `SfmrReconstruction.resolve_measurement_endpoint` and
`SfmrReconstruction.resolve_measurements` on the seoul_bull ground truth."""

import numpy as np
import pytest

from sfmtool.fileio import LENGTH_UNIT_NAMES
from sfmtool.reconstruction import SfmrReconstruction
from tests.conftest import SEOUL_BULL_GROUND_TRUTH


@pytest.fixture(scope="module")
def recon():
    return SfmrReconstruction.load(SEOUL_BULL_GROUND_TRUTH)


def _finite_rows(recon, count):
    """The first `count` observation rows of finite points, each in a
    different image."""
    w = recon.positions_xyzw[:, 3]
    images = recon.track_image_indexes
    points = recon.track_point_indexes
    rows, seen = [], set()
    for j in range(len(points)):
        if w[points[j]] != 0 and images[j] not in seen:
            rows.append(j)
            seen.add(int(images[j]))
        if len(rows) == count:
            break
    return rows


def test_an_observation_keypoint_resolves_to_its_patch_centre(recon):
    (j,) = _finite_rows(recon, 1)
    image = int(recon.track_image_indexes[j])
    point = int(recon.track_point_indexes[j])
    x, y = (float(v) for v in recon.keypoints_xy[j])

    res = recon.resolve_measurement_endpoint(image, (x, y))

    assert 0 <= res["chosen"] < len(res["candidates"])
    own = [c for c in res["candidates"] if c["point"] == point]
    assert len(own) == 1
    s, t = own[0]["patch_coords"]
    assert abs(s) < 1e-4 and abs(t) < 1e-4
    assert own[0]["direction"] is None
    np.testing.assert_allclose(
        own[0]["position"], recon.positions_xyzw[point, :3], atol=1e-6
    )
    assert own[0]["standing"] in ("alone", "ranked")
    if own[0]["standing"] == "ranked":
        assert own[0]["cost"]["cost"] > 0


def test_a_refusal_raises_its_sentence(recon):
    with pytest.raises(ValueError, match="place the endpoint in one of them"):
        recon.resolve_measurement_endpoint(len(recon.image_names), (10.0, 10.0))
    with pytest.raises(ValueError, match="place the endpoint inside the image"):
        recon.resolve_measurement_endpoint(0, (-1.0, 10.0))


def test_measurements_resolve_with_their_scene_distance(recon):
    assert recon.resolve_measurements() == []
    a, b = _finite_rows(recon, 2)
    images = recon.track_image_indexes
    kp = recon.keypoints_xy
    measured = recon.clone_with_changes(
        measurements={
            "distances": np.array([1.0], dtype=np.float32),
            "units": np.array([LENGTH_UNIT_NAMES.index("m")], dtype=np.uint8),
            "image_indexes": np.array([[images[a], images[b]]], dtype=np.uint32),
            "endpoints_xy": np.array([[kp[a], kp[b]]], dtype=np.float32),
            "labels": [""],
        }
    )

    (m,) = measured.resolve_measurements(layer_depth_factor=2.0)

    assert m["refusals"] == [None, None]
    ends = [e["candidates"][e["chosen"]]["position"] for e in m["endpoints"]]
    assert m["scene_distance"] == pytest.approx(
        float(np.linalg.norm(np.subtract(ends[0], ends[1]))), rel=1e-12
    )
    assert m["scene_distance"] > 0
