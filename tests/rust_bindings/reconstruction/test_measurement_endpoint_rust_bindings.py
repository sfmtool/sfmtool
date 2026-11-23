# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `SfmrReconstruction.resolve_measurement_endpoint`,
`SfmrReconstruction.resolve_distance_measurements` and
`SfmrReconstruction.measurement_endpoint_views` on the seoul_bull ground truth,
and of a measurement of the demo scene's cube edge, whose length and corners
are known exactly."""

import numpy as np
import pytest

from sfmtool.fileio import LENGTH_UNIT_NAMES
from sfmtool.geometry import RotQuaternion
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


# The demo scene's cube edge where the faces -a and -b meet, 1 m long: its
# base and top corners, and the images the demo's measurement places them in.
_CUBE_EDGE = ((-0.1, -0.7, 0.0), (-0.1, -0.7, 1.0))
_CUBE_EDGE_IMAGES = (2, 1)


def _project(recon, image, world_xyz):
    """The pixel at which ``image`` sees the world point ``world_xyz``."""
    rot = np.asarray(
        RotQuaternion.from_wxyz_array(
            recon.quaternions_wxyz[image]
        ).to_rotation_matrix()
    )
    camera_xyz = rot @ np.asarray(world_xyz) + recon.translations[image]
    return recon.image_intrinsics(image).ray_to_pixel_batch(camera_xyz[None])[0]


def test_distance_measurements_resolve_with_their_scene_distance(demo_workspace_once):
    """On the demo scene, a measurement placed at the projections of the
    cube's 1 m edge's corners resolves each endpoint to its corner, and to a
    scene distance of 1 m."""
    demo = SfmrReconstruction.load(demo_workspace_once)
    assert (
        demo.clone_with_changes(measurements=None).resolve_distance_measurements() == []
    )
    endpoints = [
        _project(demo, image, corner)
        for image, corner in zip(_CUBE_EDGE_IMAGES, _CUBE_EDGE)
    ]
    # The projections are the endpoints the demo stores, to their f32 rounding.
    np.testing.assert_allclose(
        demo.measurements["distance"]["endpoints_xy"][0], endpoints, atol=1e-4
    )
    measured = demo.clone_with_changes(
        measurements={
            "distance": {
                "distances": np.array([1.0], dtype=np.float32),
                "units": np.array([LENGTH_UNIT_NAMES.index("m")], dtype=np.uint8),
                "image_indexes": np.array([_CUBE_EDGE_IMAGES], dtype=np.uint32),
                "endpoints_xy": np.array([endpoints], dtype=np.float32),
                "labels": [""],
            }
        }
    )

    (m,) = measured.resolve_distance_measurements(layer_depth_factor=2.0)

    assert m["refusals"] == [None, None]
    ends = [e["candidates"][e["chosen"]]["position"] for e in m["endpoints"]]
    assert m["scene_distance"] == pytest.approx(
        float(np.linalg.norm(np.subtract(ends[0], ends[1]))), rel=1e-12
    )
    # Each endpoint is its corner, to the f32 rounding of the keypoints and
    # endpoints, about 1e-5 px at 7 mm a pixel.
    np.testing.assert_allclose(ends, _CUBE_EDGE, atol=1e-6)
    assert m["scene_distance"] == pytest.approx(1.0, abs=1e-6)


def test_the_views_of_a_keypoint_are_the_other_keypoints_of_its_track(recon):
    (j,) = _finite_rows(recon, 1)
    image = int(recon.track_image_indexes[j])
    point = int(recon.track_point_indexes[j])
    x, y = (float(v) for v in recon.keypoints_xy[j])
    res = recon.resolve_measurement_endpoint(image, (x, y))
    chosen = res["candidates"][res["chosen"]]["point"]

    views = recon.measurement_endpoint_views(image, (x, y))

    assert views[0]["point"] == chosen
    own = [v for v in views if v["point"] == point]
    rows = np.flatnonzero(recon.track_point_indexes == point)
    assert 0 < len(own) <= len(rows) - 1
    for v in own:
        # The texel is the keypoint's own, (0, 0), so each other square holds
        # it at that observation's keypoint.
        row = rows[v["observation"]]
        assert v["image"] == int(recon.track_image_indexes[row])
        assert v["image"] != image
        np.testing.assert_allclose(v["pixel"], recon.keypoints_xy[row], atol=1e-2)
        assert v["resolves_elsewhere"] == (v["resolves_to"] != point)
        if not v["resolves_elsewhere"]:
            moved = recon.resolve_measurement_endpoint(v["image"], v["pixel"])
            assert moved["candidates"][moved["chosen"]]["point"] == point


def test_the_views_refuse_as_the_resolver_does(recon):
    with pytest.raises(ValueError, match="place the endpoint inside the image"):
        recon.measurement_endpoint_views(0, (-1.0, 10.0))
