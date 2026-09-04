# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the intrinsics-estimate Rust binding
(``sfmtool._sfmtool.geometry.estimate_intrinsics``; see
``specs/core/geometry/estimate-intrinsics.md``).

The estimate is the high-level face of the focal vote: it runs the same vote
and hands back one typed answer -- the model verdict, whether that verdict is
corroborated, the consensus focal, and the votes that belong to it -- with the
raw vote nested under ``"vote"``. The synthetic scenes are the vote binding's
own, imported rather than rebuilt, so the two bindings are read against
identical captures.
"""

import numpy as np
import pytest

from sfmtool._sfmtool.geometry import estimate_intrinsics, focal_vote

from .test_focal_vote_rust_bindings import (
    F_FISH,
    H,
    W,
    _fisheye_scene,
    _rotation_scene,
)

BOTH = ("pinhole", "equidistant")


def test_estimate_dict_layout():
    cl, im, pos = _fisheye_scene(2718)
    est = estimate_intrinsics(cl, im, pos, W, H, seed=0)
    assert set(est) == {
        "camera_model",
        "confirmed",
        "focal_px",
        "verdict_votes",
        "vote",
    }
    assert est["camera_model"] in ("Pinhole", "EquidistantFisheye", None)
    assert est["confirmed"] in (True, False, None)
    assert isinstance(est["focal_px"], float)
    assert isinstance(est["verdict_votes"], list)
    # The nested vote is the full focal_vote dict, untouched.
    assert set(est["vote"]) >= {"columns", "epipolar_votes", "rotation_votes"}
    assert est["focal_px"] == est["vote"]["focal_px"]
    assert est["camera_model"] == est["vote"]["camera_model"]


def test_default_columns_are_both():
    # Unlike focal_vote, whose default is the closed-form pinhole kernel, the
    # estimate's default runs both columns: the verdict is what it is for.
    cl, im, pos = _fisheye_scene(2718)
    default = estimate_intrinsics(cl, im, pos, W, H, seed=0)
    explicit = estimate_intrinsics(cl, im, pos, W, H, seed=0, columns=BOTH)
    assert default == explicit
    assert [c["camera_model"] for c in default["vote"]["columns"]] == [
        "Pinhole",
        "EquidistantFisheye",
    ]


def test_nested_vote_is_the_focal_vote_result():
    cl, im, pos = _fisheye_scene(2718)
    est = estimate_intrinsics(cl, im, pos, W, H, seed=0, columns=BOTH)
    raw = focal_vote(cl, im, pos, W, H, seed=0, columns=BOTH)
    assert est["vote"] == raw


def test_fisheye_capture_is_confirmed():
    cl, im, pos = _fisheye_scene(2718)
    est = estimate_intrinsics(cl, im, pos, W, H, seed=0)
    assert est["camera_model"] == "EquidistantFisheye", est["vote"]["columns"]
    assert est["confirmed"] is True
    assert abs(est["focal_px"] - F_FISH) / F_FISH < 0.05


def test_min_rotation_mass_floor_is_respected():
    cl, im, pos = _fisheye_scene(2718)
    est = estimate_intrinsics(cl, im, pos, W, H, seed=0)
    (fish,) = [
        c for c in est["vote"]["columns"] if c["camera_model"] == "EquidistantFisheye"
    ]
    mass = fish["n_certified_rotation"]
    assert mass >= 1
    above = estimate_intrinsics(cl, im, pos, W, H, seed=0, min_rotation_mass=mass + 1)
    assert above["confirmed"] is False
    # Only the corroboration moved; the verdict itself did not.
    assert above["camera_model"] == est["camera_model"]
    assert above["focal_px"] == est["focal_px"]


def test_pinhole_verdict_raises_no_confirmation_question():
    cl, im, pos = _rotation_scene(2024)
    est = estimate_intrinsics(cl, im, pos, W, H, seed=0)
    assert est["camera_model"] == "Pinhole"
    assert est["confirmed"] is None


def test_single_column_arbitrates_nothing():
    cl, im, pos = _fisheye_scene(2718)
    est = estimate_intrinsics(cl, im, pos, W, H, seed=0, columns=("fisheye",))
    assert est["camera_model"] == "EquidistantFisheye"
    # One column is the verdict by construction, so there is nothing to confirm.
    assert est["confirmed"] is None


def test_verdict_votes_are_the_winning_column_s_certified_scans():
    cl, im, pos = _fisheye_scene(2718)
    est = estimate_intrinsics(cl, im, pos, W, H, seed=0)
    (fish,) = [
        c for c in est["vote"]["columns"] if c["camera_model"] == "EquidistantFisheye"
    ]
    expected = [v for v in fish["scan_votes"] if v["certified"]]
    assert est["verdict_votes"] == expected
    assert expected, "the winning column must have certified something"
    # Both cells of the winning column are represented, and every entry carries
    # a scan vote's own keys.
    assert {v["cell"] for v in est["verdict_votes"]} == {"Epipolar", "Rotation"}
    assert set(est["verdict_votes"][0]) == set(fish["scan_votes"][0])
    # They are NOT the flat vote lists, which describe the pinhole closed-form
    # kernel whichever column won.
    flat = {
        v["focal_px"]
        for v in est["vote"]["epipolar_votes"] + est["vote"]["rotation_votes"]
    }
    assert not flat & {v["focal_px"] for v in est["verdict_votes"]}


def test_seed_reproducibility():
    cl, im, pos = _fisheye_scene(2718)
    a = estimate_intrinsics(cl, im, pos, W, H, seed=7)
    b = estimate_intrinsics(cl, im, pos, W, H, seed=7)
    assert a == b


def test_rejects_unknown_column():
    cl, im, pos = _rotation_scene(2024)
    with pytest.raises(ValueError):
        estimate_intrinsics(cl, im, pos, W, H, columns=("brown-conrady",))


def test_shape_validation():
    with pytest.raises(ValueError):
        estimate_intrinsics(
            np.zeros(10, np.uint32), np.zeros(10, np.uint32), np.zeros((10, 3)), W, H
        )
    with pytest.raises(ValueError):
        estimate_intrinsics(
            np.zeros(10, np.uint32), np.zeros(9, np.uint32), np.zeros((10, 2)), W, H
        )


def test_empty_input_has_no_verdict():
    est = estimate_intrinsics(
        np.zeros(0, np.uint32), np.zeros(0, np.uint32), np.zeros((0, 2)), W, H
    )
    assert est["camera_model"] is None
    assert est["confirmed"] is None
    assert est["focal_px"] is None
    assert est["verdict_votes"] == []
