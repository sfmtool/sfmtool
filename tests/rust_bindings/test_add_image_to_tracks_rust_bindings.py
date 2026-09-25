# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``EditedReconstruction.add_image_to_tracks``.

The seoul_bull ground truth with one image's observations taken out: at the
image's ground-truth pose the operation should find the tracks it was in again,
at the keypoints it had.
"""

from pathlib import Path

import numpy as np
import pytest

from sfmtool._sfmtool.patches import ImagePyramidSet
from sfmtool._sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction
from sfmtool._workspace_image import read_workspace_image

GROUND_TRUTH = (
    Path(__file__).parents[2]
    / "test-data"
    / "images"
    / "seoul_bull_sculpture"
    / "seoul_bull_sculpture_ground_truth.sfmr"
)
IMAGE = 3


@pytest.fixture(scope="module")
def ground_truth():
    return SfmrReconstruction.load(GROUND_TRUTH)


@pytest.fixture(scope="module")
def pyramids(ground_truth):
    images = [
        read_workspace_image(GROUND_TRUTH.parent, name)
        for name in ground_truth.image_names
    ]
    return ImagePyramidSet(ground_truth, images)


@pytest.fixture(scope="module")
def without_image(ground_truth):
    """The ground truth with IMAGE's observations removed, and what they were."""
    pts = np.asarray(ground_truth.track_point_indexes)
    imgs = np.asarray(ground_truth.track_image_indexes)
    kps = np.asarray(ground_truth.keypoints_xy)
    remaining = np.bincount(pts[imgs != IMAGE], minlength=ground_truth.point_count)
    # Only points that keep two observations, so none has to be dropped and
    # every point index stays the same.
    removable = (imgs == IMAGE) & (remaining[pts] >= 2)
    keep = ~removable
    recon = ground_truth.clone_with_changes(
        track_image_indexes=np.ascontiguousarray(imgs[keep], dtype=np.uint32),
        track_feature_indexes=np.zeros(int(keep.sum()), np.uint32),
        track_point_indexes=np.ascontiguousarray(pts[keep], dtype=np.uint32),
        keypoints_xy=np.ascontiguousarray(kps[keep], dtype=np.float32),
    )
    removed = {int(p): kps[row] for row, p in enumerate(pts) if removable[row]}
    return recon, removed


def test_removed_observations_are_found_again(without_image, pyramids):
    recon, removed = without_image
    edited = EditedReconstruction(recon)
    nxt, report = edited.add_image_to_tracks(
        IMAGE, pyramids, rule="fixed", min_zncc=0.7
    )

    cands = report["candidates"]
    assert len(cands["point"]) == len(cands["zncc"]) == len(cands["refusal"])
    assert cands["keypoint"].shape == (len(cands["point"]), 2)
    assert report["accepted"] == int(np.sum(cands["accepted"]))
    assert (
        report["observations_after"]
        == report["observations_before"] + report["accepted"]
    )
    assert nxt.point_count == recon.point_count

    found = {
        int(p): cands["keypoint"][k]
        for k, p in enumerate(cands["point"])
        if cands["accepted"][k] and int(p) in removed
    }
    assert len(found) >= 0.8 * len(removed)
    errors = [float(np.hypot(*(found[p] - removed[p]))) for p in found]
    assert np.median(errors) < 0.5


def test_existing_observations_are_untouched(without_image, pyramids):
    recon, _ = without_image
    nxt, report = EditedReconstruction(recon).add_image_to_tracks(IMAGE, pyramids)
    before, _, _ = EditedReconstruction(recon).materialize()
    after, _, _ = nxt.materialize()
    assert np.array_equal(after.positions, before.positions)
    old = np.asarray(before.track_image_indexes) != IMAGE
    new = np.asarray(after.track_image_indexes) != IMAGE
    assert np.array_equal(
        np.asarray(after.keypoints_xy)[new], np.asarray(before.keypoints_xy)[old]
    )
    assert int(np.sum(~new)) - int(np.sum(~old)) == report["accepted"]


def test_every_rule_and_gate_is_accepted(without_image, pyramids):
    edited = EditedReconstruction(without_image[0])
    for kwargs in [
        dict(rule="track_basis", basis="median_minus_mad", basis_k=3.0),
        dict(rule="track_basis", basis="fraction_of_median", pair_statistic="max"),
        dict(rule="pooled_basis", basis="min"),
        dict(rule="pooled_or_track", track_basis="min", ascend_on_edge=True),
        dict(position_gate="image_mad", position_k=3.0),
        dict(position_gate="max_px", position_max_px=2.0),
        dict(subpixel=False, max_keypoint_uncertainty=0.0),
    ]:
        _, report = edited.add_image_to_tracks(IMAGE, pyramids, **kwargs)
        assert report["accepted"] >= 0
        assert set(report["refusal_counts"]) <= {
            "no_patch",
            "not_in_frame",
            "grazing",
            "back_facing",
            "too_few_references",
            "unlocalizable",
            "no_peak",
            "peak_at_edge",
            "unscorable",
            "below_floor",
            "below_bar",
            "too_far",
            "shared_keypoint",
        }


def test_bad_arguments_raise(without_image, pyramids):
    edited = EditedReconstruction(without_image[0])
    with pytest.raises(ValueError, match="rule must be"):
        edited.add_image_to_tracks(IMAGE, pyramids, rule="loose")
    with pytest.raises(ValueError, match="basis must be"):
        edited.add_image_to_tracks(IMAGE, pyramids, basis="mean")
    with pytest.raises(ValueError, match="not one of"):
        edited.add_image_to_tracks(99, pyramids)
