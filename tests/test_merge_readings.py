# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""``sfm merge`` carries each observation's readings to the merged observation
of the same image and feature, without the scores, whose bitmaps the merge
does not keep."""

from types import SimpleNamespace

import numpy as np

from sfmtool.merge.reconstructions import _merged_readings

OPTIONS = {
    "resolution": 24,
    "sampler": "per_view",
    "score_window": "gaussian_disk",
    "score_window_sigma": 0.6,
    "max_radius": 3,
    "flat_floor": 0.5,
    "noise": 2.0,
    "relative_tolerance": 0.05,
    "anisotropic_threshold": 1.5,
}


def _readings(tags, options=OPTIONS):
    tag = np.asarray(tags, dtype=np.float32)
    m = len(tag)
    return {
        "zncc_self_similarity_ellipse_axes": np.stack([tag, tag / 2], axis=1),
        "zncc_self_similarity_ellipse_axes_is_at_least": np.zeros((m, 2), np.uint8),
        "zncc_self_similarity_ellipse_major_angle": np.full(m, 0.5, np.float32),
        "zncc_self_similarity_cos_view_angle": np.full(m, 0.75, np.float32),
        "zncc_self_similarity_tilt_angle": np.full(m, 1.0, np.float32),
        "zncc_self_similarity_zoom": np.tile(np.float32([0.25, 0.5]), (m, 1)),
        "plain_bitmap_zncc": tag * 0.01,
        "blur_matched_bitmap_zncc": tag * 0.01,
        "options": dict(options),
    }


def _source(names, img, feat, tags, options=OPTIONS):
    return SimpleNamespace(
        image_names=names,
        track_image_indexes=np.asarray(img, dtype=np.uint32),
        track_feature_indexes=None if feat is None else np.asarray(feat, np.uint32),
        observation_readings=None if tags is None else _readings(tags, options),
    )


def test_rows_follow_image_and_feature_and_scores_are_cleared():
    a = _source(["x.jpg", "y.jpg"], [0, 1], [5, 6], [1.0, 2.0])
    b = _source(["y.jpg", "z.jpg"], [0, 1], [6, 7], [9.0, 3.0])
    merged_names = ["x.jpg", "y.jpg", "z.jpg"]
    tracks = {
        "image_indexes": np.array([2, 1, 0]),
        "feature_indexes": np.array([7, 6, 5]),
    }
    out = _merged_readings([a, b], merged_names, tracks)
    # z/7 from b, y/6 from a (the first source to read it), x/5 from a.
    np.testing.assert_array_equal(
        out["zncc_self_similarity_ellipse_axes"][:, 0], [3.0, 2.0, 1.0]
    )
    assert np.isnan(out["plain_bitmap_zncc"]).all()
    assert np.isnan(out["blur_matched_bitmap_zncc"]).all()
    assert out["options"] == OPTIONS


def test_rows_under_other_options_or_without_features_are_not_measured():
    a = _source(["x.jpg"], [0], [5], [1.0])
    b = _source(["y.jpg"], [0], [6], [2.0], dict(OPTIONS, resolution=12))
    embedded = _source(["z.jpg"], [0], None, [3.0])
    tracks = {
        "image_indexes": np.array([0, 1, 2]),
        "feature_indexes": np.array([5, 6, 0]),
    }
    out = _merged_readings([a, b, embedded], ["x.jpg", "y.jpg", "z.jpg"], tracks)
    axes = out["zncc_self_similarity_ellipse_axes"][:, 0]
    assert axes[0] == 1.0
    assert np.isnan(axes[1:]).all()


def test_no_readings_gives_none():
    a = _source(["x.jpg"], [0], [5], None)
    tracks = {"image_indexes": np.array([0]), "feature_indexes": np.array([5])}
    assert _merged_readings([a], ["x.jpg"], tracks) is None
