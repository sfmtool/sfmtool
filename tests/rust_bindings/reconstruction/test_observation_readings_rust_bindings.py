# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The optional per-observation readings: ``SfmrReconstruction.observation_readings``,
``clone_with_changes(observation_readings=...)`` and the ``read_sfmr`` /
``write_sfmr`` dict."""

import zipfile

import numpy as np
import pytest

from sfmtool.fileio import read_sfmr, write_sfmr
from sfmtool.reconstruction import SfmrReconstruction

COLUMNS = {
    "zncc_self_similarity_ellipse_axes": ((2,), np.float32),
    "zncc_self_similarity_ellipse_axes_is_at_least": ((2,), np.uint8),
    "zncc_self_similarity_ellipse_major_angle": ((), np.float32),
    "zncc_self_similarity_cos_view_angle": ((), np.float32),
    "zncc_self_similarity_tilt_angle": ((), np.float32),
    "zncc_self_similarity_zoom": ((2,), np.float32),
    "plain_bitmap_zncc": ((), np.float32),
    "blur_matched_bitmap_zncc": ((), np.float32),
}


def _readings(m: int) -> dict:
    """A readings dict whose every row says which observation it is."""
    tag = np.arange(1, m + 1, dtype=np.float32)
    out = {
        "zncc_self_similarity_ellipse_axes": np.stack([tag, tag / 2], axis=1),
        "zncc_self_similarity_ellipse_axes_is_at_least": np.stack(
            [(np.arange(m) % 2).astype(np.uint8), np.zeros(m, np.uint8)], axis=1
        ),
        "zncc_self_similarity_ellipse_major_angle": (tag * 0.001).astype(np.float32),
        "zncc_self_similarity_cos_view_angle": np.full(m, 0.75, np.float32),
        "zncc_self_similarity_tilt_angle": np.full(m, np.nan, np.float32),
        "zncc_self_similarity_zoom": np.tile(np.float32([0.25, 0.5]), (m, 1)),
        "plain_bitmap_zncc": (tag * 1e-4).astype(np.float32),
        "blur_matched_bitmap_zncc": (tag * 2e-4).astype(np.float32),
        "options": dict(OPTIONS),
    }
    return out


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


def _assert_rows_equal(a: dict, b: dict) -> None:
    for key in COLUMNS:
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]), key)


@pytest.fixture
def ground_truth(seoul_bull_ground_truth_sfmr) -> SfmrReconstruction:
    return SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)


def test_absent_on_a_file_without_them(ground_truth):
    assert ground_truth.observation_readings is None


def test_clone_save_load_round_trip(ground_truth, seoul_bull_ground_truth_sfmr):
    tmp_path = seoul_bull_ground_truth_sfmr.parent
    m = len(np.asarray(ground_truth.track_point_indexes))
    readings = _readings(m)
    with_readings = ground_truth.clone_with_changes(observation_readings=readings)
    got = with_readings.observation_readings
    for key, (shape, dtype) in COLUMNS.items():
        assert got[key].dtype == dtype, key
        assert got[key].shape == (m,) + shape, key
    _assert_rows_equal(got, readings)
    assert got["options"]["max_radius"] == 3
    assert got["options"]["anisotropic_threshold"] == 1.5
    assert ground_truth.observation_readings is None

    path = tmp_path / "readings.sfmr"
    with_readings.save(path)
    _assert_rows_equal(SfmrReconstruction.load(path).observation_readings, readings)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    for key in COLUMNS:
        assert any(n.startswith(f"tracks/{key}.{m}.") for n in names), key

    # read_sfmr / write_sfmr carry the same dict.
    data = read_sfmr(path)
    _assert_rows_equal(data["observation_readings"], readings)
    out = tmp_path / "rewritten.sfmr"
    write_sfmr(out, data)
    _assert_rows_equal(read_sfmr(out)["observation_readings"], readings)


def test_none_drops_and_omitting_keeps(ground_truth):
    m = len(np.asarray(ground_truth.track_point_indexes))
    with_readings = ground_truth.clone_with_changes(observation_readings=_readings(m))
    kept = with_readings.clone_with_changes(world_space_unit="mm")
    _assert_rows_equal(kept.observation_readings, _readings(m))
    assert (
        with_readings.clone_with_changes(observation_readings=None).observation_readings
        is None
    )


def test_a_wrong_shape_or_dtype_is_refused(ground_truth):
    m = len(np.asarray(ground_truth.track_point_indexes))
    short = _readings(m)
    short["plain_bitmap_zncc"] = short["plain_bitmap_zncc"][:-1]
    with pytest.raises(ValueError, match="plain_bitmap_zncc"):
        ground_truth.clone_with_changes(observation_readings=short)
    wrong = _readings(m)
    wrong["zncc_self_similarity_zoom"] = wrong["zncc_self_similarity_zoom"].astype(
        np.float64
    )
    with pytest.raises(TypeError, match="zncc_self_similarity_zoom"):
        ground_truth.clone_with_changes(observation_readings=wrong)
    missing = _readings(m)
    del missing["blur_matched_bitmap_zncc"]
    with pytest.raises(ValueError, match="blur_matched_bitmap_zncc"):
        ground_truth.clone_with_changes(observation_readings=missing)


def test_replaced_tracks_carry_each_row_with_its_observation(ground_truth):
    # Tracks handed back in another order, as a bundle adjustment's readback
    # does, keep each row with the observation of the same point and image.
    m = len(np.asarray(ground_truth.track_point_indexes))
    recon = ground_truth.clone_with_changes(observation_readings=_readings(m))
    pts = np.asarray(recon.track_point_indexes)
    imgs = np.asarray(recon.track_image_indexes)
    kxy = np.asarray(recon.keypoints_xy)
    # Reverse each point's run.
    order = np.lexsort((-imgs.astype(np.int64), pts))
    out = recon.clone_with_changes(
        track_image_indexes=np.ascontiguousarray(imgs[order]),
        track_feature_indexes=np.zeros(m, dtype=np.uint32),
        track_point_indexes=np.ascontiguousarray(pts[order]),
        keypoints_xy=np.ascontiguousarray(kxy[order]),
    )
    got = out.observation_readings
    want = _readings(m)
    for key in COLUMNS:
        np.testing.assert_array_equal(np.asarray(got[key]), want[key][order], key)


def test_a_changed_reference_clears_that_points_scores(ground_truth):
    m = len(np.asarray(ground_truth.track_point_indexes))
    recon = ground_truth.clone_with_changes(observation_readings=_readings(m))
    references = np.asarray(recon.reference_observations).copy()
    counts = np.asarray(recon.observation_counts)
    p = int(np.flatnonzero(counts >= 2)[0])
    references[p] = 1 if references[p] != 1 else 0
    out = recon.clone_with_changes(reference_observations=references)
    got = out.observation_readings
    offsets = np.concatenate([[0], np.cumsum(counts)])
    run = slice(int(offsets[p]), int(offsets[p + 1]))
    assert np.isnan(got["plain_bitmap_zncc"][run]).all()
    assert np.isnan(got["blur_matched_bitmap_zncc"][run]).all()
    # The rest of the row, and every other point's scores, stand.
    want = _readings(m)
    np.testing.assert_array_equal(
        got["zncc_self_similarity_ellipse_axes"],
        want["zncc_self_similarity_ellipse_axes"],
    )
    keep = np.ones(m, dtype=bool)
    keep[run] = False
    np.testing.assert_array_equal(
        got["plain_bitmap_zncc"][keep], want["plain_bitmap_zncc"][keep]
    )


def test_options_are_required_and_round_trip(ground_truth):
    m = len(np.asarray(ground_truth.track_point_indexes))
    readings = _readings(m)
    del readings["options"]
    with pytest.raises(ValueError, match="options"):
        ground_truth.clone_with_changes(observation_readings=readings)
    partial = _readings(m)
    del partial["options"]["resolution"]
    with pytest.raises(ValueError, match="resolution"):
        ground_truth.clone_with_changes(observation_readings=partial)
    got = ground_truth.clone_with_changes(observation_readings=_readings(m))
    assert got.observation_readings["options"] == OPTIONS


def test_a_reversed_column_writes(ground_truth, seoul_bull_ground_truth_sfmr):
    # A column handed over as a reversed view is written in its logical order,
    # rather than failing on its negative stride.
    m = len(np.asarray(ground_truth.track_point_indexes))
    path = seoul_bull_ground_truth_sfmr.parent / "with_readings.sfmr"
    ground_truth.clone_with_changes(observation_readings=_readings(m)).save(path)
    data = read_sfmr(path)
    arr = np.ascontiguousarray(data["observation_readings"]["plain_bitmap_zncc"][::-1])
    data["observation_readings"]["plain_bitmap_zncc"] = arr[::-1]
    out = seoul_bull_ground_truth_sfmr.parent / "reversed.sfmr"
    write_sfmr(out, data)
    np.testing.assert_array_equal(
        read_sfmr(out)["observation_readings"]["plain_bitmap_zncc"],
        _readings(m)["plain_bitmap_zncc"],
    )


def test_an_edit_record_carries_its_options(ground_truth):
    from sfmtool.reconstruction import EditedReconstruction

    m = len(np.asarray(ground_truth.track_point_indexes))
    recon = ground_truth.clone_with_changes(observation_readings=_readings(m))
    edited = EditedReconstruction(recon)
    record = edited.point(0)
    assert record["observation_readings"]["options"] == OPTIONS
    # Handed back as it came, the rows stand.
    kept = edited.replace_point(0, dict(record))
    np.testing.assert_array_equal(
        edited.point(kept)["observation_readings"]["plain_bitmap_zncc"],
        record["observation_readings"]["plain_bitmap_zncc"],
    )
    # Read under another resolution, they are not measured.
    other = dict(record)
    readings = {k: v for k, v in record["observation_readings"].items()}
    readings["options"] = dict(OPTIONS, resolution=12)
    other["observation_readings"] = readings
    moved = edited.replace_point(kept, other)
    got = edited.point(moved)["observation_readings"]
    assert got["options"] == OPTIONS
    assert np.isnan(got["zncc_self_similarity_ellipse_axes"]).all()
