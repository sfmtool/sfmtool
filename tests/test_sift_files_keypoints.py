# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""A ``sift_files`` reconstruction carries its observations' pixels inline.

Every writer stores the optional ``keypoints_xy`` column when the ``.sift``
files can supply it: ``from_data`` and ``load`` build it from those files when
the data lacks it, and ``clone_with_changes`` rebuilds it when the tracks are
replaced.
"""

from pathlib import Path

import numpy as np

from sfmtool.reconstruction import RangeExpr
from sfmtool.fileio import read_sfmr, write_sfmr
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.sift.file import SiftReader, get_sift_path_for_image
from sfmtool.xform import IncludeRangeFilter, apply_transforms


def _sift_pixels(recon) -> np.ndarray:
    """Each observation's ``.sift`` position, parallel to the track arrays."""
    workspace = Path(recon.workspace_dir)
    images = np.asarray(recon.track_image_indexes)
    features = np.asarray(recon.track_feature_indexes)
    out = np.zeros((len(images), 2), dtype=np.float32)
    for image in np.unique(images):
        rows = np.nonzero(images == image)[0]
        path = get_sift_path_for_image(workspace / recon.image_names[image])
        with SiftReader(path) as reader:
            positions = np.asarray(reader.read_positions(), dtype=np.float32)
        out[rows] = positions[features[rows]]
    return out


def test_a_written_file_carries_the_sift_pixels(seoul_bull_workspace):
    """The fixture is built through ``from_data`` from a dict with no
    ``keypoints_xy``; the file it saved carries the column anyway."""
    data = read_sfmr(seoul_bull_workspace)
    assert data["metadata"]["feature_source"] == "sift_files"
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    np.testing.assert_array_equal(data["keypoints_xy"], _sift_pixels(recon))


def test_load_fills_a_file_written_without_the_column(seoul_bull_workspace, tmp_path):
    data = read_sfmr(seoul_bull_workspace)
    del data["keypoints_xy"]
    bare = seoul_bull_workspace.parent / "bare.sfmr"
    write_sfmr(bare, data)
    assert read_sfmr(bare).get("keypoints_xy") is None

    recon = SfmrReconstruction.load(bare)
    np.testing.assert_array_equal(np.asarray(recon.keypoints_xy), _sift_pixels(recon))

    resaved = tmp_path / "resaved.sfmr"
    recon.save(resaved)
    assert read_sfmr(resaved)["keypoints_xy"].shape == (recon.observation_count, 2)


def test_a_filter_keeps_the_column_in_step(seoul_bull_workspace, tmp_path):
    """An xform that drops images replaces the tracks; the output still states
    every remaining observation's pixel."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    out = apply_transforms(recon, [IncludeRangeFilter(RangeExpr("1-10"))])
    path = tmp_path / "subset.sfmr"
    out.save(path, operation="xform_test")

    written = read_sfmr(path)["keypoints_xy"]
    reloaded = SfmrReconstruction.load(path)
    np.testing.assert_array_equal(written, _sift_pixels(reloaded))


def test_a_track_replacement_keeps_pixels_the_source_stated(seoul_bull_workspace):
    """A pixel a producer refined past the detection survives a track
    replacement that keeps its observation."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    refined = np.asarray(recon.keypoints_xy).copy()
    refined[0] += 0.25
    recon = recon.clone_with_changes(keypoints_xy=refined)

    # Drop the last point's observations and the point itself.
    points = np.asarray(recon.track_point_indexes)
    keep = points != recon.point_count - 1
    positions = np.asarray(recon.positions)[:-1]
    out = recon.clone_with_changes(
        positions=positions,
        colors=np.asarray(recon.colors)[:-1],
        errors=np.asarray(recon.errors)[:-1],
        track_image_indexes=np.asarray(recon.track_image_indexes)[keep],
        track_feature_indexes=np.asarray(recon.track_feature_indexes)[keep],
        track_point_indexes=points[keep],
    )
    np.testing.assert_array_equal(np.asarray(out.keypoints_xy), refined[keep])


def test_keypoints_none_drops_the_column(seoul_bull_workspace):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    assert recon.clone_with_changes(keypoints_xy=None).keypoints_xy is None
