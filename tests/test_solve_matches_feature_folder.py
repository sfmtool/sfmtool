# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""A `.matches` file is used with the `.sift` files it was matched on.

The workspace here is matched once, then its feature definition is changed
(`sfm ws init --force` with another grey conversion, as `sfm ws upgrade`
changes it) and its `.sift` files extracted again into the new folder, where
each image has a different number of features. A solve, or `sfm to-colmap-db`,
from the earlier `.matches` file must read the earlier folder, and refuse
before it builds anything when that folder's files are gone or changed.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import sfmtool.fileio
from sfmtool import load_workspace_config
from sfmtool.cli import main
from sfmtool.colmap.io import save_colmap_binary
from sfmtool.fileio import SiftFile, read_colmap_binary, read_matches
from sfmtool.merge.reconstructions import merge_reconstructions
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.visualization._epipolar_display import _require_sift_path

from .conftest import SEOUL_BULL_DIR


def _invoke(*args):
    return CliRunner().invoke(main, [str(a) for a in args])


@pytest.fixture(scope="module")
def moved_features_workspace(tmp_path_factory) -> tuple[Path, Path, str, str]:
    """A workspace with a `.matches` file made on its first feature folder,
    and its `.sift` files extracted again into a second one.

    Returns `(workspace, matches_path, old_folder, new_folder)`. Tests copy it.
    """
    workspace = tmp_path_factory.mktemp("moved_features") / "ws"
    workspace.mkdir()
    # Eight of the 17 images: enough for a solve, and half the extraction
    # and matching of all of them.
    for image in sorted(SEOUL_BULL_DIR.glob("seoul_bull_sculpture_*.jpg"))[:8]:
        shutil.copy(image, workspace / image.name)
    shutil.copy(SEOUL_BULL_DIR / "camera_config.json", workspace)

    result = _invoke("ws", "init", workspace)
    assert result.exit_code == 0, result.output
    old_folder = load_workspace_config(workspace)["feature_prefix_dir"]
    matches_path = workspace / "old.matches"
    result = _invoke("match", "--exhaustive", "--output", matches_path, workspace)
    assert result.exit_code == 0, result.output

    result = _invoke("ws", "init", "--force", "--image-to-gray", "G", workspace)
    assert result.exit_code == 0, result.output
    new_folder = load_workspace_config(workspace)["feature_prefix_dir"]
    assert new_folder != old_folder
    result = _invoke("sift", "--extract", workspace)
    assert result.exit_code == 0, result.output
    return workspace, matches_path, old_folder, new_folder


@pytest.fixture
def workspace_copy(moved_features_workspace, tmp_path):
    workspace, matches_path, old_folder, new_folder = moved_features_workspace
    copy = tmp_path / "ws"
    shutil.copytree(workspace, copy)
    return copy, copy / matches_path.name, old_folder, new_folder


def _positions(sift_path: Path) -> np.ndarray:
    with SiftFile(sift_path) as reader:
        return np.asarray(reader.read_positions(), dtype=np.float32)


def test_solve_reads_the_feature_folder_the_matches_file_was_matched_on(
    workspace_copy, monkeypatch
):
    workspace, matches_path, old_folder, new_folder = workspace_copy
    output = workspace / "solved.sfmr"
    result = _invoke("solve", "-g", "--output", output, matches_path)
    assert result.exit_code == 0, result.output
    assert f"Features: {old_folder}, which the .matches file" in result.output

    # The reconstruction records the .matches file's feature definition.
    recon = SfmrReconstruction.load(output)
    matches = read_matches(matches_path)
    contents = recon.metadata()["workspace"]["contents"]
    matched = matches["metadata"]["workspace"]["contents"]
    assert contents["feature_prefix_dir"] == old_folder
    assert contents["feature_options"] == matched["feature_options"]
    assert (
        contents["feature_options"]
        != (load_workspace_config(workspace)["feature_options"])
    )

    # The recorded .sift hashes are those the .matches file recorded.
    recorded = {
        name: bytes(h)
        for name, h in zip(matches["image_names"], matches["sift_content_hashes"])
    }
    for name, h in zip(recon.image_names, recon.sift_content_hashes):
        assert bytes(h) == recorded[name], name

    # Every observation's keypoint is its feature in the old folder's .sift
    # file, and not the feature at the same index in the new folder's.
    keypoints = np.asarray(recon.keypoints_xy)
    image_indexes = np.asarray(recon.track_image_indexes)
    feature_indexes = np.asarray(recon.track_feature_indexes)
    differs_from_new = 0
    for i, name in enumerate(recon.image_names):
        rows = image_indexes == i
        old = _positions(workspace / old_folder / f"{name}.sift")
        np.testing.assert_array_equal(keypoints[rows], old[feature_indexes[rows]])
        new = _positions(workspace / new_folder / f"{name}.sift")
        indexes = feature_indexes[rows]
        in_new = indexes < len(new)
        differs_from_new += int(
            np.any(new[indexes[in_new]] != old[indexes[in_new]]) or not in_new.all()
        )
    assert differs_from_new == recon.image_count

    # A sound solve: finite points that reproject to within a pixel or so.
    assert recon.image_count == 8
    at_infinity = np.asarray(recon.point_is_at_infinity, dtype=bool)
    assert at_infinity.mean() < 0.05
    errors = np.asarray(recon.rms_reprojection_errors)
    assert float(errors[~at_infinity].mean()) < 1.5

    # A COLMAP export, which a bundle adjustment goes through, also reads the
    # folder the reconstruction records, not the workspace's current one.
    colmap_dir = workspace / "colmap"
    save_colmap_binary(recon, colmap_dir)
    exported = read_colmap_binary(colmap_dir)
    for name, kps in zip(exported["image_names"], exported["keypoints_per_image"]):
        old = _positions(workspace / old_folder / f"{name}.sift")
        np.testing.assert_array_equal(np.asarray(kps, dtype=np.float32)[:, :2], old)

    # `sfm epipolar` and the pose refinement of `sfm merge` read the same folder.
    name = recon.image_names[0]
    assert _require_sift_path(recon, 0, name) == workspace / old_folder / (
        f"{name}.sift"
    )
    opened = []

    def recording_sift_file(path, *args, **kwargs):
        opened.append(Path(path))
        return SiftFile(path, *args, **kwargs)

    monkeypatch.setattr(sfmtool.fileio, "SiftFile", recording_sift_file)
    merge_reconstructions([recon, recon])
    assert opened
    assert {p.parent for p in opened} == {workspace / old_folder}


def test_solve_refuses_before_solving_when_the_feature_folder_is_gone(
    workspace_copy,
):
    workspace, matches_path, old_folder, _ = workspace_copy
    shutil.rmtree(workspace / old_folder)
    output = workspace / "solved.sfmr"
    result = _invoke("solve", "-g", "--output", output, matches_path)
    assert result.exit_code != 0
    assert f"Cannot use {matches_path}" in result.output
    assert f"in the feature folder {old_folder}" in result.output
    assert "8 of the 8 .sift files it needs are missing" in result.output
    assert "Run 'sfm match' on its images again" in result.output
    # Stopped before the database was written or anything solved.
    assert "Wrote" not in result.output
    assert not output.exists()


def test_solve_refuses_when_a_sift_file_differs_from_its_recorded_hash(
    workspace_copy,
):
    workspace, matches_path, old_folder, new_folder = workspace_copy
    name = "seoul_bull_sculpture_05.jpg.sift"
    shutil.copy(workspace / new_folder / name, workspace / old_folder / name)
    output = workspace / "solved.sfmr"
    result = _invoke("solve", "-i", "--output", output, matches_path)
    assert result.exit_code != 0
    assert "1 of the 8 .sift files it needs has changed" in result.output
    assert str(workspace / old_folder / name) in result.output
    assert "Wrote" not in result.output
    assert not output.exists()


def test_to_colmap_db_refuses_when_the_feature_folder_is_gone(workspace_copy):
    workspace, matches_path, old_folder, _ = workspace_copy
    shutil.rmtree(workspace / old_folder)
    db_path = workspace / "database.db"
    result = _invoke("to-colmap-db", matches_path, db_path)
    assert result.exit_code != 0
    assert f"Cannot use {matches_path}" in result.output
    assert f"in the feature folder {old_folder}" in result.output
    assert not db_path.exists()
