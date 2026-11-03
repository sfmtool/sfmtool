# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfm to-colmap-db` on a reconstruction whose record holds version 1
settings: the record is converted in memory and its `.sift` files are read
from the converted folder, or, where a new extraction cannot be shown to give
the record's features, the export is refused naming the `.sfmr`."""

import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import read_sift, write_sift
from sfmtool.sift.extract import SiftExtractionError, image_files_to_sift_files
from sfmtool.sift.extract_opencv import get_default_opencv_feature_options
from sfmtool.sift.extract_sfmtool import get_default_sfmtool_feature_options

from . import conftest
from .conftest import SEOUL_BULL_DIR, SEOUL_BULL_GROUND_TRUTH

_CONVERTED = "Converted the version 1 SIFT extraction record"


def _export(sfmr: Path, db: Path):
    return CliRunner().invoke(
        main, ["to-colmap-db", "--no-guided-matching", str(sfmr), str(db)]
    )


def test_an_embedded_patches_record_of_version_1_is_converted(tmp_path):
    """The seoul_bull ground truth (embedded patches, no .sift files, a
    version 1 sfmtool record) exports: its files are extracted into the folder
    the converted record names, and one line says so."""
    shutil.copy(SEOUL_BULL_GROUND_TRUTH, tmp_path / SEOUL_BULL_GROUND_TRUTH.name)
    shutil.copy(SEOUL_BULL_DIR / ".sfm-workspace.json", tmp_path)
    for image in SEOUL_BULL_DIR.glob("seoul_bull_sculpture_*.jpg"):
        shutil.copy(image, tmp_path / image.name)
    old_prefix = json.loads((tmp_path / ".sfm-workspace.json").read_text())[
        "feature_prefix_dir"
    ]

    result = _export(tmp_path / SEOUL_BULL_GROUND_TRUTH.name, tmp_path / "out.db")
    assert result.exit_code == 0, result.output
    line = next(line for line in result.output.splitlines() if _CONVERTED in line)
    assert f"moved from {old_prefix}" in line
    assert str(tmp_path / SEOUL_BULL_GROUND_TRUTH.name) in line
    assert "sfm ws upgrade" not in result.output
    assert (tmp_path / "out.db").is_file()
    assert not (tmp_path / old_prefix).exists()
    new_files = sorted(tmp_path.glob("features/*/*.sift"))
    assert len(new_files) == 17
    assert all(read_sift(p)["metadata"]["version"] == 2 for p in new_files)


def _as_version_1(data: dict, tool: str) -> dict:
    """A version 2 `.sift` dict rewritten as a version 1 writer wrote it."""
    data["metadata"]["version"] = 1
    del data["metadata"]["sharpness"]
    del data["metadata"]["gradient_structure_tensor_xx_yy_xy"]
    options = data["feature_tool_metadata"]["feature_options"]
    image_to_gray = options.pop("image_to_gray")
    if tool == "sfmtool":
        options["gray_formula"] = image_to_gray["formula"]
    return data


def _version_1_sift_files_reconstruction(tmp_path: Path, tool: str) -> Path:
    """A `sift_files` reconstruction of six seoul_bull images made in a
    version 1 `tool` workspace, at the ground truth's poses."""
    workspace = tmp_path / "ws"
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True)
    images = []
    for i in range(1, 7):
        name = f"seoul_bull_sculpture_{i:02d}.jpg"
        shutil.copy(SEOUL_BULL_DIR / name, images_dir / name)
        images.append(images_dir / name)

    if tool == "sfmtool":
        config = json.loads((SEOUL_BULL_DIR / ".sfm-workspace.json").read_text())
        v2_options = get_default_sfmtool_feature_options()
    else:
        v2_options = get_default_opencv_feature_options()
        v1_options = {k: v for k, v in v2_options.items() if k != "image_to_gray"}
        config = {
            "version": 1,
            "feature_tool": tool,
            "feature_type": f"sift-{tool}",
            "feature_options": v1_options,
            "feature_prefix_dir": f"features/sift-{tool}-v1",
        }
    (workspace / ".sfm-workspace.json").write_text(json.dumps(config, indent=2))
    # The version 1 files, as a version 1 extractor wrote them.
    scratch = image_files_to_sift_files(
        images,
        feature_tool=tool,
        feature_options=v2_options,
        feature_path=tmp_path / "scratch",
    )
    prefix = config["feature_prefix_dir"]
    for image, path in zip(images, scratch):
        target = images_dir / prefix / (image.name + ".sift")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_sift(target, _as_version_1(read_sift(path), tool))

    # The fixture's own workspace setup writes version 2; keep version 1.
    original = conftest.init_workspace
    conftest.init_workspace = lambda *args, **kwargs: None
    try:
        return conftest.build_reconstruction_at_poses(
            workspace, images, SEOUL_BULL_GROUND_TRUTH, workspace / "recon.sfmr"
        )
    finally:
        conftest.init_workspace = original


def test_a_version_1_sfmtool_sift_files_record_reads_the_converted_folder(tmp_path):
    """With one .sift file deleted, the export extracts into the converted
    folder, and the database's features for that image are the deleted
    file's, so the reconstruction's feature indexes still hold."""
    sfmr = _version_1_sift_files_reconstruction(tmp_path, "sfmtool")
    workspace = sfmr.parent
    prefix = json.loads((workspace / ".sfm-workspace.json").read_text())[
        "feature_prefix_dir"
    ]
    gone = workspace / "images" / prefix / "seoul_bull_sculpture_03.jpg.sift"
    original = read_sift(gone)
    gone.unlink()

    result = _export(sfmr, tmp_path / "out.db")
    assert result.exit_code == 0, result.output
    assert _CONVERTED in result.output
    assert f"moved from {prefix}" in result.output
    assert "sfm ws upgrade" not in result.output

    with pycolmap.Database.open(tmp_path / "out.db") as db:
        image = db.read_image_with_name("seoul_bull_sculpture_03.jpg")
        keypoints = db.read_keypoints(image.image_id)
        descriptors = db.read_descriptors(image.image_id)
    descriptors = getattr(descriptors, "data", descriptors)
    np.testing.assert_array_equal(keypoints[:, :2], original["positions_xy"])
    np.testing.assert_array_equal(descriptors, original["descriptors"])


def test_a_version_1_opencv_sift_files_record_with_a_missing_file_is_refused(
    tmp_path,
):
    """A new opencv or colmap extraction is not known to reproduce the
    features the record's indexes name, so a missing file is refused, naming
    the .sfmr and not pointing at sfm ws upgrade."""
    sfmr = _version_1_sift_files_reconstruction(tmp_path, "opencv")
    gone = (
        sfmr.parent / "images/features/sift-opencv-v1/seoul_bull_sculpture_03.jpg.sift"
    )
    gone.unlink()
    result = _export(sfmr, tmp_path / "out.db")
    assert result.exit_code != 0
    assert str(sfmr) in result.output
    assert "not known to give the same features" in result.output
    assert "sfm ws upgrade" not in result.output

    # With every file present, nothing is extracted and the export succeeds.
    sfmr2 = _version_1_sift_files_reconstruction(tmp_path / "again", "opencv")
    result = _export(sfmr2, tmp_path / "again.db")
    assert result.exit_code == 0, result.output
    assert _CONVERTED not in result.output


def test_an_undistort_tool_is_refused(tmp_path):
    shutil.copy(SEOUL_BULL_DIR / "seoul_bull_sculpture_01.jpg", tmp_path)
    with pytest.raises(SiftExtractionError, match="Run 'sfm undistort' again"):
        image_files_to_sift_files(
            [tmp_path / "seoul_bull_sculpture_01.jpg"],
            feature_tool="sfmtool-undistort",
            feature_options={"image_to_gray": {"formula": "G"}},
            feature_prefix_dir="features/sift-sfmtool-undistort-x",
        )
    assert not (tmp_path / "features").exists()
