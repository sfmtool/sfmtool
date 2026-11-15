# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Every extractor writes `.sift` version 2 with the image's readings, and no
writer mixes the two versions: no file from version 1 settings, and none beside
a version 1 file."""

import json
import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import SiftFile, read_image_rgb, read_sift, write_sift
from sfmtool.sift import ImageToGray, image_readings_from_rgb, version_1_sift_files
from sfmtool.sift.extract import SiftExtractionError, image_files_to_sift_files
from sfmtool.sift.extract_colmap import get_colmap_feature_options
from sfmtool.sift.extract_opencv import get_default_opencv_feature_options
from sfmtool.sift.extract_sfmtool import get_default_sfmtool_feature_options

from ..conftest import SEOUL_BULL_DIR

_IMAGES = ("seoul_bull_sculpture_01.jpg", "seoul_bull_sculpture_02.jpg")


@pytest.fixture
def images(tmp_path: Path) -> list[Path]:
    """Two seoul_bull photographs in `images/` of a temporary directory."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in _IMAGES:
        shutil.copy(SEOUL_BULL_DIR / name, image_dir / name)
    return [image_dir / name for name in _IMAGES]


def _assert_version_2_with_readings(sift_path: Path, image_path: Path, options):
    """The file is version 2, records `options`' conversion, and holds the
    readings `image_readings_from_rgb` takes from the same decode."""
    data = read_sift(sift_path)
    metadata = data["metadata"]
    assert metadata["version"] == 2
    image_to_gray = data["feature_tool_metadata"]["feature_options"]["image_to_gray"]
    assert image_to_gray == options["image_to_gray"]
    expected = image_readings_from_rgb(
        read_image_rgb(image_path), ImageToGray.from_json(image_to_gray)
    )
    assert metadata["sharpness"] == expected.sharpness
    assert metadata["gradient_structure_tensor_xx_yy_xy"] == list(
        expected.gradient_structure_tensor_xx_yy_xy
    )
    assert metadata["sharpness"] > 0


@pytest.mark.parametrize(
    "tool, options",
    [
        ("sfmtool", get_default_sfmtool_feature_options),
        ("opencv", get_default_opencv_feature_options),
        ("colmap", lambda: get_colmap_feature_options(use_gpu=False)),
        (
            "sfmtool",
            lambda: get_default_sfmtool_feature_options(
                image_to_gray={"formula": "G", "quantize_bits": 6}
            ),
        ),
    ],
)
def test_every_backend_writes_version_2_with_the_readings_of_its_decode(
    images, tool, options
):
    options = options()
    sift_paths = image_files_to_sift_files(
        images, feature_tool=tool, feature_options=options
    )
    for sift_path, image_path in zip(sift_paths, images):
        _assert_version_2_with_readings(sift_path, image_path, options)


def _write_version_1_sift(sift_path: Path, image_path: Path) -> None:
    """A version 1 `.sift` file for `image_path`, as a version 1 writer made it,
    newer than the image."""
    data = read_sift(
        image_files_to_sift_files(
            [image_path],
            feature_tool="sfmtool",
            feature_options=get_default_sfmtool_feature_options(),
            feature_path=sift_path.parent.parent / "scratch",
        )[0]
    )
    data["metadata"]["version"] = 1
    del data["metadata"]["sharpness"]
    del data["metadata"]["gradient_structure_tensor_xx_yy_xy"]
    options = data["feature_tool_metadata"]["feature_options"]
    options["gray_formula"] = options.pop("image_to_gray")["formula"]
    sift_path.parent.mkdir(parents=True, exist_ok=True)
    write_sift(sift_path, data)
    later = image_path.stat().st_mtime + 10
    os.utime(sift_path, (later, later))


def test_a_version_2_workspace_whose_folder_holds_version_1_files_is_told_to_remove_them(
    images, tmp_path
):
    """`sfm ws upgrade` does nothing to a version 2 workspace, so the advice is
    to remove the version 1 files from its folder and run again, which then
    works."""
    result = CliRunner().invoke(
        main, ["ws", "init", "--feature-tool", "sfmtool", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    prefix = json.loads((tmp_path / ".sfm-workspace.json").read_text())[
        "feature_prefix_dir"
    ]
    folder = images[1].parent / prefix
    v1_path = folder / (images[1].name + ".sift")
    _write_version_1_sift(v1_path, images[1])

    result = CliRunner().invoke(main, ["sift", "--extract", str(tmp_path / "images")])
    assert result.exit_code != 0
    assert f"the folder {folder} holds version 1 .sift files" in result.output
    assert f"({v1_path.name})" in result.output
    assert "remove those files from the folder and run again" in result.output
    assert "sfm ws upgrade" not in result.output
    assert not (folder / (images[0].name + ".sift")).exists()

    v1_path.unlink()
    result = CliRunner().invoke(main, ["sift", "--extract", str(tmp_path / "images")])
    assert result.exit_code == 0, result.output
    for image in images:
        with SiftFile(folder / (image.name + ".sift")) as reader:
            assert reader.metadata["version"] == 2


def test_version_1_files_in_the_folder_are_refused(images, tmp_path):
    options = get_default_sfmtool_feature_options()
    folder = tmp_path / "images" / "features" / "mixed"
    _write_version_1_sift(folder / (images[1].name + ".sift"), images[1])
    with pytest.raises(SiftExtractionError) as raised:
        image_files_to_sift_files(
            images[:1],
            feature_tool="sfmtool",
            feature_options=options,
            feature_prefix_dir="features/mixed",
        )
    # The message is the binding's, so Rust and Python say the same.
    found = version_1_sift_files(folder)
    assert found is not None
    assert str(raised.value) == str(found)
    assert found.folder == folder
    assert found.files == [folder / (images[1].name + ".sift")]
    assert not (folder / (images[0].name + ".sift")).exists()


def test_version_1_sift_files_passes_over_what_is_not_a_version_1_sift(tmp_path):
    folder = tmp_path / "features"
    assert version_1_sift_files(folder) is None
    folder.mkdir()
    (folder / "broken.jpg.sift").write_bytes(b"not a zip")
    (folder / "notes.txt").write_text("")
    assert version_1_sift_files(folder) is None
    with pytest.raises(OSError, match="failed to list the folder"):
        version_1_sift_files(folder / "notes.txt")


@pytest.mark.parametrize(
    "tool, options",
    [
        ("colmap", lambda: get_colmap_feature_options(use_gpu=False)),
        ("opencv", get_default_opencv_feature_options),
    ],
)
def test_a_colmap_or_opencv_conversion_other_than_the_tool_s_is_refused(
    images, tool, options
):
    options = options()
    options["image_to_gray"] = {"formula": "G"}
    with pytest.raises(SiftExtractionError, match=f"The '{tool}' feature tool"):
        image_files_to_sift_files(images, feature_tool=tool, feature_options=options)


def test_options_without_image_to_gray_are_refused(images):
    options = get_default_sfmtool_feature_options()
    del options["image_to_gray"]
    with pytest.raises(SiftExtractionError) as raised:
        image_files_to_sift_files(
            images, feature_tool="sfmtool", feature_options=options
        )
    assert str(raised.value) == (
        "The feature options carry no 'image_to_gray', so they are version 1 "
        "settings, and no .sift file is written from version 1 settings."
    )
    assert not list(images[0].parent.rglob("*.sift"))


def test_tool_outside_a_workspace_writes_version_2(images):
    """`sfm sift --extract --tool` outside a workspace uses the tool's version 2
    defaults and a folder named from them."""
    result = CliRunner().invoke(
        main, ["sift", "--extract", "--tool", "opencv", str(images[0].parent)]
    )
    assert result.exit_code == 0, result.output
    sift_paths = sorted(images[0].parent.rglob("*.sift"))
    assert len(sift_paths) == 2
    for sift_path in sift_paths:
        with SiftFile(sift_path) as reader:
            assert reader.metadata["version"] == 2
