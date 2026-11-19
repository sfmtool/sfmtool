# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Every extractor writes `.sift` version 2 with the image's readings, no file
is written from version 1 settings, and a version 1 file is refused, naming the
last release that reads it."""

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import (
    SiftFile,
    read_image_rgb,
    read_sift,
    read_sift_metadata,
    verify_sift,
)
from sfmtool.sift import ImageToGray, image_readings_from_rgb
from sfmtool.sift.extract import SiftExtractionError, image_files_to_sift_files
from sfmtool.sift.extract_colmap import get_colmap_feature_options
from sfmtool.sift.extract_opencv import get_default_opencv_feature_options
from sfmtool.sift.extract_sfmtool import get_default_sfmtool_feature_options

from ..conftest import SEOUL_BULL_DIR, rewrite_sift_as_version_1

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
    """A version 1 `.sift` file for `image_path`, as a version 1 writer made it."""
    [v2_path] = image_files_to_sift_files(
        [image_path],
        feature_tool="sfmtool",
        feature_options=get_default_sfmtool_feature_options(),
        feature_path=sift_path.parent.parent / "scratch",
    )
    sift_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(v2_path, sift_path)
    rewrite_sift_as_version_1(sift_path)


def test_a_version_1_file_is_refused_naming_the_last_release_that_reads_it(
    images, tmp_path
):
    """Every reader refuses a version 1 file with one message, raised as
    `OSError`, and `sfm inspect` prints it."""
    sift_path = tmp_path / "v1" / (images[0].name + ".sift")
    _write_version_1_sift(sift_path, images[0])
    message = (
        f"{sift_path} is a version 1 .sift file, and this sfmtool reads .sift files "
        "of version 2 or later. sfmtool 0.4.1 is the last release that reads "
        "version 1: install it, run `sfm ws upgrade` on the workspace, then use "
        "the file with this release."
    )
    for read in (read_sift, read_sift_metadata, SiftFile):
        with pytest.raises(OSError) as raised:
            read(sift_path)
        assert str(raised.value) == message
    assert verify_sift(sift_path) == (False, [message])
    result = CliRunner().invoke(main, ["inspect", str(sift_path)])
    assert result.exit_code != 0
    assert "sfmtool 0.4.1 is the last release that reads version 1" in result.output


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
        "settings, and no .sift file is written from version 1 settings. sfmtool "
        "0.4.1 is the last release that converts them: install it and run "
        "`sfm ws upgrade` on the workspace."
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
