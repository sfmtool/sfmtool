# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.sift.file import (
    SiftReader,
    get_feature_tool_xxh128,
    get_feature_type_for_tool,
)
from sfmtool.sift.extract_colmap import get_colmap_feature_options

# Compute the hash of the default feature tool metadata (COLMAP without DSP)
_default_options = get_colmap_feature_options()
_default_feature_type = get_feature_type_for_tool("colmap", _default_options)
EXPECTED_FEATURE_TOOL_HASH = get_feature_tool_xxh128(
    "colmap", _default_feature_type, _default_options
)


def test_default_feature_tool():
    """Sanity check that we're getting a consistent hash for the feature tool."""
    assert len(EXPECTED_FEATURE_TOOL_HASH) == 32
    assert all(c in "0123456789abcdef" for c in EXPECTED_FEATURE_TOOL_HASH)


def test_sift_cli_no_paths():
    """Tests that 'sift --extract' fails with --extract but no paths."""
    result = CliRunner().invoke(main, ["sift", "--extract"])
    assert result.exit_code != 0
    assert "Must provide a list of paths to process" in result.output


@pytest.mark.parametrize("provide_dir", [True, False])
def test_sift_cli_with_file(isolated_seoul_bull_image: Path, provide_dir: bool):
    """Tests 'sift --extract' with a single image file."""
    expected_sift_path = (
        isolated_seoul_bull_image.parent
        / "features"
        / f"sift-colmap-{EXPECTED_FEATURE_TOOL_HASH}"
        / (isolated_seoul_bull_image.name + ".sift")
    )

    if provide_dir:
        extract_path = str(isolated_seoul_bull_image.parent)
    else:
        extract_path = str(isolated_seoul_bull_image)

    # Run the command once, and it should create the .sift file
    result = CliRunner().invoke(
        main, ["sift", "--extract", "--tool", "colmap", extract_path]
    )
    assert result.exit_code == 0, result.output
    assert "Given 1 path(s) to process, expanded to 1 filename(s):" in result.output
    assert "test_image.jpg (1 file)" in result.output
    assert "New SIFT feature extraction: 1 / 1 image(s)" in result.output
    assert "test_image.jpg.sift (1 file)" in result.output

    assert expected_sift_path.exists()
    assert expected_sift_path.is_file()
    with SiftReader(expected_sift_path) as reader:
        assert reader.metadata["image_name"] == isolated_seoul_bull_image.name

    # Run the command a second time, and it should leave the .sift file untouched
    result = CliRunner().invoke(
        main, ["sift", "--extract", "--tool", "colmap", extract_path]
    )
    assert result.exit_code == 0, result.output
    assert "Given 1 path(s) to process, expanded to 1 filename(s)" in result.output
    assert (
        "Existing SIFT features already processed for 1 / 1 image(s)" in result.output
    )
    assert "New SIFT feature extraction: 0 / 1 image(s)" in result.output
    assert "test_image.jpg (1 file)" in result.output
    assert "test_image.jpg.sift (1 file)" in result.output

    # Touch the last-modified time the image, and it should re-compute the .sift file.
    # Sleep first to clear filesystem mtime resolution (a few ms on WSL2 ext4),
    # otherwise touch() can land on the same mtime as the just-written .sift.
    time.sleep(0.005)
    isolated_seoul_bull_image.touch()
    result = CliRunner().invoke(
        main, ["sift", "--extract", "--tool", "colmap", extract_path]
    )
    assert result.exit_code == 0, result.output
    assert "Given 1 path(s) to process, expanded to 1 filename(s)" in result.output
    assert "New SIFT feature extraction: 1 / 1 image(s)" in result.output
    assert "test_image.jpg (1 file)" in result.output
    assert "test_image.jpg.sift (1 file)" in result.output


def test_sift_cli_sfmtool_backend(isolated_seoul_bull_image: Path):
    """Tests 'sift --extract --tool sfmtool' uses the sfmtool Rust backend."""
    from sfmtool.sift.extract_sfmtool import get_default_sfmtool_feature_options

    options = get_default_sfmtool_feature_options()
    feature_type = get_feature_type_for_tool("sfmtool", options)
    assert feature_type == "sift-sfmtool"
    tool_hash = get_feature_tool_xxh128("sfmtool", feature_type, options)

    expected_sift_path = (
        isolated_seoul_bull_image.parent
        / "features"
        / f"{feature_type}-{tool_hash}"
        / (isolated_seoul_bull_image.name + ".sift")
    )

    result = CliRunner().invoke(
        main,
        ["sift", "--extract", "--tool", "sfmtool", str(isolated_seoul_bull_image)],
    )
    assert result.exit_code == 0, result.output
    assert "using the sfmtool backend" in result.output
    assert "New SIFT feature extraction (SFMTOOL): 1 / 1 image(s)" in result.output

    assert expected_sift_path.is_file()
    with SiftReader(expected_sift_path) as reader:
        assert reader.metadata["image_name"] == isolated_seoul_bull_image.name
        assert reader.metadata["feature_count"] > 0
        assert reader.feature_tool_metadata["feature_tool"] == "sfmtool"
        positions = reader.read_positions()
        descriptors = reader.read_descriptors()
    assert positions.shape == (reader.metadata["feature_count"], 2)
    assert descriptors.shape == (reader.metadata["feature_count"], 128)


def test_sfmtool_backend_streams_in_order(isolated_seoul_bull_17_images: list[Path]):
    """The sfmtool backend yields one result per image, in input order.

    Exercises the prefetch/look-ahead generator in extract_sfmtool.py across
    multiple images: it must be lazy (a generator, not a list) and preserve the
    input ordering its callers rely on for the parallel .sift filename list.
    """
    import types

    from sfmtool.sift.extract_sfmtool import (
        extract_sift_with_sfmtool,
        get_default_sfmtool_feature_options,
    )

    images = isolated_seoul_bull_17_images
    options = get_default_sfmtool_feature_options()

    results = extract_sift_with_sfmtool(images, options)
    assert isinstance(results, types.GeneratorType)

    results = list(results)
    assert len(results) == len(images)
    for image_path, result in zip(images, results):
        _, metadata, positions, _, descriptors, thumbnail = result
        assert metadata["image_name"] == image_path.name
        assert metadata["feature_count"] == len(positions) == len(descriptors)
        assert thumbnail.shape == (128, 128, 3)


def test_sfmtool_backend_propagates_decode_error(isolated_seoul_bull_image: Path):
    """A failed read+decode on the prefetch worker surfaces in input order."""
    from sfmtool.sift.extract_sfmtool import (
        extract_sift_with_sfmtool,
        get_default_sfmtool_feature_options,
    )
    from sfmtool.sift.file import SiftExtractionError

    bad_path = isolated_seoul_bull_image.parent / "not_an_image.jpg"
    bad_path.write_bytes(b"this is not a valid image")

    images = [isolated_seoul_bull_image, bad_path]
    results = extract_sift_with_sfmtool(images, get_default_sfmtool_feature_options())

    # The first (valid) image yields fine; the second raises when reached.
    first = next(results)
    assert first[1]["image_name"] == isolated_seoul_bull_image.name
    with pytest.raises(SiftExtractionError, match="Failed to load image"):
        next(results)


def test_sift_write_queue_writes_and_propagates_errors(
    isolated_seoul_bull_image: Path, tmp_path
):
    """SiftWriteQueue (rayon-pool saves) writes correctly and surfaces errors.

    This is the mechanism that overlaps a save with the next image's extract:
    submit() spawns the compression onto the shared rayon pool, join() awaits
    it. A good path produces a readable .sift; a save into a missing directory
    surfaces as an IOError from join().
    """
    from sfmtool._sfmtool import SiftWriteQueue
    from sfmtool.sift.extract_sfmtool import (
        extract_sift_with_sfmtool,
        get_default_sfmtool_feature_options,
    )
    from sfmtool.sift.file import _validate_sift_write

    result = next(
        iter(
            extract_sift_with_sfmtool(
                [isolated_seoul_bull_image], get_default_sfmtool_feature_options()
            )
        )
    )
    data = _validate_sift_write(*result)

    # Good path: the save lands on the pool, join() waits, file is readable.
    queue = SiftWriteQueue()
    good = tmp_path / "ok.sift"
    queue.submit(str(good), data, 5)
    assert queue.pending_count == 1
    queue.join()
    assert queue.pending_count == 0
    with SiftReader(good) as reader:
        assert reader.metadata["image_name"] == isolated_seoul_bull_image.name

    # A save into a nonexistent directory fails; join() reports it in order.
    bad_queue = SiftWriteQueue()
    bad_queue.submit(str(tmp_path / "missing_dir" / "bad.sift"), data, 5)
    with pytest.raises(OSError):
        bad_queue.join()


def test_sift_cli_with_empty_directory(tmp_path):
    """Tests the sift command with an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = CliRunner().invoke(
        main, ["sift", "--extract", "--tool", "colmap", str(empty_dir)]
    )
    assert result.exit_code != 0
    assert "There were no image files to process" in result.output


def test_sift_cli_with_range(isolated_seoul_bull_17_images: list[Path]):
    """Tests 'sift --extract' with the --range option."""
    image_dir = isolated_seoul_bull_17_images[0].parent
    range_expr = "5-7,10"

    result = CliRunner().invoke(
        main,
        [
            "sift",
            "--extract",
            "--tool",
            "colmap",
            "--range",
            range_expr,
            str(image_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "expanded to 4 filename(s)" in result.output
    assert "New SIFT feature extraction: 4 / 4 image(s)" in result.output

    # Check that the correct .sift files were created
    for i in [5, 6, 7, 10]:
        assert (
            image_dir
            / "features"
            / f"sift-colmap-{EXPECTED_FEATURE_TOOL_HASH}"
            / f"seoul_bull_sculpture_{i:02d}.jpg.sift"
        ).exists()
