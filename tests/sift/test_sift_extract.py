# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.sift.file import get_feature_type_for_tool
from sfmtool.fileio import SiftFile
from sfmtool.sift.extract_colmap import get_colmap_feature_options
from sfmtool.workspace import feature_prefix_dir

# The features folder of the default feature tool options (COLMAP without DSP)
_default_options = get_colmap_feature_options()
_default_feature_type = get_feature_type_for_tool("colmap", _default_options)
EXPECTED_FEATURE_PREFIX_DIR = feature_prefix_dir(
    "colmap", _default_feature_type, _default_options
)


def test_default_feature_tool():
    """The COLMAP defaults name the folder every earlier version named for
    them."""
    assert EXPECTED_FEATURE_PREFIX_DIR == (
        "features/sift-colmap-c2405a72d0875a97223cbeccadab3291"
    )


def test_sift_cli_no_paths_defaults_to_cwd(
    isolated_seoul_bull_image: Path, monkeypatch
):
    """'sift --extract' with no paths processes the current directory."""
    expected_sift_path = (
        isolated_seoul_bull_image.parent
        / EXPECTED_FEATURE_PREFIX_DIR
        / (isolated_seoul_bull_image.name + ".sift")
    )

    monkeypatch.chdir(isolated_seoul_bull_image.parent)
    result = CliRunner().invoke(main, ["sift", "--extract", "--tool", "colmap"])
    assert result.exit_code == 0, result.output
    assert "New SIFT feature extraction: 1 / 1 image(s)" in result.output
    assert expected_sift_path.is_file()


@pytest.mark.parametrize("provide_dir", [True, False])
def test_sift_cli_with_file(isolated_seoul_bull_image: Path, provide_dir: bool):
    """Tests 'sift --extract' with a single image file."""
    expected_sift_path = (
        isolated_seoul_bull_image.parent
        / EXPECTED_FEATURE_PREFIX_DIR
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
    with SiftFile(expected_sift_path) as reader:
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
    prefix = feature_prefix_dir("sfmtool", feature_type, options)

    expected_sift_path = (
        isolated_seoul_bull_image.parent
        / prefix
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
    with SiftFile(expected_sift_path) as reader:
        assert reader.metadata["image_name"] == isolated_seoul_bull_image.name
        assert reader.metadata["feature_count"] > 0
        assert reader.feature_tool_metadata["feature_tool"] == "sfmtool"
        positions = reader.read_positions()
        descriptors = reader.read_descriptors()
    assert positions.shape == (reader.metadata["feature_count"], 2)
    assert descriptors.shape == (reader.metadata["feature_count"], 128)


def test_sfmtool_backend_streams_in_order(isolated_seoul_bull_17_images: list[Path]):
    """The sfmtool backend yields one result per image, in input order.

    Exercises the concurrent decode+extract generator in extract_sfmtool.py
    across multiple images (real extract, default concurrency): it must be lazy
    (a generator, not a list) and preserve the input ordering its callers rely on
    for the parallel .sift filename list.
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
    """A failed read+decode on an extract worker surfaces in input order."""
    from sfmtool.sift.extract_sfmtool import (
        extract_sift_with_sfmtool,
        get_default_sfmtool_feature_options,
    )
    from sfmtool.sift.extract import SiftExtractionError

    bad_path = isolated_seoul_bull_image.parent / "not_an_image.jpg"
    bad_path.write_bytes(b"this is not a valid image")

    images = [isolated_seoul_bull_image, bad_path]
    results = extract_sift_with_sfmtool(images, get_default_sfmtool_feature_options())

    # The first (valid) image yields fine; the second raises when reached.
    first = next(results)
    assert first[1]["image_name"] == isolated_seoul_bull_image.name
    with pytest.raises(SiftExtractionError, match="Failed to load image"):
        next(results)


def test_default_concurrent_images_env_override(monkeypatch):
    """SFMTOOL_SIFT_EXTRACT_WORKERS sets the count, raised to 1 if lower, and a
    value that is not a whole number is ignored with a warning."""
    from sfmtool.sift import default_concurrent_images

    monkeypatch.delenv("SFMTOOL_SIFT_EXTRACT_WORKERS", raising=False)
    default = default_concurrent_images(270, 480)

    monkeypatch.setenv("SFMTOOL_SIFT_EXTRACT_WORKERS", "1")
    assert default_concurrent_images(270, 480) == 1  # one at a time
    monkeypatch.setenv("SFMTOOL_SIFT_EXTRACT_WORKERS", "7")
    assert default_concurrent_images(270, 480) == 7
    # The override wins over the memory cap too.
    assert default_concurrent_images(2**32 - 1, 2**32 - 1) == 7
    monkeypatch.setenv("SFMTOOL_SIFT_EXTRACT_WORKERS", "0")
    assert default_concurrent_images(270, 480) == 1  # never zero
    monkeypatch.setenv("SFMTOOL_SIFT_EXTRACT_WORKERS", "-3")
    assert default_concurrent_images(270, 480) == 1
    # Empty counts as unset. (On Windows, setting a variable to "" removes it.)
    monkeypatch.setenv("SFMTOOL_SIFT_EXTRACT_WORKERS", "")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # so a warning about "" fails the test
        assert default_concurrent_images(270, 480) == default

    monkeypatch.setenv("SFMTOOL_SIFT_EXTRACT_WORKERS", "notanint")
    with pytest.warns(UserWarning, match="non-integer SFMTOOL_SIFT_EXTRACT_WORKERS"):
        assert default_concurrent_images(270, 480) == default


def test_default_concurrent_images_scales_with_cores_bounded_by_memory(monkeypatch):
    """Without an override a small image gets one image per core, an image too
    large for half of physical memory gets one, and an image of unknown size
    (a side of 0) gets at most 4. The arithmetic of the memory cap is tested in
    Rust (`concurrency_is_the_core_count_capped_by_memory`)."""
    from sfmtool.sift import default_concurrent_images

    monkeypatch.delenv("SFMTOOL_SIFT_EXTRACT_WORKERS", raising=False)
    cores = os.process_cpu_count() or 1
    # 64 pixels at 192 bytes each fit any machine's memory once per core.
    assert default_concurrent_images(8, 8) == cores
    # About 1.8e19 pixels fit no machine's memory even once.
    assert default_concurrent_images(2**32 - 1, 2**32 - 1) == 1
    assert default_concurrent_images(0, 480) == min(cores, 4)


def test_stream_preserves_input_order_under_out_of_order_completion(monkeypatch):
    """The FIFO yields in input order even when later images finish first.

    Drives ``_stream_sift_with_sfmtool`` directly with lightweight fakes so the
    timing is controlled: every image runs concurrently (workers == count) and
    earlier indices sleep *longer*, so completion order is the reverse of input
    order. A correct FIFO-over-futures must still yield 0, 1, 2, ... in order.
    """
    from sfmtool.sift import extract_sfmtool

    count = 5
    monkeypatch.setenv("SFMTOOL_SIFT_EXTRACT_WORKERS", str(count))
    frame = np.zeros((4, 4, 3), np.uint8)
    monkeypatch.setattr(extract_sfmtool, "_decode_image", lambda p: (p, frame, None))

    def fake_extract_one(image_path, rgb, thumbnail, params, feature_options, rust):
        idx = int(image_path.name)
        time.sleep((count - idx) * 0.02)  # idx 0 finishes last, idx 4 first
        return ("result", idx)

    monkeypatch.setattr(extract_sfmtool, "_extract_one", fake_extract_one)

    paths = [Path(str(i)) for i in range(count)]
    out = list(extract_sfmtool._stream_sift_with_sfmtool(paths, None, {}, object()))
    assert [r[1] for r in out] == list(range(count))


def test_stream_raises_error_in_input_order_not_completion_order(monkeypatch):
    """A failure that completes first is still surfaced only at its input slot.

    Image 2 fails *immediately* while 0 and 1 are slow; with several workers in
    flight its future is the first to finish, but the FIFO must still yield 0 and
    1 before raising, and must never reach image 3 past the failure.
    """
    from sfmtool.sift import extract_sfmtool
    from sfmtool.sift.extract import SiftExtractionError

    count = 4
    monkeypatch.setenv("SFMTOOL_SIFT_EXTRACT_WORKERS", str(count))

    def fake_decode(p):
        idx = int(p.name)
        if idx == 2:
            raise SiftExtractionError("boom on image 2")
        time.sleep(0.05)
        return (p, np.zeros((4, 4, 3), np.uint8), None)

    monkeypatch.setattr(extract_sfmtool, "_decode_image", fake_decode)
    monkeypatch.setattr(
        extract_sfmtool,
        "_extract_one",
        lambda image_path, *a: ("result", int(image_path.name)),
    )

    paths = [Path(str(i)) for i in range(count)]
    gen = extract_sfmtool._stream_sift_with_sfmtool(paths, None, {}, object())

    assert next(gen)[1] == 0
    assert next(gen)[1] == 1
    with pytest.raises(SiftExtractionError, match="boom on image 2"):
        next(gen)
    # The error terminates the generator, so image 3 (past the failure in input
    # order) is never yielded, even though it may have been decoded in flight.
    with pytest.raises(StopIteration):
        next(gen)


def test_sift_write_queue_writes_and_propagates_errors(
    isolated_seoul_bull_image: Path, tmp_path
):
    """SiftWriteQueue (rayon-pool saves) writes correctly and surfaces errors.

    This is the mechanism that overlaps a save with the next image's extract:
    submit() spawns the compression onto the shared rayon pool, join() awaits
    it. A good path produces a readable .sift; a save into a missing directory
    surfaces as an OSError from join(), and from the end of a `with` block;
    data the format refuses is a ValueError from submit() itself.
    """
    from sfmtool.fileio import SiftWriteQueue
    from sfmtool.sift.extract_sfmtool import (
        extract_sift_with_sfmtool,
        get_default_sfmtool_feature_options,
    )
    from sfmtool.sift.extract import _SIFT_DATA_KEYS

    result = next(
        iter(
            extract_sift_with_sfmtool(
                [isolated_seoul_bull_image], get_default_sfmtool_feature_options()
            )
        )
    )
    data = dict(zip(_SIFT_DATA_KEYS, result))

    # Good path: the save lands on the pool, join() waits, file is readable.
    queue = SiftWriteQueue()
    good = tmp_path / "ok.sift"
    queue.submit(good, data)
    assert queue.pending_count == 1
    queue.join()
    assert queue.pending_count == 0
    with SiftFile(good) as reader:
        assert reader.metadata["image_name"] == isolated_seoul_bull_image.name

    # A save into a nonexistent directory fails; join() reports it in order.
    bad_queue = SiftWriteQueue()
    bad_queue.submit(str(tmp_path / "missing_dir" / "bad.sift"), data, 5)
    with pytest.raises(OSError):
        bad_queue.join()

    # Refused data raises at submit, before anything is spawned.
    refused = dict(data, metadata=dict(data["metadata"], extra_field=1))
    with pytest.raises(ValueError, match="unknown field `extra_field`"):
        SiftWriteQueue().submit(tmp_path / "refused.sift", refused)
    assert not (tmp_path / "refused.sift").exists()

    # As a context manager: the lookahead bounds the saves in flight, and a
    # clean exit joins them.
    with SiftWriteQueue(lookahead=1) as writer:
        assert writer.lookahead == 1
        for i in range(3):
            writer.submit(tmp_path / f"ctx_{i}.sift", data)
            assert writer.pending_count == 1
    assert writer.pending_count == 0
    for i in range(3):
        assert SiftFile(tmp_path / f"ctx_{i}.sift").feature_count == len(
            data["descriptors"]
        )

    # A clean exit raises the first save error.
    with pytest.raises(OSError):
        with SiftWriteQueue() as writer:
            writer.submit(tmp_path / "missing_dir" / "bad.sift", data)

    # An exit by exception drains instead, and the exception is the one raised.
    with pytest.raises(RuntimeError, match="extraction failed"):
        with SiftWriteQueue() as writer:
            writer.submit(tmp_path / "missing_dir" / "bad.sift", data)
            raise RuntimeError("extraction failed")
    assert writer.pending_count == 0

    with pytest.raises(ValueError, match="lookahead"):
        SiftWriteQueue(lookahead=0)


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
            / EXPECTED_FEATURE_PREFIX_DIR
            / f"seoul_bull_sculpture_{i:02d}.jpg.sift"
        ).exists()
