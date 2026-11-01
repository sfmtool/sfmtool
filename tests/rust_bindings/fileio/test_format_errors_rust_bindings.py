# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The exception each format error raises, for each file format.

The `.sfmr`, `.sift`, `.matches` and `.camrig` bindings map their errors
through one table (`crates/sfmtool-py/src/format_errors.rs`): an I/O failure is
an `OSError` of the subclass for its kind, a file that cannot be parsed is an
`OSError`, data a writer was handed that breaks a rule of the format is a
`ValueError`, and a `.sfmr` that holds something this build cannot apply is a
`ValueError`. These tests trigger each variant through each format's readers
(`SiftFile` and `MatchesFile` among them), verifier and writers (`write_sift`,
`SiftWriteQueue`, `write_matches`, `write_clusters_matches`, `write_sfmr`,
`write_camrig`), through `SfmrReconstruction`'s `load`, `from_data` and
`save`, through `SphericalTileRig`'s `.camrig` reader and writer, and through
the `.sift` reads of `bench.NearbyTrackSources`. The `.kdf` cases are in
`spatial/test_kdf_rust_bindings.py`.
"""

import json
import shutil
import zipfile
from compression import zstd

import numpy as np
import pytest

from sfmtool.fileio import (
    read_camrig,
    read_camrig_metadata,
    read_matches,
    read_matches_metadata,
    read_sfmr,
    read_sfmr_content_hash,
    read_sfmr_metadata,
    read_sift,
    read_sift_metadata,
    read_sift_partial,
    verify_camrig,
    verify_matches,
    verify_sfmr,
    verify_sift,
    write_camrig,
    write_clusters_matches,
    write_matches,
    write_sfmr,
    write_sift,
    MatchesFile,
    SiftFile,
    SiftWriteQueue,
)
from sfmtool import bench
from sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction
from sfmtool.spherical import SphericalTileRig

from tests.rust_bindings.fileio.test_matches_file_rust_bindings import _pairwise_dict
from tests.rust_bindings.fileio.test_write_clusters_matches_rust_bindings import (
    _clusters_kwargs,
)
from tests.conftest import SEOUL_BULL_DIR, SEOUL_BULL_GROUND_TRUTH

_SFMR = SEOUL_BULL_GROUND_TRUTH

_CAMERA = {
    "model": "PINHOLE",
    "width": 10,
    "height": 10,
    "parameters": {
        "focal_length_x": 5.0,
        "focal_length_y": 5.0,
        "principal_point_x": 5.0,
        "principal_point_y": 5.0,
    },
}


def _sift_dict(n=10):
    rng = np.random.default_rng(0)
    return {
        "feature_tool_metadata": {
            "feature_tool": "pytest",
            "feature_type": "sift",
            "feature_options": {},
        },
        "metadata": {
            "version": 1,
            "image_name": "t.jpg",
            "image_file_xxh128": "a" * 32,
            "image_file_size": 1,
            "image_width": 100,
            "image_height": 100,
            "feature_count": n,
        },
        "positions_xy": rng.random((n, 2), dtype=np.float32),
        "affine_shapes": rng.random((n, 2, 2), dtype=np.float32),
        "descriptors": rng.integers(0, 255, (n, 128), dtype=np.uint8),
        "thumbnail_y_x_rgb": rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
    }


def _write_camrig(path, cameras=(_CAMERA,)):
    write_camrig(
        str(path),
        "rig",
        "generic",
        list(cameras),
        [],
        [0],
        np.array([[1.0, 0.0, 0.0, 0.0]]),
        np.zeros((1, 3)),
    )


def _make_file(fmt, tmp_path):
    """A valid file of format `fmt` in `tmp_path`."""
    path = tmp_path / f"valid.{fmt}"
    if fmt == "sfmr":
        shutil.copy(_SFMR, path)
    elif fmt == "sift":
        write_sift(str(path), _sift_dict(), 3)
    elif fmt == "matches":
        write_matches(str(path), _pairwise_dict())
    else:
        _write_camrig(path)
    return path


def _rewrite_metadata(path, out, edit):
    """Copy `path` to `out` with `metadata.json.zst` replaced by `edit(bytes)`."""
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            if info.filename == "metadata.json.zst":
                data = zstd.compress(edit(zstd.decompress(data)))
            dst.writestr(info, data)
    return out


def _bump_json(field, value):
    def edit(raw):
        metadata = json.loads(raw)
        metadata[field] = value
        return json.dumps(metadata).encode()

    return edit


# Each format's readers and its verifier. A verifier reports a damaged file in
# its result where it can, so it is only checked for the failures it raises on.
_READERS = {
    "sfmr": [read_sfmr, read_sfmr_metadata, read_sfmr_content_hash],
    "sift": [
        read_sift,
        read_sift_metadata,
        lambda p: read_sift_partial(p, 1),
        SiftFile,
    ],
    "matches": [read_matches, read_matches_metadata, MatchesFile],
    "camrig": [read_camrig, read_camrig_metadata],
}
_VERIFIERS = {
    "sfmr": verify_sfmr,
    "sift": verify_sift,
    "matches": verify_matches,
    "camrig": verify_camrig,
}
# A metadata edit the full reader refuses as `InvalidFormat` or `ShapeMismatch`.
_INVALID_METADATA = {
    "sfmr": _bump_json("version", 999),
    "sift": _bump_json("version", 999),
    "matches": _bump_json("version", 999),
    "camrig": _bump_json("camera_count", 3),
}
_FORMATS = sorted(_READERS)
# What creating a file under a path whose parent is a file raises: the
# operating system names it NotADirectoryError on Linux and macOS and
# FileNotFoundError ("path not found") on Windows. Either is the OSError
# subclass for the kind, which a plain OSError is not.
_NO_DIRECTORY = (NotADirectoryError, FileNotFoundError)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_a_missing_file_is_a_file_not_found_error(fmt, tmp_path):
    missing = str(tmp_path / f"missing.{fmt}")
    for call in [*_READERS[fmt], _VERIFIERS[fmt]]:
        with pytest.raises(FileNotFoundError, match="missing"):
            call(missing)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_a_file_that_is_not_a_zip_is_an_os_error(fmt, tmp_path):
    junk = tmp_path / f"junk.{fmt}"
    junk.write_bytes(b"definitely not a zip archive")
    for call in [*_READERS[fmt], _VERIFIERS[fmt]]:
        with pytest.raises(OSError, match="ZIP error") as info:
            call(str(junk))
        assert type(info.value) is OSError


@pytest.mark.parametrize("fmt", _FORMATS)
def test_metadata_that_is_not_json_is_an_os_error(fmt, tmp_path):
    bad = _rewrite_metadata(
        _make_file(fmt, tmp_path), tmp_path / f"bad.{fmt}", lambda _: b"{not json"
    )
    for call in _READERS[fmt]:
        if call is read_sfmr_content_hash:
            continue  # reads only content_hash.json.zst
        with pytest.raises(OSError, match="JSON error") as info:
            call(str(bad))
        assert type(info.value) is OSError


@pytest.mark.parametrize("fmt", _FORMATS)
def test_a_file_that_breaks_a_format_rule_is_an_os_error(fmt, tmp_path):
    bad = _rewrite_metadata(
        _make_file(fmt, tmp_path), tmp_path / f"bad.{fmt}", _INVALID_METADATA[fmt]
    )
    full_reader = _READERS[fmt][0]
    with pytest.raises(OSError, match="Invalid format|Shape mismatch") as info:
        full_reader(str(bad))
    assert type(info.value) is OSError


def test_data_a_writer_cannot_store_is_a_value_error(tmp_path):
    # Shape mismatch: three descriptors for ten features.
    sift = _sift_dict()
    sift["descriptors"] = sift["descriptors"][:3]
    with pytest.raises(ValueError, match="Shape mismatch"):
        write_sift(str(tmp_path / "bad.sift"), sift, 3)

    # The batch writer checks the data in submit and raises the same
    # exception there, before anything is spawned.
    queue = SiftWriteQueue()
    with pytest.raises(ValueError, match="Shape mismatch"):
        queue.submit(str(tmp_path / "queued.sift"), sift)
    assert queue.pending_count == 0

    # Invalid metadata: `feature_options` that is not a JSON object, and a
    # metadata key the format does not define.
    sift = _sift_dict()
    sift["feature_tool_metadata"]["feature_options"] = []
    with pytest.raises(ValueError, match="Invalid metadata"):
        write_sift(str(tmp_path / "bad.sift"), sift, 3)
    sift = _sift_dict()
    sift["metadata"]["extra_field"] = 1
    with pytest.raises(ValueError, match="Invalid metadata"):
        SiftWriteQueue().submit(str(tmp_path / "bad.sift"), sift)

    # Invalid format: a `.sift` version this build does not write.
    sift = _sift_dict()
    sift["metadata"]["version"] = 999
    with pytest.raises(ValueError, match="Invalid format"):
        write_sift(str(tmp_path / "bad.sift"), sift, 3)
    assert not (tmp_path / "bad.sift").exists()

    # Invalid format: a rig with no cameras.
    with pytest.raises(ValueError, match="Invalid format"):
        _write_camrig(tmp_path / "bad.camrig", cameras=())

    # Shape mismatch: a match count the match table does not hold.
    matches = _pairwise_dict()
    matches["metadata"]["match_count"] = 7
    with pytest.raises(ValueError, match="Shape mismatch"):
        write_matches(str(tmp_path / "bad.matches"), matches)

    # Invalid format: a cluster member past its image's feature count.
    clusters = _clusters_kwargs()
    clusters["member_features"][1] = 150
    with pytest.raises(ValueError, match="Invalid format"):
        write_clusters_matches(tmp_path / "bad_clusters.matches", **clusters)

    # Shape mismatch: a points table shorter than its metadata says.
    sfmr = read_sfmr(str(_SFMR))
    sfmr["metadata"]["point_count"] += 1
    with pytest.raises(ValueError, match="Shape mismatch"):
        write_sfmr(str(tmp_path / "bad.sfmr"), sfmr, skip_recompute_depth_stats=True)


def test_a_write_the_filesystem_refuses_is_an_os_error(tmp_path):
    # The parent of each output path is a file, so creating the output, or
    # the parent directory for writers that create it, fails.
    parent = tmp_path / "a_file"
    parent.write_bytes(b"")
    with pytest.raises(_NO_DIRECTORY):
        write_sift(str(parent / "out.sift"), _sift_dict(), 3)
    with pytest.raises(_NO_DIRECTORY):
        _write_camrig(parent / "out.camrig")
    with pytest.raises(FileExistsError):
        write_matches(str(parent / "out.matches"), _pairwise_dict())
    with pytest.raises(FileExistsError):
        write_clusters_matches(parent / "out_clusters.matches", **_clusters_kwargs())
    with pytest.raises(FileExistsError):
        write_sfmr(
            str(parent / "out.sfmr"),
            read_sfmr(str(_SFMR)),
            skip_recompute_depth_stats=True,
        )

    # The batch writer raises the failed save's error from the wait for it.
    queue = SiftWriteQueue()
    queue.submit(str(parent / "queued.sift"), _sift_dict())
    with pytest.raises(_NO_DIRECTORY):
        queue.join()


def test_a_sift_file_whose_columns_cannot_be_read_is_an_os_error(tmp_path):
    # The metadata reads, so the file opens, but its feature count names
    # column entries the archive does not hold, so the first column read
    # fails.
    bad = _rewrite_metadata(
        _make_file("sift", tmp_path),
        tmp_path / "bad.sift",
        _bump_json("feature_count", 11),
    )
    sift = SiftFile(str(bad))
    for read in [sift.read_positions, lambda: sift.read_descriptors(count=1)]:
        with pytest.raises(OSError) as info:
            read()
        assert type(info.value) is OSError


def test_a_missing_reconstruction_is_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        SfmrReconstruction.load(str(tmp_path / "missing.sfmr"))


def test_a_reconstruction_this_build_cannot_apply_is_a_value_error(tmp_path):
    # The format allows rolling-shutter rates on any camera model, but a
    # reconstruction applies them on SFMTOOL_PINHOLE alone, so loading a
    # SIMPLE_RADIAL one with rates is `Unsupported`.
    shutil.copy(SEOUL_BULL_DIR / ".sfm-workspace.json", tmp_path)
    data = read_sfmr(str(_SFMR))
    data["per_image_adjustments"] = [["rolling_shutter"] for _ in data["cameras"]]
    data["rolling_shutter_rates_xyz"] = np.full((len(data["image_names"]), 3), 1e-3)
    data["rolling_shutter_scanlines"] = ["ROWS"] * len(data["cameras"])
    path = tmp_path / "rolling.sfmr"
    write_sfmr(str(path), data, skip_recompute_depth_stats=True)

    with pytest.raises(ValueError, match="Unsupported"):
        SfmrReconstruction.load(str(path))
    with pytest.raises(ValueError, match="Unsupported"):
        SfmrReconstruction.from_data(tmp_path, data)


def test_a_save_the_filesystem_refuses_is_an_os_error(tmp_path):
    parent = tmp_path / "a_file"
    parent.write_bytes(b"")
    recon = SfmrReconstruction.load(str(_SFMR))
    with pytest.raises(FileExistsError):
        recon.save(str(parent / "out.sfmr"))


def test_a_tile_rig_camrig_error_goes_through_the_table(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        SphericalTileRig.read_camrig(str(tmp_path / "missing.camrig"))
    junk = tmp_path / "junk.camrig"
    junk.write_bytes(b"definitely not a zip archive")
    with pytest.raises(OSError, match="ZIP error") as info:
        SphericalTileRig.read_camrig(str(junk))
    assert type(info.value) is OSError

    parent = tmp_path / "a_file"
    parent.write_bytes(b"")
    rig = SphericalTileRig(n=20, arc_per_pixel=2 * np.pi / 128, seed=1)
    with pytest.raises(_NO_DIRECTORY):
        rig.write_camrig(str(parent / "out.camrig"))


def test_a_missing_sift_for_the_nearby_sources_is_a_file_not_found_error(tmp_path):
    edited = EditedReconstruction(SfmrReconstruction.load(str(_SFMR)))
    missing = tmp_path / "missing.sift"
    with pytest.raises(FileNotFoundError, match="missing.sift"):
        bench.NearbyTrackSources(edited, sift=[str(missing)] * edited.image_count)
