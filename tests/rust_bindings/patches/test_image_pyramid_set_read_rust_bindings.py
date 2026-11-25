# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`ImagePyramidSet.read`: a set of photographs read from their files through
a photograph cache, and the cluster refinement reading one in place of its
list of arrays. See ``specs/python-bindings.md`` and
``specs/core/patch/patch-cloud.md``."""

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from sfmtool import ProgressCounter
from sfmtool.fileio import read_image_rgb
from sfmtool.geometry import CameraIntrinsics
from sfmtool.matching import refine_cluster_patches
from sfmtool.patches import CameraViews, ImagePyramidSet
from sfmtool.photographs import PhotographCache
from sfmtool.reconstruction import SfmrReconstruction
from tests.conftest import SEOUL_BULL_GROUND_TRUTH


def _texture(w: int, h: int, phase: float) -> np.ndarray:
    """A colour texture whose small patches pin a position, so a refinement
    from one image to a shifted copy finds the shift."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    v = (
        127.0
        + 45.0 * np.sin(0.11 * x + 0.06 * y + 1.3 + phase)
        + 30.0 * np.sin(0.05 * x - 0.12 * y + 0.7)
        + 20.0 * np.sin(0.17 * x + 0.13 * y + 2.9)
        + 15.0 * np.sin(0.83 * x + 0.47 * y + 0.2)
        + 12.0 * np.sin(-0.52 * x + 0.88 * y + 1.1)
        + 10.0 * np.sin(0.61 * x - 0.71 * y + 2.3)
    )
    grey = np.clip(np.round(v), 0, 255).astype(np.uint8)
    return np.stack([grey, np.roll(grey, 3, axis=1), 255 - grey], axis=-1)


def _write(path: Path, rgb: np.ndarray) -> Path:
    """Write `rgb` losslessly, so `read_image_rgb` gives it back."""
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def _two_photographs(directory: Path) -> tuple[Path, Path]:
    """Two 96 x 96 photographs, the second the first shifted by (2, 1) px."""
    first = _texture(96, 96, 0.0)
    return (
        _write(directory / "a.png", first),
        _write(directory / "b.png", np.roll(first, (1, 2), axis=(0, 1))),
    )


def _cluster():
    """One two-member cluster, a member at (48, 48) in each photograph."""
    positions = [np.array([[48.0, 48.0]], dtype=np.float32)] * 2
    affine = np.array([[[3.0, 0.0], [0.0, 3.0]]], dtype=np.float32)
    return (
        positions,
        [affine, affine],
        np.array([0, 2], dtype=np.uint32),
        np.array([0, 1], dtype=np.uint32),
        np.array([0, 0], dtype=np.uint32),
    )


def test_a_list_of_paths_is_read_in_order_as_the_kernel_reads_the_arrays(tmp_path):
    a, b = _two_photographs(tmp_path)
    counter = ProgressCounter()

    pyramids = ImagePyramidSet.read([a, str(b)], progress=counter)

    assert len(pyramids) == 2
    assert counter.value == 0
    from_set = refine_cluster_patches(pyramids, *_cluster())
    from_arrays = refine_cluster_patches(
        [read_image_rgb(a), read_image_rgb(b)], *_cluster()
    )
    assert from_set["member_status"].tolist() == from_arrays["member_status"].tolist()
    for key in ("member_positions", "member_affine_shapes", "member_zncc"):
        np.testing.assert_array_equal(from_set[key], from_arrays[key], err_msg=key)
    # The other order reads the shift the other way, so the order is the one
    # given.
    swapped = refine_cluster_patches(ImagePyramidSet.read([b, a]), *_cluster())
    assert not np.array_equal(swapped["member_positions"], from_set["member_positions"])


def test_read_refuses_what_names_no_photographs(tmp_path):
    a, _ = _two_photographs(tmp_path)
    camera = CameraIntrinsics(
        "PINHOLE",
        96,
        96,
        {
            "focal_length_x": 100.0,
            "focal_length_y": 100.0,
            "principal_point_x": 48.0,
            "principal_point_y": 48.0,
        },
    )
    views = CameraViews([camera], np.array([[1.0, 0.0, 0.0, 0.0]]), np.zeros((1, 3)))
    with pytest.raises(TypeError, match="a CameraViews names no photographs"):
        ImagePyramidSet.read(views)
    with pytest.raises(TypeError, match="or a list of photograph paths"):
        ImagePyramidSet.read(str(a))
    with pytest.raises(ValueError, match="workspace_dir is for a reconstruction"):
        ImagePyramidSet.read([a], workspace_dir=tmp_path)
    gone = tmp_path / "gone.png"
    with pytest.raises(FileNotFoundError, match="gone.png"):
        ImagePyramidSet.read([a, gone])
    with pytest.raises(TypeError, match="an ImagePyramidSet"):
        refine_cluster_patches("images", *_cluster())


def test_a_reconstruction_is_read_under_its_workspace_and_checked_against_its_cameras(
    tmp_path,
):
    recon = SfmrReconstruction.load(SEOUL_BULL_GROUND_TRUTH)
    names = recon.image_names

    assert len(ImagePyramidSet.read(recon)) == len(names)

    # Another directory: the same names under it, the first photograph not
    # its camera's size.
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(recon.workspace_dir) / name, tmp_path / name)
    _write(tmp_path / names[0], _texture(32, 24, 0.0))
    with pytest.raises(ValueError, match="image 0 is 32x24 but its camera is"):
        ImagePyramidSet.read(recon, workspace_dir=tmp_path)


def test_a_cache_passed_to_two_reads_decodes_and_makes_each_photograph_once(
    tmp_path,
):
    (tmp_path / ".sfm-workspace.json").write_text(
        json.dumps(
            {
                "version": 2,
                "mipmap_options": {
                    "out_of_core_long_side": 64,
                    "jpeg_quality": 95,
                    "full_resolution_jpeg_quality": 95,
                },
            }
        )
    )
    # Out of core, over the threshold of 64 px, and in core, under it.
    big = _write(tmp_path / "big.png", _texture(300, 200, 0.0))
    small = _write(tmp_path / "small.png", _texture(60, 40, 1.0))
    cache = PhotographCache()
    counter = ProgressCounter()

    ImagePyramidSet.read([big, small], cache=cache, progress=counter)
    first = cache.stats()
    again = ProgressCounter()
    ImagePyramidSet.read([big, small], cache=cache, progress=again)
    second = cache.stats()

    assert (counter.value, again.value) == (1, 0)
    assert (first["mipmaps_made"], first["decodes"]) == (1, 1)
    assert (second["mipmaps_made"], second["decodes"]) == (1, 1)
    assert second["misses"] == first["misses"]
    assert second["hits"] == first["hits"] + 2


def test_a_photograph_changed_after_the_read_fails_the_kernel_naming_it(tmp_path):
    a, b = _two_photographs(tmp_path)
    # A budget of one byte keeps no tile, so the kernel's reads decode the
    # files again.
    pyramids = ImagePyramidSet.read([a, b], cache=PhotographCache(budget_bytes=1))
    _write(b, _texture(96, 96, 2.0))
    later = os.stat(b).st_mtime + 10
    os.utime(b, (later, later))

    with pytest.raises(RuntimeError, match=r"b\.png changed while it was being read"):
        refine_cluster_patches(pyramids, *_cluster())
