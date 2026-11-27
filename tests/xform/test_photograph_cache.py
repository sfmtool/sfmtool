# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The photometric ``sfm xform`` steps read their photographs through one
photograph cache per run (``ImagePyramidSet.read``), as the viewer reads
them. See ``specs/cli/reconstruction/xform/xform-command.md``."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.reconstruction import SfmrReconstruction

from ..conftest import SEOUL_BULL_POINT_STEP, every_nth_point

# Every step that reads photographs, each with work to do: the conversion
# reads none and stores no bitmaps, which the second step renders, and the
# localizer drops the bitmaps the last step renders again.
_STEPS = [
    "--to-embedded-patches",
    "--add-patch-bitmaps",
    "resolution=16",
    "--refine-normals",
    "--localize-keypoints",
    "resolution=16",
    "--refine-keypoints",
    "resolution=16",
]


def _subset(seoul_bull_workspace: Path) -> Path:
    """Every eighth point of the workspace's reconstruction, saved beside it."""
    path = seoul_bull_workspace.with_name("subset.sfmr")
    every_nth_point(
        SfmrReconstruction.load(seoul_bull_workspace), SEOUL_BULL_POINT_STEP
    ).save(str(path), operation="test subset")
    return path


def _entries(path: Path) -> dict[str, bytes]:
    """Every entry of a ``.sfmr`` but its write record, whose timestamp
    differs between two saves of one value."""
    with zipfile.ZipFile(path) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if name != "written.json.zst"
        }


def test_xform_through_the_photograph_cache_writes_the_file_the_arrays_gave(
    seoul_bull_workspace: Path, monkeypatch, tmp_path: Path
):
    """In core, the steps reading their photographs through
    `ImagePyramidSet.read` write the same file, byte for byte, as the
    `read_image_rgb` arrays they read before; and the run's steps share one
    cache, which decodes each photograph once."""
    import sfmtool.patches as patches_module
    from sfmtool.fileio import read_image_rgb

    src = _subset(seoul_bull_workspace)
    runner = CliRunner()
    through_cache = runner.invoke(
        main, ["xform", str(src), str(tmp_path / "cache.sfmr"), *_STEPS]
    )
    assert through_cache.exit_code == 0, through_cache.output
    # Four steps read the 17 photographs, through one cache.
    assert "Photographs: 17 whole decodes, 0 mipmaps made" in through_cache.output
    assert "The cache decoded" not in through_cache.output

    class _Arrays:
        """The photographs as the steps read them before: whole arrays."""

        @staticmethod
        def read(recon, cache=None, progress=None):
            workspace_dir = Path(recon.workspace_dir)
            return [read_image_rgb(workspace_dir / name) for name in recon.image_names]

    monkeypatch.setattr(patches_module, "ImagePyramidSet", _Arrays)
    from_arrays = runner.invoke(
        main, ["xform", str(src), str(tmp_path / "arrays.sfmr"), *_STEPS]
    )
    assert from_arrays.exit_code == 0, from_arrays.output
    # The arrays reached the kernels: the run's cache decoded nothing.
    assert "Photographs: 0 whole decodes" in from_arrays.output

    assert _entries(tmp_path / "cache.sfmr") == _entries(tmp_path / "arrays.sfmr")


def test_xform_reads_out_of_core_photographs_from_their_mipmaps(
    seoul_bull_workspace: Path,
):
    """With `mipmap_options` in the workspace, the run makes each photograph's
    mipmap once, for its first step that reads photographs, and every step
    reads it."""
    workspace_dir = seoul_bull_workspace.parent
    marker = workspace_dir / ".sfm-workspace.json"
    config = json.loads(marker.read_text())
    # Every 270 x 480 photograph has a side over 64 px.
    config["mipmap_options"] = {
        "out_of_core_long_side": 64,
        "jpeg_quality": 95,
        "full_resolution_jpeg_quality": 95,
    }
    marker.write_text(json.dumps(config))
    src = _subset(seoul_bull_workspace)
    out = workspace_dir / "out-of-core.sfmr"

    result = CliRunner().invoke(main, ["xform", str(src), str(out), *_STEPS])

    assert result.exit_code == 0, result.output
    assert "Photographs: 0 whole decodes, 17 mipmaps made" in result.output
    assert "The cache decoded" not in result.output
    assert SfmrReconstruction.load(out).patch_bitmaps is not None
