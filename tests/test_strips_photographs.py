# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The strip montages of ``sfm compare --strips`` and ``sfm inspect --strips``
read their photographs through one photograph cache for the run. See
``specs/cli/reconstruction/compare-command.md`` and
``specs/cli/reconstruction/inspect-command.md``."""

import json
from pathlib import Path

import numpy as np
from click.testing import CliRunner

import sfmtool.strips._inspect as inspect_module
import sfmtool.strips._solve as solve_module
from sfmtool.cli import main
from sfmtool.fileio import read_image_rgb
from sfmtool.geometry import CameraIntrinsics
from sfmtool.patches import CameraViews, ImagePyramidSet
from sfmtool.reconstruction import RangeExpr, SfmrReconstruction
from sfmtool.xform import IncludeRangeFilter, apply_transforms


class _Arrays:
    """The photographs as the strips read them before: each decoded whole by
    ``read_image_rgb``, here handed over as a set built from the arrays."""

    @staticmethod
    def read(recon_or_paths, workspace_dir=None, cache=None, progress=None):
        if isinstance(recon_or_paths, list):
            arrays = [read_image_rgb(path) for path in recon_or_paths]
            height, width = arrays[0].shape[:2]
            camera = CameraIntrinsics(
                "PINHOLE",
                width,
                height,
                {
                    "focal_length_x": float(width),
                    "focal_length_y": float(width),
                    "principal_point_x": width / 2,
                    "principal_point_y": height / 2,
                },
            )
            n = len(arrays)
            views = CameraViews(
                [camera], np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)), np.zeros((n, 3))
            )
            return ImagePyramidSet(views, arrays)
        recon = recon_or_paths
        workspace = Path(workspace_dir or recon.workspace_dir)
        return ImagePyramidSet(
            recon, [read_image_rgb(workspace / name) for name in recon.image_names]
        )


def _from_arrays(monkeypatch):
    monkeypatch.setattr(solve_module, "ImagePyramidSet", _Arrays)
    monkeypatch.setattr(inspect_module, "ImagePyramidSet", _Arrays)


def _invoke(args):
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    return result.output


def test_compare_strips_read_through_one_cache_as_the_arrays_render(
    seoul_bull_workspace, tmp_path, monkeypatch
):
    """Both reconstructions read through the run's one cache, each photograph
    decoded once, and in core the montage is the one the arrays render."""
    workspace = seoul_bull_workspace.parent
    subset = apply_transforms(
        SfmrReconstruction.load(seoul_bull_workspace),
        [IncludeRangeFilter(RangeExpr("5-12"))],
    )
    subset_path = workspace / "subset.sfmr"
    subset.save(str(subset_path), operation="test")
    args = ["compare", str(seoul_bull_workspace), str(subset_path), "--strips"]
    options = ["--strips-num", "4"]
    # The montage's title names its file, so both runs write the same name.
    (tmp_path / "cache").mkdir()
    (tmp_path / "arrays").mkdir()

    caches = []
    real_read = ImagePyramidSet.read

    def spy(recon_or_paths, workspace_dir=None, cache=None, progress=None):
        caches.append(cache)
        return real_read(recon_or_paths, workspace_dir, cache, progress)

    with monkeypatch.context() as m:
        m.setattr(solve_module.ImagePyramidSet, "read", staticmethod(spy))
        output = _invoke([*args, str(tmp_path / "cache" / "strips.png"), *options])
    # The refinement reads every photograph of both reconstructions; the
    # second's are among the first's, so the one cache decodes 17.
    assert "--strips: Photographs: 17 whole decodes, 0 mipmaps made" in output
    assert "The cache decoded" not in output
    assert caches and caches[0] is not None
    assert all(cache is caches[0] for cache in caches)

    _from_arrays(monkeypatch)
    _invoke([*args, str(tmp_path / "arrays" / "strips.png"), *options])
    assert (tmp_path / "cache" / "strips.png").read_bytes() == (
        tmp_path / "arrays" / "strips.png"
    ).read_bytes()


def test_inspect_strips_read_the_photographs_as_the_arrays_render(
    seoul_bull_workspace, tmp_path, monkeypatch
):
    """A ``sift_files`` reconstruction reads every photograph, for the
    refinement; an ``embedded_patches`` one reads only the photographs its
    listed points are seen in. In core both montages are the ones the arrays
    render."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    embedded = recon.to_embedded_patches(normal="mean_viewing", extent_value=5.0)
    embedded_path = seoul_bull_workspace.parent / "embedded.sfmr"
    embedded.save(str(embedded_path), operation="test")
    listed = [0, 1, 2]
    track_points = np.asarray(embedded.track_point_indexes)
    seen_in = set(
        np.asarray(embedded.track_image_indexes)[np.isin(track_points, listed)]
    )
    assert len(seen_in) < embedded.image_count
    runs = [(seoul_bull_workspace, "sift"), (embedded_path, "embedded")]
    # The montage's title names its file, so both runs write the same name.
    (tmp_path / "cache").mkdir()
    (tmp_path / "arrays").mkdir()

    for path, name in runs:
        out = tmp_path / "cache" / f"{name}.png"
        output = _invoke(["inspect", "--strips", str(path), "0-2", "-o", str(out)])
        decodes = recon.image_count if name == "sift" else len(seen_in)
        assert f"--strips: Photographs: {decodes} whole decodes, 0 mipmaps" in output

    _from_arrays(monkeypatch)
    for path, name in runs:
        out = tmp_path / "arrays" / f"{name}.png"
        _invoke(["inspect", "--strips", str(path), "0-2", "-o", str(out)])
        assert (tmp_path / "cache" / f"{name}.png").read_bytes() == out.read_bytes()


def test_inspect_strips_read_out_of_core_photographs_from_their_mipmaps(
    seoul_bull_workspace, tmp_path
):
    """With ``mipmap_options`` in the workspace, the strips make each
    photograph's mipmap and read it, decoding none whole."""
    marker = seoul_bull_workspace.parent / ".sfm-workspace.json"
    config = json.loads(marker.read_text())
    # Every 270 x 480 photograph has a side over 64 px.
    config["mipmap_options"] = {
        "out_of_core_long_side": 64,
        "jpeg_quality": 95,
        "full_resolution_jpeg_quality": 95,
    }
    marker.write_text(json.dumps(config))
    out = tmp_path / "strips.png"

    output = _invoke(
        ["inspect", "--strips", str(seoul_bull_workspace), "0", "-o", str(out)]
    )

    assert "--strips: Photographs: 0 whole decodes, 17 mipmaps made" in output
    assert out.is_file()


def test_compare_strips_recommend_mipmaps_in_each_reconstructions_workspace(
    seoul_bull_workspace, tmp_path, monkeypatch
):
    """Two reconstructions in two workspaces, with a cache too small for their
    photographs: the recommendation names both workspaces' files."""
    import shutil

    other_workspace = tmp_path / "other"
    shutil.copytree(seoul_bull_workspace.parent, other_workspace)
    other = other_workspace / seoul_bull_workspace.name
    # 4 MB holds about half of the tiles of the 17 270 x 480 photographs.
    monkeypatch.setenv("SFM_EXPLORER_PHOTOGRAPH_CACHE_MB", "4")

    output = _invoke(
        [
            "compare",
            str(seoul_bull_workspace),
            str(other),
            "--strips",
            str(tmp_path / "strips.png"),
            "--strips-num",
            "1",
        ]
    )

    assert "--strips: The cache decoded 34 photographs whole" in output
    first = seoul_bull_workspace.parent / ".sfm-workspace.json"
    second = other_workspace / ".sfm-workspace.json"
    assert f"object to {first} and {second}, such as" in output
