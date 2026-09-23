# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``xform`` steps that drop and add the heavy optional columns.

``--drop-thumbnails``, ``--drop-patch-bitmaps``, ``--add-thumbnails``,
``--add-patch-bitmaps`` and the ``--minimal`` shorthand, end to end on the
17-image seoul_bull reconstruction. The properties under test: the drops keep
every row of everything; ``--add-thumbnails`` rebuilds the column byte for byte
from the ``.sift`` copies first and the photographs second; ``--add-patch-bitmaps`` moves nothing and renders what the
zero-step fuse renders; ``--minimal`` writes the smallest file with metadata that
names no machine and no history. See
``specs/cli/reconstruction/xform/xform-command.md``.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._sfmtool.io import read_sfmr, read_sfmr_metadata, verify_sfmr, write_sfmr
from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool.cli import main
from sfmtool.sift.file import get_sift_path_from_recon
from sfmtool.xform import (
    AddPatchBitmapsTransform,
    AddThumbnailsTransform,
    DropPatchBitmapsTransform,
    DropThumbnailsTransform,
    MinimalTransform,
)
from sfmtool.xform._arg_parser import (
    parse_add_patch_bitmaps_params,
    parse_minimal_params,
    parse_transform_args,
)
from sfmtool.xform._images import load_workspace_images


def _run(args: list[str]):
    """Invoke the CLI with ``sys.argv`` patched, as ``xform`` reparses it."""
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    return result


def _xform(input_path: Path, output_path: Path, *steps: str):
    return _run(["xform", str(input_path), str(output_path), *steps])


@pytest.fixture
def embedded_sfmr(seoul_bull_workspace) -> Path:
    """An ``embedded_patches`` reconstruction carrying thumbnails (copied from
    the ``.sift`` files by the solve) and 16x16 patch bitmaps."""
    out = seoul_bull_workspace.with_name("embedded.sfmr")
    _xform(
        seoul_bull_workspace,
        out,
        "--to-embedded-patches",
        "--add-patch-bitmaps",
        "resolution=16",
    )
    return out


# ── Argument grammar ────────────────────────────────────────────────────────


def test_parse_steps_in_command_line_order():
    steps = parse_transform_args(
        [
            "--drop-thumbnails",
            "--drop-patch-bitmaps",
            "--add-thumbnails",
            "--add-patch-bitmaps",
            "--minimal",
        ]
    )
    assert [type(s) for s in steps] == [
        DropThumbnailsTransform,
        DropPatchBitmapsTransform,
        AddThumbnailsTransform,
        AddPatchBitmapsTransform,
        MinimalTransform,
    ]
    # An --add-* before --minimal restores nothing the shorthand dropped.
    assert not steps[2].restores_minimal
    assert not steps[3].restores_minimal


def test_add_after_minimal_is_marked_as_restoring():
    steps = parse_transform_args(
        ["--minimal", "--add-thumbnails", "--add-patch-bitmaps"]
    )
    assert steps[1].restores_minimal
    assert steps[2].restores_minimal


def test_add_patch_bitmaps_params():
    t = parse_add_patch_bitmaps_params("")
    assert (t.resolution, t.sampler) == (24, "bilinear_mip")
    t = parse_add_patch_bitmaps_params("resolution=32,sampler=anisotropic")
    assert (t.resolution, t.sampler) == (32, "anisotropic")
    with pytest.raises(Exception, match="Unknown --add-patch-bitmaps key"):
        parse_add_patch_bitmaps_params("max_gn_steps=3")
    with pytest.raises(ValueError):
        parse_add_patch_bitmaps_params("resolution=1")
    with pytest.raises(ValueError):
        parse_add_patch_bitmaps_params("sampler=nearest")


def test_minimal_params():
    assert parse_minimal_params("").workspace_path is None
    assert parse_minimal_params("wspath=.").workspace_path == "."
    # The path is taken as written, so a nested one arrives whole.
    assert parse_minimal_params("wspath=../ws").workspace_path == "../ws"
    with pytest.raises(Exception, match="Unknown --minimal key"):
        parse_minimal_params("workspace_path=.")
    with pytest.raises(Exception, match="expected key=value"):
        parse_minimal_params("wspath")
    with pytest.raises(Exception, match="empty key"):
        parse_minimal_params("=.")


def test_minimal_takes_no_value_a_bare_one_or_a_joined_one():
    bare = parse_transform_args(["--minimal", "--add-thumbnails"])
    assert isinstance(bare[0], MinimalTransform)
    assert bare[0].workspace_path is None
    assert bare[1].restores_minimal
    assert parse_transform_args(["--minimal", "wspath=."])[0].workspace_path == "."
    assert parse_transform_args(["--minimal=wspath=."])[0].workspace_path == "."


def test_add_patch_bitmaps_takes_a_bare_or_joined_value():
    bare = parse_transform_args(["--add-patch-bitmaps", "--drop-thumbnails"])
    assert bare[0].resolution == 24
    assert isinstance(bare[1], DropThumbnailsTransform)
    joined = parse_transform_args(["--add-patch-bitmaps=resolution=12"])
    assert joined[0].resolution == 12


# ── Thumbnails ──────────────────────────────────────────────────────────────


def test_drop_then_add_thumbnails_reproduces_the_bytes(embedded_sfmr, tmp_path):
    # The input's thumbnails were copied out of the `.sift` files, and adding
    # them back reads those same `.sift` copies first.
    dropped = tmp_path / "dropped.sfmr"
    _xform(embedded_sfmr, dropped, "--drop-thumbnails")
    assert read_sfmr(dropped)["thumbnails_y_x_rgb"] is None
    ok, errors = verify_sfmr(dropped)
    assert ok, errors

    restored = embedded_sfmr.with_name("restored.sfmr")
    _xform(dropped, restored, "--add-thumbnails")
    before = read_sfmr(embedded_sfmr)
    after = read_sfmr(restored)
    assert after["thumbnails_y_x_rgb"] is not None
    np.testing.assert_array_equal(
        after["thumbnails_y_x_rgb"], before["thumbnails_y_x_rgb"]
    )


def test_add_thumbnails_from_the_photographs_reproduces_the_bytes(embedded_sfmr):
    # With no `.sift` to read, every row is the photograph's decode and resize,
    # which is what the extractor wrote into the `.sift` in the first place.
    recon = SfmrReconstruction.load(embedded_sfmr)
    original = np.asarray(recon.thumbnails_y_x_rgb).copy()
    for name in recon.image_names:
        get_sift_path_from_recon(recon, name).unlink()
    rebuilt = AddThumbnailsTransform().apply(DropThumbnailsTransform().apply(recon))
    np.testing.assert_array_equal(np.asarray(rebuilt.thumbnails_y_x_rgb), original)


def test_add_thumbnails_on_sift_files_reproduces_the_bytes(seoul_bull_workspace):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    original = np.asarray(recon.thumbnails_y_x_rgb).copy()
    rebuilt = AddThumbnailsTransform().apply(DropThumbnailsTransform().apply(recon))
    np.testing.assert_array_equal(np.asarray(rebuilt.thumbnails_y_x_rgb), original)


def test_drop_thumbnails_keeps_every_row(embedded_sfmr, tmp_path):
    out = tmp_path / "dropped.sfmr"
    _xform(embedded_sfmr, out, "--drop-thumbnails")
    before, after = read_sfmr(embedded_sfmr), read_sfmr(out)
    for key in (
        "positions_xyzw",
        "keypoints_xy",
        "image_indexes",
        "point_indexes",
        "patch_u_halfvec_xyz",
        "patch_bitmaps_y_x_rgba",
    ):
        np.testing.assert_array_equal(after[key], before[key])
    assert after["image_names"] == before["image_names"]


def test_add_thumbnails_is_a_no_op_when_present(embedded_sfmr, capsys):
    recon = SfmrReconstruction.load(embedded_sfmr)
    out = AddThumbnailsTransform().apply(recon)
    assert "already present" in capsys.readouterr().out
    np.testing.assert_array_equal(
        np.asarray(out.thumbnails_y_x_rgb), np.asarray(recon.thumbnails_y_x_rgb)
    )


def test_a_verified_sift_is_read_before_the_photograph(embedded_sfmr, capsys):
    # Image 3's photograph is replaced by another of the capture, which would
    # fail its hash check if it were read; its `.sift` is read first, so it
    # never is.
    recon = SfmrReconstruction.load(embedded_sfmr)
    original = np.asarray(recon.thumbnails_y_x_rgb).copy()
    workspace = Path(recon.workspace_dir)
    shutil.copy(workspace / recon.image_names[7], workspace / recon.image_names[3])
    rebuilt = AddThumbnailsTransform().apply(DropThumbnailsTransform().apply(recon))
    np.testing.assert_array_equal(np.asarray(rebuilt.thumbnails_y_x_rgb), original)
    assert (
        "17 from verified .sift copies, 0 from photographs" in capsys.readouterr().out
    )


def test_a_missing_photograph_is_read_from_its_verified_sift(embedded_sfmr):
    recon = SfmrReconstruction.load(embedded_sfmr)
    original = np.asarray(recon.thumbnails_y_x_rgb).copy()
    (Path(recon.workspace_dir) / recon.image_names[3]).unlink()
    rebuilt = AddThumbnailsTransform().apply(DropThumbnailsTransform().apply(recon))
    np.testing.assert_array_equal(np.asarray(rebuilt.thumbnails_y_x_rgb), original)


def test_a_sift_that_is_not_the_images_falls_back_to_the_photograph(
    embedded_sfmr, capsys
):
    # Image 2's `.sift` is replaced by image 7's, whose recorded photograph
    # hash is not image 2's: it is passed over, and the photograph is read.
    recon = SfmrReconstruction.load(embedded_sfmr)
    original = np.asarray(recon.thumbnails_y_x_rgb).copy()
    shutil.copy(
        get_sift_path_from_recon(recon, recon.image_names[7]),
        get_sift_path_from_recon(recon, recon.image_names[2]),
    )
    rebuilt = AddThumbnailsTransform().apply(DropThumbnailsTransform().apply(recon))
    np.testing.assert_array_equal(np.asarray(rebuilt.thumbnails_y_x_rgb), original)
    assert (
        "16 from verified .sift copies, 1 from photographs" in capsys.readouterr().out
    )


def test_a_missing_photograph_with_no_sift_fails_naming_it(embedded_sfmr):
    recon = SfmrReconstruction.load(embedded_sfmr)
    name = recon.image_names[5]
    (Path(recon.workspace_dir) / name).unlink()
    get_sift_path_from_recon(recon, name).unlink()
    bare = DropThumbnailsTransform().apply(recon)
    with pytest.raises(ValueError, match=r"1 of 17 images") as error:
        AddThumbnailsTransform().apply(bare)
    assert name in str(error.value)


def test_a_different_photograph_fails_naming_it(embedded_sfmr):
    recon = SfmrReconstruction.load(embedded_sfmr)
    workspace = Path(recon.workspace_dir)
    name = recon.image_names[2]
    # Another photograph of the capture in its place: decodable, and not the one
    # the reconstruction was built from. Its `.sift` is gone, so the photograph
    # is what is read.
    shutil.copy(workspace / recon.image_names[7], workspace / name)
    get_sift_path_from_recon(recon, name).unlink()
    bare = DropThumbnailsTransform().apply(recon)
    with pytest.raises(ValueError, match="image_file_hashes") as error:
        AddThumbnailsTransform().apply(bare)
    assert name in str(error.value)


# ── Patch bitmaps ───────────────────────────────────────────────────────────


def test_drop_patch_bitmaps_keeps_frames_and_normals(embedded_sfmr, tmp_path):
    out = tmp_path / "no_bitmaps.sfmr"
    _xform(embedded_sfmr, out, "--drop-patch-bitmaps")
    before, after = read_sfmr(embedded_sfmr), read_sfmr(out)
    assert after["patch_bitmaps_y_x_rgba"] is None
    for key in ("patch_u_halfvec_xyz", "patch_v_halfvec_xyz", "normals_xyz"):
        np.testing.assert_array_equal(after[key], before[key])
    np.testing.assert_array_equal(
        after["thumbnails_y_x_rgb"], before["thumbnails_y_x_rgb"]
    )


def test_drop_patch_bitmaps_is_a_no_op_without_them(seoul_bull_workspace, capsys):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    out = DropPatchBitmapsTransform().apply(recon)
    assert "No patch bitmaps to drop" in capsys.readouterr().out
    assert out.point_count == recon.point_count


def test_add_patch_bitmaps_moves_nothing_and_matches_the_zero_step_fuse(
    embedded_sfmr,
):
    recon = SfmrReconstruction.load(embedded_sfmr)
    bare = DropPatchBitmapsTransform().apply(recon)
    added = AddPatchBitmapsTransform(resolution=16).apply(bare)

    assert added.patch_bitmap_resolution == 16
    np.testing.assert_array_equal(
        np.asarray(added.keypoints_xy), np.asarray(recon.keypoints_xy)
    )
    np.testing.assert_array_equal(
        np.asarray(added.positions), np.asarray(recon.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(added.track_point_indexes), np.asarray(recon.track_point_indexes)
    )

    # The sub-pixel refiner with no Gauss-Newton step is the fuse, reached here
    # through its own per-patch loop rather than the cloud form the step calls.
    images = load_workspace_images(bare)
    fused = bare.patches.refine_keypoints(
        bare, images, resolution=16, max_gn_steps=0, render_bitmaps=True
    )
    expected = np.zeros_like(np.asarray(added.patch_bitmaps))
    for entry in fused:
        np.testing.assert_array_equal(
            np.asarray(entry["keypoints"], dtype=np.float32).reshape(-1, 2),
            np.asarray(bare.keypoints_xy)[
                np.asarray(bare.track_point_indexes) == entry["point_index"]
            ],
        )
        if entry["bitmap"] is not None:
            expected[int(entry["point_index"])] = entry["bitmap"]
    np.testing.assert_array_equal(np.asarray(added.patch_bitmaps), expected)


def test_add_patch_bitmaps_is_a_no_op_when_present(embedded_sfmr, capsys):
    recon = SfmrReconstruction.load(embedded_sfmr)
    out = AddPatchBitmapsTransform().apply(recon)
    assert "already present" in capsys.readouterr().out
    assert out.patch_bitmap_resolution == 16


def test_add_patch_bitmaps_requires_embedded_patches(seoul_bull_workspace):
    args = [
        "xform",
        str(seoul_bull_workspace),
        str(seoul_bull_workspace.with_name("out.sfmr")),
        "--add-patch-bitmaps",
    ]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "embedded_patches" in result.output


# ── --minimal ───────────────────────────────────────────────────────────────


def _with_lineage(path: Path, out: Path) -> Path:
    data = read_sfmr(path)
    data["metadata"]["lineage"] = [
        {
            "hash": "0123456789abcdef0123456789abcdef",
            "kind": "base",
            "map": {
                "form": "monotone",
                "source_rows": data["metadata"]["point_count"] + 1,
                "deleted": [0],
                "created": [],
            },
        }
    ]
    write_sfmr(out, data)
    return out


def test_minimal_writes_the_whole_reconstruction_and_no_history(embedded_sfmr):
    source = _with_lineage(embedded_sfmr, embedded_sfmr.with_name("lineage.sfmr"))
    assert "lineage" in read_sfmr_metadata(source)
    out_dir = embedded_sfmr.parent / "published"
    out = out_dir / "minimal.sfmr"
    _xform(source, out, "--minimal")

    before, after = read_sfmr(source), read_sfmr(out)
    assert after["thumbnails_y_x_rgb"] is None
    assert after["patch_bitmaps_y_x_rgba"] is None
    meta = after["metadata"]
    assert meta["workspace"]["absolute_path"] == ""
    assert meta["workspace"]["relative_path"] == ".."
    assert "lineage" not in meta
    assert meta["tool_options"] == {
        "transforms": ["Minimal (drop patch bitmaps and thumbnails; minimal metadata)"]
    }
    assert meta["operation"] == "xform"
    assert meta["tool"] == "sfmtool"
    for key in (
        "positions_xyzw",
        "colors_rgb",
        "quaternions_wxyz",
        "translations_xyz",
        "keypoints_xy",
        "image_indexes",
        "point_indexes",
        "patch_u_halfvec_xyz",
        "patch_v_halfvec_xyz",
        "normals_xyz",
    ):
        np.testing.assert_array_equal(after[key], before[key])
    assert after["image_file_hashes"] == before["image_file_hashes"]
    assert after["image_names"] == before["image_names"]
    assert out.stat().st_size < source.stat().st_size / 10
    ok, errors = verify_sfmr(out)
    assert ok, errors
    # The relative path alone finds the workspace.
    assert Path(SfmrReconstruction.load(out).workspace_dir) == Path(
        before["metadata"]["workspace"]["absolute_path"]
    )


def test_minimal_then_add_thumbnails_keeps_them_and_minimal_metadata(
    embedded_sfmr, tmp_path
):
    out = embedded_sfmr.with_name("minimal_thumbs.sfmr")
    result = _xform(embedded_sfmr, out, "--minimal", "--add-thumbnails")
    assert "after --minimal" in result.output
    before, after = read_sfmr(embedded_sfmr), read_sfmr(out)
    np.testing.assert_array_equal(
        after["thumbnails_y_x_rgb"], before["thumbnails_y_x_rgb"]
    )
    assert after["patch_bitmaps_y_x_rgba"] is None
    assert after["metadata"]["workspace"]["absolute_path"] == ""
    assert list(after["metadata"]["tool_options"]) == ["transforms"]


def test_minimal_twice_has_one_content_hash(embedded_sfmr):
    first = embedded_sfmr.with_name("minimal_a.sfmr")
    second = embedded_sfmr.with_name("minimal_b.sfmr")
    _xform(embedded_sfmr, first, "--minimal")
    _xform(embedded_sfmr, second, "--minimal")
    assert (
        SfmrReconstruction.load(first).content_xxh128
        == SfmrReconstruction.load(second).content_xxh128
    )


def test_minimal_states_the_workspace_path(embedded_sfmr):
    # The motivating case: a ground truth written inside its own workspace, to be
    # checked in there, records the workspace it sits in.
    out = embedded_sfmr.parent / "ground_truth.sfmr"
    _xform(embedded_sfmr, out, "--minimal", "wspath=.")

    meta = read_sfmr_metadata(out)
    assert meta["workspace"]["relative_path"] == "."
    assert meta["workspace"]["absolute_path"] == ""
    # The description is the record of the invocation, so it names the statement.
    assert meta["tool_options"]["transforms"] == [
        "Minimal (drop patch bitmaps and thumbnails; minimal metadata; "
        "workspace path '.')"
    ]
    # The stated path alone finds the workspace.
    loaded = Path(SfmrReconstruction.load(out).workspace_dir)
    assert loaded.resolve() == embedded_sfmr.parent.resolve()


def test_minimal_states_the_path_a_measurement_would_not_produce(embedded_sfmr):
    # Written to a staging directory inside the workspace, for a file whose home
    # is the workspace root: measuring would record "..", and the statement wins.
    out = embedded_sfmr.parent / "staging" / "ground_truth.sfmr"
    _xform(embedded_sfmr, out, "--minimal", "wspath=.")
    assert read_sfmr_metadata(out)["workspace"]["relative_path"] == "."


def test_minimal_with_a_joined_workspace_path(embedded_sfmr):
    out = embedded_sfmr.parent / "joined.sfmr"
    _xform(embedded_sfmr, out, "--minimal=wspath=.")
    assert read_sfmr_metadata(out)["workspace"]["relative_path"] == "."


# ── inspect ─────────────────────────────────────────────────────────────────


def test_inspect_reports_the_heavy_columns(embedded_sfmr):
    full = _run(["inspect", str(embedded_sfmr)]).output
    assert "Thumbnails:    yes" in full
    assert "Patch bitmaps: 16x16" in full

    minimal = embedded_sfmr.with_name("minimal_inspect.sfmr")
    _xform(embedded_sfmr, minimal, "--minimal")
    bare = _run(["inspect", str(minimal)]).output
    assert "Thumbnails:    no" in bare
    assert "Patch bitmaps: no" in bare

    verbose = _run(["inspect", "-v", str(minimal)]).output
    assert "Absolute path: (none recorded)" in verbose
    assert "Thumbnails: no" in verbose
