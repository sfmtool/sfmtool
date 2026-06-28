# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the ``sfm embed-patches`` command and the ``embed_patches``
orchestration (steps 1-7 of the sift_files -> embedded_patches pipeline).

Runs the real photometric kernels on the ``seoul_bull_workspace`` fixture (which
carries the ``.sift`` files + source images) at a low patch resolution to keep the
end-to-end cost reasonable. See ``specs/cli/embed-patches-command.md`` and
``specs/core/sift-to-patch-reconstruction.md``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch as mock_patch

import numpy as np
from click.testing import CliRunner

import sfmtool._embed_patches as ep
from sfmtool._embed_patches import (
    image_file_hashes_from_images,
    image_file_hashes_from_sift,
)
from sfmtool._sfmtool import SfmrReconstruction
from sfmtool._sfmtool.io import verify_sfmr
from sfmtool.cli import main


def test_embed_patches_cli_round_trips(monkeypatch, seoul_bull_workspace, tmp_path):
    """The CLI converts sift_files -> embedded_patches and the output loads + verifies
    with no .sift companion."""
    real = ep.embed_patches
    monkeypatch.setattr(
        ep,
        "embed_patches",
        lambda recon, images, **kw: real(recon, images, **{**kw, "resolution": 12}),
    )
    out = tmp_path / "out.sfmr"
    args = ["embed-patches", str(seoul_bull_workspace), str(out)]
    with mock_patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert out.exists()

    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"

    reloaded = SfmrReconstruction.load(str(out))
    assert reloaded.feature_source == "embedded_patches"
    assert reloaded.point_count > 0
    assert reloaded.keypoints_xy is not None
    assert reloaded.patches is not None
    # image hashes match the source .sift metadata (one per image).
    assert len(reloaded.image_file_hashes) == reloaded.image_count


def test_embed_patches_default_output_path(monkeypatch, seoul_bull_workspace):
    """Omitting OUTPUT writes <stem>-embedded.sfmr next to the input."""
    real = ep.embed_patches
    monkeypatch.setattr(
        ep,
        "embed_patches",
        lambda recon, images, **kw: real(recon, images, **{**kw, "resolution": 12}),
    )
    src = Path(seoul_bull_workspace)
    expected = src.with_name(f"{src.stem}-embedded.sfmr")
    try:
        args = ["embed-patches", str(src)]
        with mock_patch("sys.argv", ["sfm"] + args):
            result = CliRunner().invoke(main, args)
        assert result.exit_code == 0, result.output
        assert expected.exists()
        assert (
            SfmrReconstruction.load(str(expected)).feature_source == "embedded_patches"
        )
    finally:
        expected.unlink(missing_ok=True)


def test_embed_patches_rejects_already_embedded(seoul_bull_workspace, tmp_path):
    """Running on an already-embedded reconstruction errors (nothing to convert)."""
    # Build a cheap embedded_patches input via the non-photometric baseline.
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    emb = recon.to_embedded_patches()
    embedded_path = tmp_path / "embedded.sfmr"
    emb.save(str(embedded_path), operation="xform")

    args = ["embed-patches", str(embedded_path), str(tmp_path / "out.sfmr")]
    with mock_patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "already embedded_patches" in result.output


def test_image_file_hashes_from_sift_matches_metadata(seoul_bull_workspace):
    """image_file_hashes_from_sift reads the .sift image_file_xxh128 (16 bytes),
    matching the value re-hashing the image bytes would produce."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    from_sift = image_file_hashes_from_sift(recon)
    assert len(from_sift) == recon.image_count
    assert all(isinstance(h, bytes) and len(h) == 16 for h in from_sift)
    # Same identity hash as re-hashing the image bytes (the .sift records exactly
    # that digest).
    assert from_sift == image_file_hashes_from_images(recon)


def test_embed_patches_handles_points_at_infinity(seoul_bull_workspace):
    """The full orchestration runs on an infinity-bearing input (feature_size
    sizing doesn't choke on w=0 points) and produces a valid embedded_patches
    reconstruction; any surviving infinity point stays at infinity."""
    import cv2

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    # Turn one well-observed point into a point at infinity.
    pos = np.asarray(recon.positions_xyzw, dtype=np.float64)
    counts = np.bincount(np.asarray(recon.track_point_ids), minlength=recon.point_count)
    pi = int(np.argmax(counts))
    xyz = pos[pi, :3]
    pos[pi] = np.append(xyz / np.linalg.norm(xyz), 0.0)
    recon = recon.clone_with_changes(positions=pos)
    assert bool(np.asarray(recon.point_is_at_infinity)[pi])

    ws = recon.workspace_dir
    images = [
        np.ascontiguousarray(cv2.imread(f"{ws}/{name}", cv2.IMREAD_COLOR))
        for name in recon.image_names
    ]

    out = ep.embed_patches(recon, images, resolution=12)
    assert out.feature_source == "embedded_patches"
    # The run completed and verifies; any kept infinity point is still w = 0.
    assert int(np.asarray(out.point_is_at_infinity).sum()) <= 1


def test_embed_patches_sources_hashes_from_embedded_not_sift(
    monkeypatch, seoul_bull_workspace
):
    """The re-layered pipeline's only ``.sift`` read is the ``to_embedded_patches``
    bridge; it sources image hashes from the embedded recon, not by re-reading the
    ``.sift`` files. Make the sift-hash helper blow up — the run must still succeed.
    """
    import cv2

    def _boom(_recon):
        raise AssertionError("embed_patches should not re-read .sift for hashes")

    monkeypatch.setattr(ep, "image_file_hashes_from_sift", _boom)

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    ws = recon.workspace_dir
    images = [
        np.ascontiguousarray(cv2.imread(f"{ws}/{name}", cv2.IMREAD_COLOR))
        for name in recon.image_names
    ]
    out = ep.embed_patches(recon, images, resolution=12)
    assert out.feature_source == "embedded_patches"
    # The hashes are exactly the bridge's (images aren't culled, only points), so
    # they match what to_embedded_patches set — not a re-hash or a .sift re-read.
    expected = recon.to_embedded_patches(extent_value=5.0).image_file_hashes
    assert [bytes(h) for h in out.image_file_hashes] == [bytes(h) for h in expected]


def test_embed_patches_refine_anchors_on_stored_keypoints(
    monkeypatch, seoul_bull_workspace
):
    """The re-layer's intent: normal refinement runs with use_stored_keypoints=True
    (anchoring on the carried-in SIFT detections), not the reprojected center.
    Spy on PatchCloud.refine_normals to capture the flag the pipeline passes."""
    import cv2

    from sfmtool._sfmtool import PatchCloud

    captured: dict = {}
    orig = PatchCloud.refine_normals

    def spy(self, recon, images, **kwargs):
        captured["use_stored_keypoints"] = kwargs.get("use_stored_keypoints")
        return orig(self, recon, images, **kwargs)

    monkeypatch.setattr(PatchCloud, "refine_normals", spy)

    recon = SfmrReconstruction.load(seoul_bull_workspace)
    ws = recon.workspace_dir
    images = [
        np.ascontiguousarray(cv2.imread(f"{ws}/{name}", cv2.IMREAD_COLOR))
        for name in recon.image_names
    ]
    ep.embed_patches(recon, images, resolution=12)
    assert captured.get("use_stored_keypoints") is True


def test_embed_patches_cli_subpixel_and_search_resolution_multiplier(
    monkeypatch, seoul_bull_workspace, tmp_path
):
    """End-to-end CLI plumbing for the two new opt-in knobs: the values
    parsed from the command line reach `embed_patches`'s kwargs. Spying on
    the function rather than running the full pipeline keeps this test cheap
    while covering the Click choice validation + kwarg threading."""
    captured: dict = {}
    real = ep.embed_patches

    def spy(recon, images, **kwargs):
        captured["subpixel"] = kwargs.get("subpixel")
        captured["search_resolution_multiplier"] = kwargs.get(
            "search_resolution_multiplier"
        )
        return real(recon, images, **{**kwargs, "resolution": 12})

    monkeypatch.setattr(ep, "embed_patches", spy)

    out = tmp_path / "out.sfmr"
    args = [
        "embed-patches",
        str(seoul_bull_workspace),
        str(out),
        "--subpixel",
        "lk",
        "--search-resolution-multiplier",
        "2.0",
    ]
    with mock_patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert captured["subpixel"] == "lk"
    assert captured["search_resolution_multiplier"] == 2.0


def test_embed_patches_cli_rejects_unknown_subpixel(seoul_bull_workspace, tmp_path):
    """Click validates `--subpixel` against the known choices; a typo errors
    out before any work happens."""
    args = [
        "embed-patches",
        str(seoul_bull_workspace),
        str(tmp_path / "out.sfmr"),
        "--subpixel",
        "lq",  # typo for "lk"
    ]
    with mock_patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "subpixel" in result.output.lower()
