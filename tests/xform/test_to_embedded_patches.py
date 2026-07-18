# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the baseline sift_files → embedded_patches conversion
(``SfmrReconstruction.to_embedded_patches`` and the ``sfm xform
--to-embedded-patches`` op).

The conversion copies each observation's keypoint and each image's identity hash
straight from the ``.sift`` files (no photometric adaptation), so it runs against
the full ``seoul_bull_workspace`` fixture (which carries the ``.sift`` files). See
``specs/core/sift-to-patch-reconstruction.md``.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._sfmtool.reconstruction import SfmrReconstruction
from sfmtool._sfmtool.io import verify_sfmr
from sfmtool.cli import main
from sfmtool.sift.file import SiftReader, get_sift_path_from_recon
from sfmtool.xform import ToEmbeddedPatchesTransform

from .conftest import apply_transforms_to_file


def test_to_embedded_patches_copies_sift_keypoints(
    seoul_bull_workspace: Path, tmp_path: Path
):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    assert recon.feature_source == "sift_files"

    emb = recon.to_embedded_patches()  # mean_viewing normal, feature_size extent

    # Baseline conversion keeps every point and image; only the representation
    # changes.
    assert emb.feature_source == "embedded_patches"
    assert emb.point_count == recon.point_count
    assert emb.image_count == recon.image_count

    n_obs = int(np.asarray(recon.track_image_indexes).shape[0])
    kxy = np.asarray(emb.keypoints_xy)
    assert kxy.shape == (n_obs, 2)
    assert np.all(np.isfinite(kxy))

    # The observation order is preserved, and each keypoint is exactly the source
    # SIFT detection (sift.positions_xy[feature_index] for that image).
    rec_img = np.asarray(recon.track_image_indexes)
    rec_feat = np.asarray(recon.track_feature_indexes)
    np.testing.assert_array_equal(np.asarray(emb.track_image_indexes), rec_img)

    pos_cache: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(n_obs, size=min(250, n_obs), replace=False))
    for j in sample.tolist():
        img, feat = int(rec_img[j]), int(rec_feat[j])
        if img not in pos_cache:
            sr = SiftReader(get_sift_path_from_recon(recon, recon.image_names[img]))
            pos_cache[img] = np.asarray(sr.read_positions(), dtype=np.float32)
        np.testing.assert_allclose(kxy[j], pos_cache[img][feat], atol=1e-4)

    # Each image hash is the .sift's recorded image_file_xxh128 (no re-hashing).
    for i, name in enumerate(recon.image_names):
        sr = SiftReader(get_sift_path_from_recon(recon, name))
        expected = bytes.fromhex(sr.metadata["image_file_xxh128"])
        assert bytes(emb.image_file_hashes[i]) == expected

    # A mean-view patch frame is attached for every point — no refinement.
    assert emb.patches is not None
    assert len(emb.patches) == emb.point_count

    # Round-trips through the v4 format (the writer requires the patch frame).
    out = tmp_path / "embedded.sfmr"
    emb.save(str(out), operation="xform")
    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"
    reloaded = SfmrReconstruction.load(str(out))
    assert reloaded.feature_source == "embedded_patches"
    np.testing.assert_allclose(np.asarray(reloaded.keypoints_xy), kxy, atol=1e-4)


def test_to_embedded_patches_frames_points_at_infinity(
    seoul_bull_workspace: Path, tmp_path: Path
):
    """Points at infinity are kept and get a tangent-sphere frame (normal -d)."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    # Turn one point into a point at infinity: keep its direction, set w = 0.
    pos = np.asarray(recon.positions_xyzw, dtype=np.float64)
    counts = np.bincount(
        np.asarray(recon.track_point_indexes), minlength=recon.point_count
    )
    pi = int(np.argmax(counts))  # a well-observed point (scales available)
    xyz = pos[pi, :3]
    pos[pi] = np.append(xyz / np.linalg.norm(xyz), 0.0)
    recon_inf = recon.clone_with_changes(positions=pos)
    assert bool(np.asarray(recon_inf.point_is_at_infinity)[pi])

    emb = recon_inf.to_embedded_patches()

    # The whole point set is preserved — no point is dropped or left frameless.
    assert emb.point_count == recon_inf.point_count
    assert int(np.asarray(emb.point_is_at_infinity).sum()) == 1
    assert len(emb.patches) == emb.point_count

    # The infinity point's frame is tangent to the unit sphere around its
    # direction d: u, v ⊥ d, the outward normal is normalize(-d), and the patch
    # is flagged at infinity (w == 0) so rendering treats its corners as
    # directions.
    patches = emb.patches
    ids = list(patches.point_indexes)
    patch = patches[ids.index(pi)]
    d = np.asarray(emb.positions_xyzw, dtype=np.float64)[pi, :3]
    d = d / np.linalg.norm(d)
    np.testing.assert_allclose(np.asarray(patch.normal), -d, atol=1e-5)
    assert abs(float(np.dot(np.asarray(patch.u_axis), d))) < 1e-5
    assert abs(float(np.dot(np.asarray(patch.v_axis), d))) < 1e-5
    assert patch.w == 0.0
    # A finite point's patch keeps w == 1.
    finite_pi = next(p for p in ids if p != pi)
    assert patches[ids.index(finite_pi)].w == 1.0

    # Still a valid embedded_patches file.
    out = tmp_path / "infinity.sfmr"
    emb.save(str(out), operation="xform")
    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"
    # The frame survives a save/load round-trip (w is re-derived from the
    # reloaded point's homogeneous coordinate).
    reloaded = SfmrReconstruction.load(str(out))
    rids = list(reloaded.patches.point_indexes)
    assert reloaded.patches[rids.index(pi)].w == 0.0


def test_apply_halves_full_cli_extent_to_library_half_extent(
    seoul_bull_workspace: Path,
):
    """``apply`` must convert the full CLI ``extent_value`` to the library
    half-extent (divide by 2). With ``extent=fixed`` the world half-extent is
    exactly the library value, so a full CLI size of ``W`` must yield patches
    whose ``half_extent`` is ``W / 2``. (This sizing used to live on
    ``--refine-normals``; it now happens only here.)"""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    full_size = 0.1
    out = ToEmbeddedPatchesTransform(extent="fixed", extent_value=full_size).apply(
        recon
    )
    cloud = out.patches
    assert cloud is not None and len(cloud) > 0
    half = np.asarray([cloud[i].half_extent for i in range(len(cloud))])
    np.testing.assert_allclose(half, full_size / 2.0, rtol=1e-6)


def test_apply_maps_pixel_size_to_library_policy(seoul_bull_workspace: Path):
    """The CLI ``pixel_size`` policy must reach the library (whose policy is
    named ``pixel_radius``); a broken mapping would raise ``unknown extent
    policy`` from the binding."""
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    out = ToEmbeddedPatchesTransform(extent="pixel_size", extent_value=8.0).apply(recon)
    assert out.patches is not None and len(out.patches) > 0


def test_to_embedded_patches_rejects_already_embedded(seoul_bull_workspace: Path):
    recon = SfmrReconstruction.load(seoul_bull_workspace)
    emb = recon.to_embedded_patches()
    with pytest.raises(ValueError, match="already embedded_patches"):
        emb.to_embedded_patches()


def test_to_embedded_patches_xform_op(seoul_bull_workspace: Path, tmp_path: Path):
    """The transform produces the same embedded_patches result through the chain."""
    out = tmp_path / "out.sfmr"
    apply_transforms_to_file(seoul_bull_workspace, out, [ToEmbeddedPatchesTransform()])
    valid, errors = verify_sfmr(str(out))
    assert valid, f"integrity check failed: {errors}"
    recon = SfmrReconstruction.load(str(out))
    assert recon.feature_source == "embedded_patches"
    assert recon.keypoints_xy is not None
    assert recon.patches is not None


def test_xform_to_embedded_patches_cli(seoul_bull_workspace: Path, tmp_path: Path):
    out = tmp_path / "cli.sfmr"
    args = [
        "xform",
        str(seoul_bull_workspace),
        str(out),
        "--to-embedded-patches",
        "extent=fixed,extent_value=1.0",
    ]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    recon = SfmrReconstruction.load(str(out))
    assert recon.feature_source == "embedded_patches"
