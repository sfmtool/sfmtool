# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--refine-keypoints`` xform operation.

Covers the ``key=value`` argument grammar (and its error cases) and integration
runs over the real 17-image seoul_bull reconstruction. The key property is
**structural invariance**: the op rewrites ``keypoints_xy`` values only — the
track arrays, observation counts, and point count are byte-identical to the
input. See specs/cli/xform-refine-keypoints-command.md.
"""

from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.cli import main
from sfmtool.xform import RefineKeypointsTransform
from sfmtool.xform._arg_parser import parse_refine_keypoints_params


# ── Argument grammar ────────────────────────────────────────────────────────


def test_parse_empty_runs_defaults():
    """No params yields a transform on the binding defaults."""
    t = parse_refine_keypoints_params("")
    assert isinstance(t, RefineKeypointsTransform)
    assert t.resolution == 24
    assert t.window == "gaussian_disk"
    assert t.window_sigma == 0.6
    assert t.sampler == "bilinear"
    assert t.robust_iters == 3
    assert t.max_outer_sweeps == 1
    assert t.outer_convergence_px == 0.005
    assert t.max_gn_steps == 10
    assert t.convergence_px == 0.01
    assert t.max_offset_px == 2.0
    assert t.consensus_refresh == "per_sweep"
    assert t.bitmaps is False


def test_parse_key_value_overrides():
    """Each key=value token overrides the matching default with the right type."""
    t = parse_refine_keypoints_params(
        "resolution=16,window=gaussian,window_sigma=0.8,sampler=anisotropic,"
        "robust_iters=2,max_outer_sweeps=3,outer_convergence_px=0.01,"
        "max_gn_steps=5,convergence_px=0.02,max_offset_px=1.5,"
        "consensus_refresh=per_move"
    )
    assert t.resolution == 16
    assert isinstance(t.resolution, int)
    assert t.window == "gaussian"
    assert t.window_sigma == 0.8
    assert t.sampler == "anisotropic"
    assert t.robust_iters == 2
    assert t.max_outer_sweeps == 3
    assert t.outer_convergence_px == 0.01
    assert t.max_gn_steps == 5
    assert t.convergence_px == 0.02
    assert t.max_offset_px == 1.5
    assert t.consensus_refresh == "per_move"


def test_parse_tolerates_blank_segments():
    """Trailing/empty comma segments are ignored, not errors."""
    t = parse_refine_keypoints_params("max_gn_steps=5,")
    assert t.max_gn_steps == 5


def test_parse_bilinear_mip_sampler():
    """The single-tap mip sampler round-trips through the parser."""
    t = parse_refine_keypoints_params("sampler=bilinear_mip")
    assert t.sampler == "bilinear_mip"


def test_parse_unknown_key_rejected():
    import click

    with pytest.raises(click.UsageError, match="Unknown --refine-keypoints key"):
        parse_refine_keypoints_params("angular_range_deg=25")


@pytest.mark.parametrize("key", ["extent", "extent_value", "normal"])
def test_frame_building_keys_rejected(key):
    """Frame-sizing knobs live on ``--to-embedded-patches``, not
    ``--refine-keypoints`` (which reuses the stored frame). They are rejected
    as unknown keys here."""
    import click

    with pytest.raises(click.UsageError, match="Unknown --refine-keypoints key"):
        parse_refine_keypoints_params(f"{key}=foo")


def test_parse_malformed_token_rejected():
    import click

    with pytest.raises(click.UsageError, match="expected key=value"):
        parse_refine_keypoints_params("resolution")


def test_parse_empty_key_rejected():
    import click

    with pytest.raises(click.UsageError, match="empty key"):
        parse_refine_keypoints_params("=5")


def test_parse_bad_value_type_rejected():
    import click

    with pytest.raises(click.UsageError, match="not a valid int"):
        parse_refine_keypoints_params("resolution=foo")


def test_parse_duplicate_key_rejected():
    import click

    with pytest.raises(click.UsageError, match="Duplicate"):
        parse_refine_keypoints_params("resolution=12,resolution=16")


@pytest.mark.parametrize(
    "param",
    [
        "resolution=1",
        "window=bogus",
        "window_sigma=0",
        "sampler=bogus",
        "robust_iters=0",
        "max_outer_sweeps=0",
        "outer_convergence_px=0",
        "max_gn_steps=0",
        "convergence_px=0",
        "max_offset_px=0",
        "consensus_refresh=bogus",
    ],
)
def test_parse_out_of_range_or_bad_enum_rejected(param):
    """Constructor range/enum validation surfaces as ValueError."""
    with pytest.raises(ValueError):
        parse_refine_keypoints_params(param)


def test_parse_bitmaps():
    """``bitmaps`` is a recognized boolean key (default False); it controls
    whether the per-point RGBA patch textures are rendered and persisted."""
    assert parse_refine_keypoints_params("").bitmaps is False
    assert parse_refine_keypoints_params("bitmaps=true").bitmaps is True
    assert parse_refine_keypoints_params("bitmaps=false").bitmaps is False
    import click

    with pytest.raises(click.UsageError):
        parse_refine_keypoints_params("bitmaps=maybe")


def test_constructor_description_mentions_key_settings():
    desc = RefineKeypointsTransform(
        max_outer_sweeps=2, sampler="anisotropic"
    ).description()
    assert "Refine keypoints" in desc
    assert "sweeps=2" in desc
    assert "anisotropic" in desc
    assert "bitmaps" not in desc
    assert "bitmaps" in RefineKeypointsTransform(bitmaps=True).description()


# ── Integration over a real reconstruction ──────────────────────────────────


def _modest_params(**overrides) -> RefineKeypointsTransform:
    """Cheap solve params: correctness, not quality."""
    kwargs = dict(resolution=12, max_gn_steps=3)
    kwargs.update(overrides)
    return RefineKeypointsTransform(**kwargs)


def _embedded(workspace) -> SfmrReconstruction:
    """Convert the sift_files workspace recon to embedded_patches.

    ``--refine-keypoints`` requires an ``embedded_patches`` reconstruction (it
    seeds each view from the stored inline keypoint), so the integration tests
    feed it the converted recon — the same bridge the CLI's
    ``--to-embedded-patches --refine-keypoints`` chain runs.
    """
    return SfmrReconstruction.load(workspace).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )


def test_refine_keypoints_structural_invariance(seoul_bull_workspace, tmp_path):
    """The key property: only ``keypoints_xy`` values change — the track
    arrays, observation counts, point count, positions, and normals are
    identical to the input's; and the result still saves cleanly (the in-frame
    clamp keeps the writer's keypoint checks green)."""
    from sfmtool._sfmtool.io import verify_sfmr

    recon = _embedded(seoul_bull_workspace)
    orig_track_images = np.asarray(recon.track_image_indexes).copy()
    orig_track_points = np.asarray(recon.track_point_indexes).copy()
    orig_obs_counts = np.asarray(recon.observation_counts).copy()
    orig_positions = np.asarray(recon.positions).copy()
    orig_normals = np.asarray(recon.normals).copy()
    orig_keypoints = np.asarray(recon.keypoints_xy).copy()

    out = _modest_params().apply(recon)

    # Structural invariance: no view or point dropped, nothing reordered.
    assert out.point_count == recon.point_count
    assert out.image_count == recon.image_count
    assert out.feature_source == "embedded_patches"
    np.testing.assert_array_equal(
        np.asarray(out.track_image_indexes), orig_track_images
    )
    np.testing.assert_array_equal(
        np.asarray(out.track_point_indexes), orig_track_points
    )
    np.testing.assert_array_equal(np.asarray(out.observation_counts), orig_obs_counts)
    np.testing.assert_array_equal(np.asarray(out.positions), orig_positions)
    np.testing.assert_array_equal(np.asarray(out.normals), orig_normals)

    # Only the keypoint values moved — same shape, at least one refined.
    new_keypoints = np.asarray(out.keypoints_xy)
    assert new_keypoints.shape == orig_keypoints.shape
    assert not np.array_equal(new_keypoints, orig_keypoints)

    # The refined recon round-trips through a .sfmr save (in-frame clamp).
    path = tmp_path / "refined_keypoints.sfmr"
    out.save(path)
    is_valid, errors = verify_sfmr(str(path))
    assert is_valid, errors
    reloaded = SfmrReconstruction.load(path)
    np.testing.assert_array_equal(np.asarray(reloaded.keypoints_xy), new_keypoints)


def test_refine_keypoints_stay_in_frame(seoul_bull_workspace):
    """Refined keypoints are within each image's [0, width) x [0, height)."""
    recon = _embedded(seoul_bull_workspace)

    out = _modest_params().apply(recon)

    kxy = np.asarray(out.keypoints_xy)
    im = np.asarray(out.track_image_indexes)
    cam_idx = np.asarray(out.camera_indexes)
    cams = out.cameras
    widths = np.array([cams[int(c)].width for c in cam_idx], dtype=np.float32)
    heights = np.array([cams[int(c)].height for c in cam_idx], dtype=np.float32)
    assert (kxy[:, 0] >= 0).all()
    assert (kxy[:, 1] >= 0).all()
    assert (kxy[:, 0] < widths[im]).all()
    assert (kxy[:, 1] < heights[im]).all()


def test_refine_keypoints_bitmaps(seoul_bull_workspace):
    """``bitmaps=true`` attaches a ``(point_count, R, R, 4)`` uint8 texture
    array; without it none is attached."""
    recon = _embedded(seoul_bull_workspace)

    plain = _modest_params().apply(recon)
    assert plain.patch_bitmaps is None

    out = _modest_params(bitmaps=True).apply(recon)
    assert out.patches is not None
    bitmaps = out.patch_bitmaps
    assert bitmaps is not None
    assert bitmaps.shape == (recon.point_count, 12, 12, 4)
    assert bitmaps.dtype == np.uint8
    # At least one patch was rendered (non-zero RGBA somewhere).
    assert bitmaps.any()


def test_refine_keypoints_prints_summary(seoul_bull_workspace, capsys):
    """The one-line summary reports the refined-view count and mean offset."""
    recon = _embedded(seoul_bull_workspace)
    _modest_params().apply(recon)
    summary = capsys.readouterr().out
    assert "Refined" in summary
    assert "keypoints" in summary


def test_missing_image_is_hard_error(seoul_bull_workspace):
    recon = _embedded(seoul_bull_workspace)
    from pathlib import Path

    img = Path(recon.workspace_dir) / recon.image_names[0]
    img.unlink()
    with pytest.raises(FileNotFoundError):
        _modest_params().apply(recon)


def test_cli_refine_keypoints(seoul_bull_workspace):
    """End-to-end CLI run rewrites keypoints without touching the structure;
    the sys.argv reparse needs patching.

    ``--refine-keypoints`` requires embedded_patches, so the run converts first
    in the same pipeline (``--to-embedded-patches --refine-keypoints``)."""
    input_sfmr = seoul_bull_workspace
    output_sfmr = input_sfmr.with_name("refined_kpts.sfmr")

    args = [
        "xform",
        str(input_sfmr),
        str(output_sfmr),
        "--to-embedded-patches",
        "--refine-keypoints",
        "resolution=12,max_gn_steps=3",
    ]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, result.output
    assert output_sfmr.exists()

    embedded = _embedded(input_sfmr)
    refined = SfmrReconstruction.load(output_sfmr)
    assert refined.point_count == embedded.point_count
    assert refined.observation_count == embedded.observation_count
    np.testing.assert_array_equal(
        np.asarray(refined.track_point_indexes),
        np.asarray(embedded.track_point_indexes),
    )
    # At least one keypoint changed.
    assert not np.array_equal(
        np.asarray(embedded.keypoints_xy), np.asarray(refined.keypoints_xy)
    )


def test_cli_refine_keypoints_bare_before_other_option(seoul_bull_workspace):
    """A bare --refine-keypoints followed by another option runs the defaults
    and leaves the following option intact (optional-value tokenization).

    This is a CLI *tokenization* test, not a refinement-quality one, so the
    expensive default-resolution refinement is stubbed out: what matters is
    that the bare option parsed to the documented defaults and did not swallow
    the trailing ``--scale 2.0`` as its value."""
    input_sfmr = seoul_bull_workspace
    output_sfmr = input_sfmr.with_name("refined_kpts_bare.sfmr")

    args = [
        "xform",
        str(input_sfmr),
        str(output_sfmr),
        "--to-embedded-patches",
        "--refine-keypoints",
        "--scale",
        "2.0",
    ]

    captured: list[RefineKeypointsTransform] = []

    def _stub_apply(self, recon):
        # Record the parsed transform, then skip the heavy refinement and pass
        # the reconstruction through unchanged.
        captured.append(self)
        return recon

    with (
        patch("sys.argv", ["sfm"] + args),
        patch.object(RefineKeypointsTransform, "apply", _stub_apply),
    ):
        result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, result.output
    assert output_sfmr.exists()

    # The bare --refine-keypoints parsed to the documented defaults.
    assert len(captured) == 1
    t = captured[0]
    assert (t.resolution, t.max_outer_sweeps, t.max_gn_steps) == (24, 1, 10)

    original = SfmrReconstruction.load(input_sfmr)
    refined = SfmrReconstruction.load(output_sfmr)
    # The trailing --scale 2.0 still applied (proves it wasn't consumed as the
    # bare option's value).
    np.testing.assert_allclose(
        np.asarray(refined.positions), np.asarray(original.positions) * 2.0, rtol=1e-5
    )


def test_refine_keypoints_rejects_sift_files(seoul_bull_workspace):
    """``--refine-keypoints`` on a sift_files recon is rejected up front
    (before any image load or refinement) with a pointer to the conversion
    bridge."""
    input_sfmr = seoul_bull_workspace
    output_sfmr = input_sfmr.with_name("rejected_kpts.sfmr")

    args = [
        "xform",
        str(input_sfmr),
        str(output_sfmr),
        "--refine-keypoints",
    ]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)

    assert result.exit_code != 0
    assert "embedded_patches" in result.output
    assert not output_sfmr.exists()
