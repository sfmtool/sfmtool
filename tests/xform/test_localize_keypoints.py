# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--localize-keypoints`` xform operation.

Covers the ``key=value`` argument grammar (and its error cases) and integration
runs over the real 17-image seoul_bull reconstruction. Unlike
``--refine-keypoints`` (a pure in-place modifier), this op is **structural**:
the localizer drops views that won't co-register, the ``min_views`` cull can
drop points, and the track arrays + ``keypoints_xy`` are rebuilt from the
survivors — so the invariants tested here are shrinking counts and output
validity, not byte-identity. See specs/cli/xform-localize-keypoints-command.md.
"""

from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.cli import main
from sfmtool.xform import LocalizeKeypointsTransform
from sfmtool.xform._arg_parser import (
    parse_localize_keypoints_params,
    parse_transform_args,
)

# ── Argument grammar ────────────────────────────────────────────────────────


def test_parse_empty_runs_defaults():
    """No params yields a transform on the binding defaults plus min_views=2."""
    t = parse_localize_keypoints_params("")
    assert isinstance(t, LocalizeKeypointsTransform)
    assert t.min_views == 2
    assert t.max_iters == 5
    assert t.search == 6.0
    assert t.max_shift_px == 3.0
    assert t.min_relative_zncc == 0.7
    assert t.min_grazing_cos == 0.1
    assert t.resolution == 24
    assert t.window == "gaussian_disk"
    assert t.window_sigma == 0.6
    assert t.sampler == "bilinear"
    assert t.robust_iters == 3
    assert t.convergence_px == 0.05
    assert t.search_resolution_multiplier == 1.0
    assert t.search_strategy == "plus_descent"


def test_parse_key_value_overrides():
    """Each key=value token overrides the matching default with the right type."""
    t = parse_localize_keypoints_params(
        "min_views=3,max_iters=2,search=8,max_shift_px=2.5,"
        "min_relative_zncc=0.5,min_grazing_cos=0.2,resolution=16,"
        "window=gaussian,window_sigma=0.8,sampler=anisotropic,robust_iters=2,"
        "convergence_px=0.1,search_resolution_multiplier=2.0,"
        "search_strategy=exhaustive"
    )
    assert t.min_views == 3
    assert isinstance(t.min_views, int)
    assert t.max_iters == 2
    assert t.search == 8.0
    assert t.max_shift_px == 2.5
    assert t.min_relative_zncc == 0.5
    assert t.min_grazing_cos == 0.2
    assert t.resolution == 16
    assert isinstance(t.resolution, int)
    assert t.window == "gaussian"
    assert t.window_sigma == 0.8
    assert t.sampler == "anisotropic"
    assert t.robust_iters == 2
    assert t.convergence_px == 0.1
    assert t.search_resolution_multiplier == 2.0
    assert t.search_strategy == "exhaustive"


def test_parse_tolerates_blank_segments():
    """Trailing/empty comma segments are ignored, not errors."""
    t = parse_localize_keypoints_params("max_iters=2,")
    assert t.max_iters == 2


def test_parse_unknown_key_rejected():
    import click

    with pytest.raises(click.UsageError, match="Unknown --localize-keypoints key"):
        parse_localize_keypoints_params("max_outer_sweeps=2")


def test_parse_bitmaps_key_rejected():
    """There is deliberately no ``bitmaps`` key: the localizer renders none and
    the structural rebuild drops stored ones as stale (re-run
    ``--refine-keypoints bitmaps=true`` to regenerate)."""
    import click

    with pytest.raises(click.UsageError, match="Unknown --localize-keypoints key"):
        parse_localize_keypoints_params("bitmaps=true")


def test_parse_malformed_token_rejected():
    import click

    with pytest.raises(click.UsageError, match="expected key=value"):
        parse_localize_keypoints_params("search")


def test_parse_empty_key_rejected():
    import click

    with pytest.raises(click.UsageError, match="empty key"):
        parse_localize_keypoints_params("=5")


def test_parse_bad_value_type_rejected():
    import click

    with pytest.raises(click.UsageError, match="not a valid int"):
        parse_localize_keypoints_params("resolution=foo")


def test_parse_duplicate_key_rejected():
    import click

    with pytest.raises(click.UsageError, match="Duplicate"):
        parse_localize_keypoints_params("search=4,search=8")


@pytest.mark.parametrize(
    "param",
    [
        "min_views=0",
        "max_iters=0",
        "search=0",
        "max_shift_px=0",
        "min_relative_zncc=1.5",
        "min_grazing_cos=-0.1",
        "resolution=1",
        "window=bogus",
        "window_sigma=0",
        "sampler=bogus",
        "robust_iters=0",
        "convergence_px=0",
        "search_resolution_multiplier=0",
        "search_strategy=bogus",
    ],
)
def test_parse_out_of_range_or_bad_enum_rejected(param):
    """Constructor range/enum validation surfaces as ValueError."""
    with pytest.raises(ValueError):
        parse_localize_keypoints_params(param)


def test_bare_flag_tokenization():
    """A bare ``--localize-keypoints`` (trailing, or followed by another
    option) parses to the defaults without swallowing the next token; the
    joined ``=`` form carries its params."""
    transforms = parse_transform_args(["--localize-keypoints"])
    assert len(transforms) == 1
    assert isinstance(transforms[0], LocalizeKeypointsTransform)
    assert transforms[0].min_views == 2

    transforms = parse_transform_args(["--localize-keypoints", "--bundle-adjust"])
    assert len(transforms) == 2
    assert isinstance(transforms[0], LocalizeKeypointsTransform)
    assert transforms[0].search == 6.0

    transforms = parse_transform_args(["--localize-keypoints=search=8,min_views=3"])
    assert len(transforms) == 1
    assert transforms[0].search == 8.0
    assert transforms[0].min_views == 3


def test_constructor_description_mentions_key_settings():
    desc = LocalizeKeypointsTransform(search=8.0, min_views=3).description()
    assert "Localize keypoints" in desc
    assert "search=8.0" in desc
    assert "min_views=3" in desc
    assert "plus_descent" in desc


# ── Integration over a real reconstruction ──────────────────────────────────


def _modest_params(**overrides) -> LocalizeKeypointsTransform:
    """Cheap search params: correctness, not quality."""
    kwargs = dict(resolution=12, max_iters=2)
    kwargs.update(overrides)
    return LocalizeKeypointsTransform(**kwargs)


def _embedded(workspace) -> SfmrReconstruction:
    """Convert the sift_files workspace recon to embedded_patches.

    ``--localize-keypoints`` requires an ``embedded_patches`` reconstruction
    (it searches over the stored per-point patch frames), so the integration
    tests feed it the converted recon — the same bridge the CLI's
    ``--to-embedded-patches --localize-keypoints`` chain runs.
    """
    return SfmrReconstruction.load(workspace).to_embedded_patches(
        normal="mean_viewing", extent_value=5.0
    )


def test_localize_keypoints_structural_cull(seoul_bull_workspace, tmp_path):
    """The key property: the output is a valid, re-loadable embedded_patches
    recon whose point/observation counts are <= the input's, with every
    surviving point keeping at least ``min_views`` observations. The track
    arrays are rebuilt, so no byte-identity is asserted."""
    from sfmtool._sfmtool.io import verify_sfmr

    recon = _embedded(seoul_bull_workspace)
    obs_before = len(np.asarray(recon.track_point_indexes))

    out = _modest_params().apply(recon)

    # Structural cull: counts shrink (or stay), never grow.
    assert out.feature_source == "embedded_patches"
    assert out.image_count == recon.image_count
    assert 0 < out.point_count <= recon.point_count
    obs_after = len(np.asarray(out.track_point_indexes))
    assert 0 < obs_after <= obs_before

    # Every survivor respects the min_views floor (default 2).
    obs_counts = np.asarray(out.observation_counts)
    assert obs_counts.shape == (out.point_count,)
    assert (obs_counts >= 2).all()

    # Track arrays are consistent with the dense renumbering.
    tpi = np.asarray(out.track_point_indexes)
    assert tpi.max() == out.point_count - 1
    np.testing.assert_array_equal(np.bincount(tpi), obs_counts)

    # The rebuilt recon round-trips through a .sfmr save (in-frame clamp).
    path = tmp_path / "localized_keypoints.sfmr"
    out.save(path)
    is_valid, errors = verify_sfmr(str(path))
    assert is_valid, errors
    reloaded = SfmrReconstruction.load(path)
    assert reloaded.point_count == out.point_count
    np.testing.assert_array_equal(
        np.asarray(reloaded.keypoints_xy), np.asarray(out.keypoints_xy)
    )


def test_localize_keypoints_min_views_respected(seoul_bull_workspace):
    """A higher ``min_views`` culls at least as many points as the default,
    and every survivor meets the raised floor."""
    recon = _embedded(seoul_bull_workspace)

    out_default = _modest_params().apply(recon)
    out_strict = _modest_params(min_views=4).apply(recon)

    assert out_strict.point_count <= out_default.point_count
    assert (np.asarray(out_strict.observation_counts) >= 4).all()


def test_localize_keypoints_stay_in_frame(seoul_bull_workspace):
    """Localized keypoints are within each image's [0, width) x [0, height)."""
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


def test_localize_keypoints_drops_bitmaps(seoul_bull_workspace):
    """The output keeps patch frames but no bitmaps — the localizer renders
    none, and stored ones would be stale after the keypoints move and views
    drop (re-run --refine-keypoints bitmaps=true to regenerate)."""
    from sfmtool.xform import RefineKeypointsTransform

    recon = _embedded(seoul_bull_workspace)
    # Attach bitmaps first so the drop is observable.
    with_bitmaps = RefineKeypointsTransform(
        bitmaps=True, resolution=12, max_gn_steps=1
    ).apply(recon)
    assert with_bitmaps.patch_bitmaps is not None

    out = _modest_params().apply(with_bitmaps)

    assert out.patch_bitmaps is None
    assert out.patches is not None
    assert len(out.patches) == out.point_count


def test_localize_keypoints_prints_summary(seoul_bull_workspace, capsys):
    """The summary reports the structural point/observation shrink."""
    recon = _embedded(seoul_bull_workspace)
    _modest_params().apply(recon)
    summary = capsys.readouterr().out
    assert "Localized keypoints:" in summary
    assert "points" in summary
    assert "observations" in summary


def test_missing_image_is_hard_error(seoul_bull_workspace):
    recon = _embedded(seoul_bull_workspace)
    from pathlib import Path

    img = Path(recon.workspace_dir) / recon.image_names[0]
    img.unlink()
    with pytest.raises(FileNotFoundError):
        _modest_params().apply(recon)


def test_cli_localize_keypoints(seoul_bull_workspace):
    """End-to-end CLI run: convert then localize in one chain; the output is a
    valid embedded_patches recon with shrunk (or equal) counts and no bitmaps.
    The sys.argv reparse needs patching."""
    input_sfmr = seoul_bull_workspace
    output_sfmr = input_sfmr.with_name("localized_kpts.sfmr")

    args = [
        "xform",
        str(input_sfmr),
        str(output_sfmr),
        "--to-embedded-patches",
        "--localize-keypoints",
        "resolution=12,max_iters=2",
    ]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, result.output
    assert output_sfmr.exists()

    embedded = _embedded(input_sfmr)
    localized = SfmrReconstruction.load(output_sfmr)
    assert localized.feature_source == "embedded_patches"
    assert 0 < localized.point_count <= embedded.point_count
    assert 0 < localized.observation_count <= embedded.observation_count
    assert localized.patch_bitmaps is None
    assert localized.patches is not None


def test_localize_keypoints_rejects_sift_files(seoul_bull_workspace):
    """``--localize-keypoints`` on a sift_files recon is rejected up front
    (before any image load or search) with a pointer to the conversion
    bridge."""
    input_sfmr = seoul_bull_workspace
    output_sfmr = input_sfmr.with_name("rejected_localize.sfmr")

    args = [
        "xform",
        str(input_sfmr),
        str(output_sfmr),
        "--localize-keypoints",
    ]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)

    assert result.exit_code != 0
    assert "embedded_patches" in result.output
    assert not output_sfmr.exists()
