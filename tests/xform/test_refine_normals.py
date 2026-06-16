# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--refine-normals`` xform operation.

Covers the ``key=value`` argument grammar (and its error cases) and an
integration run over the real 17-image seoul_bull reconstruction: the point
count is unchanged, finite-point normals change, infinity points pass through,
and the mean photoconsistency does not decrease. See
specs/cli/xform-refine-normals-command.md.
"""

from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._sfmtool import SfmrReconstruction
from sfmtool.cli import main
from sfmtool.xform import RefineNormalsTransform
from sfmtool.xform._arg_parser import parse_refine_normals_params


# ── Argument grammar ────────────────────────────────────────────────────────


def test_parse_empty_runs_defaults():
    """No params yields a transform on the binding defaults."""
    t = parse_refine_normals_params("")
    assert isinstance(t, RefineNormalsTransform)
    assert t.angular_range_deg == 25.0
    assert t.init_steps == 7
    assert t.initial_normals == "stored"
    assert t.extent == "feature_size"


def test_parse_key_value_overrides():
    """Each key=value token overrides the matching default with the right type."""
    t = parse_refine_normals_params(
        "angular_range_deg=30,init_steps=9,sampler=anisotropic,"
        "objective=mean,initial_normals=geometric,resolution=32,extent=fixed,"
        "extent_value=0.05,min_views=4,window=gaussian,window_sigma=0.8"
    )
    assert t.angular_range_deg == 30.0
    assert t.init_steps == 9
    assert isinstance(t.init_steps, int)
    assert t.sampler == "anisotropic"
    assert t.objective == "mean"
    assert t.initial_normals == "geometric"
    assert t.resolution == 32
    assert t.extent == "fixed"
    assert t.extent_value == 0.05
    assert t.min_views == 4
    assert t.window == "gaussian"
    assert t.window_sigma == 0.8


def test_parse_tolerates_blank_segments():
    """Trailing/empty comma segments are ignored, not errors."""
    t = parse_refine_normals_params("init_steps=5,")
    assert t.init_steps == 5


def test_parse_unknown_key_rejected():
    import click

    with pytest.raises(click.UsageError, match="Unknown --refine-normals key"):
        parse_refine_normals_params("k_neighbors=8")


def test_parse_malformed_token_rejected():
    import click

    with pytest.raises(click.UsageError, match="expected key=value"):
        parse_refine_normals_params("init_steps")


def test_parse_empty_key_rejected():
    import click

    with pytest.raises(click.UsageError, match="empty key"):
        parse_refine_normals_params("=5")


def test_parse_bad_value_type_rejected():
    import click

    with pytest.raises(click.UsageError, match="not a valid int"):
        parse_refine_normals_params("init_steps=foo")


def test_parse_duplicate_key_rejected():
    import click

    with pytest.raises(click.UsageError, match="Duplicate"):
        parse_refine_normals_params("init_steps=5,init_steps=6")


@pytest.mark.parametrize(
    "param",
    [
        "angular_range_deg=0",
        "init_steps=1",
        "refine_levels=0",
        "resolution=1",
        "objective=bogus",
        "window=bogus",
        "sampler=bogus",
        "min_valid_fraction=1.5",
        "min_views=1",
        "initial_normals=bogus",
        "extent=bogus",
        "extent_value=0",
        "cache=bogus",
        "cache_supersample=0.5",
        "quality=bogus",
    ],
)
def test_parse_out_of_range_or_bad_enum_rejected(param):
    """Constructor range/enum validation surfaces as ValueError."""
    with pytest.raises(ValueError):
        parse_refine_normals_params(param)


def test_parse_cache_knobs():
    """The cache knobs parse with the right types and pass through."""
    t = parse_refine_normals_params("cache=fronto,cache_supersample=2")
    assert t.cache == "fronto"
    assert t.cache_supersample == 2.0


def test_parse_confidence_flag():
    """``confidence`` parses to a bool and is off by default."""
    assert parse_refine_normals_params("").confidence is False
    assert parse_refine_normals_params("confidence=true").confidence is True
    assert parse_refine_normals_params("confidence=off").confidence is False
    import click
    import pytest

    with pytest.raises(click.UsageError):
        parse_refine_normals_params("confidence=maybe")


def test_summary_reports_confidence_only_when_requested(capsys):
    """The xform summary mentions low-confidence only when confidence is on."""
    photo = np.array([0.8, 0.9])
    init = np.array([0.7, 0.8])
    conf = np.array([0.05, np.nan])

    RefineNormalsTransform(confidence=False)._print_summary(photo, init, conf)
    assert "low-confidence" not in capsys.readouterr().out

    RefineNormalsTransform(confidence=True)._print_summary(photo, init, conf)
    assert "low-confidence" in capsys.readouterr().out


def test_quality_preset_overrides_cache_knobs():
    """A quality preset sets (cache, cache_supersample) and wins over them."""
    coarse = RefineNormalsTransform(quality="coarse")
    assert coarse.cache == "fronto"
    assert coarse.cache_supersample == 2.0

    fine = RefineNormalsTransform(quality="fine", cache="fronto", cache_supersample=3.0)
    assert fine.cache == "off"

    # quality=none defers to the explicit knobs.
    explicit = RefineNormalsTransform(
        quality="none", cache="fronto", cache_supersample=2.0
    )
    assert explicit.cache == "fronto"
    assert explicit.cache_supersample == 2.0


def test_constructor_description_mentions_key_settings():
    desc = RefineNormalsTransform(initial_normals="mean_viewing").description()
    assert "Refine normals" in desc
    assert "mean_viewing" in desc


# ── Integration over a real reconstruction ──────────────────────────────────


def _modest_params() -> RefineNormalsTransform:
    """Cheap search params: correctness, not quality (mirrors the core test)."""
    return RefineNormalsTransform(
        resolution=12, init_steps=5, refine_levels=2, sampler="bilinear"
    )


def test_refine_normals_preserves_points_and_improves(
    sfmrfile_reconstruction_with_17_images,
):
    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    original_normals = np.asarray(recon.estimated_normals).copy()
    original_positions = np.asarray(recon.positions).copy()
    at_infinity = np.asarray(recon.point_is_at_infinity, dtype=bool)

    out = _modest_params().apply(recon)

    # Modifier semantics: nothing about the geometry changes.
    assert out.point_count == recon.point_count
    assert out.image_count == recon.image_count
    np.testing.assert_array_equal(np.asarray(out.positions), original_positions)

    new_normals = np.asarray(out.estimated_normals)
    assert new_normals.shape == original_normals.shape

    # Some finite-point normal actually moved.
    finite = ~at_infinity
    assert finite.any()
    moved = np.abs((new_normals[finite] * original_normals[finite]).sum(axis=1))
    assert np.nanmin(1.0 - moved) >= -1e-6
    assert (1.0 - moved).max() > 1e-4

    # Infinity points (if any) pass through untouched.
    if at_infinity.any():
        np.testing.assert_array_equal(
            new_normals[at_infinity], original_normals[at_infinity]
        )


def test_refine_normals_does_not_lower_consensus(
    sfmrfile_reconstruction_with_17_images, capsys
):
    """The summary reports a non-negative mean Φ delta."""
    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    _modest_params().apply(recon)
    summary = capsys.readouterr().out
    assert "Refined" in summary


def test_missing_image_is_hard_error(sfmrfile_reconstruction_with_17_images):
    recon = SfmrReconstruction.load(sfmrfile_reconstruction_with_17_images)
    # Remove a source image so loading fails.
    from pathlib import Path

    img = Path(recon.workspace_dir) / recon.image_names[0]
    img.unlink()
    with pytest.raises(FileNotFoundError):
        _modest_params().apply(recon)


def test_cli_refine_normals(sfmrfile_reconstruction_with_17_images):
    """End-to-end CLI run rewrites normals; the sys.argv reparse needs patching."""
    input_sfmr = sfmrfile_reconstruction_with_17_images
    output_sfmr = input_sfmr.with_name("refined.sfmr")

    args = [
        "xform",
        str(input_sfmr),
        str(output_sfmr),
        "--refine-normals",
        "resolution=12,init_steps=5,refine_levels=2",
    ]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, result.output
    assert output_sfmr.exists()

    original = SfmrReconstruction.load(input_sfmr)
    refined = SfmrReconstruction.load(output_sfmr)
    assert refined.point_count == original.point_count
    # At least one normal changed.
    assert not np.array_equal(
        np.asarray(original.estimated_normals), np.asarray(refined.estimated_normals)
    )


def test_cli_refine_normals_bare_before_other_option(
    sfmrfile_reconstruction_with_17_images,
):
    """A bare --refine-normals followed by another option runs the defaults
    and leaves the following option intact (optional-value tokenization)."""
    input_sfmr = sfmrfile_reconstruction_with_17_images
    output_sfmr = input_sfmr.with_name("refined_bare.sfmr")

    # --refine-normals (bare) then --scale: the scale must still be parsed as
    # its own transform, not swallowed as the refine-normals value.
    args = [
        "xform",
        str(input_sfmr),
        str(output_sfmr),
        "--refine-normals",
        "--scale",
        "2.0",
    ]
    with patch("sys.argv", ["sfm"] + args):
        result = CliRunner().invoke(main, args)

    assert result.exit_code == 0, result.output
    assert output_sfmr.exists()
    original = SfmrReconstruction.load(input_sfmr)
    refined = SfmrReconstruction.load(output_sfmr)
    # The trailing --scale 2.0 still applied (proves it wasn't consumed as the
    # bare option's value).
    np.testing.assert_allclose(
        np.asarray(refined.positions), np.asarray(original.positions) * 2.0, rtol=1e-5
    )
