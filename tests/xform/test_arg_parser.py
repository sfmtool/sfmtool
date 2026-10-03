# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ordered xform command-line parser."""

import click
import pytest

from sfmtool._commands.xform import xform
from sfmtool.xform._arg_parser import (
    _GLOBAL_OPTIONS,
    _TRANSFORM_OPTIONS,
    check_against_click,
    parse_transform_args,
    parse_xform_args,
)


@pytest.mark.parametrize(
    "option",
    [
        "--rotate",
        "--translate",
        "--scale",
        "--remove-short-tracks",
        "--remove-narrow-tracks",
        "--remove-isolated",
        "--align-to",
        "--remove-large-features",
        "--filter-by-reprojection-error",
        "--filter-by-zncc-self-similarity-radius",
        "--filter-by-patch-size",
        "--include-range",
        "--exclude-range",
        "--scale-by-measurements",
        "--include-glob",
        "--exclude-glob",
        "--include-by-distribution",
        "--camera-model",
        "--find-points-at-infinity",
    ],
)
def test_required_value_option_reports_missing_argument(option):
    with pytest.raises(click.UsageError) as exc_info:
        parse_transform_args([option])
    assert str(exc_info.value) == f"{option} requires an argument"


def test_interleaved_options_keep_order_and_repeated_values():
    transforms = parse_transform_args(
        ["--scale", "2", "--bundle-adjust", "--scale", "3", "--drop-thumbnails"]
    )
    assert [type(transform).__name__ for transform in transforms] == [
        "ScaleTransform",
        "BundleAdjustTransform",
        "ScaleTransform",
        "DropThumbnailsTransform",
    ]
    assert [transforms[0].scale, transforms[2].scale] == [2.0, 3.0]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ["--rotate", "0,1,90deg"],
            "--rotate expects 4 comma-separated values (axisX,axisY,axisZ,angle), got: 0,1,90deg",
        ),
        (
            ["--include-by-distribution", "2,other"],
            "Unknown --include-by-distribution modifier 'other' (expected 'verbose')",
        ),
        (
            ["--minimal=unknown"],
            "Invalid --minimal token 'unknown': expected key=value",
        ),
    ],
)
def test_option_specific_errors(args, message):
    with pytest.raises(click.UsageError) as exc_info:
        parse_transform_args(args)
    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--translate", "-17.17,-0.42,-0.27"),
        ("--translate", "3,5,-2"),
        ("--rotate", "0,0,1,87deg"),
        ("--rotate", "-1,0,0,-90deg"),
        ("--scale", "1.44"),
        ("--remove-short-tracks", "2"),
        ("--remove-narrow-tracks", "5deg"),
        ("--include-range", "1-10"),
        ("--include-glob", "*left*"),
        ("--camera-model", "SFMTOOL_FISHEYE,coeffs=6"),
        ("--find-points-at-infinity", "0.1,200,2"),
        ("--classify-points-at-infinity", "1.0"),
    ],
)
def test_joined_and_separated_required_values_agree(option, value):
    separated = parse_transform_args([option, value])
    joined = parse_transform_args([f"{option}={value}"])
    assert len(separated) == len(joined) == 1
    assert type(separated[0]) is type(joined[0])
    assert separated[0].description() == joined[0].description()


def test_a_negative_translation_reads_in_both_spellings():
    for args in (["--translate", "-1,2,3"], ["--translate=-1,2,3"]):
        (transform,) = parse_transform_args(args)
        assert transform.translation.tolist() == [-1.0, 2.0, 3.0]


def test_the_georeferencing_chain_keeps_every_step():
    transforms = parse_transform_args(
        [
            "--rotate",
            "0,0,1,87deg",
            "--scale",
            "1.44",
            "--translate=-17.17,-0.42,-0.27",
        ]
    )
    assert [type(t).__name__ for t in transforms] == [
        "RotateTransform",
        "ScaleTransform",
        "TranslateTransform",
    ]
    assert transforms[2].translation.tolist() == [-17.17, -0.42, -0.27]


@pytest.mark.parametrize(
    "option",
    [
        "--drop-thumbnails",
        "--drop-patch-bitmaps",
        "--add-thumbnails",
        "--align-to-input",
    ],
)
def test_a_value_free_option_refuses_a_joined_value(option):
    with pytest.raises(click.UsageError) as exc_info:
        parse_transform_args([f"{option}=x"])
    assert str(exc_info.value) == f"{option} does not take a value"


@pytest.mark.parametrize(
    "token", ["--tranlsate", "--trans", "--no-such-option=1", "-5"]
)
def test_an_unknown_option_is_refused(token):
    with pytest.raises(click.UsageError) as exc_info:
        parse_transform_args(["--scale", "2", token])
    assert str(exc_info.value) == f"No such option: {token.partition('=')[0]}"


def test_a_positional_token_is_refused():
    with pytest.raises(click.UsageError) as exc_info:
        parse_transform_args(["--scale", "2", "stray"])
    assert str(exc_info.value) == "Unexpected argument: stray"


def test_the_walk_collects_positionals_and_global_options():
    parsed = parse_xform_args(
        [
            "in.sfmr",
            "--find-points-at-infinity",
            "0.1",
            "--max-features",
            "500",
            "out.sfmr",
            "--max-features=600",
            "--",
            "--odd-name.sfmr",
        ]
    )
    assert parsed.positionals == ["in.sfmr", "out.sfmr", "--odd-name.sfmr"]
    assert parsed.values == {"--find-points-at-infinity": ["0.1"]}


def test_an_optional_value_follows_click():
    # The next token is the value unless it is an option; a lone "-" is a value,
    # so --minimal reads it (and refuses it as a key=value token).
    parsed = parse_xform_args(
        ["--bundle-adjust", "--scale", "2", "--minimal", "wspath=."]
    )
    assert parsed.values == {
        "--bundle-adjust": [""],
        "--scale": ["2"],
        "--minimal": ["wspath=."],
    }
    with pytest.raises(click.UsageError, match="token '-'"):
        parse_xform_args(["--minimal", "-"])


def test_every_command_option_is_known_to_the_walk():
    declared = {
        opt
        for param in xform.params
        if isinstance(param, click.Option)
        for opt in param.opts + param.secondary_opts
    }
    assert declared == set(_TRANSFORM_OPTIONS) | set(_GLOBAL_OPTIONS)
    for option, (value_rule, _build) in _TRANSFORM_OPTIONS.items():
        (param,) = [p for p in xform.params if option in getattr(p, "opts", ())]
        assert param.multiple
        if value_rule == "none":
            assert param.is_flag
        elif value_rule == "optional":
            assert not param.is_flag and param._flag_needs_value
        else:
            assert not param.is_flag and not param._flag_needs_value


def test_a_disagreement_with_click_is_refused():
    parsed = parse_xform_args(["in.sfmr", "--scale", "2"])
    check_against_click(parsed, {"scale": ("2",)}, ["in.sfmr", None])
    with pytest.raises(click.UsageError, match="--translate"):
        check_against_click(
            parsed, {"scale": ("2",), "translate": ("1,2,3",)}, ["in.sfmr", None]
        )
    with pytest.raises(click.UsageError, match="arguments"):
        check_against_click(parsed, {"scale": ("2",)}, ["in.sfmr", "out.sfmr"])


def test_classify_points_at_infinity_takes_an_optional_noise_level():
    (bare,) = parse_transform_args(["--classify-points-at-infinity"])
    assert bare.sigma_px is None
    (given,) = parse_transform_args(["--classify-points-at-infinity", "0.5"])
    assert given.sigma_px == 0.5
    (joined,) = parse_transform_args(["--classify-points-at-infinity=0.25"])
    assert joined.sigma_px == 0.25
    # Bare, followed by another option: the option is not its value.
    first, second = parse_transform_args(
        ["--classify-points-at-infinity", "--scale", "2"]
    )
    assert first.sigma_px is None
    assert type(second).__name__ == "ScaleTransform"


@pytest.mark.parametrize("value", ["abc", "0", "-1", "nan", "inf"])
def test_classify_points_at_infinity_rejects_a_bad_noise_level(value):
    with pytest.raises(click.UsageError, match="--classify-points-at-infinity"):
        parse_transform_args([f"--classify-points-at-infinity={value}"])


def test_find_points_at_infinity_takes_an_optional_noise_level():
    """The fourth component is a sigma_px override; without it the noise is
    measured."""
    (measured,) = parse_transform_args(["--find-points-at-infinity", "0.1,200,2"])
    assert measured.sigma_px is None
    (given,) = parse_transform_args(["--find-points-at-infinity", "0.1,200,2,0.5"])
    assert given.sigma_px == 0.5
    assert "sigma_px=0.5px" in given.description()


@pytest.mark.parametrize("value", ["abc", "0", "-1", "nan", "inf"])
def test_find_points_at_infinity_rejects_a_bad_noise_level(value):
    with pytest.raises(click.UsageError, match="--find-points-at-infinity"):
        parse_transform_args([f"--find-points-at-infinity=0.1,200,2,{value}"])
