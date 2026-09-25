# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ordered xform command-line parser."""

import click
import pytest

from sfmtool.xform._arg_parser import parse_transform_args


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
        "--filter-by-keypoint-uncertainty",
        "--filter-by-patch-size",
        "--include-range",
        "--exclude-range",
        "--scale-by-measurements",
        "--include-glob",
        "--exclude-glob",
        "--include-by-distribution",
        "--camera-model",
        "--find-points-at-infinity",
        "--classify-points-at-infinity",
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
