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
