# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfmtool.workspace.feature_prefix_dir`: which options name a folder.

The expected folder names were computed by `get_feature_tool_xxh128`, the
Python function that named every folder before the binding did.
"""

import pytest

from sfmtool.workspace import feature_prefix_dir


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"x": 2**64 - 1}, "features/sift-t-ff6f0648e87cb2820020981e0f5478e2"),
        ({"x": -(2**63)}, "features/sift-t-b406d562b90d6bfec34babf6872396a3"),
        (
            {"edges": [2**64 - 1, -(2**63)]},
            "features/sift-t-fcf30ab5bd949771364656514caf1776",
        ),
    ],
)
def test_integers_at_the_64_bit_edges_name_the_folder_python_gave(options, expected):
    assert feature_prefix_dir("t", "sift-t", options) == expected


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"x": float("nan")}, r"feature_options\['x'\] is NaN"),
        ({"x": float("inf")}, r"feature_options\['x'\] is inf"),
        ({"x": float("-inf")}, r"feature_options\['x'\] is -inf"),
        ({"x": 2**64}, r"feature_options\['x'\] is an integer outside"),
        ({"x": -(2**63) - 1}, r"feature_options\['x'\] is an integer outside"),
        ({"a": {"b": [1, 2**64]}}, r"feature_options\['a'\]\['b'\]\[1\] is an integer"),
        ({"x": {1: "a"}}, r"feature_options\['x'\] has the key 1, which is not a str"),
        ({1: "a"}, r"feature_options has the key 1, which is not a str"),
    ],
)
def test_options_json_cannot_hold_name_no_folder(options, message):
    with pytest.raises(ValueError, match=message):
        feature_prefix_dir("t", "sift-t", options)


def _nested(depth):
    options = {"x": None}
    for _ in range(depth):
        options = {"x": options}
    return options


def test_options_nested_past_the_limit_name_no_folder():
    assert feature_prefix_dir("t", "sift-t", _nested(127)).startswith("features/")
    with pytest.raises(
        ValueError, match="nested more than 128 lists or dicts deep, which names"
    ):
        feature_prefix_dir("t", "sift-t", _nested(128))
    looped = []
    looped.append(looped)
    with pytest.raises(ValueError, match="nested more than 128"):
        feature_prefix_dir("t", "sift-t", {"x": looped})


def test_a_value_of_another_type_names_no_folder():
    with pytest.raises(ValueError, match=r"feature_options\['x'\] has the type set"):
        feature_prefix_dir("t", "sift-t", {"x": {1}})
