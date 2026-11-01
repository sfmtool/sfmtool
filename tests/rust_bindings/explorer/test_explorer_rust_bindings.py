# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The viewer's tool catalog and call parse, through `sfmtool.explorer`."""

import math

import pytest

from sfmtool.explorer import tool_catalog, validate_tool_call


def test_the_catalog_lists_each_tool_with_its_kind_and_advertised_entry():
    catalog = tool_catalog()
    names = [tool["name"] for tool in catalog]
    assert len(names) == len(set(names))
    assert {"get_scene", "get_point", "screenshot", "save_reconstruction"} <= set(names)
    kinds = {tool["name"]: tool["kind"] for tool in catalog}
    assert kinds["get_scene"] == "read"
    assert kinds["select_point"] == "write"
    assert kinds["save_reconstruction"] == "save"
    assert kinds["click"] == "input"
    for tool in catalog:
        advertised = tool["advertised"]
        assert advertised["name"] == tool["name"]
        assert advertised["description"]
        assert advertised["inputSchema"]["type"] == "object"
        assert advertised["annotations"]["openWorldHint"] is False
        assert advertised["annotations"]["readOnlyHint"] is (tool["kind"] == "read")


def test_arguments_that_parse_validate_to_none():
    assert validate_tool_call("get_scene") is None
    assert validate_tool_call("get_scene", None) is None
    assert validate_tool_call("get_scene", {}) is None
    assert validate_tool_call("get_point", {"point": 12}) is None


def test_arguments_that_do_not_parse_give_the_viewer_s_message():
    message = validate_tool_call("list_camera_images", {"offset": "soon"})
    assert message is not None and "offset" in message
    message = validate_tool_call("get_scene", {"labl": "x"})
    assert message.startswith('get_scene has no argument "labl"')
    message = validate_tool_call("get_scen", {})
    assert message == (
        'There is no tool named "get_scen". Call tools/list for what this '
        "viewer offers."
    )


def test_arguments_that_are_not_a_json_object_are_refused():
    with pytest.raises(ValueError, match="must be a dict"):
        validate_tool_call("get_scene", [1, 2])
    with pytest.raises(ValueError):
        validate_tool_call("get_point", {"point": math.nan})
    with pytest.raises(TypeError):
        validate_tool_call("get_point", {"point": object()})
