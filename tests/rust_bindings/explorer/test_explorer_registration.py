# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.explorer`."""

import sfmtool._sfmtool.explorer as explorer

_EXPECTED_FUNCTIONS = (
    "tool_catalog",
    "validate_tool_call",
)


def test_all_explorer_functions_registered():
    """Every expected free function is present on `_sfmtool.explorer`, and
    nothing else is."""
    assert sorted(explorer.__all__) == sorted(_EXPECTED_FUNCTIONS)
    for name in _EXPECTED_FUNCTIONS:
        assert callable(getattr(explorer, name)), f"{name} is not callable"


def test_explorer_submodule_public_name():
    """The submodule reports its public `__name__`."""
    assert explorer.__name__ == "sfmtool.explorer"
