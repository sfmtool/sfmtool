# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.photographs`."""

import sfmtool._sfmtool.photographs as photographs

_EXPECTED_CLASSES = ("PhotographCache", "PhotographLayout")

_EXPECTED_FUNCTIONS = ("check_mipmaps", "prune_mipmaps")

_EXPECTED_METHODS = {
    "PhotographCache": (
        "build_mipmaps",
        "layout",
        "read_level",
        "read_region",
        "read_tile",
        "stats",
    )
}


def test_all_photographs_bindings_registered():
    """Every expected class, method and function is present on
    `_sfmtool.photographs`."""
    missing = [
        name
        for name in _EXPECTED_CLASSES + _EXPECTED_FUNCTIONS
        if not hasattr(photographs, name)
    ]
    assert not missing, f"missing photographs bindings: {missing}"
    for name in _EXPECTED_CLASSES:
        assert isinstance(getattr(photographs, name), type), f"{name} is not a class"
    for name, methods in _EXPECTED_METHODS.items():
        for method in methods:
            assert callable(getattr(getattr(photographs, name), method, None)), (
                f"{name}.{method} is missing"
            )
    for name in _EXPECTED_FUNCTIONS:
        assert callable(getattr(photographs, name)), f"{name} is not callable"


def test_photographs_submodule_public_name():
    """The submodule reports its public `__name__` so binding objects'
    `__module__` reads `sfmtool.photographs`."""
    assert photographs.__name__ == "sfmtool.photographs"
    for name in _EXPECTED_CLASSES:
        assert getattr(photographs, name).__module__ == "sfmtool.photographs"
