# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.sift`."""

import sfmtool._sfmtool.sift as sift

_EXPECTED_FUNCTIONS = (
    "detect_sift_keypoints",
    "extract_sift",
)


def test_all_sift_bindings_registered():
    """Every expected function is present on `_sfmtool.sift`."""
    missing = [name for name in _EXPECTED_FUNCTIONS if not hasattr(sift, name)]
    assert not missing, f"missing sift bindings: {missing}"
    for name in _EXPECTED_FUNCTIONS:
        assert callable(getattr(sift, name)), f"{name} is not callable"


def test_sift_submodule_public_name():
    """The submodule reports its public `__name__` so binding objects'
    `__module__` reads `sfmtool.sift` — the same name under which the real
    `sfmtool.sift` package re-exports them."""
    assert sift.__name__ == "sfmtool.sift"
    for name in _EXPECTED_FUNCTIONS:
        assert getattr(sift, name).__module__ == "sfmtool.sift"


def test_bindings_reexported_through_sift_package():
    """The native bindings are reachable from the public `sfmtool.sift`
    package (so their `__module__` is truthful, not just a label)."""
    import sfmtool.sift

    for name in _EXPECTED_FUNCTIONS:
        assert hasattr(sfmtool.sift, name), f"{name} not re-exported by sfmtool.sift"
