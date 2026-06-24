# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.spatial`."""

import sfmtool._sfmtool.spatial as spatial

_EXPECTED_CLASSES = (
    "KdTree2d",
    "KdTree3d",
    "KdForest",
)


def test_all_spatial_bindings_registered():
    """Every expected class is present on `_sfmtool.spatial`."""
    missing = [name for name in _EXPECTED_CLASSES if not hasattr(spatial, name)]
    assert not missing, f"missing spatial bindings: {missing}"
    for name in _EXPECTED_CLASSES:
        assert isinstance(getattr(spatial, name), type), f"{name} is not a class"


def test_spatial_submodule_public_name():
    """The submodule reports its public `__name__` so binding objects'
    `__module__` reads `sfmtool.spatial`."""
    assert spatial.__name__ == "sfmtool.spatial"
    for name in _EXPECTED_CLASSES:
        assert getattr(spatial, name).__module__ == "sfmtool.spatial"
