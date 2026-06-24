# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.spherical`."""

import sfmtool._sfmtool.spherical as spherical

_EXPECTED_FUNCTIONS = ("evenly_distributed_sphere_points",)

_EXPECTED_CLASSES = (
    "SphericalTileRig",
    "PerSphericalTileSourceStack",
)


def test_all_spherical_bindings_registered():
    """Every expected function and class is present on `_sfmtool.spherical`."""
    missing = [
        name
        for name in (*_EXPECTED_FUNCTIONS, *_EXPECTED_CLASSES)
        if not hasattr(spherical, name)
    ]
    assert not missing, f"missing spherical bindings: {missing}"
    for name in _EXPECTED_FUNCTIONS:
        assert callable(getattr(spherical, name)), f"{name} is not callable"
    for name in _EXPECTED_CLASSES:
        assert isinstance(getattr(spherical, name), type), f"{name} is not a class"


def test_spherical_submodule_public_name():
    """The submodule reports its public `__name__` so binding objects'
    `__module__` reads `sfmtool.spherical`."""
    assert spherical.__name__ == "sfmtool.spherical"
    for name in (*_EXPECTED_FUNCTIONS, *_EXPECTED_CLASSES):
        assert getattr(spherical, name).__module__ == "sfmtool.spherical"
