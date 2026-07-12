# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.geometry`."""

import sfmtool._sfmtool.geometry as geometry

_EXPECTED_CLASSES = (
    "AffineFactorization",
    "CameraIntrinsics",
    "MetricHypothesis",
    "RigidTransform",
    "RotQuaternion",
    "Se3Transform",
)


def test_all_geometry_bindings_registered():
    """Every expected class is present on `_sfmtool.geometry`."""
    missing = [name for name in _EXPECTED_CLASSES if not hasattr(geometry, name)]
    assert not missing, f"missing geometry bindings: {missing}"
    for name in _EXPECTED_CLASSES:
        assert isinstance(getattr(geometry, name), type), f"{name} is not a class"


def test_geometry_submodule_public_name():
    """The submodule reports its public `__name__` so binding objects'
    `__module__` reads `sfmtool.geometry`."""
    assert geometry.__name__ == "sfmtool.geometry"
    for name in _EXPECTED_CLASSES:
        assert getattr(geometry, name).__module__ == "sfmtool.geometry"
