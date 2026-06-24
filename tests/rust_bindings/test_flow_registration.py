# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.flow`."""

import sfmtool._sfmtool.flow as flow

_EXPECTED_FUNCTIONS = (
    "gpu_available",
    "compute_optical_flow",
    "compute_optical_flow_timed",
    "compute_optical_flow_with_init",
    "compose_flow",
    "advect_points",
)

_EXPECTED_CLASSES = (
    "WarpMap",
    "ImagePyramid",
)


def test_all_flow_bindings_registered():
    """Every expected function and class is present on `_sfmtool.flow`."""
    missing = [
        name
        for name in (*_EXPECTED_FUNCTIONS, *_EXPECTED_CLASSES)
        if not hasattr(flow, name)
    ]
    assert not missing, f"missing flow bindings: {missing}"
    for name in _EXPECTED_FUNCTIONS:
        assert callable(getattr(flow, name)), f"{name} is not callable"
    for name in _EXPECTED_CLASSES:
        assert isinstance(getattr(flow, name), type), f"{name} is not a class"


def test_flow_submodule_public_name():
    """The submodule reports its public `__name__` so binding objects'
    `__module__` reads `sfmtool.flow` in tracebacks, IPython, and Sphinx."""
    assert flow.__name__ == "sfmtool.flow"
    for name in _EXPECTED_FUNCTIONS:
        assert getattr(flow, name).__module__ == "sfmtool.flow"
    for name in _EXPECTED_CLASSES:
        assert getattr(flow, name).__module__ == "sfmtool.flow"
