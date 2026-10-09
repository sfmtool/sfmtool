# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.reconstruction` and `_sfmtool.patches`,
plus the deliberate root-level surface left after the submodule migration."""

import importlib
from types import ModuleType

import sfmtool
import sfmtool._sfmtool as _sfmtool
import sfmtool._sfmtool.patches as patches
import sfmtool._sfmtool.reconstruction as reconstruction

_RECONSTRUCTION_CLASSES = ("SfmrReconstruction", "RangeExpr")
_PATCHES_CLASSES = (
    "OrientedPatch",
    "PatchCloud",
    "CameraViews",
    "ImagePyramidSet",
    "RansacPhotometricOutput",
)
_PATCHES_FUNCTIONS = (
    "refine_photometric_ransac",
    "render_consensus_atlas",
    "spawn_candidate_tracks",
)


def test_reconstruction_bindings_registered():
    """Every expected class is present on `_sfmtool.reconstruction`."""
    missing = [n for n in _RECONSTRUCTION_CLASSES if not hasattr(reconstruction, n)]
    assert not missing, f"missing reconstruction bindings: {missing}"
    for name in _RECONSTRUCTION_CLASSES:
        assert isinstance(getattr(reconstruction, name), type), f"{name} is not a class"


def test_patches_bindings_registered():
    """Every expected class and function is present on `_sfmtool.patches`."""
    missing = [n for n in _PATCHES_CLASSES if not hasattr(patches, n)]
    assert not missing, f"missing patches bindings: {missing}"
    for name in _PATCHES_CLASSES:
        assert isinstance(getattr(patches, name), type), f"{name} is not a class"
    missing = [n for n in _PATCHES_FUNCTIONS if not hasattr(patches, n)]
    assert not missing, f"missing patches functions: {missing}"
    for name in _PATCHES_FUNCTIONS:
        assert callable(getattr(patches, name)), f"{name} is not callable"


def test_submodule_public_names():
    """Both submodules report public `__name__`s so binding objects'
    `__module__` reads the public location."""
    assert reconstruction.__name__ == "sfmtool.reconstruction"
    for name in _RECONSTRUCTION_CLASSES:
        assert getattr(reconstruction, name).__module__ == "sfmtool.reconstruction"
    assert patches.__name__ == "sfmtool.patches"
    for name in _PATCHES_CLASSES:
        assert getattr(patches, name).__module__ == "sfmtool.patches"


def test_root_surface_is_deliberate_and_minimal():
    """The `_sfmtool` root registers only its deliberate root-level names:
    `build_profile`, `ProgressCounter` and `THUMBNAIL_SIZE`, which the package
    root re-exports, and `run_explorer`, the viewer entry point that `sfm
    explorer` calls and the package root does not re-export. Every other
    binding is registered on a submodule, and the package root does not
    re-export it flat."""
    assert callable(_sfmtool.build_profile)
    assert isinstance(_sfmtool.ProgressCounter, type)
    assert isinstance(_sfmtool.THUMBNAIL_SIZE, int)
    assert callable(_sfmtool.run_explorer)
    assert not hasattr(sfmtool, "run_explorer")
    for name in ("ProgressCounter", "build_profile", "THUMBNAIL_SIZE"):
        assert getattr(sfmtool, name) is getattr(_sfmtool, name), name
    for stale in (
        "SfmrReconstruction",
        "RangeExpr",
        "PatchCloud",
        "OrientedPatch",
        "ImagePyramidSet",
        "CameraViews",
        "RansacPhotometricOutput",
        "refine_photometric_ransac",
        "render_consensus_atlas",
        "image_dimensions",
    ):
        assert not hasattr(_sfmtool, stale), f"{stale} still registered flat"
        assert not hasattr(sfmtool, stale), f"sfmtool.{stale} still re-exported flat"


_SUBMODULES = (
    "analysis",
    "bench",
    "fileio",
    "flow",
    "geometry",
    "matching",
    "patches",
    "reconstruction",
    "sift",
    "spatial",
    "spherical",
)


def test_each_submodule_has_a_public_module_with_the_same_names():
    """Each `_sfmtool` submodule has a public module `sfmtool.<name>` that
    exports exactly the submodule's `__all__`, as the same objects. `sfmtool.sift`
    also holds the Python SIFT code, so only its binding names are compared."""
    registered = {
        name
        for name in vars(_sfmtool)
        if isinstance(getattr(_sfmtool, name), ModuleType)
    }
    assert registered == set(_SUBMODULES)
    for name in _SUBMODULES:
        extension = getattr(_sfmtool, name)
        public = importlib.import_module(f"sfmtool.{name}")
        if name != "sift":
            assert list(public.__all__) == list(extension.__all__), name
        for binding in extension.__all__:
            assert getattr(public, binding) is getattr(extension, binding), (
                name,
                binding,
            )
