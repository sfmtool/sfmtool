# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.reconstruction` and `_sfmtool.patches`,
plus the deliberate root-level surface left after the submodule migration."""

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
_PATCHES_FUNCTIONS = ("refine_photometric_ransac", "render_consensus_atlas")


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
    """The `_sfmtool` root registers exactly the two cross-cutting names
    (`build_profile`, `ProgressCounter`); the old flat class registrations are
    gone, and the package root still re-exports the public API explicitly."""
    assert callable(_sfmtool.build_profile)
    assert isinstance(_sfmtool.ProgressCounter, type)
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
    # The public package-root surface is preserved via explicit re-exports.
    for name in (
        "SfmrReconstruction",
        "RangeExpr",
        "PatchCloud",
        "OrientedPatch",
        "ImagePyramidSet",
        "CameraViews",
        "ProgressCounter",
        "build_profile",
        "image_dimensions",
    ):
        assert hasattr(sfmtool, name), f"sfmtool.{name} re-export missing"


def test_image_dimensions_lives_in_io():
    """`image_dimensions` moved into the `io` submodule."""
    import sfmtool._sfmtool.io as io_mod

    assert callable(io_mod.image_dimensions)
