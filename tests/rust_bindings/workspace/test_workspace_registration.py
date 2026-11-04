# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.workspace`."""

import sfmtool._sfmtool.workspace as workspace

_EXPECTED_FUNCTIONS = (
    "is_named_image_list_name",
    "recorded_image_to_gray",
    "split_named_image_list_value",
    "tool_image_to_gray",
    "upgrade_feature_options",
)

_EXPECTED_CLASSES = (
    "NamedImageLists",
    "NamedImageListMember",
    "NamedImageListSlice",
)

_EXPECTED_CONSTANTS = ("NAMED_IMAGE_LISTS_FILE_NAME", "WORKSPACE_VERSION")


def test_all_workspace_bindings_registered():
    """Every expected function, class and constant is on `_sfmtool.workspace`,
    and nothing else is."""
    assert sorted(workspace.__all__) == sorted(
        (*_EXPECTED_FUNCTIONS, *_EXPECTED_CLASSES, *_EXPECTED_CONSTANTS)
    )
    for name in _EXPECTED_FUNCTIONS:
        assert callable(getattr(workspace, name)), f"{name} is not callable"
    for name in _EXPECTED_CLASSES:
        assert isinstance(getattr(workspace, name), type), f"{name} is not a class"
    assert workspace.NAMED_IMAGE_LISTS_FILE_NAME == ".sfm-named-image-lists.json"
    assert workspace.WORKSPACE_VERSION == 2


def test_workspace_submodule_public_name():
    """The submodule reports its public `__name__` so binding objects'
    `__module__` reads `sfmtool.workspace`."""
    assert workspace.__name__ == "sfmtool.workspace"
    for name in (*_EXPECTED_FUNCTIONS, *_EXPECTED_CLASSES):
        assert getattr(workspace, name).__module__ == "sfmtool.workspace"
