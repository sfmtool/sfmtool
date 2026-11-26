# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The bindings `read_mipmap_options` and `enable_mipmap_options`: what each
returns, and the exceptions they raise. What the writer keeps of the file,
and every invalid object, are tested in Rust
(`crates/sfmtool-core/src/workspace/mipmap_options.rs`)."""

import json

import pytest

from sfmtool.workspace import (
    enable_mipmap_options,
    new_workspace_mipmap_options,
    read_mipmap_options,
)


def _workspace(tmp_path, **config):
    (tmp_path / ".sfm-workspace.json").write_text(json.dumps({"version": 2, **config}))
    return tmp_path


def test_enable_returns_and_writes_the_object_that_read_returns(tmp_path):
    root = _workspace(tmp_path)
    assert read_mipmap_options(root) is None
    written = enable_mipmap_options(root, jpeg_quality=80)
    expected = {**new_workspace_mipmap_options(), "jpeg_quality": 80}
    assert written == expected
    assert list(written) == list(expected)
    assert read_mipmap_options(root) == expected
    assert list(read_mipmap_options(root)) == list(expected)


@pytest.mark.parametrize(
    "values, shown",
    [
        (
            {"jpeg_quality": 300},
            "mipmap_options.jpeg_quality is 300; it must be an integer from 1 to 100",
        ),
        (
            {"out_of_core_long_side": -1},
            "mipmap_options.out_of_core_long_side is -1; it must be an integer of "
            "at least 1",
        ),
        (
            {"out_of_core_long_side": 2**40},
            f"mipmap_options.out_of_core_long_side is {2**40}",
        ),
    ],
)
def test_enable_refuses_a_value_out_of_range(tmp_path, values, shown):
    """A value that does not fit the field's type is refused with the
    reader's words, not an `OverflowError`."""
    root = _workspace(tmp_path)
    with pytest.raises(ValueError, match="^" + shown.replace("(", r"\(")):
        enable_mipmap_options(root, **values)
    assert read_mipmap_options(root) is None


def test_the_errors_of_each_binding(tmp_path):
    with pytest.raises(OSError):
        enable_mipmap_options(tmp_path)
    with pytest.raises(OSError):
        read_mipmap_options(tmp_path)
    root = _workspace(tmp_path, mipmap_options={"jpeg_quality": 95})
    with pytest.raises(ValueError, match="already has mipmap_options"):
        enable_mipmap_options(root)
    with pytest.raises(
        ValueError, match="mipmap_options.out_of_core_long_side is missing"
    ):
        read_mipmap_options(root)
