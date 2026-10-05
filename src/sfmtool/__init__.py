# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""SfM Tool: Structure from Motion on a Rust core.

The names on this package come from two places. The names of the compiled
``sfmtool._sfmtool`` extension are bound when the package is imported, since
loading the extension takes about 10 ms. The names defined in the Python
submodules are bound on first use, through the module ``__getattr__`` below
(PEP 562): those submodules import numpy, OpenCV and pycolmap, which take
hundreds of milliseconds on a warm disk cache and seconds on a cold one, and a
program that imports one part of the package, such as ``sfm explorer`` or
``sfm --help``, should not pay for the rest. ``from sfmtool import X``,
``sfmtool.X``, ``from sfmtool import *`` and ``dir(sfmtool)`` all see the same
names either way.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

# The root-level `_sfmtool` names; everything else lives on a submodule.
# `run_explorer`, also root-level, is left out: `sfm explorer` calls it.
from sfmtool._sfmtool import THUMBNAIL_SIZE, ProgressCounter, build_profile  # noqa: F401
from sfmtool._sfmtool.reconstruction import *  # noqa: F401, F403
from sfmtool._sfmtool.patches import *  # noqa: F401, F403
from sfmtool._sfmtool.geometry import *  # noqa: F401, F403
from sfmtool._sfmtool.io import *  # noqa: F401, F403
from sfmtool._sfmtool.sift import *  # noqa: F401, F403
from sfmtool._sfmtool.matching import *  # noqa: F401, F403
from sfmtool._sfmtool.analysis import *  # noqa: F401, F403
from sfmtool._sfmtool.flow import *  # noqa: F401, F403
from sfmtool._sfmtool.spatial import *  # noqa: F401, F403
from sfmtool._sfmtool.spherical import *  # noqa: F401, F403

# Each name bound on first use, and the submodule it is read from.
_LAZY_NAMES = {
    "sfmtool._filenames": (
        "expand_paths",
        "normalize_workspace_path",
        "number_from_filename",
    ),
    "sfmtool._workspace": (
        "find_workspace_for_path",
        "init_workspace",
        "load_workspace_config",
    ),
    "sfmtool.sift.file": (
        "SiftExtractionError",
        "SiftReader",
        "compute_orientation",
        "draw_sift_features",
        "feature_size",
        "feature_size_x",
        "feature_size_y",
        "get_feature_tool_xxh128",
        "get_feature_type_for_tool",
        "get_sift_path_for_image",
        "get_used_features_from_reconstruction",
        "image_files_to_sift_files",
        "image_files_to_sift_files_opencv",
        "write_sift",
        "xxh128_of_file",
    ),
    "sfmtool.sift.extract_colmap": (
        "extract_sift_with_colmap",
        "get_colmap_feature_options",
        "read_colmap_db_sift",
    ),
    "sfmtool.sift.extract_opencv": (
        "extract_sift_with_opencv",
        "get_default_opencv_feature_options",
        "opencv_keypoint_to_affine_shape",
    ),
    "sfmtool.rig.spherical_tile": ("resample_atlas_to_equirect",),
}
_LAZY = {name: module for module, names in _LAZY_NAMES.items() for name in names}

# Subpackages that are bound as attributes of the package root on first read,
# as they also are once one of the names above has imported them.
_LAZY_SUBPACKAGES = ("rig", "sift")

# Where a Python submodule defines a name that an extension submodule also
# exports, the package root gives the Python one: `sfmtool.write_sift` is the
# function in `sfmtool.sift.file`, which validates its arguments before calling
# the `_sfmtool.io` binding of the same name. Unbind each such extension name so
# it goes through `__getattr__`.
for _name in _LAZY.keys() & globals().keys():
    del globals()[_name]
del _name

# Type checkers and editors do not run `__getattr__`, so they read the lazy
# names from these imports, which never run. `tests/test_lazy_loading.py`
# checks that they match `_LAZY_NAMES` and `_LAZY_SUBPACKAGES`.
if TYPE_CHECKING:
    from sfmtool import rig, sift  # noqa: F401
    from sfmtool._filenames import (  # noqa: F401
        expand_paths,
        normalize_workspace_path,
        number_from_filename,
    )
    from sfmtool._workspace import (  # noqa: F401
        find_workspace_for_path,
        init_workspace,
        load_workspace_config,
    )
    from sfmtool.rig.spherical_tile import resample_atlas_to_equirect  # noqa: F401
    from sfmtool.sift.extract_colmap import (  # noqa: F401
        extract_sift_with_colmap,
        get_colmap_feature_options,
        read_colmap_db_sift,
    )
    from sfmtool.sift.extract_opencv import (  # noqa: F401
        extract_sift_with_opencv,
        get_default_opencv_feature_options,
        opencv_keypoint_to_affine_shape,
    )
    from sfmtool.sift.file import (  # noqa: F401
        SiftExtractionError,
        SiftReader,
        compute_orientation,
        draw_sift_features,
        feature_size,
        feature_size_x,
        feature_size_y,
        get_feature_tool_xxh128,
        get_feature_type_for_tool,
        get_sift_path_for_image,
        get_used_features_from_reconstruction,
        image_files_to_sift_files,
        image_files_to_sift_files_opencv,
        write_sift,
        xxh128_of_file,
    )
del TYPE_CHECKING

__all__ = sorted(
    {name for name in globals() if not name.startswith("_")}
    | set(_LAZY)
    | set(_LAZY_SUBPACKAGES)
)


def __getattr__(name):
    if name in _LAZY:
        value = getattr(_import_module(_LAZY[name]), name)
    elif name in _LAZY_SUBPACKAGES:
        value = _import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(_LAZY_SUBPACKAGES))
