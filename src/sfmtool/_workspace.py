# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

# The file that marks a directory as a workspace and holds its feature
# configuration.
WORKSPACE_CONFIG_FILE = ".sfm-workspace.json"


# The last sfmtool release that upgrades a version 1 workspace, which the
# refusal of one names; `WORKSPACE_LAST_RELEASE_READING_VERSION_1` in
# `sfmtool_core::workspace`, which `sfm ws upgrade`'s refusal names.
_LAST_RELEASE_READING_VERSION_1 = "0.4.1"


class Version1WorkspaceError(RuntimeError):
    """A command that uses a workspace was given a version 1 workspace, which
    only sfmtool 0.4.1's ``sfm ws upgrade`` and ``sfm ws init --force`` act
    on."""


def _version_1_workspace_error(config_file: Path) -> Version1WorkspaceError:
    """The one refusal of a version 1 workspace, naming its file."""
    return Version1WorkspaceError(
        f"The workspace file {config_file} is version 1, and sfmtool works only "
        f"in a version 2 workspace. sfmtool {_LAST_RELEASE_READING_VERSION_1} is "
        "the last release that upgrades a version 1 workspace: install it, run "
        "`sfm ws upgrade` on the workspace, then use the workspace with this "
        "release."
    )


def require_version_2_workspace(workspace_dir: str | Path) -> None:
    """Refuse a version 1 workspace, for a command that uses the workspace's
    files without its settings, such as one that reads the ``.sift`` files a
    ``.sfmr`` names.

    Raises ``Version1WorkspaceError`` when ``workspace_dir`` holds a
    ``.sfm-workspace.json`` whose ``version`` is 1, with the message
    ``load_workspace_config`` gives. A directory with no workspace file, or one
    that cannot be read, passes: the check is about the version only, and the
    reader that needs the file reports what is wrong with it.
    """
    config_file = Path(workspace_dir) / WORKSPACE_CONFIG_FILE
    try:
        with open(config_file) as f:
            config = json.load(f)
    except OSError, ValueError:
        return
    if isinstance(config, dict):
        version = config.get("version")
        if version == 1 and not isinstance(version, bool):
            raise _version_1_workspace_error(config_file)


def init_workspace(
    workspace_dir: str | Path,
    *,
    feature_tool: str = "sfmtool",
    domain_size_pooling: bool = False,
    max_num_features: int | None = None,
    estimate_affine_shape: bool = False,
    use_gpu: bool = True,
    image_to_gray_formula: str | None = None,
    quantize_bits: int | None = None,
) -> dict:
    """Initialize an SfM workspace.

    Creates a version 2 .sfm-workspace.json file with feature extraction
    settings: the tool's default options, the given overrides, and the tool's
    ``image_to_gray`` object, with a new ``feature_prefix_dir`` named from
    them. An existing file is replaced; its folder and the named image lists
    file stay on disk. The ``sfmtool`` tool's file is written by
    ``sfmtool.workspace.write_sfmtool_workspace_marker``.

    Args:
        workspace_dir: The SfM workspace directory.
        feature_tool: Name of the feature tool (default: "sfmtool")
        domain_size_pooling: Enable domain size pooling for SIFT (default: False)
        max_num_features: Maximum number of features per image (COLMAP and
                         sfmtool backends). If None, uses the backend default
                         (8192).
        estimate_affine_shape: Enable affine shape estimation (COLMAP only, default: False).
        use_gpu: Use GPU SIFT extraction (COLMAP only, default: True). Persisted
                 into the config and honored at extraction time, but excluded
                 from the feature-cache hash.
        image_to_gray_formula: The grey conversion's formula (sfmtool only).
                 ``None`` keeps the default, BT.709 luma.
        quantize_bits: The bit depth, 1 to 16, the formula's value is rounded
                 to (sfmtool only). ``None`` for no rounding.

    Returns:
        The workspace configuration dict that was written.

    Raises:
        ValueError: for an unsupported tool, a formula that does not parse, a
            ``quantize_bits`` outside 1 to 16, or either of the two given with
            a tool other than sfmtool.
    """
    from sfmtool.sift.extract_colmap import get_colmap_feature_options
    from sfmtool.sift.extract_opencv import get_default_opencv_feature_options
    from sfmtool.workspace import feature_prefix_dir as named_feature_prefix_dir
    from sfmtool.workspace import (
        feature_type_for_tool,
        tool_image_to_gray,
        write_sfmtool_workspace_marker,
    )

    workspace_dir = Path(workspace_dir)

    image_to_gray = None
    if image_to_gray_formula is not None or quantize_bits is not None:
        if feature_tool.lower() != "sfmtool":
            raise ValueError(
                f"The {feature_tool} feature tool makes its own grey conversion, so "
                "image_to_gray and quantize_bits can only be set for sfmtool."
            )
        from sfmtool.sift import ImageToGray

        formula = image_to_gray_formula
        if formula is None:
            formula = tool_image_to_gray("sfmtool").formula
        image_to_gray = ImageToGray(formula, quantize_bits)

    if feature_tool.lower() == "sfmtool":
        # The one writer of a new sfmtool workspace is in Rust, which also
        # names its features folder.
        marker = write_sfmtool_workspace_marker(
            workspace_dir,
            max_num_features=max_num_features,
            image_to_gray=image_to_gray,
        )
        return json.loads(marker)

    workspace_dir.mkdir(parents=True, exist_ok=True)

    if feature_tool.lower() == "colmap":
        feature_options = get_colmap_feature_options(
            domain_size_pooling=domain_size_pooling,
            max_num_features=max_num_features,
            estimate_affine_shape=estimate_affine_shape,
            use_gpu=use_gpu,
        )
    elif feature_tool.lower() == "opencv":
        feature_options = get_default_opencv_feature_options()
    else:
        raise ValueError(f"Unsupported feature tool: {feature_tool}")

    feature_tool_lower = feature_tool.lower()
    feature_type = feature_type_for_tool(feature_tool_lower, feature_options)
    feature_prefix_dir = named_feature_prefix_dir(
        feature_tool_lower, feature_type, feature_options
    )

    workspace_config = {
        "version": 2,
        "feature_tool": feature_tool_lower,
        "feature_type": feature_type,
        "feature_options": feature_options,
        "feature_prefix_dir": feature_prefix_dir,
    }

    config_path = workspace_dir / WORKSPACE_CONFIG_FILE
    with open(config_path, "w") as f:
        json.dump(workspace_config, f, indent=2)

    return workspace_config


def find_sfmr_by_content_hash(workspace: Path, hash_prefix: str) -> Path | None:
    """First .sfmr under `workspace` whose content hash starts with `hash_prefix`.

    Search order follows the sfmtool-sfmr-format spec: the conventional ``sfmr/``
    subdirectory first, then the workspace root, then the rest of the tree
    (skipping hidden directories). Reading each candidate's hash decompresses
    only ``content_hash.json.zst``, not the reconstruction data.

    Returns the matching path, or None if no reconstruction matches.
    """
    from ._sfmtool.fileio import read_sfmr_content_hash

    def matches(path: Path) -> bool:
        try:
            return read_sfmr_content_hash(str(path))[:8].lower() == hash_prefix
        except Exception:
            return False

    # 1. The conventional sfmr/ subdirectory.
    sfmr_dir = workspace / "sfmr"
    if sfmr_dir.is_dir():
        for path in sorted(sfmr_dir.glob("*.sfmr")):
            if matches(path):
                return path
    # 2. The workspace root.
    for path in sorted(workspace.glob("*.sfmr")):
        if matches(path):
            return path
    # 3. The rest of the tree, skipping hidden / already-searched directories.
    for path in sorted(workspace.rglob("*.sfmr")):
        if path.parent == workspace or path.parent == sfmr_dir:
            continue
        if any(part.startswith(".") for part in path.relative_to(workspace).parts):
            continue
        if matches(path):
            return path
    return None


def load_workspace_config(workspace_dir: Path) -> dict:
    """
    Load workspace configuration from .sfm-workspace.json.

    This is the one reader of the file for every command that uses a
    workspace, and it requires version 2, so such a command stops before any
    work: a version 1 file raises ``Version1WorkspaceError``, whose message
    names the file and says to upgrade it with sfmtool 0.4.1. A version 2 file's
    ``feature_options`` must carry a valid ``image_to_gray`` object at their
    top level and no ``gray_formula``; these rules apply to the top level of
    ``feature_options`` only. The recorded ``feature_prefix_dir`` is returned
    as it is, never computed again.

    Args:
        workspace_dir: Path to workspace directory

    Returns:
        Dictionary with workspace configuration fields

    Raises:
        Version1WorkspaceError: If the file is version 1.
        RuntimeError: If config file is missing or invalid, naming the file
            and what is wrong
    """
    config_file = workspace_dir / WORKSPACE_CONFIG_FILE

    if not config_file.exists():
        raise RuntimeError(
            f"The SfM workspace {workspace_dir} is not initialized. "
            f"Initialize it with 'sfm ws init {workspace_dir}'."
        )

    try:
        with open(config_file) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in workspace config at {config_file}: {e}")

    def invalid(problem: str) -> RuntimeError:
        return RuntimeError(f"Invalid workspace config at {config_file}: {problem}")

    if not isinstance(config, dict):
        raise invalid("it is not a JSON object")

    # Validate required fields
    if "feature_tool" not in config or "feature_options" not in config:
        raise invalid("missing required fields 'feature_tool' or 'feature_options'")

    if "version" not in config:
        raise invalid("it has no 'version'")
    version = config["version"]
    if isinstance(version, bool) or version not in (1, 2):
        raise invalid(f"it is version {version!r}, and this build reads version 2")
    if version == 1:
        raise _version_1_workspace_error(config_file)
    feature_options = config["feature_options"]
    if not isinstance(feature_options, dict):
        raise invalid("its 'feature_options' is not a JSON object")
    if "gray_formula" in feature_options:
        raise invalid(
            "a version 2 file's feature_options carry 'gray_formula', which "
            "belongs to version 1; version 2 records the conversion as "
            "'image_to_gray'"
        )
    if "image_to_gray" not in feature_options:
        raise invalid("a version 2 file's feature_options carry no 'image_to_gray'")
    from sfmtool.sift import ImageToGray

    try:
        ImageToGray.from_json(feature_options["image_to_gray"])
    except ValueError as e:
        raise invalid(f"its feature_options 'image_to_gray' is not valid: {e}")

    return config
