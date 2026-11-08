# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

# The file that marks a directory as a workspace and holds its feature
# configuration.
WORKSPACE_CONFIG_FILE = ".sfm-workspace.json"

# The `.sfm-workspace.json` versions this build reads.
_READABLE_VERSIONS = (1, 2)


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
    file stay on disk.

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
    from sfmtool.sift.extract_sfmtool import get_default_sfmtool_feature_options
    from sfmtool.sift.file import get_feature_tool_xxh128, get_feature_type_for_tool

    workspace_dir = Path(workspace_dir)

    image_to_gray = None
    if image_to_gray_formula is not None or quantize_bits is not None:
        if feature_tool.lower() != "sfmtool":
            raise ValueError(
                f"The {feature_tool} feature tool makes its own grey conversion, so "
                "image_to_gray and quantize_bits can only be set for sfmtool."
            )
        from sfmtool.sift import ImageToGray
        from sfmtool.sift.extract_sfmtool import get_default_sfmtool_feature_options

        formula = image_to_gray_formula
        if formula is None:
            formula = get_default_sfmtool_feature_options()["image_to_gray"]["formula"]
        image_to_gray = ImageToGray(formula, quantize_bits).to_dict()

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
    elif feature_tool.lower() == "sfmtool":
        feature_options = get_default_sfmtool_feature_options(
            max_num_features=max_num_features,
            image_to_gray=image_to_gray,
        )
    else:
        raise ValueError(f"Unsupported feature tool: {feature_tool}")

    feature_tool_lower = feature_tool.lower()
    feature_type = get_feature_type_for_tool(feature_tool_lower, feature_options)
    feature_tool_xxh128 = get_feature_tool_xxh128(
        feature_tool_lower, feature_type, feature_options
    )
    feature_prefix_dir = f"features/{feature_type}-{feature_tool_xxh128}"

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


class WorkspaceUpgradeError(Exception):
    """``upgrade_workspace`` could not upgrade a workspace file, and changed
    nothing."""


def upgrade_workspace(path: str | Path) -> tuple[Path, dict, dict | None]:
    """Rewrite the version 1 workspace file found from ``path`` as version 2.

    The workspace is found by the upward search from ``path``. Every option is
    kept, hand-edited ones included; the ``sfmtool`` tool's ``gray_formula``
    becomes ``image_to_gray: {"formula": <gray_formula>}`` and a ``colmap`` or
    ``opencv`` workspace gains its tool's conversion
    (``sfmtool.workspace.upgrade_feature_options``); and a new
    ``feature_prefix_dir`` is named from the new options. The old folder, its
    ``.sift`` files and the named image lists file stay on disk.

    Returns:
        ``(config_path, old_config, new_config)``, with ``new_config`` None
        when the file is already version 2 and nothing was changed.

    Raises:
        WorkspaceUpgradeError: naming the file and what stops the upgrade: no
            workspace file, no ``version``, a version other than 1 or 2, a
            feature tool other than sfmtool, colmap or opencv (an
            ``sfmtool-undistort`` workspace is told to run ``sfm undistort``
            again), or a ``gray_formula`` that does not parse.
    """
    from sfmtool.sift.file import get_feature_tool_xxh128
    from sfmtool.workspace import find_workspace_for_path, upgrade_feature_options

    path = Path(path).resolve()
    workspace_dir = find_workspace_for_path(path)
    if workspace_dir is None:
        raise WorkspaceUpgradeError(
            f"No workspace file {WORKSPACE_CONFIG_FILE} was found in {path} or any "
            "directory above it."
        )
    config_path = workspace_dir / WORKSPACE_CONFIG_FILE
    try:
        with open(config_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise WorkspaceUpgradeError(f"{config_path} is not valid JSON: {e}") from e
    if not isinstance(config, dict) or "version" not in config:
        raise WorkspaceUpgradeError(f"{config_path} has no 'version'.")
    version = config["version"]
    if version == 2:
        return config_path, config, None
    if version != 1:
        raise WorkspaceUpgradeError(
            f"{config_path} is version {version!r}, and this build upgrades version 1 "
            "files to version 2."
        )
    for key in ("feature_tool", "feature_type", "feature_options"):
        if key not in config:
            raise WorkspaceUpgradeError(f"{config_path} has no '{key}'.")
    feature_tool = config["feature_tool"]
    try:
        feature_options = upgrade_feature_options(
            feature_tool, config["feature_options"]
        )
    except ValueError as e:
        raise WorkspaceUpgradeError(f"{config_path} cannot be upgraded: {e}.") from e

    feature_type = config["feature_type"]
    xxh128 = get_feature_tool_xxh128(feature_tool, feature_type, feature_options)
    new_config = dict(config)
    new_config["version"] = 2
    new_config["feature_options"] = feature_options
    new_config["feature_prefix_dir"] = f"features/{feature_type}-{xxh128}"
    with open(config_path, "w") as f:
        json.dump(new_config, f, indent=2)
    return config_path, config, new_config


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

    The file's ``version`` is checked here, so a file this build cannot use
    fails before any work: it must be 1 or 2. A version 2 file's
    ``feature_options`` must carry a valid ``image_to_gray`` object at their
    top level and no ``gray_formula``; a version 1 file's must carry no
    ``image_to_gray``. These rules apply to the top level of
    ``feature_options`` only. The recorded ``feature_prefix_dir`` is returned
    as it is, never computed again.

    Args:
        workspace_dir: Path to workspace directory

    Returns:
        Dictionary with workspace configuration fields

    Raises:
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
    if isinstance(version, bool) or version not in _READABLE_VERSIONS:
        raise invalid(
            f"it is version {version!r}, and this build reads versions 1 and 2"
        )
    feature_options = config["feature_options"]
    if not isinstance(feature_options, dict):
        raise invalid("its 'feature_options' is not a JSON object")
    if version == 2:
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
    elif "image_to_gray" in feature_options:
        raise invalid(
            "a version 1 file's feature_options carry 'image_to_gray', which belongs "
            "to version 2; run 'sfm ws upgrade' on a version 1 file instead of "
            "editing it"
        )

    return config
