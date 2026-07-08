# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Optional


def init_workspace(
    workspace_dir: str | Path,
    *,
    feature_tool: str = "sfmtool",
    domain_size_pooling: bool = False,
    max_num_features: int | None = None,
    estimate_affine_shape: bool = False,
    use_gpu: bool = True,
) -> dict:
    """Initialize an SfM workspace.

    Creates a .sfm-workspace.json file with feature extraction settings.

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

    Returns:
        The workspace configuration dict that was written.
    """
    from sfmtool.sift.extract_colmap import get_colmap_feature_options
    from sfmtool.sift.extract_opencv import get_default_opencv_feature_options
    from sfmtool.sift.extract_sfmtool import get_default_sfmtool_feature_options
    from sfmtool.sift.file import get_feature_tool_xxh128, get_feature_type_for_tool

    workspace_dir = Path(workspace_dir)
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
        "version": 1,
        "feature_tool": feature_tool_lower,
        "feature_type": feature_type,
        "feature_options": feature_options,
        "feature_prefix_dir": feature_prefix_dir,
    }

    config_path = workspace_dir / ".sfm-workspace.json"
    with open(config_path, "w") as f:
        json.dump(workspace_config, f, indent=2)

    return workspace_config


def find_workspace_for_path(path: Path) -> Optional[Path]:
    """
    Find the workspace directory by searching upward for .sfm-workspace.json.

    Args:
        path: A path within the workspace (file or directory)

    Returns:
        Workspace directory path, or None if no workspace found
    """
    # Start from the directory containing the path
    if path.is_file():
        current = path.parent
    else:
        current = path

    # Search upward through parent directories
    while True:
        workspace_file = current / ".sfm-workspace.json"
        if workspace_file.exists():
            return current

        # Check if we've reached the root
        parent = current.parent
        if parent == current:
            return None
        current = parent


def find_sfmr_by_content_hash(workspace: Path, hash_prefix: str) -> Optional[Path]:
    """First .sfmr under `workspace` whose content hash starts with `hash_prefix`.

    Search order follows the sfmr-format spec: the conventional ``sfmr/``
    subdirectory first, then the workspace root, then the rest of the tree
    (skipping hidden directories). Reading each candidate's hash decompresses
    only ``content_hash.json.zst``, not the reconstruction data.

    Returns the matching path, or None if no reconstruction matches.
    """
    from ._sfmtool.io import read_sfmr_content_hash

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

    Args:
        workspace_dir: Path to workspace directory

    Returns:
        Dictionary with workspace configuration fields

    Raises:
        RuntimeError: If config file is missing or invalid
    """
    config_file = workspace_dir / ".sfm-workspace.json"

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

    # Validate required fields
    if "feature_tool" not in config or "feature_options" not in config:
        raise RuntimeError(
            f"Invalid workspace config at {config_file}: "
            "missing required fields 'feature_tool' or 'feature_options'"
        )

    return config
