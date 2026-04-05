# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Optional


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
            f"Initialize it with 'sfm init {workspace_dir}'."
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
