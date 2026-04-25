# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Camera config (`camera_config.json`) loading and per-directory resolution.

A `camera_config.json` file commits explicit camera intrinsics to a workspace
so subsequent solves start from known-good values instead of EXIF guesses.
Resolution is closest-ancestor-wins, capped at the workspace root.

See `specs/workspace/camera-config.md` for the file format and semantics.
"""

import json
from pathlib import Path

from ._cameras import _CAMERA_PARAM_NAMES, FOCAL_PRINCIPAL_PARAM_NAMES


class CameraConfigError(Exception):
    """Raised when a camera_config.json file is malformed or unusable."""


_TOP_LEVEL_KEYS = {"version", "camera_intrinsics"}
_INTRINSICS_KEYS = {"model", "width", "height", "parameters"}


def load_camera_config(path: Path) -> dict | None:
    """Load and validate a camera_config.json file.

    Returns the `camera_intrinsics` block (which may itself be `None` if the
    file omits it). Raises `CameraConfigError` on schema violations.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise CameraConfigError(f"Invalid JSON in {path}: {e}")

    if not isinstance(data, dict):
        raise CameraConfigError(f"{path}: top level must be an object")

    unknown = set(data.keys()) - _TOP_LEVEL_KEYS
    if unknown:
        raise CameraConfigError(f"{path}: unknown top-level keys: {sorted(unknown)}")

    version = data.get("version")
    if version != 1:
        raise CameraConfigError(f"{path}: unsupported version {version!r} (expected 1)")

    intrinsics = data.get("camera_intrinsics")
    if intrinsics is None:
        return None

    if not isinstance(intrinsics, dict):
        raise CameraConfigError(f"{path}: 'camera_intrinsics' must be an object")

    unknown = set(intrinsics.keys()) - _INTRINSICS_KEYS
    if unknown:
        raise CameraConfigError(
            f"{path}: unknown keys in 'camera_intrinsics': {sorted(unknown)}"
        )

    model = intrinsics.get("model")
    if model is None:
        raise CameraConfigError(f"{path}: 'camera_intrinsics.model' is required")
    if model not in _CAMERA_PARAM_NAMES:
        raise CameraConfigError(
            f"{path}: unknown camera model {model!r}; must be one of "
            f"{sorted(_CAMERA_PARAM_NAMES.keys())}"
        )

    parameters = intrinsics.get("parameters")
    if parameters is not None:
        if not isinstance(parameters, dict):
            raise CameraConfigError(f"{path}: 'parameters' must be an object")
        param_names = set(_CAMERA_PARAM_NAMES[model])
        unknown_params = set(parameters.keys()) - param_names
        if unknown_params:
            raise CameraConfigError(
                f"{path}: unknown parameters for model {model!r}: "
                f"{sorted(unknown_params)}"
            )
        has_focal_or_principal = any(
            name in parameters for name in FOCAL_PRINCIPAL_PARAM_NAMES
        )
        if has_focal_or_principal:
            if "width" not in intrinsics or "height" not in intrinsics:
                raise CameraConfigError(
                    f"{path}: 'width' and 'height' are required when "
                    f"'parameters' includes a focal length or principal point"
                )

    width = intrinsics.get("width")
    height = intrinsics.get("height")
    if (width is None) != (height is None):
        raise CameraConfigError(f"{path}: 'width' and 'height' must be set together")
    if width is not None and (not isinstance(width, int) or width <= 0):
        raise CameraConfigError(f"{path}: 'width' must be a positive integer")
    if height is not None and (not isinstance(height, int) or height <= 0):
        raise CameraConfigError(f"{path}: 'height' must be a positive integer")

    return intrinsics


def find_camera_config_for_directory(
    image_dir: Path, workspace_dir: Path
) -> tuple[Path, dict] | None:
    """Walk from `image_dir` up to (and including) `workspace_dir`, returning
    the first `camera_config.json` found.

    Returns `(file_path, parsed_intrinsics_dict)` or `None` if no file is
    found or `image_dir` is not inside `workspace_dir`. Stops cleanly without
    raising when `image_dir` is outside the workspace.
    """
    image_dir = image_dir.resolve()
    workspace_dir = workspace_dir.resolve()

    try:
        image_dir.relative_to(workspace_dir)
    except ValueError:
        return None

    current = image_dir
    while True:
        candidate = current / "camera_config.json"
        if candidate.exists():
            intrinsics = load_camera_config(candidate)
            if intrinsics is not None:
                return (candidate, intrinsics)
        if current == workspace_dir:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


class CameraConfigResolver:
    """Per-directory cache for `camera_config.json` resolution.

    Build one fresh per CLI invocation. The first lookup in any directory
    walks the parent chain up to the workspace root; every subsequent lookup
    in that same directory is an O(1) cache hit.
    """

    def __init__(self, workspace_dir: Path | None):
        self._workspace_dir = (
            Path(workspace_dir).resolve() if workspace_dir is not None else None
        )
        self._cache: dict[Path, tuple[Path, dict] | None] = {}

    @property
    def workspace_dir(self) -> Path | None:
        return self._workspace_dir

    def resolve_for_image(self, image_path: Path) -> tuple[Path, dict] | None:
        """Resolve the `camera_config.json` for an image, by its parent directory."""
        return self.resolve_for_directory(Path(image_path).parent)

    def resolve_for_directory(self, image_dir: Path) -> tuple[Path, dict] | None:
        """Resolve the `camera_config.json` for an image directory."""
        if self._workspace_dir is None:
            return None
        key = Path(image_dir).resolve()
        if key in self._cache:
            return self._cache[key]
        result = find_camera_config_for_directory(key, self._workspace_dir)
        self._cache[key] = result
        return result
