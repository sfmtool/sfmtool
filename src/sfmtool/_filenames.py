# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

import ntpath
import posixpath
import re
from pathlib import Path
from typing import Iterable

__all__ = ["number_from_filename", "expand_paths", "normalize_workspace_path"]


_NUMBER_FROM_FILENAME_RE = re.compile(r"\((\d+)\)$|(\d+)$")


def number_from_filename(filename: str | Path) -> int | None:
    """
    If a file is numbered, extract the number and returns it as an integer.
    The number of a file is all the digits next to its extension, either
    in parentheses or directly adjacent.

    Parenthesized numbers take precedence over non-parenthesized numbers.

    "file1.jpg" -> 1
    "file123.jpg" -> 123
    "file (1).jpg" -> 1
    "file (123).jpg" -> 123
    "file123(456).jpg" -> 456  (parenthesized takes precedence)
    "file.jpg" -> None
    """
    match = _NUMBER_FROM_FILENAME_RE.search(Path(filename).stem)
    if match:
        # Group 1 is for parenthesized numbers, group 2 is for non-parenthesized
        return int(match.group(1) or match.group(2))
    return None


def expand_paths(
    paths: Iterable[str | Path],
    *,
    extensions: tuple[str, ...] | None = None,
    numbers: Iterable[int] | None = None,
) -> list[Path]:
    """
    Given a list of file and directory paths, keeps all the files and expands
    the directories recursively. If extensions or numbers are not None, they
    filter the list of paths from directories.
    """
    if numbers and not isinstance(numbers, set):
        numbers = set(numbers)

    filenames: list[Path] = []

    for path_item in paths:
        path = Path(path_item)

        if path.is_dir():
            # Recursively iterate over directory contents
            try:
                dir_contents = [p for p in path.rglob("*") if p.is_file()]

                # Filter by extension if needed
                if extensions:
                    # extensions are typically passed as (".jpg", ".png")
                    # path.suffix includes the dot
                    dir_contents = [
                        p
                        for p in dir_contents
                        if p.suffix.lower() in extensions
                        or p.name.lower().endswith(extensions)
                    ]

                # Filter by number if needed
                if numbers:
                    dir_contents = [
                        p for p in dir_contents if number_from_filename(p) in numbers
                    ]

                filenames.extend(dir_contents)
            except OSError:
                # Handle permission errors or other OS errors gracefully
                continue
        else:
            # It's a file (or doesn't exist, but we treat as file path)
            if numbers and number_from_filename(path) not in numbers:
                continue
            filenames.append(path)

    return filenames


def normalize_workspace_path(relative_path: str | Path) -> str:
    """
    Validate and normalize a relative path for use within a workspace.

    This function ensures that a path is:
    1. Relative (not absolute)
    2. Uses POSIX separators (forward slashes)
    3. Does not escape the workspace using ".."

    Args:
        relative_path: A path that should be relative to a workspace

    Returns:
        Normalized path string with POSIX separators

    Raises:
        ValueError: If the path is absolute or attempts to escape the workspace

    Examples:
        >>> normalize_workspace_path("images/img001.jpg")
        'images/img001.jpg'

        >>> normalize_workspace_path("images/../img001.jpg")
        'img001.jpg'

        >>> normalize_workspace_path("../outside/file.jpg")
        Traceback (most recent call last):
        ...
        ValueError: Path cannot escape workspace directory. Got path with leading '..': ../outside/file.jpg
    """
    path_str = str(relative_path)

    # 1. Check that the path is relative, not absolute
    # This catches:
    # - Unix absolute paths: "/home/user/file"
    # - Windows absolute paths: "C:\\path\\to\\file"
    # - Windows UNC paths: "\\\\server\\share\\file"
    # - Windows partial relative paths: "D:file\\path" (drive letter without backslash)

    # Check for Unix/POSIX absolute paths using posixpath for cross-platform validation
    if posixpath.isabs(path_str):
        raise ValueError(
            f"Path must be relative to workspace, not absolute. Got: {path_str}"
        )

    # Check for Windows absolute paths using ntpath for cross-platform validation
    # This catches: "C:\path" and "\\server\share" (UNC)
    if ntpath.isabs(path_str):
        raise ValueError(
            f"Path must be relative to workspace, not absolute. Got: {path_str}"
        )

    # Check for Windows paths starting with backslash (absolute from current drive root)
    # These are like "\home\path" which are absolute on Windows but not caught by ntpath.isabs
    # on all platforms
    if path_str.startswith("\\"):
        raise ValueError(
            f"Path must be relative to workspace, not absolute. Got: {path_str}"
        )

    # Check for Windows drive-relative paths like "D:file"
    # These are not considered absolute by ntpath but are platform-specific and problematic
    # They refer to the current directory on a specific drive, which is ambiguous
    if len(path_str) >= 2 and path_str[1] == ":" and not ntpath.isabs(path_str):
        raise ValueError(
            f"Path must be relative to workspace, not a Windows drive path. "
            f"Got: {path_str}"
        )

    # 2. Convert Windows path separators to POSIX
    # Replace all backslashes with forward slashes
    path_str = path_str.replace("\\", "/")

    # 3. Normalize the relative path by collapsing ".." and "."
    # We manually process the path to collapse ".." while preserving POSIX separators

    # Split the path and process components
    parts = path_str.split("/")
    normalized_parts = []

    for part in parts:
        if part == "..":
            # Try to go up one level
            # Only pop if we have a real directory to go up from (not another '..')
            if normalized_parts and normalized_parts[-1] != "..":
                normalized_parts.pop()
            else:
                # Can't resolve this '..', keep it (it means escaping the workspace)
                normalized_parts.append(part)
        elif part == "." or part == "":
            # Skip "." and empty parts (from double slashes)
            continue
        else:
            normalized_parts.append(part)

    # Reconstruct the path
    normalized = "/".join(normalized_parts)

    # 4. Validate that the path does not start with ".."
    # After normpath, any remaining ".." at the start means the path escapes the workspace
    if normalized.startswith(".."):
        raise ValueError(
            f"Path cannot escape workspace directory. "
            f"Got path with leading '..': {normalized}\n"
            f"Original path: {relative_path}"
        )

    # Also check for ".." after a slash (shouldn't happen after normalization, but be safe)
    if "/.." in normalized:
        raise ValueError(
            f"Path cannot escape workspace directory. "
            f"Got path with internal '..': {normalized}\n"
            f"Original path: {relative_path}"
        )

    # 5. Return the normalized and validated result
    return normalized
