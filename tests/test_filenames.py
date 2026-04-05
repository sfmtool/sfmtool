# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from sfmtool._filenames import (
    expand_paths,
    normalize_workspace_path,
    number_from_filename,
)


@pytest.mark.parametrize(
    "filename, expected",
    [
        # Original patterns (name#.ext)
        ("file1.jpg", 1),
        ("file123.jpg", 123),
        ("photo_007.png", 7),
        ("image42", 42),
        (Path("some/dir/image99.tiff"), 99),
        # Parenthesized patterns (name(#).ext)
        ("file (1).jpg", 1),
        ("Conch (1).jpg", 1),
        ("Conch (123).jpg", 123),
        ("photo (007).png", 7),
        ("image (42)", 42),
        (Path("some/dir/image (99).tiff"), 99),
        # Precedence: parenthesized takes priority
        ("file123(456).jpg", 456),
        ("image1(2).png", 2),
        # Invalid cases
        ("file.jpg", None),
        ("file1a.jpg", None),
        ("123file.txt", None),
        ("archive.123.tar.gz", None),
        ("no_extension", None),
        ("file().jpg", None),  # Empty parentheses
        ("file(abc).jpg", None),  # Non-numeric in parentheses
        ("file(1)(2).jpg", 2),  # Multiple groups: extracts last one
        ("", None),
    ],
)
def test_number_from_filename(filename, expected):
    assert number_from_filename(filename) == expected


def test_expand_paths(tmp_path: Path):
    """Tests that expand_paths correctly expands directories and filters files."""
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    (dir1 / "image1.jpg").touch()
    (dir1 / "image2.png").touch()
    (dir1 / "image3.JPG").touch()  # Uppercase extension
    (dir1 / "data.txt").touch()

    dir2 = tmp_path / "dir2"
    dir2.mkdir()
    (dir2 / "photo10.jpeg").touch()
    (dir2 / "photo11.jpg").touch()

    file_outside = tmp_path / "other_file.txt"
    file_outside.touch()

    # Case 1: Basic expansion of directories and files
    paths = [dir1, file_outside]
    result = expand_paths(paths)
    assert len(result) == 5  # 4 in dir1 + file_outside
    assert file_outside in result
    assert all(isinstance(p, Path) for p in result)

    # Case 2: Filter by extensions
    paths = [dir1, dir2]
    result = expand_paths(paths, extensions=(".jpg", ".jpeg"))
    assert set(result) == {
        dir1 / "image1.jpg",
        dir1 / "image3.JPG",
        dir2 / "photo10.jpeg",
        dir2 / "photo11.jpg",
    }

    # Case 3: Filter by numbers
    paths = [dir1, dir2]
    result = expand_paths(paths, numbers={1, 10})
    assert set(result) == {dir1 / "image1.jpg", dir2 / "photo10.jpeg"}

    # Case 4: Filter individual file paths by numbers (e.g., from glob expansion)
    file1 = dir1 / "image1.jpg"
    file2 = dir1 / "image2.png"
    file10 = dir2 / "photo10.jpeg"
    paths = [file1, file2, file10]
    result = expand_paths(paths, numbers={1, 10})
    assert set(result) == {file1, file10}


def test_expand_paths_recursive(tmp_path: Path):
    """Tests that expand_paths recursively finds images in subdirectories."""
    # Simulate rig layout: images/fisheye_left/, images/fisheye_right/
    images_dir = tmp_path / "images"
    left_dir = images_dir / "fisheye_left"
    right_dir = images_dir / "fisheye_right"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)

    (left_dir / "frame_000001.jpg").touch()
    (left_dir / "frame_000002.jpg").touch()
    (right_dir / "frame_000001.jpg").touch()
    (right_dir / "frame_000002.jpg").touch()

    # Also put a non-image file in a subdirectory
    (left_dir / "thumbs.db").touch()

    # Case 1: Recursive discovery finds all files
    result = expand_paths([images_dir])
    assert len(result) == 5  # 4 jpg + 1 thumbs.db

    # Case 2: Recursive discovery with extension filter
    result = expand_paths([images_dir], extensions=(".jpg", ".jpeg"))
    assert len(result) == 4
    assert left_dir / "frame_000001.jpg" in result
    assert right_dir / "frame_000002.jpg" in result

    # Case 3: Recursive discovery with number filter
    result = expand_paths([images_dir], extensions=(".jpg",), numbers={1})
    assert set(result) == {
        left_dir / "frame_000001.jpg",
        right_dir / "frame_000001.jpg",
    }

    # Case 4: Deeper nesting
    deep_dir = images_dir / "seq1" / "sensor_a"
    deep_dir.mkdir(parents=True)
    (deep_dir / "img1.png").touch()
    result = expand_paths([images_dir], extensions=(".png",))
    assert deep_dir / "img1.png" in result


class TestNormalizeWorkspacePath:
    """Tests for normalize_workspace_path function."""

    def test_simple_relative_path(self):
        """Test that simple relative paths are preserved."""
        assert normalize_workspace_path("images/img001.jpg") == "images/img001.jpg"
        assert (
            normalize_workspace_path("data/subset/file.txt") == "data/subset/file.txt"
        )
        assert normalize_workspace_path("file.jpg") == "file.jpg"

    def test_path_object_input(self):
        """Test that Path objects are accepted."""
        assert (
            normalize_workspace_path(Path("images/img001.jpg")) == "images/img001.jpg"
        )

    def test_windows_backslashes_converted(self):
        """Test that Windows backslashes are converted to POSIX forward slashes."""
        assert normalize_workspace_path("images\\img001.jpg") == "images/img001.jpg"
        assert (
            normalize_workspace_path("data\\subset\\file.txt") == "data/subset/file.txt"
        )

    def test_mixed_separators(self):
        """Test that mixed separators are normalized."""
        assert (
            normalize_workspace_path("images\\subset/file.txt")
            == "images/subset/file.txt"
        )

    def test_dotdot_normalization_ok(self):
        """Test that internal '..' are normalized when they don't escape."""
        # "images/../file.jpg" -> "file.jpg"
        assert normalize_workspace_path("images/../file.jpg") == "file.jpg"
        # "a/b/../c/file.jpg" -> "a/c/file.jpg"
        assert normalize_workspace_path("a/b/../c/file.jpg") == "a/c/file.jpg"

    def test_dot_normalization(self):
        """Test that '.' components are removed."""
        assert normalize_workspace_path("./images/file.jpg") == "images/file.jpg"
        assert normalize_workspace_path("images/./file.jpg") == "images/file.jpg"
        assert normalize_workspace_path("./file.jpg") == "file.jpg"

    def test_double_slashes(self):
        """Test that double slashes are collapsed."""
        assert normalize_workspace_path("images//file.jpg") == "images/file.jpg"
        assert normalize_workspace_path("a///b//c/file.jpg") == "a/b/c/file.jpg"

    def test_trailing_slash_removed(self):
        """Test that trailing slashes are handled."""
        # Trailing slashes in directory paths
        assert normalize_workspace_path("images/") == "images"

    def test_escape_attempt_leading_dotdot(self):
        """Test that paths starting with '..' are rejected."""
        with pytest.raises(ValueError, match="cannot escape workspace"):
            normalize_workspace_path("../outside/file.jpg")

        with pytest.raises(ValueError, match="cannot escape workspace"):
            normalize_workspace_path("../../file.jpg")

    def test_escape_attempt_after_normalization(self):
        """Test that paths that escape after normalization are rejected."""
        # "a/../../../b.jpg" normalizes to "../b.jpg" which escapes
        with pytest.raises(ValueError, match="cannot escape workspace"):
            normalize_workspace_path("a/../../b.jpg")

    def test_absolute_unix_path_rejected(self):
        """Test that absolute Unix paths are rejected."""
        with pytest.raises(ValueError, match="must be relative"):
            normalize_workspace_path("/home/user/file.jpg")

        with pytest.raises(ValueError, match="must be relative"):
            normalize_workspace_path("/etc/passwd")

    def test_absolute_windows_path_rejected(self):
        """Test that absolute Windows paths are rejected."""
        with pytest.raises(ValueError, match="must be relative"):
            normalize_workspace_path("C:\\Users\\file.jpg")

        with pytest.raises(ValueError, match="must be relative"):
            normalize_workspace_path("D:\\data\\file.txt")

    def test_absolute_windows_backslash_path_rejected(self):
        """Test that Windows paths starting with backslash are rejected."""
        # Paths starting with single backslash are absolute on Windows
        # (relative to root of current drive)
        with pytest.raises(ValueError, match="must be relative"):
            normalize_workspace_path("\\home\\path\\to\\file.jpg")

        with pytest.raises(ValueError, match="must be relative"):
            normalize_workspace_path("\\data\\file.txt")

    def test_windows_unc_path_rejected(self):
        """Test that Windows UNC paths are rejected."""
        with pytest.raises(ValueError, match="must be relative"):
            normalize_workspace_path("\\\\server\\share\\file.jpg")

    def test_windows_drive_letter_without_backslash(self):
        """Test that Windows partial relative paths like 'D:file' are rejected."""
        with pytest.raises(ValueError, match="Windows drive path"):
            normalize_workspace_path("D:file.jpg")

        with pytest.raises(ValueError, match="Windows drive path"):
            normalize_workspace_path("C:data/file.txt")

    def test_empty_path(self):
        """Test handling of empty paths."""
        # Empty string normalizes to empty string
        assert normalize_workspace_path("") == ""

    def test_complex_valid_path(self):
        """Test a complex but valid path."""
        # This should normalize: "a/b/c/../d/./e" -> "a/b/d/e"
        assert normalize_workspace_path("a/b/c/../d/./e/file.jpg") == "a/b/d/e/file.jpg"
