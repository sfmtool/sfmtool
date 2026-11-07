# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfm version`: the line it prints, and the two `git describe` strings it
reads, checked without depending on the state of this repository."""

import re
import shutil
import subprocess

import pytest
from click.testing import CliRunner

import sfmtool
from sfmtool._version import checkout_describe, version_text
from sfmtool.cli import main

DESCRIBE = re.compile(r"(.+-\d+-g)?[0-9a-f]{7,}(-dirty)?")


def test_release_prints_the_version_alone():
    release = "v0.3-0-g276e8d986"
    assert version_text("0.3", None, None) == "sfmtool 0.3"
    assert version_text("0.3", release, release) == "sfmtool 0.3"
    assert version_text("0.3", None, release) == "sfmtool 0.3"
    assert version_text("0.3", release, None) == "sfmtool 0.3"
    assert version_text("0.3", "0.3-0-g276e8d986", None) == "sfmtool 0.3"


def test_past_the_tag_prints_the_describe():
    past = "v0.3-385-g276e8d986"
    assert version_text("0.3", past, past) == f"sfmtool 0.3 ({past})"
    assert version_text("0.3", past, None) == f"sfmtool 0.3 ({past})"
    assert version_text("0.3", None, past) == f"sfmtool 0.3 ({past})"
    # A shallow clone with no tag: `--always` gives the hash alone.
    assert version_text("0.3", "276e8d986", None) == "sfmtool 0.3 (276e8d986)"


def test_a_tag_other_than_the_version_is_printed():
    other = "v0.4-0-g276e8d986"
    assert version_text("0.3", other, other) == f"sfmtool 0.3 ({other})"
    assert version_text("0.3", "v0.30-0-g276e8d986", None) == (
        "sfmtool 0.3 (v0.30-0-g276e8d986)"
    )


def test_dirty_is_printed_even_at_the_tag():
    dirty = "v0.3-0-g276e8d986-dirty"
    assert version_text("0.3", dirty, dirty) == f"sfmtool 0.3 ({dirty})"
    past_dirty = "v0.3-385-g276e8d986-dirty"
    assert version_text("0.3", past_dirty, past_dirty) == (
        f"sfmtool 0.3 ({past_dirty})"
    )


def test_extension_differing_from_the_checkout_prints_both():
    assert version_text("0.3", "v0.3-386-g1a2b3c4d5", "v0.3-385-g276e8d986") == (
        "sfmtool 0.3 (checkout v0.3-386-g1a2b3c4d5, "
        "extension built from v0.3-385-g276e8d986)"
    )
    # Both labelled even when one of them is the release.
    assert version_text("0.3", "v0.3-0-g276e8d986-dirty", "v0.3-0-g276e8d986") == (
        "sfmtool 0.3 (checkout v0.3-0-g276e8d986-dirty, "
        "extension built from v0.3-0-g276e8d986)"
    )


def test_checkout_describe_skips_a_directory_outside_a_checkout(tmp_path, monkeypatch):
    """No `.git` two levels above the package: git is not run at all."""
    package_dir = tmp_path / "src" / "sfmtool"
    package_dir.mkdir(parents=True)

    def no_subprocess(*args, **kwargs):
        raise AssertionError("git was run outside a checkout")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    assert checkout_describe(package_dir) is None


def test_checkout_describe_skips_a_failing_git(tmp_path):
    """A `.git` that is not a repository: git exits with an error."""
    package_dir = tmp_path / "src" / "sfmtool"
    package_dir.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    assert checkout_describe(package_dir) is None


def test_checkout_describe_skips_a_missing_git(tmp_path, monkeypatch):
    """No `git` on PATH: starting it fails, and that is not an error."""
    package_dir = tmp_path / "src" / "sfmtool"
    package_dir.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert checkout_describe(package_dir) is None


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_checkout_describe_reads_a_real_checkout(tmp_path):
    """In a repository laid out as `<root>/src/sfmtool`, at a tag and then
    with an uncommitted change."""
    package_dir = tmp_path / "src" / "sfmtool"
    package_dir.mkdir(parents=True)
    module = package_dir / "__init__.py"
    module.write_text("x = 1\n", encoding="utf-8")

    def git(*args):
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "first")
    git("tag", "v0.3")
    describe = checkout_describe(package_dir)
    assert re.fullmatch(r"v0\.3-0-g[0-9a-f]+", describe), describe
    assert version_text("0.3", describe, None) == "sfmtool 0.3"

    module.write_text("x = 2\n", encoding="utf-8")
    dirty = checkout_describe(package_dir)
    assert dirty == f"{describe}-dirty"
    assert version_text("0.3", dirty, None) == f"sfmtool 0.3 ({dirty})"


def test_extension_records_its_build_describe():
    """The extension under test was built from this repository's git
    checkout, so `build.rs` recorded a describe."""
    describe = sfmtool.build_git_describe()
    assert isinstance(describe, str)
    assert DESCRIBE.fullmatch(describe), describe


def test_version_command_prints_one_line():
    result = CliRunner().invoke(main, ["version"])
    assert result.exit_code == 0, result.output
    assert re.fullmatch(r"sfmtool \S+( \(.+\))?\n", result.output), result.output
