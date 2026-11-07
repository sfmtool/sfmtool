# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The text `sfm version` prints: the package version, and which commit the
running code comes from when that is not the release the version names.

specs/cli/other/version-command.md gives the rules.
"""

import os
import re
import subprocess
from pathlib import Path

# How long each git command of the run-time describe may take before it is
# given up on. Each takes a few tens of milliseconds on this repository.
GIT_TIMEOUT_S = 2.0


def checkout_describe(package_dir: Path) -> str | None:
    """`git describe` of the git checkout the `sfmtool` package is run from.

    `package_dir` is the directory of the `sfmtool` package. It is in a
    checkout when it is `<root>/src/sfmtool` and `<root>` holds `.git`, as in
    an editable install; then this returns the output of
    `git describe --tags --long --always` run in `<root>`, with `-dirty` added
    when `git status --porcelain --untracked-files=no` lists a change, as
    `build.rs` records it for the extension. It returns None when there is no
    `.git` there (an install into site-packages, even one inside some other
    repository), when git is not installed, and when git fails or runs longer
    than `GIT_TIMEOUT_S`.
    """
    root = package_dir.parent.parent
    if not (root / ".git").exists():
        return None
    describe = _git(root, "describe", "--tags", "--long", "--always")
    changes = _git(root, "status", "--porcelain", "--untracked-files=no")
    if describe is None or changes is None:
        return None
    return f"{describe}-dirty" if changes else describe


def _git(root: Path, *args: str) -> str | None:
    """The stripped standard output of `git -C <root> <args>`, or None when git
    cannot be started, exits with an error, or runs past `GIT_TIMEOUT_S`."""
    env = dict(os.environ)
    # Read-only queries: `git status` then leaves the index as it is, where
    # `git describe --dirty` would rewrite it. `-C` picks the repository, which
    # a GIT_DIR or GIT_WORK_TREE from the environment would override.
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            env=env,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def is_release_describe(describe: str, package_version: str) -> bool:
    """Whether `describe` is a clean tree exactly at the release tag of
    `package_version`: `v<version>-0-g<hash>` (or `<version>-0-g<hash>`)."""
    tag = re.escape(package_version)
    return re.fullmatch(rf"v?{tag}-0-g[0-9a-f]+", describe) is not None


def sfmtool_facts() -> dict:
    """The version facts `sfm version --bare` prints, as the `sfmtool` object
    of the JSON report: the package metadata version (`"unknown"` without
    metadata), the checkout's and the extension's `git describe` (each None
    when unknown), the extension's build profile and the line `version_text`
    makes of them. It runs git, but imports nothing beyond the extension."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version

    from ._sfmtool import _build_git_describe, build_profile

    try:
        version = package_version("sfmtool")
    except PackageNotFoundError:
        version = "unknown"
    checkout = checkout_describe(Path(__file__).resolve().parent)
    extension = _build_git_describe()
    return {
        "version": version,
        "line": version_text(version, checkout, extension),
        "checkout_describe": checkout,
        "extension_describe": extension,
        "build_profile": build_profile(),
    }


def version_text(
    package_version: str, checkout: str | None, extension: str | None
) -> str:
    """The line `sfm version` prints.

    `checkout` is the describe of the checkout the package runs from and
    `extension` the one recorded when the extension was built, each None when
    unknown. When both are known and differ, both are shown, labelled. Otherwise
    the one known is shown in parentheses after the version, unless it is the
    release the version names; with neither known, the version stands alone.
    """
    text = f"sfmtool {package_version}"
    if checkout is not None and extension is not None and checkout != extension:
        return f"{text} (checkout {checkout}, extension built from {extension})"
    describe = checkout if checkout is not None else extension
    if describe is None or is_release_describe(describe, package_version):
        return text
    return f"{text} ({describe})"
