#!/usr/bin/env bash
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
#
# Append `.dev<N>` to the `version` in pyproject.toml's `[project]` table, so
# `0.3` becomes `0.3.dev<N>`. A manual run of publish_to_pypi.yml calls this
# with the workflow's run number before it builds, because maturin takes the
# wheel and sdist version from that line and TestPyPI refuses a version it
# already has. PEP 440 sorts `0.3.dev<N>` before `0.3`.
#
# Usage: scripts/ci_set_dev_version.sh N [path/to/pyproject.toml]
#
# Fails, leaving the file unchanged, unless exactly one `version = "..."` line
# in `[project]` was rewritten. Accepts the CRLF line endings a Windows
# checkout gives the file; Git Bash's awk writes the result with LF endings,
# which TOML reads the same.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^[0-9]+$ ]]; then
    echo "usage: $0 RUN_NUMBER [pyproject.toml]" >&2
    exit 2
fi
dev_number=$1
pyproject=${2:-pyproject.toml}

awk -v suffix=".dev${dev_number}" '
    /^\[/ { in_project = ($0 ~ /^\[project\]\r?$/) }
    in_project && /^version = "[^"]*"\r?$/ { sub(/"\r?$/, suffix "&"); n++ }
    { print }
    END { if (n != 1) exit 1 }
' "$pyproject" > "$pyproject.new" || {
    rm -f "$pyproject.new"
    echo "error: found no single version line in the [project] table of $pyproject" >&2
    exit 1
}
mv "$pyproject.new" "$pyproject"
grep -n '^version = ' "$pyproject"
