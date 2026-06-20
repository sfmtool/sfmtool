#!/usr/bin/env bash
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
#
# Generate combined Rust + Python coverage.
# Run via: pixi run -e test coverage-all

set -euo pipefail

# Set up LLVM coverage environment
eval "$(cargo llvm-cov show-env --sh)"

# Clear stale coverage counters from prior runs, but keep compiled artifacts
# so a restored build cache (CI) or an incremental local build stays warm.
cargo llvm-cov clean --workspace --profraw-only

# Build the Python extension with coverage instrumentation.
# --release so the Rust kernels run at shipping speed (a debug build is ~10-15x
# slower); instrument-coverage still emits valid region mapping under release,
# though optimization/inlining can make Rust-side line coverage slightly coarser.
maturin develop --release

# Run Rust tests (generates Rust-side coverage).
# sfm-explorer is excluded because its ui_basic integration tests require
# --test-threads=1 and are run separately in the ui-test CI job.
cargo test --workspace --exclude sfm-explorer

# Run Python tests (generates Rust coverage from Python calls + Python coverage)
pytest --cov=sfmtool --cov-report=lcov:python-lcov.info

# Generate the Rust coverage report
cargo llvm-cov report --lcov --output-path lcov.info

echo ""
echo "Coverage reports written to lcov.info (Rust) and python-lcov.info (Python)"
