#!/usr/bin/env bash
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
#
# Generate combined Rust + Python coverage.
# Run via: pixi run -e test coverage-all

set -euo pipefail

# Set up LLVM coverage environment
eval "$(cargo llvm-cov show-env --sh)"

# Clean previous coverage data
cargo llvm-cov clean --workspace

# Build the Python extension with coverage instrumentation
maturin develop

# Run Rust tests (generates Rust-side coverage)
cargo test --workspace

# Run Python tests (generates Rust coverage from Python calls + Python coverage)
pytest --cov=sfmtool --cov-report=lcov:python-lcov.info

# Generate the Rust coverage report
cargo llvm-cov report --lcov --output-path lcov.info

echo ""
echo "Coverage reports written to lcov.info (Rust) and python-lcov.info (Python)"
