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

# Rayon is a *loss* under coverage instrumentation, so both test phases below run
# it single-threaded. `-C instrument-coverage` gives every coverage region one
# non-atomic counter in a process-wide `__llvm_prf_cnts` section; the number of
# increments is fixed by the data, so parallelism does not divide the counter
# traffic, it just makes several threads write the *same* address. A hot loop
# that was thread-private without coverage becomes one cache line ping-ponging
# between cores, and the parallel speedup inverts. Measured on a 4-CPU pinned
# box, holding everything else fixed:
#
#                                     RAYON=1   RAYON=4
#   sfmtool-core --lib, plain           14.8 s     9.6 s  (rayon worth 1.5x)
#   sfmtool-core --lib, instrumented    76.7 s    94.6 s  (rayon a 23% loss)
#   pytest -n 4 --cov, plain           388.4 s   302.7 s  (rayon worth 28%)
#   pytest -n 4 --cov, instrumented    851.5 s  1205.6 s  (rayon a 42% loss)
#
# Nothing is given up. Coverage is unaffected -- thread count does not change
# which regions execute, and the one place this workspace branches on rayon
# state (`warp_map::parallelize_rows`) keys off
# `rayon::current_thread_index().is_none()`, i.e. *where* the caller runs rather
# than how wide the pool is, so both of its arms stay reachable. Wall-clock
# parallelism is not given up either: both phases keep their task parallelism
# (cargo's test threads, xdist's workers), which already fills the runner's
# cores, and the `test-os` jobs run these same two suites uninstrumented at full
# rayon width on every PR.
#
# This is scoped to the instrumented run. `pixi run test-rust` is left alone:
# the numbers above are for the 4-core CI shape, and a many-core dev box was not
# measured.
export RAYON_NUM_THREADS=1

# Run Rust tests (generates Rust-side coverage).
# sfm-explorer is excluded because its ui_basic integration tests require
# --test-threads=1 and are run separately in the ui-test CI job. Its *lib*
# tests are headless and run in the test-os jobs (Windows/macOS); they stay
# out of this instrumented run so uninstrumented artifacts can't land in the
# shared target dir and degrade the next run's coverage. Locally, use
# `cargo test -p sfm-explorer --lib`.
cargo test --workspace --exclude sfm-explorer

# Run Python tests (generates Rust coverage from Python calls + Python coverage).
# Parallelize across min(4, cpu) xdist workers: the suite's wall time is set by a
# few long reconstruction/patch tests, and the runner has 4 cores with plenty of
# RAM headroom (peak ~1.4 GB per worker, measured). pytest-cov auto-detects the
# xdist workers and combines their coverage data at the end.
workers=$(python -c 'import os; print(min(4, os.cpu_count() or 1))')
pytest -n "$workers" --cov=sfmtool --cov-report=lcov:python-lcov.info

# Generate the Rust coverage report
cargo llvm-cov report --lcov --output-path lcov.info

echo ""
echo "Coverage reports written to lcov.info (Rust) and python-lcov.info (Python)"
