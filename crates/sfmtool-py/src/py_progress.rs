// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A shared work counter for reporting progress out of a long-running,
//! GIL-releasing Rust pass.
//!
//! The heavy patch-cloud kernels (e.g. [`PyPatchCloud::refine_normals`]) run a
//! single `par_iter().collect()` over hundreds of thousands of patches inside
//! `py.detach()`, so from Python the call is one opaque blocking step. A caller
//! that wants intra-pass feedback constructs a [`ProgressCounter`], passes it to
//! the kernel (which bumps it once per patch as work completes), and polls
//! [`ProgressCounter::value`] from a background thread while the GIL is released.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use pyo3::prelude::*;

/// A thread-safe completed-work counter shared between a Rust kernel and a
/// Python poller.
///
/// Construct one, hand it to a kernel that accepts a ``progress`` argument, and
/// read :attr:`value` (typically from a background thread) while the kernel runs
/// — it climbs from ``0`` toward the kernel's item count as work finishes. The
/// counter is monotonic within a single kernel call; reuse across calls should
/// :meth:`reset` first.
// `from_py_object`: this class is accepted by value as a kernel argument (e.g.
// `refine_normals(progress=...)`), which extracts it via a `Clone` — cheap here,
// as the clone only bumps the `Arc` refcount onto the shared counter.
#[pyclass(name = "ProgressCounter", module = "sfmtool._sfmtool", from_py_object)]
#[derive(Clone, Default)]
pub struct ProgressCounter {
    inner: Arc<AtomicUsize>,
}

#[pymethods]
impl ProgressCounter {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// The number of items completed so far (a relaxed atomic load).
    #[getter]
    fn value(&self) -> usize {
        self.inner.load(Ordering::Relaxed)
    }

    /// Reset the counter to zero (before reusing it for a second kernel call).
    fn reset(&self) {
        self.inner.store(0, Ordering::Relaxed);
    }
}

impl ProgressCounter {
    /// A cheap handle to the shared atomic, for a kernel to bump per item. Held
    /// by the caller across `py.detach()` so the Python poller and the Rust
    /// workers reference the same counter.
    pub fn handle(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.inner)
    }
}
