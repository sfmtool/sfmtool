// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Running Python's signal handlers during a run with the GIL released, so
//! that Ctrl+C stops it.
//!
//! A binding makes a [`SignalWatch`], cancels the run's `Progress` by
//! [`SignalWatch::cancel_flag`], calls [`SignalWatch::check`] from wherever
//! it can take the GIL while the run goes, and passes the run's result
//! through [`SignalWatch::finish`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use pyo3::prelude::*;

/// The cancel flag of one run with the GIL released, and the exception a
/// signal handler raised while it went.
pub(crate) struct SignalWatch {
    cancel: AtomicBool,
    raised: Mutex<Option<PyErr>>,
}

impl SignalWatch {
    pub(crate) fn new() -> Self {
        Self {
            cancel: AtomicBool::new(false),
            raised: Mutex::new(None),
        }
    }

    /// The flag [`check`](Self::check) sets when a handler raises, for the
    /// run's `Progress` to be cancelled by.
    pub(crate) fn cancel_flag(&self) -> &AtomicBool {
        &self.cancel
    }

    /// Take the GIL and run Python's signal handlers, unless the flag is set
    /// already. A handler that raises (`KeyboardInterrupt` for Ctrl+C) sets
    /// the flag, and its exception is kept for [`finish`](Self::finish).
    pub(crate) fn check(&self) {
        if self.cancel.load(Ordering::Relaxed) {
            return;
        }
        if let Err(e) = Python::attach(|py| py.check_signals()) {
            *self.raised.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
            self.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// The exception a handler raised while the run went, whatever the run's
    /// `result`, or else `result`. A handler can raise after the run's last
    /// check of the flag, so the run ends without being cancelled; its
    /// exception is raised all the same, or the signal would be lost.
    pub(crate) fn finish<T, E>(self, result: Result<T, E>) -> PyResult<Result<T, E>> {
        match self.raised.into_inner().unwrap_or_else(|p| p.into_inner()) {
            Some(e) => Err(e),
            None => Ok(result),
        }
    }
}
