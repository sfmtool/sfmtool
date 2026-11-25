// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the photograph cache (`sfmtool_core::camera::PhotographCache`):
//! the `PhotographCache` class, which a script or command holds so that its
//! calls share one build count, and through which it makes the mipmaps of
//! out-of-core photographs.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use pyo3::exceptions::{PyInterruptedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sfmtool_core::camera::photograph_cache::{default_budget_bytes, default_build_budget_bytes};
use sfmtool_core::camera::{PhotographCache, PhotographCacheOptions};
use sfmtool_core::progress::{Event, Progress};

use crate::signals::SignalWatch;
use crate::ProgressCounter;

/// A budget given in bytes: the default when `None`, refused when negative.
fn budget(name: &str, value: Option<i64>, default: fn() -> u64) -> PyResult<u64> {
    match value {
        None => Ok(default()),
        Some(bytes) => u64::try_from(bytes)
            .map_err(|_| PyValueError::new_err(format!("{name} must be 0 or more, not {bytes}"))),
    }
}

/// The photograph cache: decoded photographs kept as tiles under a byte
/// budget, and the mipmaps of out-of-core photographs made under a build
/// budget.
///
/// Every call on one cache shares its count of whole decodes and mipmap
/// builds in progress, so a script that holds one cache keeps them within
/// one build budget. Nothing is shared between two caches, and there is no
/// process-wide cache.
///
/// Args:
///     budget_bytes: The bytes of photograph tiles and mip level file bytes
///         the cache keeps; 0 keeps nothing. ``None`` is the core default, a
///         quarter of physical memory between 1 GiB and 16 GiB, or the
///         ``SFM_EXPLORER_PHOTOGRAPH_CACHE_MB`` environment variable.
///     build_budget_bytes: The estimated bytes of the whole decodes and
///         mipmap builds in progress on the cache at once. ``None`` is the
///         core default, worked out as ``budget_bytes``'s is, or the
///         ``SFM_EXPLORER_BUILD_BUDGET_MB`` environment variable.
///
/// Raises:
///     ValueError: for a negative budget.
#[pyclass(name = "PhotographCache", module = "sfmtool.photographs", frozen)]
pub struct PyPhotographCache {
    inner: PhotographCache,
}

#[pymethods]
impl PyPhotographCache {
    #[new]
    #[pyo3(signature = (budget_bytes=None, build_budget_bytes=None))]
    fn new(budget_bytes: Option<i64>, build_budget_bytes: Option<i64>) -> PyResult<Self> {
        let options = PhotographCacheOptions {
            budget_bytes: budget("budget_bytes", budget_bytes, default_budget_bytes)?,
            build_budget_bytes: budget(
                "build_budget_bytes",
                build_budget_bytes,
                default_build_budget_bytes,
            )?,
            ..PhotographCacheOptions::default()
        };
        Ok(Self {
            inner: PhotographCache::new(options),
        })
    }

    /// Make the mipmaps of the out-of-core photographs among `paths` whose
    /// mip level files are missing or stale.
    ///
    /// Each photograph's header and workspace are read first; an in-core
    /// photograph is not decoded. A mipmap already current, made by this
    /// cache, another cache or another process, is read rather than made
    /// again, and the cache's build budget bounds the builds in progress.
    /// The GIL is released while it runs.
    ///
    /// Args:
    ///     paths: The photographs' paths.
    ///     progress: A ``ProgressCounter``, set to the number of mipmaps made
    ///         so far.
    ///
    /// Returns:
    ///     A dict with the keys ``made``, ``current`` and ``in_core``, lists
    ///     of the paths whose mipmap this call made, whose mipmap was current,
    ///     and that are in core, and ``failed``, a list of ``(path,
    ///     message)`` pairs for the paths that could not be read or whose
    ///     mipmap could not be made, each message naming the path. Each path
    ///     is as given, in the order given.
    ///
    /// Raises:
    ///     KeyboardInterrupt: for Ctrl+C, which stops the builds within one
    ///         stage; the mipmaps made before it stay made.
    ///     InterruptedError: when the run was cancelled otherwise.
    #[pyo3(signature = (paths, *, progress=None))]
    fn build_mipmaps(
        &self,
        py: Python<'_>,
        paths: Vec<PathBuf>,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Py<PyAny>> {
        let counter = progress.as_ref().map(ProgressCounter::handle);
        let sink = |event: Event<'_>| {
            if let (
                Event::Value {
                    name: "mipmaps_made",
                    value,
                },
                Some(counter),
            ) = (event, &counter)
            {
                counter.fetch_max(value as usize, Ordering::Relaxed);
            }
        };
        // The builds run on the rayon pool, whose workers cannot run Python's
        // signal handlers, so the call goes on a thread of its own while the
        // calling thread waits for it, taking the GIL every `PATIENCE` to run
        // them. One that raises stops the builds, and its exception is raised
        // in place of the call's result.
        const PATIENCE: Duration = Duration::from_millis(50);
        let watch = SignalWatch::new();
        let cache = &self.inner;
        let result = py.detach(|| {
            std::thread::scope(|scope| {
                let (sender, receiver) = mpsc::channel();
                let (sink, cancel, paths) = (&sink, watch.cancel_flag(), &paths);
                let run = scope.spawn(move || {
                    let progress = Progress::to(sink).cancelled_by(cancel);
                    let paths: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
                    // The calling thread receives until this arrives, so the
                    // send succeeds.
                    let _ = sender.send(cache.build_mipmaps(&paths, &progress));
                });
                loop {
                    match receiver.recv_timeout(PATIENCE) {
                        Ok(result) => return result,
                        Err(mpsc::RecvTimeoutError::Timeout) => watch.check(),
                        // The call's thread ended without sending: it panicked.
                        Err(mpsc::RecvTimeoutError::Disconnected) => match run.join() {
                            Err(panic) => std::panic::resume_unwind(panic),
                            Ok(()) => unreachable!("the call sends its result before it ends"),
                        },
                    }
                }
            })
        });
        let report = watch
            .finish(result)?
            .map_err(|e| PyInterruptedError::new_err(e.to_string()))?;
        let dict = PyDict::new(py);
        dict.set_item("made", report.made)?;
        dict.set_item("current", report.current)?;
        dict.set_item("in_core", report.in_core)?;
        let failed = PyList::empty(py);
        for (path, error) in report.failed {
            failed.append((path, error.to_string()))?;
        }
        dict.set_item("failed", failed)?;
        Ok(dict.into_any().unbind())
    }

    /// What the cache holds and has done, as a dict.
    ///
    /// Returns:
    ///     A dict with the keys ``entries`` (the photograph tiles and file
    ///     entries kept), ``bytes`` (their bytes), ``budget_bytes``, ``hits``
    ///     and ``misses`` (answers that decoded nothing, and decodes and
    ///     builds started), ``decodes`` (whole decodes of photographs in
    ///     core), ``build_budget_bytes``, ``build_reserved_bytes`` (the
    ///     estimated bytes of the decodes and builds in progress now) and
    ///     ``mipmaps_made``.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stats = self.inner.stats();
        let dict = PyDict::new(py);
        dict.set_item("entries", stats.tiles + stats.files)?;
        dict.set_item("bytes", stats.bytes)?;
        dict.set_item("budget_bytes", stats.budget_bytes)?;
        dict.set_item("hits", stats.hits)?;
        dict.set_item("misses", stats.misses)?;
        dict.set_item("decodes", stats.decodes)?;
        dict.set_item("build_budget_bytes", stats.build_budget_bytes)?;
        dict.set_item("build_reserved_bytes", stats.build_reserved_bytes)?;
        dict.set_item("mipmaps_made", stats.mipmaps_made)?;
        Ok(dict.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "PhotographCache(budget_bytes={}, build_budget_bytes={})",
            stats.budget_bytes, stats.build_budget_bytes
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPhotographCache>()?;
    Ok(())
}
