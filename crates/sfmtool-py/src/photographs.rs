// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the photograph cache (`sfmtool_core::camera::PhotographCache`):
//! the `PhotographCache` class, which a script or command holds so that its
//! calls share one build count, and through which it makes the mipmaps of
//! out-of-core photographs; and `check_mipmaps` and `prune_mipmaps`, which
//! check the mipmaps of photographs and delete a mipmap directory's orphan
//! files, as `sfm ws mipmaps --check` and `--prune` do.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use pyo3::exceptions::{PyInterruptedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sfmtool_core::camera::mipmap::{
    check_mipmaps, prune_mipmaps, PruneMipmapsOptions, PHOTOGRAPH_TILE_SIZE,
};
use sfmtool_core::camera::photograph_cache::{default_budget_bytes, default_build_budget_bytes};
use sfmtool_core::camera::{PhotographCache, PhotographCacheOptions};
use sfmtool_core::progress::{Cancelled, Event, Progress};

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
        let cache = &self.inner;
        let report = run_watched(py, &sink, |progress| {
            let paths: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
            cache.build_mipmaps(&paths, progress)
        })?;
        let dict = PyDict::new(py);
        dict.set_item("made", report.made)?;
        dict.set_item("current", report.current)?;
        dict.set_item("in_core", report.in_core)?;
        dict.set_item("failed", pairs(py, report.failed)?)?;
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

/// Run `run` with the GIL released on a thread of its own, its `Progress`
/// reporting to `sink` and cancelled when one of Python's signal handlers
/// raises, while the calling thread waits for it and runs the handlers every
/// 50 ms. The work runs on the rayon pool, whose workers cannot run them, so
/// this is what lets Ctrl+C stop it within one stage.
///
/// A handler's exception (`KeyboardInterrupt` for Ctrl+C) is raised in place
/// of the result, and a run cancelled otherwise raises `InterruptedError`.
fn run_watched<T: Send>(
    py: Python<'_>,
    sink: &(dyn Fn(Event<'_>) + Sync),
    run: impl FnOnce(&Progress<'_>) -> Result<T, Cancelled> + Send,
) -> PyResult<T> {
    const PATIENCE: Duration = Duration::from_millis(50);
    let watch = SignalWatch::new();
    let result = py.detach(|| {
        std::thread::scope(|scope| {
            let (sender, receiver) = mpsc::channel();
            let cancel = watch.cancel_flag();
            let thread = scope.spawn(move || {
                let progress = Progress::to(sink).cancelled_by(cancel);
                // The calling thread receives until this arrives, so the
                // send succeeds.
                let _ = sender.send(run(&progress));
            });
            loop {
                match receiver.recv_timeout(PATIENCE) {
                    Ok(result) => return result,
                    Err(mpsc::RecvTimeoutError::Timeout) => watch.check(),
                    // The thread ended without sending: it panicked.
                    Err(mpsc::RecvTimeoutError::Disconnected) => match thread.join() {
                        Err(panic) => std::panic::resume_unwind(panic),
                        Ok(()) => unreachable!("the run sends its result before it ends"),
                    },
                }
            }
        })
    });
    watch
        .finish(result)?
        .map_err(|e| PyInterruptedError::new_err(e.to_string()))
}

/// A `sink` that sets `counter` to the count the run reports.
fn count_into(counter: Option<&ProgressCounter>) -> impl Fn(Event<'_>) + Sync {
    let counter = counter.map(ProgressCounter::handle);
    move |event: Event<'_>| {
        if let (Event::Count { done, .. }, Some(counter)) = (event, &counter) {
            counter.store(done as usize, Ordering::Relaxed);
        }
    }
}

/// A list of `(path, message)` pairs.
fn pairs<'py, M: ToString>(
    py: Python<'py>,
    items: impl IntoIterator<Item = (PathBuf, M)>,
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for (path, message) in items {
        list.append((path, message.to_string()))?;
    }
    Ok(list)
}

/// Check the mipmaps of the photographs at `paths`, writing nothing.
///
/// Each photograph's header and workspace are read first; an in-core
/// photograph is read no further and no photograph is decoded. An
/// out-of-core photograph's mipmap is missing when there is no file at its
/// coarsest mip level's name, and otherwise current or stale as the files'
/// records say, the original hashed only when its length matches a record
/// and its modification time does not. The GIL is released while it runs.
///
/// Args:
///     paths: The photographs' paths; a directory is not expanded.
///     progress: A ``ProgressCounter``, set to the number of photographs
///         checked so far.
///
/// Returns:
///     A dict with the keys ``current``, ``missing`` and ``in_core``, lists
///     of the paths whose mipmap is current, whose mipmap is missing, and
///     that are in core; ``stale``, a list of ``(path, message)`` pairs
///     naming the first mip level file found not current and why; and
///     ``failed``, a list of ``(path, message)`` pairs for the paths that
///     could not be read or whose workspace file is invalid, each message
///     naming the path. Each path is as given, in the order given.
///
/// Raises:
///     KeyboardInterrupt: for Ctrl+C, which stops the check.
///     InterruptedError: when the run was cancelled otherwise.
#[pyfunction(name = "check_mipmaps")]
#[pyo3(signature = (paths, *, progress=None))]
fn check_mipmaps_py(
    py: Python<'_>,
    paths: Vec<PathBuf>,
    progress: Option<ProgressCounter>,
) -> PyResult<Py<PyAny>> {
    let sink = count_into(progress.as_ref());
    let report = run_watched(py, &sink, |progress| {
        let paths: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        check_mipmaps(&paths, PHOTOGRAPH_TILE_SIZE, progress)
    })?;
    let dict = PyDict::new(py);
    dict.set_item("current", report.current)?;
    dict.set_item("missing", report.missing)?;
    dict.set_item("stale", pairs(py, report.stale)?)?;
    dict.set_item("in_core", report.in_core)?;
    dict.set_item("failed", pairs(py, report.failed)?)?;
    Ok(dict.into_any().unbind())
}

/// Delete the orphan files of the mipmap directories that hold the mipmaps
/// of the photographs under `paths`.
///
/// Each path, a photograph or a directory, existing or not, searches the
/// part of its workspace's mipmap directory that mirrors it, and a directory
/// also the whole mipmap directory of each workspace nested below it. A file
/// there is an orphan when its name is a mip level file's and the
/// workspace's file has no ``mipmap_options``, its quality is not the one
/// the object gives its level, its photograph no longer exists, is in core
/// or is too large, the photograph's mipmap is kept in a nested workspace,
/// the mipmap stores no such level, or the file is not current for its
/// photograph; or when it is a build's temporary file older than an hour.
/// Only regular files inside a mipmap directory whose names are a mip level
/// file's or a temporary file's are deleted; a link is neither entered nor
/// deleted, and no directory is deleted. The GIL is released while it runs.
///
/// Args:
///     paths: Photographs and directories.
///     dry_run: Judge every file and delete none; ``orphans`` then lists
///         the files a prune would delete.
///     progress: A ``ProgressCounter``, set to the number of files judged so
///         far.
///
/// Returns:
///     A dict, each list sorted by path: ``orphans``, ``(path, why)`` pairs
///     of the files deleted (in a dry run, to be deleted); ``kept``, the
///     current mip level files and temporary files too new to delete;
///     ``other_files``, the files whose names are not a mip level file's,
///     left as they are; ``links``, the links met and not entered;
///     ``failed``, ``(path, why)`` pairs of what could not be searched,
///     judged or deleted, nothing of which is deleted; and
///     ``outside_workspaces``, the paths in no workspace.
///
/// Raises:
///     KeyboardInterrupt: for Ctrl+C, which stops the prune; the files
///         deleted before it stay deleted.
///     InterruptedError: when the run was cancelled otherwise.
#[pyfunction(name = "prune_mipmaps")]
#[pyo3(signature = (paths, *, dry_run=false, progress=None))]
fn prune_mipmaps_py(
    py: Python<'_>,
    paths: Vec<PathBuf>,
    dry_run: bool,
    progress: Option<ProgressCounter>,
) -> PyResult<Py<PyAny>> {
    let sink = count_into(progress.as_ref());
    let options = PruneMipmapsOptions {
        dry_run,
        ..PruneMipmapsOptions::default()
    };
    let report = run_watched(py, &sink, |progress| {
        let paths: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        prune_mipmaps(&paths, &options, progress)
    })?;
    let dict = PyDict::new(py);
    dict.set_item("orphans", pairs(py, report.orphans)?)?;
    dict.set_item("kept", report.kept)?;
    dict.set_item("other_files", report.other_files)?;
    dict.set_item("links", report.links)?;
    dict.set_item("failed", pairs(py, report.failed)?)?;
    dict.set_item("outside_workspaces", report.outside_workspaces)?;
    Ok(dict.into_any().unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPhotographCache>()?;
    m.add_function(wrap_pyfunction!(check_mipmaps_py, m)?)?;
    m.add_function(wrap_pyfunction!(prune_mipmaps_py, m)?)?;
    Ok(())
}
