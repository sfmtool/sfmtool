// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the photograph cache (`sfmtool_core::camera::PhotographCache`):
//! the `PhotographCache` class, which a script or command holds so that its
//! calls share one build count, through which it reads a photograph's layout
//! (`PhotographLayout`) and the levels, tiles and regions every kernel
//! samples, and makes the mipmaps of out-of-core photographs;
//! `check_mipmaps` and `prune_mipmaps`, which check the mipmaps of
//! photographs and delete a mipmap directory's orphan files, as
//! `sfm ws mipmaps --check` and `--prune` do; and [`get_photographs`], the
//! read behind `ImagePyramidSet.read`.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use numpy::{IntoPyArray, PyArray3};
use pyo3::exceptions::{
    PyFileNotFoundError, PyIOError, PyInterruptedError, PyRuntimeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use sfmtool_core::camera::image::ImageU8;
use sfmtool_core::camera::mipmap::{
    check_mipmaps, prune_mipmaps, PruneMipmapsOptions, PHOTOGRAPH_TILE_SIZE,
};
use sfmtool_core::camera::photograph_cache::{default_budget_bytes, default_build_budget_bytes};
use sfmtool_core::camera::sampler::LevelRect;
use sfmtool_core::camera::{
    GetError, PhotographCache, PhotographCacheOptions, PhotographLayout, TileError, TileKey,
    TiledPhotograph,
};
use sfmtool_core::progress::{Cancelled, Event, Progress};

use crate::helpers::read_only;
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

/// An index or a coordinate given from Python: refused when negative or past
/// what any photograph has.
fn index(name: &str, value: i64) -> PyResult<u32> {
    u32::try_from(value).map_err(|_| match value {
        ..0 => PyValueError::new_err(format!("{name} must be 0 or more, not {value}")),
        _ => PyValueError::new_err(format!("{name} {value} is past every photograph")),
    })
}

/// The Python exception for a photograph that could not be read: as
/// `read_image_rgb` raises them, `FileNotFoundError` for a missing file and
/// `OSError` for one that cannot be read or is too large, and
/// `RuntimeError` for an invalid workspace file, as `load_workspace_config`
/// raises it. Each message names the file.
fn get_err(e: GetError) -> PyErr {
    match e {
        GetError::NotFound(_) => PyFileNotFoundError::new_err(e.to_string()),
        GetError::Workspace(_) => PyRuntimeError::new_err(e.to_string()),
        GetError::Unreadable(_) | GetError::TooLarge { .. } => PyIOError::new_err(e.to_string()),
        GetError::Cancelled => PyInterruptedError::new_err(e.to_string()),
    }
}

/// The Python exception for a read that failed: `ValueError` for a level,
/// tile or rectangle the photograph does not have, `RuntimeError` naming the
/// file that changed during the read, and [`get_err`]'s for a photograph
/// that could not be read.
fn tile_err(e: TileError) -> PyErr {
    match e {
        TileError::OutOfRange { .. } => PyValueError::new_err(e.to_string()),
        TileError::Changed { .. } => PyRuntimeError::new_err(e.to_string()),
        TileError::Unreadable { why, .. } => get_err(why),
        TileError::Cancelled => PyInterruptedError::new_err(e.to_string()),
    }
}

/// The layout of one photograph, as the cache reads it: its size, its
/// levels and their sizes, whether it is out of core, how many levels its
/// mipmap stores, and whether its original is read by region.
///
/// Made by :meth:`PhotographCache.layout`.
#[pyclass(name = "PhotographLayout", module = "sfmtool.photographs", frozen)]
pub struct PyPhotographLayout {
    inner: PhotographLayout,
}

#[pymethods]
impl PyPhotographLayout {
    /// Level 0's width in pixels.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.layout.level_size(0).0
    }

    /// Level 0's height in pixels.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.layout.level_size(0).1
    }

    /// Whether the photograph is read from its mipmap: its workspace has
    /// ``mipmap_options``, a side is over their ``out_of_core_long_side``,
    /// and it is large enough to store a reduced level.
    #[getter]
    fn out_of_core(&self) -> bool {
        self.inner.layout.is_out_of_core()
    }

    /// The number of levels, down to one or two pixels on the short side.
    #[getter]
    fn levels(&self) -> usize {
        self.inner.layout.levels()
    }

    /// Each level's ``(width, height)``, level 0 first.
    #[getter]
    fn level_sizes(&self) -> Vec<(u32, u32)> {
        let layout = &self.inner.layout;
        (0..layout.levels()).map(|l| layout.level_size(l)).collect()
    }

    /// ``K``, the coarsest level the mipmap stores; 0 in core.
    #[getter]
    fn stored_levels(&self) -> usize {
        self.inner.layout.stored_levels()
    }

    /// ``None`` in core; out of core, whether level 0 is read from the
    /// original by region (``True``) or from its full-resolution mip level
    /// file (``False``).
    #[getter]
    fn region_readable(&self) -> Option<bool> {
        self.inner.region_readable
    }

    /// The side of the photograph tiles :meth:`PhotographCache.read_tile`
    /// reads, in level pixels.
    #[getter]
    fn tile_size(&self) -> u32 {
        self.inner.layout.tile_size()
    }

    fn __repr__(&self) -> String {
        let (width, height) = self.inner.layout.level_size(0);
        format!(
            "PhotographLayout(width={width}, height={height}, levels={}, stored_levels={}, \
             region_readable={})",
            self.inner.layout.levels(),
            self.inner.layout.stored_levels(),
            match self.inner.region_readable {
                None => "None",
                Some(true) => "True",
                Some(false) => "False",
            }
        )
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
    pub(crate) inner: PhotographCache,
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

    /// The layout of the photograph at `path`, the one every read of it is
    /// served under.
    ///
    /// Reads the header, the workspace's ``mipmap_options`` and, out of
    /// core, whether the original is read by region, from a current mip
    /// level file's record or by testing the original. Makes no mipmap and
    /// decodes no pixel. The GIL is released while it runs.
    ///
    /// Args:
    ///     path: The photograph's path.
    ///
    /// Returns:
    ///     A :class:`PhotographLayout`.
    ///
    /// Raises:
    ///     FileNotFoundError: when the file does not exist.
    ///     OSError: when it cannot be read, or has a side of 65536 px or more.
    ///     RuntimeError: when its workspace file holds an invalid
    ///         ``mipmap_options``, naming the key.
    fn layout(&self, py: Python<'_>, path: PathBuf) -> PyResult<PyPhotographLayout> {
        let inner = py
            .detach(|| self.inner.read_layout(&path))
            .map_err(get_err)?;
        Ok(PyPhotographLayout { inner })
    }

    /// Level `level` of the photograph at `path`, whole.
    ///
    /// The pixels every kernel samples for the photograph, as
    /// :meth:`read_region` reads them.
    ///
    /// Args:
    ///     path: The photograph's path.
    ///     level: The level, 0 at full resolution.
    ///     progress: A ``ProgressCounter``, set to 1 when the read made the
    ///         photograph's mipmap.
    ///
    /// Returns:
    ///     An ``(H, W, 3)`` read-only ``uint8`` array.
    ///
    /// Raises:
    ///     As :meth:`read_region`.
    #[pyo3(signature = (path, level, *, progress=None))]
    fn read_level<'py>(
        &self,
        py: Python<'py>,
        path: PathBuf,
        level: i64,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let level = index("level", level)? as usize;
        let cache = &self.inner;
        self.read(py, progress, |progress| {
            cache.read_level(&path, level, progress)
        })
    }

    /// The photograph tile at column `col` and row `row` of level `level`,
    /// without its border: the rectangle of the level from
    /// ``(col * tile_size, row * tile_size)``, ``tile_size`` pixels a side
    /// and clipped to the level, as :meth:`read_region` reads it.
    /// :attr:`PhotographLayout.tile_size` is the side.
    ///
    /// Args:
    ///     path: The photograph's path.
    ///     level: The level, 0 at full resolution.
    ///     col: The tile's column.
    ///     row: The tile's row.
    ///     progress: A ``ProgressCounter``, set to 1 when the read made the
    ///         photograph's mipmap.
    ///
    /// Returns:
    ///     An ``(H, W, 3)`` read-only ``uint8`` array.
    ///
    /// Raises:
    ///     As :meth:`read_region`.
    #[pyo3(signature = (path, level, col, row, *, progress=None))]
    fn read_tile<'py>(
        &self,
        py: Python<'py>,
        path: PathBuf,
        level: i64,
        col: i64,
        row: i64,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let level = index("level", level)?;
        let key = TileKey {
            level: u8::try_from(level).map_err(|_| {
                PyValueError::new_err(format!("level {level} is past every photograph"))
            })?,
            col: index("col", col)?,
            row: index("row", row)?,
        };
        let cache = &self.inner;
        self.read(py, progress, |progress| {
            cache.read_tile(&path, key, progress)
        })
    }

    /// The pixels ``x0..x1`` by ``y0..y1`` of level `level` of the
    /// photograph at `path`.
    ///
    /// The pixels every kernel samples for the photograph: in core the exact
    /// levels of its decode; out of core level 0 from the original, or from
    /// its full-resolution mip level file when the original cannot be read
    /// by region, the stored levels from its mip level files, and the levels
    /// past them reduced from the coarsest. An out-of-core photograph whose
    /// mipmap is missing or stale has it made first, and a read of a current
    /// mipmap decodes only the tiles the rectangle covers. The GIL is
    /// released while it runs.
    ///
    /// Args:
    ///     path: The photograph's path.
    ///     level: The level, 0 at full resolution.
    ///     x0, y0, x1, y1: The rectangle, in the level's pixels, ``x1`` and
    ///         ``y1`` past its last column and row.
    ///     progress: A ``ProgressCounter``, set to 1 when the read made the
    ///         photograph's mipmap.
    ///
    /// Returns:
    ///     An ``(y1 - y0, x1 - x0, 3)`` read-only ``uint8`` array in the
    ///     layout of ``read_image_rgb``.
    ///
    /// Raises:
    ///     ValueError: for a level, tile or rectangle the photograph does not
    ///         have, an empty rectangle, or a negative index.
    ///     FileNotFoundError: when the file does not exist.
    ///     OSError: when it cannot be read, or has a side of 65536 px or more.
    ///     RuntimeError: when its workspace file holds an invalid
    ///         ``mipmap_options``, naming the key, or when a file changed
    ///         during the read, naming it.
    ///     KeyboardInterrupt: for Ctrl+C, which stops the read within one
    ///         stage.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path, level, x0, y0, x1, y1, *, progress=None))]
    fn read_region<'py>(
        &self,
        py: Python<'py>,
        path: PathBuf,
        level: i64,
        x0: i64,
        y0: i64,
        x1: i64,
        y1: i64,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let level = index("level", level)? as usize;
        let rect = LevelRect {
            x0: index("x0", x0)?,
            y0: index("y0", y0)?,
            x1: index("x1", x1)?,
            y1: index("y1", y1)?,
        };
        let cache = &self.inner;
        self.read(py, progress, |progress| {
            cache.read_region(&path, level, rect, progress)
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
        let sink = mipmaps_into(progress.as_ref());
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

impl PyPhotographCache {
    /// Run the read `read` as [`run_watched`] runs it, counting the mipmaps
    /// it made into `progress`, and hand its pixels to Python as a read-only
    /// `(H, W, 3)` array.
    fn read<'py>(
        &self,
        py: Python<'py>,
        progress: Option<ProgressCounter>,
        read: impl FnOnce(&Progress<'_>) -> Result<ImageU8, TileError> + Send,
    ) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let sink = mipmaps_into(progress.as_ref());
        let read = run_watched(py, &sink, |progress| match read(progress) {
            Err(TileError::Cancelled) => Err(Cancelled),
            read => Ok(read),
        })?;
        let image = read.map_err(tile_err)?;
        let shape = (
            image.height() as usize,
            image.width() as usize,
            image.channels() as usize,
        );
        let array = ndarray::Array3::from_shape_vec(shape, image.into_data())
            .expect("an ImageU8 holds height * width * channels bytes");
        Ok(read_only(array.into_pyarray(py)))
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

/// A `sink` that sets `counter` to the mipmaps the run reports it has made.
fn mipmaps_into(counter: Option<&ProgressCounter>) -> impl Fn(Event<'_>) + Sync {
    let counter = counter.map(ProgressCounter::handle);
    move |event: Event<'_>| {
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
    }
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

/// The photographs at `paths`, in order, read through `cache`'s `get_many`
/// as the viewer reads them: each holds no pixel and reads its tiles through
/// the cache, which decodes an in-core photograph again for a tile it has
/// evicted, and an out-of-core photograph's mipmap is made first when it is
/// missing or stale. `progress` is set to the mipmaps made; the GIL is
/// released and Ctrl+C raises `KeyboardInterrupt`, as for [`run_watched`].
///
/// A path `get_many` could not answer is asked of `get`, which answers why
/// from the failure the cache remembered, and raises as [`get_err`] says,
/// naming the path.
pub(crate) fn get_photographs(
    py: Python<'_>,
    cache: &PhotographCache,
    paths: &[PathBuf],
    progress: Option<&ProgressCounter>,
) -> PyResult<Vec<TiledPhotograph>> {
    let sink = mipmaps_into(progress);
    let photographs = run_watched(py, &sink, |progress| {
        let paths: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        let (found, _) = cache.get_many(&paths, progress)?;
        let photographs = found
            .into_iter()
            .zip(&paths)
            .map(|(photograph, path)| match photograph {
                Some(photograph) => Ok(photograph),
                None => cache.get(path, progress),
            })
            // A clone copies no pixel: it reads through the same cache.
            .map(|photograph| photograph.map(Arc::unwrap_or_clone))
            .collect::<Result<Vec<_>, _>>();
        match photographs {
            Err(GetError::Cancelled) => Err(Cancelled),
            photographs => Ok(photographs),
        }
    })?;
    photographs.map_err(get_err)
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
    m.add_class::<PyPhotographLayout>()?;
    m.add_function(wrap_pyfunction!(check_mipmaps_py, m)?)?;
    m.add_function(wrap_pyfunction!(prune_mipmaps_py, m)?)?;
    Ok(())
}
