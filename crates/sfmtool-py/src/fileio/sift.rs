// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `.sift` file I/O: the dict-shaped `read_sift` /
//! `write_sift` functions, the `SiftFile` handle most of the package reads
//! through, and the `SiftWriteQueue` that `sfm sift` saves through.
//!
//! Every `SiftError` raises through the format-error table in
//! [`crate::format_errors`], so these raise what the other formats' bindings
//! raise: a failed filesystem call is the `OSError` subclass for its kind
//! (`FileNotFoundError` for a missing file), a file that cannot be parsed or
//! breaks a rule of the format is an `OSError`, and data a caller hands a
//! writer that the format does not allow (a metadata key the format does not
//! define, a missing or mistyped value, an array whose shape disagrees with
//! `feature_count`) is a `ValueError`, raised before anything is written. An
//! array of the wrong dtype raises `TypeError` naming its key, as
//! `write_matches` does.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::exceptions::{PyIOError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use sfmtool_sift_format::{
    self, FeatureToolMetadata, SiftContentHash, SiftData, SiftError, SiftMetadata, SIFT_ZSTD_LEVEL,
};

use crate::format_errors::{read_err, write_err};
use crate::helpers::{dtype_name, get_item, py_to_serde, serde_to_py};

/// Convert SiftData to a Python dict.
fn sift_data_to_py(py: Python<'_>, data: SiftData) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    dict.set_item(
        "feature_tool_metadata",
        serde_to_py(py, &data.feature_tool_metadata)?,
    )?;
    dict.set_item("metadata", serde_to_py(py, &data.metadata)?)?;
    dict.set_item("content_hash", serde_to_py(py, &data.content_hash)?)?;
    dict.set_item("positions_xy", data.positions_xy.into_pyarray(py))?;
    dict.set_item("affine_shapes", data.affine_shapes.into_pyarray(py))?;
    dict.set_item("descriptors", data.descriptors.into_pyarray(py))?;
    dict.set_item("thumbnail_y_x_rgb", data.thumbnail_y_x_rgb.into_pyarray(py))?;

    Ok(dict.into())
}

/// Read a complete .sift file, returning a dict with numpy arrays and metadata.
///
/// Returns a dict with keys:
///   feature_tool_metadata, metadata, content_hash (dicts),
///   positions_xy (N,2 float32), affine_shapes (N,2,2 float32),
///   descriptors (N,128 uint8), thumbnail_y_x_rgb
///   (THUMBNAIL_SIZE,THUMBNAIL_SIZE,3 uint8).
///
/// Raises `FileNotFoundError` for a missing file and `OSError` for one that
/// cannot be read.
#[pyfunction]
pub fn read_sift(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let data = py
        .detach(|| sfmtool_sift_format::read_sift(&path))
        .map_err(read_err)?;
    sift_data_to_py(py, data)
}

/// Read only metadata from a .sift file (fast, no binary data).
///
/// Returns a dict with keys: feature_tool_metadata, metadata, content_hash.
#[pyfunction]
pub fn read_sift_metadata(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let (tool_meta, meta, hash) = py
        .detach(|| sfmtool_sift_format::read_sift_metadata(&path))
        .map_err(read_err)?;

    let dict = PyDict::new(py);
    dict.set_item("feature_tool_metadata", serde_to_py(py, &tool_meta)?)?;
    dict.set_item("metadata", serde_to_py(py, &meta)?)?;
    dict.set_item("content_hash", serde_to_py(py, &hash)?)?;
    Ok(dict.into())
}

/// Read the first `count` features from a .sift file.
///
/// If `count` exceeds the feature count, returns all features.
#[pyfunction]
pub fn read_sift_partial(py: Python<'_>, path: PathBuf, count: usize) -> PyResult<Py<PyAny>> {
    let data = py
        .detach(|| sfmtool_sift_format::read_sift_partial(&path, count))
        .map_err(read_err)?;
    sift_data_to_py(py, data)
}

/// One metadata dict of a `write_sift` dict as JSON, for the format crate's
/// strict `from_json_for_write` parse.
fn metadata_json(
    py: Python<'_>,
    data: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<serde_json::Value> {
    py_to_serde(py, &get_item(data, key)?)
}

/// One array of a `write_sift` dict, with an array of the wrong dtype or rank
/// (or no array at all) as a `TypeError` naming its key and what it should be.
fn column<'py, T: FromPyObjectOwned<'py>>(
    data: &Bound<'py, PyDict>,
    key: &str,
    expected: &str,
) -> PyResult<T> {
    let item = get_item(data, key)?;
    item.extract().map_err(|_| {
        let actual = dtype_name(&item)
            .map(|dtype| format!("dtype {dtype}"))
            .unwrap_or_else(|_| item.get_type().to_string());
        PyTypeError::new_err(format!(
            "{key} must be a {expected} numpy array, got {actual}"
        ))
    })
}

/// Copy a Python dict of numpy arrays + metadata into an owned `SiftData`,
/// with its metadata parsed by the format crate's strict `from_json_for_write`
/// and the whole checked with `validate_sift_data`.
///
/// Holds the GIL (it reads the numpy buffers), but the result is fully owned
/// and `Send`, so the caller can compress/write it off the GIL — or on another
/// thread / rayon worker. A missing key raises `KeyError`, an array of the
/// wrong dtype or rank `TypeError`, and anything the format refuses
/// `ValueError`.
fn sift_data_from_dict(py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<SiftData> {
    let feature_tool_metadata =
        FeatureToolMetadata::from_json_for_write(metadata_json(py, data, "feature_tool_metadata")?)
            .map_err(write_err)?;
    let metadata = SiftMetadata::from_json_for_write(metadata_json(py, data, "metadata")?)
        .map_err(write_err)?;

    let positions_xy: PyReadonlyArray2<f32> = column(data, "positions_xy", "(N, 2) float32")?;
    let affine_shapes: PyReadonlyArray3<f32> = column(data, "affine_shapes", "(N, 2, 2) float32")?;
    let descriptors: PyReadonlyArray2<u8> = column(data, "descriptors", "(N, 128) uint8")?;
    let thumbnail_y_x_rgb: PyReadonlyArray3<u8> = column(
        data,
        "thumbnail_y_x_rgb",
        "(THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3) uint8",
    )?;

    let sift_data = SiftData {
        feature_tool_metadata,
        metadata,
        content_hash: SiftContentHash::default(),
        positions_xy: positions_xy.as_array().as_standard_layout().into_owned(),
        affine_shapes: affine_shapes.as_array().as_standard_layout().into_owned(),
        descriptors: descriptors.as_array().as_standard_layout().into_owned(),
        thumbnail_y_x_rgb: thumbnail_y_x_rgb
            .as_array()
            .as_standard_layout()
            .into_owned(),
    };
    sfmtool_sift_format::validate_sift_data(&sift_data).map_err(write_err)?;
    Ok(sift_data)
}

/// Write a .sift file from a dict of numpy arrays and metadata.
///
/// The dict has the keys `read_sift` returns; `content_hash` is ignored
/// (recomputed on write). `feature_tool_metadata` holds exactly
/// `feature_tool`, `feature_type` and `feature_options` (a dict), and
/// `metadata` exactly `version`, `image_name`, `image_file_xxh128`,
/// `image_file_size`, `image_width`, `image_height` and `feature_count`.
///
/// Raises `ValueError`, before anything is written, for a metadata key the
/// format does not define, a missing or mistyped metadata value, an
/// unsupported `version`, or an array whose shape disagrees with
/// `feature_count` (or, for the thumbnail, the `THUMBNAIL_SIZE` square);
/// `TypeError` for an array of the wrong dtype; `KeyError` for a missing top-
/// level key; and `OSError` when the write itself fails.
#[pyfunction]
#[pyo3(signature = (path, data, zstd_level=SIFT_ZSTD_LEVEL))]
pub fn write_sift(
    py: Python<'_>,
    path: PathBuf,
    data: &Bound<'_, PyDict>,
    zstd_level: i32,
) -> PyResult<()> {
    let sift_data = sift_data_from_dict(py, data)?;

    // The arrays are now owned, so release the GIL around the CPU-bound
    // zstd/ZIP compression and file write.
    py.detach(|| sfmtool_sift_format::write_sift(&path, &sift_data, zstd_level))
        .map_err(write_err)
}

/// An open `.sift` file: its metadata, read when it is opened, and its
/// feature columns and thumbnail, read on request.
///
/// `SiftFile(path)` reads the three metadata entries and raises
/// `FileNotFoundError` for a missing file and `OSError` for one it cannot
/// read. The first `read_*` call without a `count` reads the whole file and
/// keeps it until `close`; a call with a `count` reads the first `count`
/// features (all of them when `count` exceeds `feature_count`) without
/// keeping them. Each array returned is the caller's own copy. Every file
/// read releases the GIL. A `with` block closes the file at its end.
#[pyclass(name = "SiftFile", module = "sfmtool.fileio", frozen)]
pub struct PySiftFile {
    path: PathBuf,
    feature_tool_metadata: FeatureToolMetadata,
    metadata: SiftMetadata,
    content_hash: SiftContentHash,
    /// The whole file, once a `read_*` call without a `count` has read it.
    /// Never locked across a file read: a thread holding it while it waits
    /// for the GIL back would block every other thread that wants it.
    data: Mutex<Option<Arc<SiftData>>>,
}

impl PySiftFile {
    /// Run `f` on the file's data: the first `count` features when a count
    /// is given, otherwise the whole file, read once and kept.
    fn with_data<R>(
        &self,
        py: Python<'_>,
        count: Option<usize>,
        f: impl FnOnce(&SiftData) -> R,
    ) -> PyResult<R> {
        if let Some(count) = count {
            let path = &self.path;
            let data = py
                .detach(|| sfmtool_sift_format::read_sift_partial(path, count))
                .map_err(read_err)?;
            return Ok(f(&data));
        }
        let cached = self.lock_data().clone();
        let data = match cached {
            Some(data) => data,
            None => {
                let path = &self.path;
                let data = Arc::new(
                    py.detach(|| sfmtool_sift_format::read_sift(path))
                        .map_err(read_err)?,
                );
                // Another thread may have read it meanwhile; keep the first.
                self.lock_data().get_or_insert(data).clone()
            }
        };
        Ok(f(&data))
    }

    fn lock_data(&self) -> std::sync::MutexGuard<'_, Option<Arc<SiftData>>> {
        // A panic while the lock was held leaves only a cache behind it.
        self.data.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn open(path: &Path) -> Result<Self, SiftError> {
        let (feature_tool_metadata, metadata, content_hash) =
            sfmtool_sift_format::read_sift_metadata(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            feature_tool_metadata,
            metadata,
            content_hash,
            data: Mutex::new(None),
        })
    }
}

#[pymethods]
impl PySiftFile {
    /// Open a `.sift` file, reading its metadata.
    ///
    /// Args:
    ///     path: `.sift` file path (str or Path).
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        py.detach(|| Self::open(&path)).map_err(read_err)
    }

    /// The path the file was opened from.
    #[getter]
    fn path(&self) -> PathBuf {
        self.path.clone()
    }

    /// `feature_tool_metadata.json` as a dict: `feature_tool`,
    /// `feature_type` and `feature_options`.
    #[getter]
    fn feature_tool_metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.feature_tool_metadata)
    }

    /// `metadata.json` as a dict: `version`, `image_name`,
    /// `image_file_xxh128`, `image_file_size`, `image_width`, `image_height`
    /// and `feature_count`.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.metadata)
    }

    /// `content_hash.json` as a dict: `metadata_xxh128`,
    /// `feature_tool_xxh128` and `content_xxh128`, as the file stores them.
    #[getter]
    fn content_hash(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serde_to_py(py, &self.content_hash)
    }

    /// The number of features the file holds, `metadata["feature_count"]`.
    #[getter]
    fn feature_count(&self) -> u32 {
        self.metadata.feature_count
    }

    /// `(N, 2)` float32 keypoint centres, the first `count` when given.
    #[pyo3(signature = (count=None))]
    fn read_positions(&self, py: Python<'_>, count: Option<usize>) -> PyResult<Py<PyAny>> {
        self.with_data(py, count, |d| {
            d.positions_xy.to_pyarray(py).into_any().unbind()
        })
    }

    /// `(N, 2, 2)` float32 affine shapes, the first `count` when given.
    #[pyo3(signature = (count=None))]
    fn read_affine_shapes(&self, py: Python<'_>, count: Option<usize>) -> PyResult<Py<PyAny>> {
        self.with_data(py, count, |d| {
            d.affine_shapes.to_pyarray(py).into_any().unbind()
        })
    }

    /// `(N, 128)` uint8 descriptors, the first `count` when given.
    #[pyo3(signature = (count=None))]
    fn read_descriptors(&self, py: Python<'_>, count: Option<usize>) -> PyResult<Py<PyAny>> {
        self.with_data(py, count, |d| {
            d.descriptors.to_pyarray(py).into_any().unbind()
        })
    }

    /// `(positions, affine_shapes)`, as `read_positions` and
    /// `read_affine_shapes` return them, from one read.
    #[pyo3(signature = (count=None))]
    fn read_positions_and_shapes<'py>(
        &self,
        py: Python<'py>,
        count: Option<usize>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let (positions, shapes) = self.with_data(py, count, |d| {
            (
                d.positions_xy.to_pyarray(py).into_any(),
                d.affine_shapes.to_pyarray(py).into_any(),
            )
        })?;
        PyTuple::new(py, [positions, shapes])
    }

    /// The `(THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)` uint8 RGB thumbnail.
    fn read_thumbnail(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.with_data(py, None, |d| {
            d.thumbnail_y_x_rgb.to_pyarray(py).into_any().unbind()
        })
    }

    /// Drop the whole-file data a `read_*` call kept. The metadata stays, and
    /// a later `read_*` call reads the file again.
    fn close(&self) {
        *self.lock_data() = None;
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: &Bound<'_, PyAny>,
        _exc: &Bound<'_, PyAny>,
        _tb: &Bound<'_, PyAny>,
    ) -> bool {
        self.close();
        false
    }

    fn __repr__(&self) -> String {
        format!("SiftFile({:?})", self.path.display().to_string())
    }
}

/// How many saves a `SiftWriteQueue` lets be in flight unless told otherwise.
/// Two lets the save of image *i* overlap the extract of image *i + 1* while
/// bounding buffered memory; a deeper queue buys no more overlap. See
/// `specs/core/features/sift.md`.
const DEFAULT_LOOKAHEAD: usize = 2;

/// A queue that compresses + writes `.sift` files on the shared rayon pool.
///
/// `submit` checks and copies the data (GIL held) and `rayon::spawn`s the
/// compression+write onto the **same** global thread pool the SIFT extract
/// uses, then returns. Because the save is a pool *task* rather than a
/// separate OS thread, it never oversubscribes the cores: one worker runs the
/// ~tens-of-ms save while the extract's `par_iter` proceeds on the rest, so the
/// save of image *i* overlaps the extract of image *i+1* without the
/// barrier busy-spin a contending external thread would cause. When
/// `lookahead` saves are already pending, `submit` first waits for the oldest
/// (backpressure). `join_oldest` and `join` wait for saves and surface their
/// errors in order.
///
/// Used as a context manager, a clean exit from the `with` block `join`s,
/// raising the first save error, and an exit by exception `drain`s instead,
/// which never raises, so it cannot replace the exception already unwinding.
/// Either way no spawned save outlives the block.
// `unsendable`: the queue holds mpsc Receivers (Send but not Sync) and is only
// ever used from the single Python thread that owns it; the spawned rayon tasks
// hold the Senders. This avoids the default pyclass `Send + Sync` requirement.
#[pyclass(unsendable, module = "sfmtool.fileio")]
pub struct SiftWriteQueue {
    pending: std::collections::VecDeque<std::sync::mpsc::Receiver<Result<(), SiftError>>>,
    lookahead: usize,
}

fn recv_write(rx: std::sync::mpsc::Receiver<Result<(), SiftError>>) -> PyResult<()> {
    match rx.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(write_err(e)),
        Err(_) => Err(PyIOError::new_err(
            "sift write task panicked before reporting a result",
        )),
    }
}

/// Await every save in `receivers`, discarding results (no GIL needed — the
/// tasks are pure Rust). Shared by `drain` and `Drop`, which only need to
/// guarantee no spawned save outlives the queue.
fn await_all(receivers: Vec<std::sync::mpsc::Receiver<Result<(), SiftError>>>) {
    for rx in receivers {
        let _ = rx.recv();
    }
}

#[pymethods]
impl SiftWriteQueue {
    /// Args:
    ///     lookahead: How many saves may be pending at once, at least 1
    ///         (default 2).
    #[new]
    #[pyo3(signature = (lookahead=DEFAULT_LOOKAHEAD))]
    fn new(lookahead: usize) -> PyResult<Self> {
        if lookahead == 0 {
            return Err(PyValueError::new_err("lookahead must be at least 1"));
        }
        Ok(Self {
            pending: std::collections::VecDeque::new(),
            lookahead,
        })
    }

    /// Number of submitted saves not yet joined.
    #[getter]
    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// How many saves may be pending at once.
    #[getter]
    fn lookahead(&self) -> usize {
        self.lookahead
    }

    /// Spawn a save onto the rayon pool, first waiting for the oldest pending
    /// save when `lookahead` are already pending.
    ///
    /// `data` is the dict `write_sift` takes, checked the same way and with
    /// the same errors, raised here before anything is spawned. An earlier
    /// save's error is raised by the wait for it.
    #[pyo3(signature = (path, data, zstd_level=SIFT_ZSTD_LEVEL))]
    fn submit(
        &mut self,
        py: Python<'_>,
        path: PathBuf,
        data: &Bound<'_, PyDict>,
        zstd_level: i32,
    ) -> PyResult<()> {
        let sift_data = sift_data_from_dict(py, data)?;
        while self.pending.len() >= self.lookahead {
            self.join_oldest(py)?;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        rayon::spawn(move || {
            let result = sfmtool_sift_format::write_sift(&path, &sift_data, zstd_level);
            let _ = tx.send(result);
        });
        self.pending.push_back(rx);
        Ok(())
    }

    /// Wait for the oldest outstanding save (backpressure); no-op if empty.
    fn join_oldest(&mut self, py: Python<'_>) -> PyResult<()> {
        match self.pending.pop_front() {
            Some(rx) => py.detach(|| recv_write(rx)),
            None => Ok(()),
        }
    }

    /// Wait for all outstanding saves, raising the first error encountered.
    fn join(&mut self, py: Python<'_>) -> PyResult<()> {
        let drained: Vec<_> = self.pending.drain(..).collect();
        py.detach(|| {
            let mut first_err = None;
            // Drain every receiver so all tasks are awaited even after an error.
            for rx in drained {
                if let Err(e) = recv_write(rx) {
                    first_err.get_or_insert(e);
                }
            }
            match first_err {
                Some(e) => Err(e),
                None => Ok(()),
            }
        })
    }

    /// Best-effort wait for all outstanding saves without raising; for a
    /// `finally`/cleanup path where a save error must not mask the exception
    /// already unwinding. Guarantees no spawned save outlives this call. No-op
    /// after a clean `join`.
    fn drain(&mut self, py: Python<'_>) {
        let drained: Vec<_> = self.pending.drain(..).collect();
        py.detach(|| await_all(drained));
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// `join` on a clean exit, `drain` on an exit by exception; never
    /// suppresses the exception.
    fn __exit__(
        &mut self,
        py: Python<'_>,
        exc_type: &Bound<'_, PyAny>,
        _exc: &Bound<'_, PyAny>,
        _tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if exc_type.is_none() {
            self.join(py)?;
        } else {
            self.drain(py);
        }
        Ok(false)
    }
}

impl Drop for SiftWriteQueue {
    /// Structural safety net: a dropped queue never leaves a spawned save
    /// racing interpreter shutdown, even if the owner skipped `join`/`drain`
    /// (the queue may also be kept alive in a traceback past its scope). The
    /// tasks are pure Rust, so awaiting them here needs no GIL.
    fn drop(&mut self) {
        await_all(self.pending.drain(..).collect());
    }
}

/// Verify integrity of a .sift file.
///
/// Returns a tuple (is_valid, error_messages). Raises `FileNotFoundError`
/// for a missing file and `OSError` for one that cannot be opened as an
/// archive.
#[pyfunction]
pub fn verify_sift(py: Python<'_>, path: PathBuf) -> PyResult<(bool, Vec<String>)> {
    py.detach(|| sfmtool_sift_format::verify_sift(&path))
        .map_err(read_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_sift, m)?)?;
    m.add_function(wrap_pyfunction!(read_sift_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(read_sift_partial, m)?)?;
    m.add_function(wrap_pyfunction!(write_sift, m)?)?;
    m.add_function(wrap_pyfunction!(verify_sift, m)?)?;
    m.add_class::<PySiftFile>()?;
    m.add_class::<SiftWriteQueue>()?;
    Ok(())
}
