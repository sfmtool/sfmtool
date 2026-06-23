// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for `.sift` file I/O.

use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

use sift_format::{self, FeatureToolMetadata, SiftContentHash, SiftData, SiftMetadata};

use crate::helpers::{get_item, py_to_serde, serde_to_py};

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
///   descriptors (N,128 uint8), thumbnail_y_x_rgb (128,128,3 uint8).
#[pyfunction]
pub fn read_sift(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let data = sift_format::read_sift(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    sift_data_to_py(py, data)
}

/// Read only metadata from a .sift file (fast, no binary data).
///
/// Returns a dict with keys: feature_tool_metadata, metadata, content_hash.
#[pyfunction]
pub fn read_sift_metadata(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let (tool_meta, meta, hash) = sift_format::read_sift_metadata(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

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
    let data = sift_format::read_sift_partial(&path, count)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    sift_data_to_py(py, data)
}

/// Copy a Python dict of numpy arrays + metadata into an owned `SiftData`.
///
/// Holds the GIL (it reads the numpy buffers), but the result is fully owned
/// and `Send`, so the caller can compress/write it off the GIL — or on another
/// thread / rayon worker.
fn sift_data_from_dict(py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<SiftData> {
    let feature_tool_metadata: FeatureToolMetadata =
        py_to_serde(py, &get_item(data, "feature_tool_metadata")?)?;
    let metadata: SiftMetadata = py_to_serde(py, &get_item(data, "metadata")?)?;

    let positions_xy: PyReadonlyArray2<f32> = get_item(data, "positions_xy")?.extract()?;
    let affine_shapes: PyReadonlyArray3<f32> = get_item(data, "affine_shapes")?.extract()?;
    let descriptors: PyReadonlyArray2<u8> = get_item(data, "descriptors")?.extract()?;
    let thumbnail_y_x_rgb: PyReadonlyArray3<u8> = get_item(data, "thumbnail_y_x_rgb")?.extract()?;

    Ok(SiftData {
        feature_tool_metadata,
        metadata,
        content_hash: SiftContentHash::default(),
        positions_xy: positions_xy.as_array().to_owned(),
        affine_shapes: affine_shapes.as_array().to_owned(),
        descriptors: descriptors.as_array().to_owned(),
        thumbnail_y_x_rgb: thumbnail_y_x_rgb.as_array().to_owned(),
    })
}

/// Write a .sift file from a dict of numpy arrays and metadata.
///
/// The dict should have the same keys as returned by `read_sift`.
/// The `content_hash` key is ignored (recomputed on write).
#[pyfunction]
#[pyo3(signature = (path, data, zstd_level=5))]
pub fn write_sift(
    py: Python<'_>,
    path: PathBuf,
    data: &Bound<'_, PyDict>,
    zstd_level: i32,
) -> PyResult<()> {
    let sift_data = sift_data_from_dict(py, data)?;

    // The arrays are now owned, so release the GIL around the CPU-bound
    // zstd/ZIP compression and file write.
    py.detach(|| sift_format::write_sift(&path, &sift_data, zstd_level))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// A queue that compresses + writes `.sift` files on the shared rayon pool.
///
/// `submit` copies the data (GIL held) and `rayon::spawn`s the
/// compression+write onto the **same** global thread pool the SIFT extract
/// uses, then returns immediately. Because the save is a pool *task* rather than
/// a separate OS thread, it never oversubscribes the cores: one worker runs the
/// ~tens-of-ms save while the extract's `par_iter` proceeds on the rest, so the
/// save of image *i* overlaps the extract of image *i+1* without the
/// barrier busy-spin a contending external thread would cause. `join_oldest`
/// (backpressure) and `join` wait for saves and surface their errors in order.
// `unsendable`: the queue holds mpsc Receivers (Send but not Sync) and is only
// ever used from the single Python thread that owns it; the spawned rayon tasks
// hold the Senders. This avoids the default pyclass `Send + Sync` requirement.
#[pyclass(unsendable, module = "sfmtool.io")]
pub struct SiftWriteQueue {
    pending: std::collections::VecDeque<std::sync::mpsc::Receiver<Result<(), String>>>,
}

fn recv_write(rx: std::sync::mpsc::Receiver<Result<(), String>>) -> PyResult<()> {
    match rx.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(pyo3::exceptions::PyIOError::new_err(e)),
        Err(_) => Err(pyo3::exceptions::PyIOError::new_err(
            "sift write task panicked before reporting a result",
        )),
    }
}

/// Await every save in `receivers`, discarding results (no GIL needed — the
/// tasks are pure Rust). Shared by `drain` and `Drop`, which only need to
/// guarantee no spawned save outlives the queue.
fn await_all(receivers: Vec<std::sync::mpsc::Receiver<Result<(), String>>>) {
    for rx in receivers {
        let _ = rx.recv();
    }
}

#[pymethods]
impl SiftWriteQueue {
    #[new]
    fn new() -> Self {
        Self {
            pending: std::collections::VecDeque::new(),
        }
    }

    /// Number of submitted saves not yet joined.
    #[getter]
    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Spawn a save onto the rayon pool; returns without waiting.
    #[pyo3(signature = (path, data, zstd_level=5))]
    fn submit(
        &mut self,
        py: Python<'_>,
        path: PathBuf,
        data: &Bound<'_, PyDict>,
        zstd_level: i32,
    ) -> PyResult<()> {
        let sift_data = sift_data_from_dict(py, data)?;
        let (tx, rx) = std::sync::mpsc::channel();
        rayon::spawn(move || {
            let result =
                sift_format::write_sift(&path, &sift_data, zstd_level).map_err(|e| e.to_string());
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
/// Returns a tuple (is_valid, error_messages).
#[pyfunction]
pub fn verify_sift(path: PathBuf) -> PyResult<(bool, Vec<String>)> {
    sift_format::verify_sift(&path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_sift, m)?)?;
    m.add_function(wrap_pyfunction!(read_sift_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(read_sift_partial, m)?)?;
    m.add_function(wrap_pyfunction!(write_sift, m)?)?;
    m.add_function(wrap_pyfunction!(verify_sift, m)?)?;
    m.add_class::<SiftWriteQueue>()?;
    Ok(())
}
