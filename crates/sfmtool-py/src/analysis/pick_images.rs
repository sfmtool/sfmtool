// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for choosing the frames of a video capture: one frame's
//! features with its forest, the motion statistics of a pair of frames, and
//! the walk over a capture's sequences (``sfmtool.analysis.FrameFeatures`` /
//! ``match_frame_pair`` / ``pick_images``, with ``CaptureImage``,
//! ``CaptureFrame`` and ``CaptureSequence``; see
//! ``specs/core/analysis/frame-pair-matching.md`` and
//! ``specs/core/analysis/pick-images.md``).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use numpy::ndarray::Ix2;
use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::{
    PyFileNotFoundError, PyInterruptedError, PyOSError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use sfmtool_core::analysis::pick_images::{
    match_frame_pair as core_match_frame_pair, pick_images as core_pick_images, CaptureFrame,
    CaptureImage, CaptureSequence, ExtractingSource, FrameFeatures, FramePairOptions, FrameSource,
    FrameSourceError, PickImagesError, PickImagesOptions, ReadOnlySource,
};
use sfmtool_core::features::kdforest::KdForestParams;
use sfmtool_core::features::sift::{BuildSiftFilesError, BuildSiftFilesOptions};
use sfmtool_core::progress::{Event, Progress};
use sfmtool_sift_format::{read_sift_features, read_sift_metadata, SiftMetadata, DESCRIPTOR_DIM};

use crate::csr_args::float_array_f32;
use crate::fileio::sift::PySiftFile;
use crate::format_errors::{fault_to_py, read_err, Direction, FormatError};
use crate::helpers::{serde_to_py, value_err};
use crate::sift::extract::{feature_options_error, feature_options_from_dict};
use crate::spatial::kdforest::{extract_u8_2d, resolve_forest_params};
use crate::ProgressCounter;

/// One frame's features, with the KD-forest over its descriptors built once,
/// so that a frame matched with several others is indexed once.
///
/// Args:
///     positions_xy: (N, 2) float32 feature positions in pixels, as a
///         ``.sift`` file stores them. A float64 array is accepted and cast.
///     descriptors: (N, 128) uint8 descriptors.
///     width: The image's width in pixels.
///     height: The image's height in pixels.
///     preset: The forest preset, "balanced" (default), "fast" or
///         "accurate", as for ``sfmtool.spatial.KdForest``.
///     num_trees: Override the preset's number of trees.
///     leaf_size: Override the preset's leaf size.
///     max_leaf_checks: Override the preset's per-query budget, which
///         ``match_frame_pair`` uses for the queries against this frame's
///         forest unless it is given its own.
///     seed: Override the preset's seed for building the trees.
///
/// Raises:
///     TypeError: an array of the wrong dtype.
///     ValueError: arrays of the wrong shape or that disagree, a position
///         that is not finite, an image side of 0 or a forest parameter of 0.
///
/// See specs/core/analysis/frame-pair-matching.md.
// This is a Python docstring (rendered by `help()`), not Rust prose: its
// indented `Args:` continuation paragraphs read as Markdown indented code
// blocks, which rustdoc then tries to parse as Rust.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyclass(name = "FrameFeatures", module = "sfmtool.analysis", frozen)]
pub struct PyFrameFeatures {
    inner: FrameFeatures,
}

/// The forest parameters a `FrameFeatures` constructor was given.
fn forest_params(
    preset: Option<&str>,
    num_trees: Option<usize>,
    leaf_size: Option<usize>,
    max_leaf_checks: Option<usize>,
    seed: Option<u64>,
) -> PyResult<KdForestParams> {
    resolve_forest_params(
        preset,
        "balanced",
        num_trees,
        leaf_size,
        max_leaf_checks,
        seed,
    )
}

#[pymethods]
impl PyFrameFeatures {
    #[new]
    #[pyo3(signature = (positions_xy, descriptors, width, height, *, preset=None, num_trees=None,
                        leaf_size=None, max_leaf_checks=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        positions_xy: &Bound<'_, PyAny>,
        descriptors: &Bound<'_, PyAny>,
        width: u32,
        height: u32,
        preset: Option<&str>,
        num_trees: Option<usize>,
        leaf_size: Option<usize>,
        max_leaf_checks: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let forest = forest_params(preset, num_trees, leaf_size, max_leaf_checks, seed)?;
        let positions = float_array_f32::<Ix2>(
            positions_xy,
            "positions_xy must be a (N, 2) float32 or float64 array",
        )?;
        if positions.shape[1] != 2 {
            return Err(PyValueError::new_err(format!(
                "positions_xy must have shape (N, 2), got {:?}",
                positions.shape
            )));
        }
        let descriptors = extract_u8_2d(descriptors, "descriptors")?;
        if descriptors.shape()[1] != DESCRIPTOR_DIM {
            return Err(PyValueError::new_err(format!(
                "descriptors must have shape (N, {DESCRIPTOR_DIM}), got {:?}",
                descriptors.shape()
            )));
        }
        let descriptors = to_contiguous!(descriptors).into_owned();
        let positions: Vec<[f32; 2]> = positions.data.as_chunks::<2>().0.to_vec();
        let inner = py
            .detach(move || FrameFeatures::new(positions, descriptors, width, height, forest))
            .map_err(value_err)?;
        Ok(Self { inner })
    }

    /// The features of a ``.sift`` file, with the image size its metadata
    /// records.
    ///
    /// Args:
    ///     sift: A ``.sift`` path (str or Path) or a ``SiftFile``.
    ///     preset, num_trees, leaf_size, max_leaf_checks, seed: The forest,
    ///         as for the constructor.
    ///
    /// Raises:
    ///     TypeError: ``sift`` is neither a path nor a ``SiftFile``.
    ///     OSError: the file cannot be read, as ``sfmtool.fileio.read_sift``
    ///         raises it.
    ///     ValueError: a forest parameter of 0, or a file whose features and
    ///         metadata disagree.
    #[allow(rustdoc::invalid_rust_codeblocks)]
    #[staticmethod]
    #[pyo3(signature = (sift, *, preset=None, num_trees=None, leaf_size=None,
                        max_leaf_checks=None, seed=None))]
    fn from_sift(
        py: Python<'_>,
        sift: &Bound<'_, PyAny>,
        preset: Option<&str>,
        num_trees: Option<usize>,
        leaf_size: Option<usize>,
        max_leaf_checks: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let forest = forest_params(preset, num_trees, leaf_size, max_leaf_checks, seed)?;
        let (path, metadata): (PathBuf, Option<SiftMetadata>) =
            if let Ok(file) = sift.cast::<PySiftFile>() {
                let file = file.get();
                (
                    file.sift_path().to_path_buf(),
                    Some(file.sift_metadata().clone()),
                )
            } else if let Ok(path) = sift.extract::<PathBuf>() {
                (path, None)
            } else {
                return Err(PyTypeError::new_err(
                    "sift must be a .sift path (str or Path) or a SiftFile",
                ));
            };
        let features = py
            .detach(|| {
                let metadata = match metadata {
                    Some(metadata) => metadata,
                    None => read_sift_metadata(&path)?.1,
                };
                read_sift_features(&path).map(|features| (features, metadata))
            })
            .map_err(read_err)?;
        let inner = py
            .detach(move || FrameFeatures::from_sift(features.0, &features.1, forest))
            .map_err(value_err)?;
        Ok(Self { inner })
    }

    /// The image's width in pixels.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    /// The image's height in pixels.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    /// The number of features.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameFeatures({} features, {} x {})",
            self.inner.len(),
            self.inner.width(),
            self.inner.height()
        )
    }
}

/// Match two frames' features and measure the motion between them.
///
/// The matches are the mutual best matches of a k = 2 query each way that
/// pass Lowe's ratio test both ways. Options left as ``None`` take the
/// defaults of the Rust ``FramePairOptions``: ``ratio`` 0.8 and the queried
/// forest's own ``max_leaf_checks``.
///
/// Args:
///     a: The first frame's ``FrameFeatures``.
///     b: The second frame's, of the same image size.
///     ratio: Lowe's ratio on descriptor distances, in (0, 1].
///     max_leaf_checks: The per-query budget of distance computations.
///
/// Returns:
///     A dict: ``matches`` (int, the number of mutual matches) and
///     ``step_percent`` (float, their median displacement in percent of the
///     long side, NaN with no matches).
///
/// Raises:
///     ValueError: an option out of range, or frames of different sizes.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyfunction]
#[pyo3(signature = (a, b, *, ratio=None, max_leaf_checks=None))]
pub fn match_frame_pair<'py>(
    py: Python<'py>,
    a: PyRef<'py, PyFrameFeatures>,
    b: PyRef<'py, PyFrameFeatures>,
    ratio: Option<f32>,
    max_leaf_checks: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let defaults = FramePairOptions::default();
    let options = FramePairOptions {
        ratio: ratio.unwrap_or(defaults.ratio),
        max_leaf_checks: max_leaf_checks.or(defaults.max_leaf_checks),
    };
    let (a, b) = (&a.inner, &b.inner);
    let stats = py
        .detach(|| core_match_frame_pair(a, b, &options))
        .map_err(value_err)?;
    let out = PyDict::new(py);
    out.set_item("matches", stats.matches)?;
    out.set_item("step_percent", stats.step_percent)?;
    Ok(out)
}

/// Where one image of a capture is and where its ``.sift`` file goes.
///
/// Args:
///     image_path: The image file (str or Path).
///     sift_path: Its ``.sift`` file, which ``pick_images`` extracts when it
///         is given feature options, and otherwise reads.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyclass(name = "CaptureImage", module = "sfmtool.analysis", frozen)]
pub struct PyCaptureImage {
    inner: CaptureImage,
}

#[pymethods]
impl PyCaptureImage {
    #[new]
    fn new(image_path: PathBuf, sift_path: PathBuf) -> Self {
        Self {
            inner: CaptureImage {
                image_path,
                sift_path,
            },
        }
    }

    /// The image file.
    #[getter]
    fn image_path(&self) -> PathBuf {
        self.inner.image_path.clone()
    }

    /// The image's ``.sift`` file.
    #[getter]
    fn sift_path(&self) -> PathBuf {
        self.inner.sift_path.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CaptureImage({:?}, {:?})",
            self.inner.image_path.display().to_string(),
            self.inner.sift_path.display().to_string()
        )
    }
}

/// One instant of a capture: one image, or one per rig sensor with the
/// reference sensor's first.
///
/// Args:
///     number: The file number the frame's images carry.
///     images: The frame's ``CaptureImage`` objects, one per sensor.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyclass(name = "CaptureFrame", module = "sfmtool.analysis", frozen)]
pub struct PyCaptureFrame {
    inner: CaptureFrame,
    images: Py<PyTuple>,
}

#[pymethods]
impl PyCaptureFrame {
    #[new]
    fn new(py: Python<'_>, number: i64, images: Vec<Bound<'_, PyCaptureImage>>) -> PyResult<Self> {
        let inner = CaptureFrame {
            number,
            images: images.iter().map(|i| i.get().inner.clone()).collect(),
        };
        Ok(Self {
            inner,
            images: PyTuple::new(py, images)?.unbind(),
        })
    }

    /// The file number.
    #[getter]
    fn number(&self) -> i64 {
        self.inner.number
    }

    /// The frame's images, a tuple of ``CaptureImage``, the reference
    /// sensor's first.
    #[getter]
    fn images(&self, py: Python<'_>) -> Py<PyTuple> {
        self.images.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "CaptureFrame({}, {} images)",
            self.inner.number,
            self.inner.images.len()
        )
    }
}

/// One sequence of a capture, in file-number order, with the pattern each
/// sensor's list entry uses.
///
/// Args:
///     patterns: One pattern per sensor, in the order of each frame's images.
///     frames: The ``CaptureFrame`` objects, in file-number order.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyclass(name = "CaptureSequence", module = "sfmtool.analysis", frozen)]
pub struct PyCaptureSequence {
    inner: CaptureSequence,
    frames: Py<PyTuple>,
}

#[pymethods]
impl PyCaptureSequence {
    #[new]
    fn new(
        py: Python<'_>,
        patterns: Vec<String>,
        frames: Vec<Bound<'_, PyCaptureFrame>>,
    ) -> PyResult<Self> {
        let inner = CaptureSequence {
            patterns,
            frames: frames.iter().map(|f| f.get().inner.clone()).collect(),
        };
        Ok(Self {
            inner,
            frames: PyTuple::new(py, frames)?.unbind(),
        })
    }

    /// One pattern per sensor.
    #[getter]
    fn patterns(&self) -> Vec<String> {
        self.inner.patterns.clone()
    }

    /// The frames, a tuple of ``CaptureFrame``, so that
    /// ``sequence.frames[id["frame"]]`` is the frame a report's ``id`` names.
    #[getter]
    fn frames(&self, py: Python<'_>) -> Py<PyTuple> {
        self.frames.clone_ref(py)
    }

    fn __len__(&self) -> usize {
        self.inner.frames.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "CaptureSequence({:?}, {} frames)",
            self.inner.patterns,
            self.inner.frames.len()
        )
    }
}

/// Choose the frames of a video capture a Structure-from-Motion solve should
/// use, from the SIFT features of the frames.
///
/// Walks each sequence from its first frame, choosing frames about
/// ``target_step_percent`` of the image's long side apart, with no step
/// beyond the next frame keeping fewer than ``min_step_matches`` mutual
/// matches, and a frame blurred against its neighbouring frames swapped for
/// a sharper one. Options left as ``None`` take the defaults of the Rust
/// ``PickImagesOptions``.
///
/// With ``feature_options`` and ``feature_prefix_dir``, the workspace's
/// settings for the ``sfmtool`` feature tool, each frame's ``.sift`` files
/// are extracted as ``build_sift_files`` extracts them, on a pool of workers
/// that runs ahead of the walk on the frames it predicts it will read, and
/// stay on disk; a file at least as new as its image is used as it is.
/// Without them, every ``.sift`` file the walk reads must already exist.
/// The frames chosen are the same either way, and for any number of workers.
///
/// Args:
///     sequences: The capture's ``CaptureSequence`` objects.
///     feature_options: The workspace's ``feature_options`` dict for the
///         ``sfmtool`` tool, to extract the files the walk reads.
///     feature_prefix_dir: The workspace's ``feature_prefix_dir``, given
///         with ``feature_options``. Each image's ``sift_path`` must be
///         ``<image dir>/<feature_prefix_dir>/<image name>.sift``.
///     concurrent_images: How many frames the workers extract at once, with
///         ``feature_options``; ``None`` takes ``SFMTOOL_SIFT_EXTRACT_WORKERS``
///         or the core count capped by memory, as ``build_sift_files`` does.
///     target_step_percent: The step to aim for, in percent of the long
///         side (10).
///     step_band: How far a step may be from the target, as a fraction
///         (0.3).
///     neighbour_blur_ratio: A frame about to be accepted is blurred against
///         its neighbours when its sharpness is below this times the sharpest
///         frame's within 2 positions (0.85).
///     min_step_matches: The match floor, the fewest mutual matches a step
///         beyond the next frame keeps (600), at least 1.
///     ratio: Lowe's ratio on descriptor distances, as for
///         ``match_frame_pair``.
///     preset, num_trees, leaf_size, max_leaf_checks: Each frame's forest,
///         as for ``FrameFeatures``.
///     progress: A ``ProgressCounter``, set to the number of frames read.
///
/// Returns:
///     The report as a dict of plain values, which ``json.dump`` writes as
///     it is: ``sequences`` (per sequence: ``chosen``, ``steps``, ``swaps`` and
///     ``frames_read``), ``frames``
///     (every frame read, with its readings), ``pairs`` (every pair matched,
///     with its statistics), ``frames_extracted`` (per entry of ``frames``,
///     whether its ``.sift`` files were written in this call) and
///     ``frames_extracted_unread`` (how many frames had ``.sift`` files
///     written ahead of the walk that it never read). A number that is not
///     finite is ``None``.
///
/// Raises:
///     The exception a signal handler raised while the walk ran
///     (``KeyboardInterrupt`` for Ctrl+C), which stops it and its workers.
///     FileNotFoundError: without feature options, a ``.sift`` file the walk
///         is about to read does not exist.
///     OSError: a ``.sift`` file that cannot be read, as
///         ``sfmtool.fileio.read_sift`` raises it, a folder the ``.sift``
///         files go in that cannot be listed, or an image that cannot be
///         read or a ``.sift`` file that cannot be written while extracting.
///     InterruptedError: the walk was cancelled otherwise.
///     ValueError: an option out of range, invalid or version 1 feature
///         options, one of ``feature_options`` and ``feature_prefix_dir``
///         without the other, a malformed sequence, a ``sift_path`` that is
///         not where the extraction writes it, version 1 ``.sift`` files in a
///         folder or read, or two frames whose reference images differ in
///         size.
///
/// The GIL is released while the walk runs, and taken for each check for a
/// signal.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyfunction]
#[pyo3(signature = (sequences, *, feature_options=None, feature_prefix_dir=None,
                    concurrent_images=None, target_step_percent=None, step_band=None,
                    neighbour_blur_ratio=None, min_step_matches=None, ratio=None, preset=None,
                    num_trees=None, leaf_size=None, max_leaf_checks=None, progress=None))]
#[allow(clippy::too_many_arguments)]
pub fn pick_images<'py>(
    py: Python<'py>,
    sequences: Vec<Bound<'py, PyCaptureSequence>>,
    feature_options: Option<Bound<'py, PyDict>>,
    feature_prefix_dir: Option<PathBuf>,
    concurrent_images: Option<usize>,
    target_step_percent: Option<f64>,
    step_band: Option<f64>,
    neighbour_blur_ratio: Option<f64>,
    min_step_matches: Option<usize>,
    ratio: Option<f32>,
    preset: Option<&str>,
    num_trees: Option<usize>,
    leaf_size: Option<usize>,
    max_leaf_checks: Option<usize>,
    progress: Option<ProgressCounter>,
) -> PyResult<Py<PyAny>> {
    let defaults = PickImagesOptions::default();
    let pair = FramePairOptions::default();
    let options = PickImagesOptions {
        target_step_percent: target_step_percent.unwrap_or(defaults.target_step_percent),
        step_band: step_band.unwrap_or(defaults.step_band),
        neighbour_blur_ratio: neighbour_blur_ratio.unwrap_or(defaults.neighbour_blur_ratio),
        min_step_matches: min_step_matches.unwrap_or(defaults.min_step_matches),
        forest: forest_params(preset, num_trees, leaf_size, max_leaf_checks, None)?,
        pair: FramePairOptions {
            ratio: ratio.unwrap_or(pair.ratio),
            max_leaf_checks: pair.max_leaf_checks,
        },
        prefetch: defaults.prefetch,
    };
    let mut source: Box<dyn FrameSource + Send> = match (feature_options, feature_prefix_dir) {
        (Some(feature_options), Some(feature_prefix_dir)) => Box::new(
            ExtractingSource::new(BuildSiftFilesOptions {
                feature_options: feature_options_from_dict(&feature_options)?,
                feature_prefix_dir,
                concurrent_images,
            })
            .map_err(|e| match e {
                BuildSiftFilesError::FeatureOptions(e) => feature_options_error(e),
                other => PyValueError::new_err(other.to_string()),
            })?,
        ),
        (None, None) if concurrent_images.is_none() => Box::new(ReadOnlySource),
        (None, None) => {
            return Err(PyValueError::new_err(
                "concurrent_images sets how many frames are extracted at once; give it with \
                 feature_options and feature_prefix_dir",
            ))
        }
        _ => {
            return Err(PyValueError::new_err(
                "give feature_options and feature_prefix_dir together, to extract the .sift \
                 files, or neither, to read .sift files that already exist",
            ))
        }
    };
    let sequences: Vec<CaptureSequence> = sequences.iter().map(|s| s.get().inner.clone()).collect();

    // Every event, among them each pair matched and every 50 ms of a wait on
    // the workers, takes the GIL to run Python's signal handlers. One that
    // raises stops the walk, and its exception is raised in place of the
    // cancellation.
    let counter = progress.as_ref().map(ProgressCounter::handle);
    let cancel = AtomicBool::new(false);
    let signal: Mutex<Option<PyErr>> = Mutex::new(None);
    let sink = |event: Event<'_>| {
        if let (
            Event::Count {
                done,
                unit: "frames",
                ..
            },
            Some(counter),
        ) = (event, &counter)
        {
            counter.store(done as usize, Ordering::Relaxed);
        }
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        if let Err(e) = Python::attach(|py| py.check_signals()) {
            *signal.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
            cancel.store(true, Ordering::Relaxed);
        }
    };
    let result = py.detach(|| {
        let progress = Progress::to(&sink).cancelled_by(&cancel);
        core_pick_images(&sequences, &mut *source, &options, &progress)
    });
    match result {
        Ok(report) => serde_to_py(py, &report),
        Err(e) => Err(match e {
            PickImagesError::Cancelled => signal
                .into_inner()
                .unwrap_or_else(|p| p.into_inner())
                .unwrap_or_else(|| PyInterruptedError::new_err(e.to_string())),
            PickImagesError::ReadSift { ref source, .. } => {
                fault_to_py(source.fault(), e.to_string(), Direction::Read)
            }
            PickImagesError::ListFolder { ref source, .. } => {
                PyErr::from(std::io::Error::new(source.kind(), e.to_string()))
            }
            PickImagesError::Source(FrameSourceError::Missing { .. }) => {
                PyFileNotFoundError::new_err(e.to_string())
            }
            PickImagesError::Source(FrameSourceError::Extract { .. }) => {
                PyOSError::new_err(e.to_string())
            }
            PickImagesError::Source(FrameSourceError::Panicked { .. }) => {
                PyRuntimeError::new_err(e.to_string())
            }
            other => value_err(other),
        }),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFrameFeatures>()?;
    m.add_function(wrap_pyfunction!(match_frame_pair, m)?)?;
    m.add_class::<PyCaptureImage>()?;
    m.add_class::<PyCaptureFrame>()?;
    m.add_class::<PyCaptureSequence>()?;
    m.add_function(wrap_pyfunction!(pick_images, m)?)?;
    Ok(())
}
