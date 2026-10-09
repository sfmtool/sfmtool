// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `write_web_export`: the data files of `sfm web-export`.

use std::path::PathBuf;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sfmtool_core::progress::Progress;
use sfmtool_core::web_export::{write_web_export as core_write, WebExportError, WebExportOptions};

use crate::PySfmrReconstruction;

/// Write the data files of a web export of ``recon`` into ``out_dir``:
/// ``scene.json`` and the JPEG atlas pages ``patches-<n>.jpg`` and
/// ``thumbs-<n>.jpg``. The viewer page and script are not written here; the
/// ``sfm web-export`` command copies them in beside these.
///
/// The directory is created when missing. Other files in it are left alone.
///
/// Args:
///     recon: The reconstruction.
///     out_dir: The directory to write into.
///     patches: Write the patch atlas; off, every point is a splat.
///     thumbnails: Write the thumbnail atlas; off, frustums are wireframes.
///     patch_size: Resample patch tiles to this edge in texels.
///     jpeg_quality: JPEG quality of both atlases, 1 to 100.
///     max_points: Keep only this many points, those with the most
///         observations.
///     start_image: Name of the image whose camera the page opens through.
///     source_name: The file name ``scene.json`` records as its source.
///     generator: What ``scene.json`` records as having written it.
///
/// Returns:
///     A dict: ``files`` (list of ``(name, bytes)`` in write order,
///     ``scene.json`` first), ``points``, ``points_at_infinity``,
///     ``points_left_out``, ``patches``, ``patch_size``,
///     ``patch_bitmaps_rendered``, ``cameras``, ``thumbnails`` (a dict of
///     ``file``, ``sift``, ``photographs``, ``placeholders`` row counts),
///     ``decoded_atlas_bytes`` and ``warnings`` (list of sentences).
///
/// Raises:
///     ValueError: An option does not fit ``recon`` (an unknown
///         ``start_image``, a quality out of range).
///     IOError: A file could not be written.
// This is a Python docstring (rendered by `help()`), not Rust prose: its
// indented continuation paragraphs read as Markdown code blocks.
#[allow(rustdoc::invalid_rust_codeblocks)]
#[pyfunction]
#[pyo3(signature = (
    recon, out_dir, *, patches=true, thumbnails=true, patch_size=None, jpeg_quality=85,
    max_points=None, start_image=None, source_name=None, generator=None
))]
#[allow(clippy::too_many_arguments)]
pub fn write_web_export<'py>(
    py: Python<'py>,
    recon: &PySfmrReconstruction,
    out_dir: PathBuf,
    patches: bool,
    thumbnails: bool,
    patch_size: Option<usize>,
    jpeg_quality: u8,
    max_points: Option<usize>,
    start_image: Option<String>,
    source_name: Option<String>,
    generator: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut options = WebExportOptions {
        patches,
        thumbnails,
        patch_size,
        jpeg_quality,
        max_points,
        start_image,
        source_name,
        ..WebExportOptions::default()
    };
    if let Some(generator) = generator {
        options.generator = generator;
    }
    let inner = &recon.inner;
    let report = py
        .detach(|| core_write(inner, &out_dir, &options, &Progress::none()))
        .map_err(|e| match e {
            WebExportError::Io(_) => PyIOError::new_err(e.to_string()),
            _ => PyValueError::new_err(e.to_string()),
        })?;
    let dict = PyDict::new(py);
    dict.set_item("files", PyList::new(py, &report.files)?)?;
    dict.set_item("points", report.points)?;
    dict.set_item("points_at_infinity", report.points_at_infinity)?;
    dict.set_item("points_left_out", report.points_left_out)?;
    dict.set_item("patches", report.patches)?;
    dict.set_item("patch_size", report.patch_size)?;
    dict.set_item("patch_bitmaps_rendered", report.patch_bitmaps_rendered)?;
    dict.set_item("cameras", report.cameras)?;
    let thumbs = PyDict::new(py);
    thumbs.set_item("file", report.thumbnails.file)?;
    thumbs.set_item("sift", report.thumbnails.sift)?;
    thumbs.set_item("photographs", report.thumbnails.photographs)?;
    thumbs.set_item("placeholders", report.thumbnails.placeholders)?;
    dict.set_item("thumbnails", thumbs)?;
    dict.set_item("decoded_atlas_bytes", report.decoded_atlas_bytes)?;
    dict.set_item("warnings", PyList::new(py, &report.warnings)?)?;
    Ok(dict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(write_web_export, m)?)?;
    Ok(())
}
