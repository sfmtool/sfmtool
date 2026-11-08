// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the files in a workspace root that configure the workspace as
//! a whole (`sfmtool_core::workspace`): its named image lists and their slices,
//! the rules of the feature options its `.sfm-workspace.json` records, and the
//! upward search that finds the workspace a path is in.

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use pyo3::exceptions::{PyInterruptedError, PyValueError};
use pyo3::prelude::*;
use sfmtool_core::progress::{Event, Progress};
use sfmtool_core::workspace::{
    is_named_image_list_name, split_named_image_list_value, NamedImageList, NamedImageListEntry,
    NamedImageListMember, NamedImageListSlice, NamedImageLists, NamedImageListsError,
    NAMED_IMAGE_LISTS_FILE_NAME, WORKSPACE_VERSION,
};

use crate::helpers::{py_to_serde, serde_to_py};
use crate::sift::image_readings::PyImageToGray;
use crate::{ProgressCounter, PyRangeExpr};

/// The Python exception for `e`: the `OSError` subclass for a failed read or
/// write, `InterruptedError` for a cancelled resolve, and `ValueError` for a
/// file or new lists that break the format's rules, a list that cannot be
/// resolved, and a list the file already defines.
fn to_py_err(e: NamedImageListsError) -> PyErr {
    match &e {
        NamedImageListsError::Io { source, .. } => {
            PyErr::from(std::io::Error::new(source.kind(), e.to_string()))
        }
        NamedImageListsError::Cancelled => PyInterruptedError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// One member of a resolved named image list.
///
/// `path` is the image's workspace-relative, forward-slash path; `entry` is
/// the index, from 0, of the list entry that matched it; `frame` is the frame
/// number that entry's frame field captured, or `None` for an entry without
/// one.
#[pyclass(
    name = "NamedImageListMember",
    module = "sfmtool.workspace",
    frozen,
    get_all
)]
pub struct PyNamedImageListMember {
    path: String,
    entry: usize,
    frame: Option<i64>,
}

#[pymethods]
impl PyNamedImageListMember {
    fn __repr__(&self) -> String {
        let frame = self
            .frame
            .map_or_else(|| "None".to_string(), |f| f.to_string());
        format!(
            "NamedImageListMember(path={:?}, entry={}, frame={frame})",
            self.path, self.entry
        )
    }
}

fn to_py_members(members: Vec<NamedImageListMember>) -> Vec<PyNamedImageListMember> {
    members
        .into_iter()
        .map(|m| PyNamedImageListMember {
            path: m.path,
            entry: m.entry,
            frame: m.frame,
        })
        .collect()
}

/// A slice of a named image list's images by position, Python's
/// `[start:stop:step]`.
///
/// Build one with `NamedImageListSlice(start=None, stop=None, step=None)`, or
/// read its bracketed text with `NamedImageListSlice.parse("[:30]")`; a `-r`
/// value with a name before the slice is read by
/// `split_named_image_list_value`. `start`, `stop` and `step` are `None` where
/// the part was left empty, and the step is never 0. `str()` writes it back as
/// `[start:stop]` or `[start:stop:step]`.
#[pyclass(
    name = "NamedImageListSlice",
    module = "sfmtool.workspace",
    frozen,
    eq,
    hash
)]
#[derive(PartialEq, Hash)]
pub struct PyNamedImageListSlice {
    inner: NamedImageListSlice,
}

#[pymethods]
impl PyNamedImageListSlice {
    /// The slice `[start:stop:step]`. Raises `ValueError` for a step of 0.
    #[new]
    #[pyo3(signature = (start=None, stop=None, step=None))]
    fn new(start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> PyResult<Self> {
        NamedImageListSlice::new(start, stop, step)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Read the bracketed text of a slice, `[` and `]` included, such as
    /// `"[:30]"` or `"[::3]"`. Raises `ValueError`, naming `text` and what is
    /// wrong with it, for text that is not `[start:stop]` or
    /// `[start:stop:step]` with each part empty or an integer, and for a step
    /// of 0.
    #[staticmethod]
    fn parse(text: &str) -> PyResult<Self> {
        NamedImageListSlice::parse(text)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn start(&self) -> Option<i64> {
        self.inner.start()
    }

    #[getter]
    fn stop(&self) -> Option<i64> {
        self.inner.stop()
    }

    #[getter]
    fn step(&self) -> Option<i64> {
        self.inner.step()
    }

    /// The indexes the slice keeps in a sequence of `length` items, in the
    /// slice's order: `list(range(*slice(start, stop, step).indices(length)))`.
    fn indices(&self, length: usize) -> Vec<usize> {
        self.inner.indices(length)
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        let part = |p: Option<i64>| p.map_or_else(|| "None".to_string(), |v| v.to_string());
        format!(
            "NamedImageListSlice(start={}, stop={}, step={})",
            part(self.inner.start()),
            part(self.inner.stop()),
            part(self.inner.step())
        )
    }
}

/// One entry `write_lists` takes from Python: a pattern and a `RangeExpr` or
/// `None`.
type PyEntry = (String, Option<PyRangeExpr>);

/// A workspace's named image lists, read from the
/// `.sfm-named-image-lists.json` in its root and checked whole.
///
/// Build one with `NamedImageLists.read(workspace_dir)`. The format, the
/// validation rules and the order of a list's members are in
/// `specs/workspace/named-image-lists.md`.
#[pyclass(name = "NamedImageLists", module = "sfmtool.workspace", frozen)]
pub struct PyNamedImageLists {
    inner: NamedImageLists,
}

#[pymethods]
impl PyNamedImageLists {
    /// Read the lists file of the workspace whose root is `workspace_dir`.
    ///
    /// Returns `None` when the workspace has no lists file. Raises
    /// `ValueError` when `workspace_dir` holds no `.sfm-workspace.json`, or
    /// when the file is not valid JSON, has a version other than 1 or breaks
    /// a rule of the format, with every problem found in the message; and the
    /// `OSError` subclass for its kind when the file cannot be read.
    #[staticmethod]
    fn read(py: Python<'_>, workspace_dir: PathBuf) -> PyResult<Option<Self>> {
        py.detach(|| NamedImageLists::read(&workspace_dir))
            .map(|lists| lists.map(|inner| Self { inner }))
            .map_err(to_py_err)
    }

    /// Add `lists` to the lists file of the workspace whose root is
    /// `workspace_dir`, or replace lists of the same names in it, creating the
    /// file when the workspace has none, and return the file's path.
    ///
    /// `lists` is a sequence of `(name, entries)` pairs, and each entry a
    /// `(pattern, range)` pair whose range is a `RangeExpr` or `None`; a
    /// dict's `items()` in a `list` will do. The file is rewritten whole:
    /// every list it held keeps its place and order and its text, a replaced
    /// list takes the place of the one it replaces, and new lists follow in
    /// the order given. It is written to a temporary file beside it and
    /// renamed over it, so a failed write leaves the old file.
    ///
    /// Raises `ValueError`, and writes nothing, for new lists that break the
    /// format's rules or a name given twice, for a name the file already
    /// defines unless `replace` is true, for a `workspace_dir` that holds no
    /// `.sfm-workspace.json`, and for an existing file that fails the
    /// reader's checks; and the `OSError` subclass for its kind when the file
    /// cannot be read or written.
    #[staticmethod]
    #[pyo3(signature = (workspace_dir, lists, replace=false))]
    fn write_lists(
        py: Python<'_>,
        workspace_dir: PathBuf,
        lists: Vec<(String, Vec<PyEntry>)>,
        replace: bool,
    ) -> PyResult<PathBuf> {
        let lists: Vec<NamedImageList> = lists
            .into_iter()
            .map(|(name, entries)| {
                let entries = entries
                    .into_iter()
                    .map(|(pattern, range)| {
                        NamedImageListEntry::new(pattern, range.map(|r| r.inner().clone()))
                    })
                    .collect();
                NamedImageList::new(name, entries)
            })
            .collect();
        py.detach(|| NamedImageLists::write_lists(&workspace_dir, &lists, replace))
            .map_err(to_py_err)
    }

    /// The lists file's path.
    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.path().to_path_buf()
    }

    /// The workspace root the lists resolve against.
    #[getter]
    fn workspace_dir(&self) -> PathBuf {
        self.inner.workspace_dir().to_path_buf()
    }

    /// The list names, in the file's order.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner
            .lists()
            .iter()
            .map(|list| list.name().to_string())
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.lists().len()
    }

    /// Whether a list is called `name`, compared exactly, case included.
    fn __contains__(&self, name: &str) -> bool {
        self.inner.get(name).is_some()
    }

    fn __repr__(&self) -> String {
        format!(
            "NamedImageLists({:?}, names={:?})",
            self.inner.path(),
            self.names()
        )
    }

    /// Resolve the list called `name` against the files under the workspace
    /// root, and return its members in the list's order.
    ///
    /// `narrow`, a `RangeExpr`, keeps only the members whose frame number it
    /// also holds. `progress`, a `ProgressCounter`, counts the directory
    /// entries the walk has read. Raises `ValueError` for an unknown name, a
    /// list that matches no file, a file two entries match, and a `narrow`
    /// given for a list with an entry without a frame field; the `OSError`
    /// subclass for a directory that cannot be listed. The GIL is released
    /// while the walk runs.
    #[pyo3(signature = (name, narrow=None, progress=None))]
    fn resolve(
        &self,
        py: Python<'_>,
        name: &str,
        narrow: Option<PyRef<'_, PyRangeExpr>>,
        progress: Option<ProgressCounter>,
    ) -> PyResult<Vec<PyNamedImageListMember>> {
        let narrow = narrow.as_ref().map(|r| r.inner().clone());
        let counter = progress.as_ref().map(ProgressCounter::handle);
        let sink = |event: Event<'_>| {
            if let (Event::Count { done, .. }, Some(counter)) = (event, &counter) {
                counter.store(done as usize, Ordering::Relaxed);
            }
        };
        py.detach(|| {
            self.inner
                .resolve(name, narrow.as_ref(), &Progress::to(&sink))
        })
        .map(to_py_members)
        .map_err(to_py_err)
    }

    /// Resolve the list called `name` against `image_names`, the
    /// workspace-relative, forward-slash paths a reconstruction records, and
    /// return the names that are members, in the list's order. The image
    /// files need not exist.
    ///
    /// `narrow` and the errors are as for `resolve`; a list that matches none
    /// of the names is an error.
    #[pyo3(signature = (name, image_names, narrow=None))]
    fn resolve_in_names(
        &self,
        name: &str,
        image_names: Vec<String>,
        narrow: Option<PyRef<'_, PyRangeExpr>>,
    ) -> PyResult<Vec<PyNamedImageListMember>> {
        let narrow = narrow.as_ref().map(|r| r.inner());
        self.inner
            .resolve_in_names(name, &image_names, narrow)
            .map(to_py_members)
            .map_err(to_py_err)
    }

    /// The items `slice` keeps of `items`, in the slice's order: the images a
    /// command takes from the list called `name`, already resolved, narrowed
    /// by `narrow` when one was given, and reduced to what the command uses.
    /// The items may be of any type, such as members or paths.
    ///
    /// `name` and `narrow` name the list in the error. Raises `ValueError`
    /// for an unknown name and for a slice that keeps no item, naming the
    /// lists file, the list, the slice and the number of items.
    #[pyo3(signature = (name, slice, items, narrow=None))]
    fn slice_members(
        &self,
        name: &str,
        slice: PyRef<'_, PyNamedImageListSlice>,
        items: Vec<Py<PyAny>>,
        narrow: Option<PyRef<'_, PyRangeExpr>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let narrow = narrow.as_ref().map(|r| r.inner());
        self.inner
            .slice_members(name, narrow, &slice.inner, items)
            .map_err(to_py_err)
    }
}

/// The nearest directory at or above `path` that holds `.sfm-workspace.json`,
/// as a `pathlib.Path`, or `None` when none does.
///
/// The search walks up `path` as given, made absolute against the current
/// directory but with no link resolved, so a directory reached through a link
/// belongs to the workspace around the link. `path` may be a directory, a
/// file, or a path that does not exist yet. The rule is the last step by which
/// a `.sfmr`, `.matches` or `.kdf` file finds the workspace it records.
#[pyfunction(name = "find_workspace_for_path")]
fn find_workspace_for_path_py(path: PathBuf) -> Option<PathBuf> {
    sfmtool_core::workspace::find_workspace_for_path(&path)
}

/// Whether `name` is a valid named image list name: one or more of the ASCII
/// letters, the digits, `-`, `_` and `.`, with at least one letter.
#[pyfunction(name = "is_named_image_list_name")]
fn is_named_image_list_name_py(name: &str) -> bool {
    is_named_image_list_name(name)
}

/// Split a `-r` or `--named-image-list` value into the list's name and its
/// slice, or `None` when it has none: `"walk1[:30]"` gives
/// `("walk1", NamedImageListSlice(stop=30))`, and `"walk1"` gives
/// `("walk1", None)`.
///
/// The text before the first `[` is the name, and the text from the `[` to the
/// end is the slice. Raises `ValueError`, naming `value`, when the name breaks
/// the name rule (a range, or nothing, before the `[` included), when the
/// slice is malformed, and for a step of 0.
#[pyfunction(name = "split_named_image_list_value")]
fn split_named_image_list_value_py(
    value: &str,
) -> PyResult<(String, Option<PyNamedImageListSlice>)> {
    split_named_image_list_value(value)
        .map(|(name, slice)| {
            (
                name.to_string(),
                slice.map(|inner| PyNamedImageListSlice { inner }),
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The grey conversion a version 2 record of `feature_tool` carries when made
/// from the tool's defaults, or None for a tool with none of its own.
///
/// BT.709 luma, unrounded, for ``sfmtool`` (a setting a workspace may
/// change); BT.709 luma rounded to 8 bits for ``colmap`` and BT.601 luma
/// rounded to 8 bits for ``opencv``, the conversions those tools make. The
/// name is matched without regard to case.
#[pyfunction(name = "tool_image_to_gray")]
fn tool_image_to_gray_py(feature_tool: &str) -> Option<PyImageToGray> {
    sfmtool_core::workspace::tool_image_to_gray(feature_tool).map(|inner| PyImageToGray { inner })
}

/// The grey conversion a record of either version states.
///
/// The top-level ``image_to_gray`` object of ``feature_options`` when there
/// is one; otherwise a ``gray_formula`` string with no rounding; otherwise the
/// tool's conversion (``tool_image_to_gray``). An ``sfmtool-undistort``
/// record without its own object states its source's, read from
/// ``source_feature_tool`` and ``source_feature_options``.
///
/// Raises:
///     ValueError: for options that are not a dict, an object or a formula
///         that does not parse, an undistort record with no source, or a tool
///         with no conversion of its own and no ``gray_formula``.
#[pyfunction(name = "recorded_image_to_gray")]
fn recorded_image_to_gray_py(
    py: Python<'_>,
    feature_tool: &str,
    feature_options: &Bound<'_, PyAny>,
) -> PyResult<PyImageToGray> {
    let options = py_to_serde(py, feature_options)?;
    sfmtool_core::workspace::recorded_image_to_gray(feature_tool, &options)
        .map(|inner| PyImageToGray { inner })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Version 1 ``feature_options`` of ``feature_tool`` made into version 2
/// ones, every other option kept as it is: the conversion ``sfm ws upgrade``
/// makes.
///
/// For ``sfmtool``, a ``gray_formula`` becomes
/// ``image_to_gray: {"formula": <gray_formula>}``, and options without one
/// gain the tool's default. For ``colmap`` and ``opencv``, the options gain
/// the tool's conversion.
///
/// Raises:
///     ValueError: for options that already carry ``image_to_gray``, an
///         ``sfmtool-undistort`` record (whose ``.sift`` files cannot be
///         extracted again), any other tool than the three, a
///         ``gray_formula`` that does not parse, or options that are not a
///         dict. The message says which.
#[pyfunction(name = "upgrade_feature_options")]
fn upgrade_feature_options_py(
    py: Python<'_>,
    feature_tool: &str,
    feature_options: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let options = py_to_serde(py, feature_options)?;
    let upgraded = sfmtool_core::workspace::upgrade_feature_options(feature_tool, &options)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_to_py(py, &upgraded)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNamedImageLists>()?;
    m.add_class::<PyNamedImageListMember>()?;
    m.add_class::<PyNamedImageListSlice>()?;
    m.add_function(wrap_pyfunction!(find_workspace_for_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_named_image_list_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(split_named_image_list_value_py, m)?)?;
    m.add_function(wrap_pyfunction!(tool_image_to_gray_py, m)?)?;
    m.add_function(wrap_pyfunction!(recorded_image_to_gray_py, m)?)?;
    m.add_function(wrap_pyfunction!(upgrade_feature_options_py, m)?)?;
    m.add("NAMED_IMAGE_LISTS_FILE_NAME", NAMED_IMAGE_LISTS_FILE_NAME)?;
    m.add("WORKSPACE_VERSION", WORKSPACE_VERSION)?;
    Ok(())
}
