// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the files in a workspace root that configure the workspace as
//! a whole: its named image lists (`sfmtool_core::workspace`).

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use pyo3::exceptions::{PyInterruptedError, PyValueError};
use pyo3::prelude::*;
use sfmtool_core::progress::{Event, Progress};
use sfmtool_core::workspace::{
    is_named_image_list_name, NamedImageListMember, NamedImageLists, NamedImageListsError,
    NAMED_IMAGE_LISTS_FILE_NAME,
};

use crate::{ProgressCounter, PyRangeExpr};

/// The Python exception for `e`: the `OSError` subclass for a failed read,
/// `InterruptedError` for a cancelled resolve, and `ValueError` for a file
/// that breaks the format's rules or a list that cannot be resolved.
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
}

/// Whether `name` is a valid named image list name: one or more of the ASCII
/// letters, the digits, `-`, `_` and `.`, with at least one letter.
#[pyfunction(name = "is_named_image_list_name")]
fn is_named_image_list_name_py(name: &str) -> bool {
    is_named_image_list_name(name)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNamedImageLists>()?;
    m.add_class::<PyNamedImageListMember>()?;
    m.add_function(wrap_pyfunction!(is_named_image_list_name_py, m)?)?;
    m.add("NAMED_IMAGE_LISTS_FILE_NAME", NAMED_IMAGE_LISTS_FILE_NAME)?;
    Ok(())
}
