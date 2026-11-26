// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for the files in a workspace root that configure the workspace as
//! a whole (`sfmtool_core::workspace`): its named image lists and their slices,
//! the rules of the feature options its `.sfm-workspace.json` records, the
//! upward search that finds the workspace a path is in, the `mipmap_options`
//! a new workspace records, and the upgrade that brings a workspace's files
//! to the current versions.

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use pyo3::exceptions::{PyInterruptedError, PyValueError};
use pyo3::prelude::*;
use sfmtool_core::progress::{Event, Progress};
use sfmtool_core::workspace::{
    is_named_image_list_name, sfmtool_workspace_marker, split_named_image_list_value,
    write_sfmtool_workspace_marker, NamedImageList, NamedImageListEntry, NamedImageListMember,
    NamedImageListSlice, NamedImageLists, NamedImageListsError, SfmtoolWorkspaceOptions,
    NAMED_IMAGE_LISTS_FILE_NAME, NEW_WORKSPACE_MIPMAP_OPTIONS, WORKSPACE_VERSION,
};

use crate::helpers::py_to_serde;
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

/// The grey conversion a record states.
///
/// The top-level ``image_to_gray`` object of ``feature_options`` when there
/// is one; otherwise a version 1 record's ``gray_formula`` string with no
/// rounding; otherwise the
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

/// The ``feature_type`` a workspace records for ``feature_tool`` with
/// ``feature_options``: ``sift-opencv``, ``sift-sfmtool``, and for
/// ``colmap`` ``sift-colmap`` followed by ``-dsp`` when
/// ``domain_size_pooling`` is set and ``-max<N>`` when ``max_num_features``
/// is given and not 8192; any other tool gives ``sift-<tool>``. The tool name
/// is matched in lower case. It is the name ``sfm ws init`` records.
///
/// Raises:
///     ValueError: for options that a JSON record cannot hold.
#[pyfunction(name = "feature_type_for_tool")]
fn feature_type_for_tool_py(
    py: Python<'_>,
    feature_tool: &str,
    feature_options: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let options = py_to_serde(py, feature_options)?;
    Ok(sfmtool_core::workspace::feature_type_for_tool(
        feature_tool,
        &options,
    ))
}

/// The folder, relative to an image's directory, that a new record's
/// ``.sift`` files go in: ``features/{feature_type}-{hash}``.
///
/// The hash is the XXH3-128 of the compact JSON of ``feature_tool``,
/// ``feature_type`` and ``feature_options``, with every object's keys sorted
/// and without the top-level ``use_gpu``, written as
/// ``json.dumps(..., sort_keys=True, separators=(",", ":"))`` writes it, so
/// it is the name every earlier version of sfmtool gave the same options.
///
/// Some options name no folder, because a JSON record cannot hold them as
/// they are and so could not tell them apart from other options: a NaN or an
/// infinite float, an integer outside the 64-bit range (below ``-2**63`` or
/// above ``2**64 - 1``), a dict key that is not a ``str``, and a list or
/// dict nested more than 128 levels deep, which no JSON record of sfmtool
/// reads back.
///
/// Raises:
///     ValueError: for options that name no folder, naming where the value
///         is, as in ``feature_options['image_to_gray']['quantize_bits']``,
///         and for a value that is not ``None``, a ``bool``, an ``int``, a
///         ``float``, a ``str``, a ``list``, a ``tuple`` or a ``dict``.
#[pyfunction(name = "feature_prefix_dir")]
fn feature_prefix_dir_py(
    feature_tool: &str,
    feature_type: &str,
    feature_options: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let options = record_json(feature_options, "feature_options", 0)?;
    Ok(sfmtool_core::workspace::feature_prefix_dir(
        feature_tool,
        feature_type,
        &options,
    ))
}

/// The options of a new ``sfmtool`` workspace: the feature cap, 8192 when
/// ``None``, and the grey conversion, BT.709 luma unrounded when ``None``.
fn sfmtool_workspace_options(
    max_num_features: Option<usize>,
    image_to_gray: Option<PyRef<'_, PyImageToGray>>,
) -> SfmtoolWorkspaceOptions {
    let default = SfmtoolWorkspaceOptions::default();
    SfmtoolWorkspaceOptions {
        max_num_features: max_num_features.unwrap_or(default.max_num_features),
        image_to_gray: image_to_gray.map_or(default.image_to_gray, |g| g.inner.clone()),
    }
}

/// The text of a new ``sfmtool`` workspace's ``.sfm-workspace.json``.
///
/// The record is version 2 for the ``sfmtool`` feature tool: its default
/// SIFT options with ``double_image_below_long_side`` at 4096, the feature
/// cap ``max_num_features`` (8192 when ``None``) and the grey conversion
/// ``image_to_gray`` (BT.709 luma, unrounded, when ``None``), the
/// ``feature_prefix_dir`` named from them, and the ``mipmap_options`` of
/// ``new_workspace_mipmap_options``. It is written as
/// ``json.dump(record, f, indent=2)`` writes it, with ``\n`` line breaks, so
/// ``json.loads`` of it gives the record with its keys in the order written.
///
/// Raises:
///     OverflowError: for a negative ``max_num_features``.
#[pyfunction(name = "sfmtool_workspace_marker")]
#[pyo3(signature = (*, max_num_features=None, image_to_gray=None))]
fn sfmtool_workspace_marker_py(
    max_num_features: Option<usize>,
    image_to_gray: Option<PyRef<'_, PyImageToGray>>,
) -> String {
    sfmtool_workspace_marker(&sfmtool_workspace_options(max_num_features, image_to_gray))
}

/// Write a new ``sfmtool`` workspace's ``.sfm-workspace.json`` into
/// ``workspace_dir`` and return its text, ``sfmtool_workspace_marker``'s for
/// the same options.
///
/// The directory and its parents are made when they do not exist, and a
/// workspace file already there is replaced; its folder of features and its
/// named image lists stay on disk.
///
/// Raises:
///     OSError: when the directory cannot be made or the file written.
///     OverflowError: for a negative ``max_num_features``.
#[pyfunction(name = "write_sfmtool_workspace_marker")]
#[pyo3(signature = (workspace_dir, *, max_num_features=None, image_to_gray=None))]
fn write_sfmtool_workspace_marker_py(
    workspace_dir: PathBuf,
    max_num_features: Option<usize>,
    image_to_gray: Option<PyRef<'_, PyImageToGray>>,
) -> PyResult<String> {
    let options = sfmtool_workspace_options(max_num_features, image_to_gray);
    Ok(write_sfmtool_workspace_marker(&workspace_dir, &options)?)
}

/// The ``mipmap_options`` object a new workspace records, as a dict with its
/// keys in the order they are written: ``out_of_core_long_side`` 4096,
/// ``jpeg_quality`` 95 and ``full_resolution_jpeg_quality`` 95.
///
/// ``sfmtool_workspace_marker`` writes it for the ``sfmtool`` tool, and
/// ``init_workspace`` writes it for the ``colmap`` and ``opencv`` tools. A new
/// dict is returned on each call. See ``specs/core/camera/mipmaps.md`` § "The
/// workspace setting".
#[pyfunction(name = "new_workspace_mipmap_options")]
fn new_workspace_mipmap_options_py(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let options = NEW_WORKSPACE_MIPMAP_OPTIONS;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("out_of_core_long_side", options.out_of_core_long_side)?;
    dict.set_item("jpeg_quality", options.jpeg_quality)?;
    dict.set_item(
        "full_resolution_jpeg_quality",
        options.full_resolution_jpeg_quality,
    )?;
    Ok(dict)
}

/// Bring the tree of workspaces at `path` to the current file versions, and
/// return a report of every file and what became of it.
///
/// The tree is the workspace at or above `path` and every workspace nested in
/// it. A version 1 workspace file is refused before anything else is read,
/// naming sfmtool 0.4.1 as the last release that upgrades it. Each
/// ``.sfmr``, ``.matches``, ``.camrig`` and ``.kdf`` file below the current
/// version that the current reader reads is rewritten at its own path as the
/// current version; a file the reader refuses is left as it is. The walk does
/// not enter a symbolic link or junction to a directory. A stopped run can be
/// run again to finish. See ``specs/cli/workspace/ws-upgrade-command.md``.
///
/// Args:
///     path: a directory or file in the tree's root workspace.
///     dry_run: find the files and decide what to do with each, and write
///         nothing; the report then gives what the run would do.
///     progress: a ``ProgressCounter`` set to the count of the phase that is
///         running: the folders walked, then the files rewritten.
///
/// Returns:
///     A dict: ``workspace_dir``, the tree's root; ``workspaces``, the root
///     then each nested workspace; ``dry_run``; ``files``, a list of dicts
///     with ``path``, ``kind`` (``sfmr``, ``matches``, ``camrig`` or ``kdf``),
///     ``outcome`` (``converted``, ``already_current``, ``not_converted`` or
///     ``failed``) and, for ``not_converted`` and ``failed``, ``reason``;
///     ``linked_directories``; ``notes``, lines for the user; and
///     ``nothing_to_do``, true when no file is converted or failed.
///
/// Raises:
///     ValueError: when a workspace file of the tree is version 1, cannot be
///         read or is of a version this build does not know; nothing is
///         written in each case.
///     FileNotFoundError: when no workspace is found at or above ``path``.
///     OSError: when a directory of the tree cannot be listed.
///     The exception a signal handler raised while the run went
///     (``KeyboardInterrupt`` for Ctrl+C), which stops it.
///     InterruptedError: when the run was cancelled otherwise.
#[pyfunction(name = "upgrade_workspace")]
#[pyo3(signature = (path, *, dry_run=false, progress=None))]
fn upgrade_workspace_py(
    py: Python<'_>,
    path: PathBuf,
    dry_run: bool,
    progress: Option<ProgressCounter>,
) -> PyResult<Py<PyAny>> {
    use sfmtool_core::workspace::{
        upgrade_workspace, UpgradeWorkspaceError, UpgradeWorkspaceOptions,
    };
    let options = UpgradeWorkspaceOptions { dry_run };
    let counter = progress.as_ref().map(ProgressCounter::handle);
    let watch = crate::signals::SignalWatch::new();
    let sink = |event: Event<'_>| {
        if let (Event::Count { done, .. }, Some(counter)) = (event, &counter) {
            counter.store(done as usize, Ordering::Relaxed);
        }
        watch.check();
    };
    let result = py.detach(|| {
        let progress = Progress::to(&sink).cancelled_by(watch.cancel_flag());
        upgrade_workspace(&path, &options, &progress)
    });
    match watch.finish(result)? {
        Ok(report) => {
            let dict = crate::helpers::serde_to_py(py, &report)?;
            dict.bind(py)
                .set_item("nothing_to_do", report.nothing_to_do())?;
            Ok(dict)
        }
        Err(e) => Err(match e {
            UpgradeWorkspaceError::NoWorkspace { .. } => {
                pyo3::exceptions::PyFileNotFoundError::new_err(e.to_string())
            }
            UpgradeWorkspaceError::ListFolder { .. } => {
                pyo3::exceptions::PyOSError::new_err(e.to_string())
            }
            UpgradeWorkspaceError::Cancelled => PyInterruptedError::new_err(e.to_string()),
            _ => PyValueError::new_err(e.to_string()),
        }),
    }
}

/// How many lists and dicts deep [`record_json`] converts a value: the
/// recursion limit `serde_json` reads JSON with, which stops at the 128th
/// nested list or object, so no sfmtool record holding a deeper value could be
/// read back; it is far below what would overflow the stack.
const RECORD_JSON_MAX_DEPTH: usize = 128;

/// The JSON value a workspace record holds for `value`, the one at `path`,
/// refusing what JSON cannot hold as it is rather than changing it: a
/// `serde_json` number is an `i64`, a `u64` or a finite `f64`, so a larger
/// integer would become a float and a NaN would have no value, and an object's
/// keys are strings, so an `int` key would become one. `depth` counts the
/// lists and dicts `value` is inside; past [`RECORD_JSON_MAX_DEPTH`] it is
/// refused, which also stops a list or dict that holds itself.
fn record_json(value: &Bound<'_, PyAny>, path: &str, depth: usize) -> PyResult<serde_json::Value> {
    use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
    if depth > RECORD_JSON_MAX_DEPTH {
        return Err(PyValueError::new_err(format!(
            "{path} is nested more than {RECORD_JSON_MAX_DEPTH} lists or dicts deep, \
             which names no features folder"
        )));
    }
    if value.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = value.cast::<PyBool>() {
        Ok(serde_json::Value::Bool(b.is_true()))
    } else if value.is_instance_of::<PyInt>() {
        if let Ok(v) = value.extract::<u64>() {
            Ok(v.into())
        } else if let Ok(v) = value.extract::<i64>() {
            Ok(v.into())
        } else {
            Err(PyValueError::new_err(format!(
                "{path} is an integer outside the 64-bit range, which names no features folder"
            )))
        }
    } else if value.is_instance_of::<PyFloat>() {
        let v: f64 = value.extract()?;
        serde_json::Number::from_f64(v)
            .map(serde_json::Value::Number)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "{path} is {v}, and a NaN or infinite value names no features folder"
                ))
            })
    } else if let Ok(s) = value.cast::<PyString>() {
        Ok(serde_json::Value::String(s.to_str()?.to_owned()))
    } else if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, item) in dict.iter() {
            let key: String = key
                .cast::<PyString>()
                .map_err(|_| {
                    PyValueError::new_err(format!(
                        "{path} has the key {}, which is not a str; a dict with a key \
                         that is not a str names no features folder",
                        key.repr()
                            .map_or_else(|_| "?".to_owned(), |r| r.to_string())
                    ))
                })?
                .to_str()?
                .to_owned();
            let at = format!("{path}['{key}']");
            map.insert(key, record_json(&item, &at, depth + 1)?);
        }
        Ok(serde_json::Value::Object(map))
    } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let items = value
            .try_iter()?
            .enumerate()
            .map(|(i, item)| record_json(&item?, &format!("{path}[{i}]"), depth + 1))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(serde_json::Value::Array(items))
    } else {
        Err(PyValueError::new_err(format!(
            "{path} has the type {}, which a workspace record cannot hold",
            value
                .get_type()
                .fully_qualified_name()
                .map_or_else(|_| "?".to_owned(), |n| n.to_string())
        )))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNamedImageLists>()?;
    m.add_class::<PyNamedImageListMember>()?;
    m.add_class::<PyNamedImageListSlice>()?;
    m.add_function(wrap_pyfunction!(feature_prefix_dir_py, m)?)?;
    m.add_function(wrap_pyfunction!(feature_type_for_tool_py, m)?)?;
    m.add_function(wrap_pyfunction!(find_workspace_for_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_named_image_list_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(split_named_image_list_value_py, m)?)?;
    m.add_function(wrap_pyfunction!(tool_image_to_gray_py, m)?)?;
    m.add_function(wrap_pyfunction!(recorded_image_to_gray_py, m)?)?;
    m.add_function(wrap_pyfunction!(sfmtool_workspace_marker_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_sfmtool_workspace_marker_py, m)?)?;
    m.add_function(wrap_pyfunction!(new_workspace_mipmap_options_py, m)?)?;
    m.add_function(wrap_pyfunction!(upgrade_workspace_py, m)?)?;
    m.add("NAMED_IMAGE_LISTS_FILE_NAME", NAMED_IMAGE_LISTS_FILE_NAME)?;
    m.add("WORKSPACE_VERSION", WORKSPACE_VERSION)?;
    Ok(())
}
