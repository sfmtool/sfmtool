// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A workspace's named image lists: the `.sfm-named-image-lists.json` file in
//! the workspace root, its validation, the resolution of one list to the
//! workspace's image files or to a reconstruction's image names, and the slice
//! that takes part of a resolved list by position ([`NamedImageListSlice`]),
//! and the writer that adds or replaces lists in the file
//! ([`NamedImageLists::write_lists`]).
//!
//! The file format, its rules and the order of a list's members are specified
//! in `specs/workspace/named-image-lists.md`. A pattern is checked and matched
//! by the `.camrig` image-pattern grammar in `sfmtool-camrig-format`, and a
//! range is an `openjd_expr::RangeExpr`, so neither rule has a second copy
//! here.

mod json;
mod slice;
mod walk;
mod write;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use openjd_expr::RangeExpr;
use sfmtool_camrig_format::{
    count_frame_fields, pattern_frame_index, pattern_literal, pattern_matches,
};
use sfmtool_progress::Progress;

use json::Json;
pub use slice::{split_named_image_list_value, NamedImageListSlice, NamedImageListSliceError};

/// The name of the lists file, in the workspace root beside
/// [`WORKSPACE_FILE_NAME`].
pub const NAMED_IMAGE_LISTS_FILE_NAME: &str = ".sfm-named-image-lists.json";

/// The name of the file that makes a directory a workspace root.
pub const WORKSPACE_FILE_NAME: &str = ".sfm-workspace.json";

/// The one version of the lists file this code reads.
pub const NAMED_IMAGE_LISTS_VERSION: i64 = 1;

/// What a list name may contain, as the error for a bad name says it.
const NAME_RULE: &str =
    "a name is one or more of the ASCII letters, the digits, '-', '_' and '.', with at least one letter";

/// Whether `name` is a valid named image list name: one or more of the ASCII
/// letters, the digits, `-`, `_` and `.`, with at least one letter.
///
/// The letter is what tells a name from a range expression, which holds only
/// digits, `-`, `:`, `,` and whitespace.
pub fn is_named_image_list_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && name.bytes().any(|b| b.is_ascii_alphabetic())
}

/// One entry of a named image list: an image pattern and an optional range of
/// the frame numbers its frame field captures.
#[derive(Debug, Clone)]
pub struct NamedImageListEntry {
    pattern: String,
    /// The range and its text: as the file wrote it for an entry read from a
    /// file, so that [`NamedImageLists::write_lists`] writes it back
    /// unchanged, and as `RangeExpr`'s `Display` writes it for one made by
    /// [`Self::new`].
    range: Option<(RangeExpr, String)>,
    has_frame_field: bool,
}

impl NamedImageListEntry {
    /// An entry to write with [`NamedImageLists::write_lists`]: `pattern` and
    /// `range` are checked by the reader's rules when it is written, not
    /// here. The range is written as `RangeExpr`'s `Display` writes it.
    pub fn new(pattern: impl Into<String>, range: Option<RangeExpr>) -> Self {
        let pattern = pattern.into();
        let has_frame_field = count_frame_fields(&pattern) == 1;
        Self {
            pattern,
            range: range.map(|range| {
                let text = range.to_string();
                (range, text)
            }),
            has_frame_field,
        }
    }

    /// The image pattern, relative to the workspace root, in the `.camrig`
    /// pattern syntax.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// The entry's range of frame numbers; `None` takes every frame number.
    /// Only an entry whose pattern has a frame field has one.
    pub fn range(&self) -> Option<&RangeExpr> {
        self.range.as_ref().map(|(range, _)| range)
    }

    /// Whether the pattern has a frame field (`%d` or `%0Nd`).
    pub fn has_frame_field(&self) -> bool {
        self.has_frame_field
    }

    /// Whether this entry takes the file at `relative_path`, and the frame
    /// number it captures. `Err` carries a frame number too large for an
    /// `i64`.
    fn take(&self, relative_path: &str) -> Result<Option<Option<i64>>, ()> {
        if !pattern_matches(&self.pattern, relative_path, false) {
            return Ok(None);
        }
        if !self.has_frame_field {
            return Ok(Some(None));
        }
        let frame = pattern_frame_index(&self.pattern, relative_path, false)
            .and_then(|frame| i64::try_from(frame).ok())
            .ok_or(())?;
        let in_range = self.range().is_none_or(|range| range.contains(frame));
        Ok(in_range.then_some(Some(frame)))
    }
}

/// One named image list: its name and its entries, in the file's order.
#[derive(Debug, Clone)]
pub struct NamedImageList {
    name: String,
    entries: Vec<NamedImageListEntry>,
}

impl NamedImageList {
    /// A list to write with [`NamedImageLists::write_lists`]: `name` and the
    /// entries are checked by the reader's rules when it is written, not
    /// here.
    pub fn new(name: impl Into<String>, entries: Vec<NamedImageListEntry>) -> Self {
        Self {
            name: name.into(),
            entries,
        }
    }

    /// The list's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The list's entries, in the file's order.
    pub fn entries(&self) -> &[NamedImageListEntry] {
        &self.entries
    }
}

/// One member of a resolved list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedImageListMember {
    /// The workspace-relative, forward-slash path of the image.
    pub path: String,
    /// The index, from 0, of the entry that matched it.
    pub entry: usize,
    /// The frame number the entry's frame field captured; `None` for an entry
    /// without a frame field.
    pub frame: Option<i64>,
}

/// A file that two entries of one list match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateMatch {
    /// The workspace-relative path of the file.
    pub file: String,
    /// The index, from 0, of the first entry that matches it.
    pub first_entry: usize,
    /// The index, from 0, of the later entry that matches it too.
    pub second_entry: usize,
}

/// Why reading, validating or resolving named image lists failed. Every
/// variant's message names the lists file.
#[derive(Debug)]
pub enum NamedImageListsError {
    /// The directory given as the workspace root holds no
    /// `.sfm-workspace.json`, so it is not a workspace and has no lists.
    NotAWorkspace { workspace_dir: PathBuf },
    /// Reading or writing the lists file, or listing a directory while
    /// resolving, failed.
    Io {
        /// The lists file.
        lists_path: PathBuf,
        /// The file or directory whose read failed.
        path: PathBuf,
        source: std::io::Error,
    },
    /// The file is not valid JSON. Nothing else in it was checked.
    InvalidJson {
        path: PathBuf,
        line: usize,
        column: usize,
        message: String,
    },
    /// The file's `version` is an integer other than 1. Nothing else in it was
    /// checked, since its layout is unknown.
    UnsupportedVersion { path: PathBuf, version: String },
    /// The file breaks the format's rules. `problems` holds every problem
    /// found, each without the path.
    Invalid {
        path: PathBuf,
        problems: Vec<String>,
    },
    /// Lists given to [`NamedImageLists::write_lists`] break the format's
    /// rules, so nothing was written. `problems` holds every problem found,
    /// each without the path.
    InvalidNewLists {
        path: PathBuf,
        problems: Vec<String>,
    },
    /// [`NamedImageLists::write_lists`] was asked to write lists the file
    /// already defines without being asked to replace them, so nothing was
    /// written. `names` are those lists, in the order they were given.
    ListExists { path: PathBuf, names: Vec<String> },
    /// No list has this name.
    UnknownList {
        path: PathBuf,
        name: String,
        /// The names the file defines, in the file's order.
        defined: Vec<String>,
    },
    /// A narrowing range was given, and these entries of the list have no
    /// frame field for it to narrow: `(entry index from 0, pattern)`.
    NarrowedEntryWithoutFrameField {
        path: PathBuf,
        list: String,
        entries: Vec<(usize, String)>,
    },
    /// Files that two entries of the list both match.
    DuplicateMatches {
        path: PathBuf,
        list: String,
        duplicates: Vec<DuplicateMatch>,
    },
    /// An entry matched a file whose frame number does not fit an `i64`.
    FrameNumberTooLarge {
        path: PathBuf,
        list: String,
        entry: usize,
        file: String,
    },
    /// The list matched no file.
    NoMatch { path: PathBuf, list: String },
    /// The list matched files, and the narrowing range held the frame number
    /// of none of them. `range` is the narrowing range as written by
    /// `RangeExpr`'s `Display`.
    NarrowedToNothing {
        path: PathBuf,
        list: String,
        range: String,
    },
    /// A slice kept none of the `len` images the caller took from the list.
    /// `range` is the narrowing range, as written by `RangeExpr`'s `Display`,
    /// when one was given, and `slice` the slice as its `Display` writes it.
    SlicedToNothing {
        path: PathBuf,
        list: String,
        range: Option<String>,
        slice: String,
        len: usize,
    },
    /// The caller's progress asked the resolution to stop.
    Cancelled,
}

impl fmt::Display for NamedImageListsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAWorkspace { workspace_dir } => write!(
                f,
                "{}: not a workspace root, since it holds no {WORKSPACE_FILE_NAME}, so it has no \
                 named image lists",
                workspace_dir.display()
            ),
            Self::Io {
                lists_path,
                path,
                source,
            } => {
                if lists_path == path {
                    write!(f, "{}: {source}", path.display())
                } else {
                    write!(
                        f,
                        "{}: could not list {}: {source}",
                        lists_path.display(),
                        path.display()
                    )
                }
            }
            Self::InvalidJson {
                path,
                line,
                column,
                message,
            } => write!(
                f,
                "{}: not valid JSON at line {line}, column {column}: {message}",
                path.display()
            ),
            Self::UnsupportedVersion { path, version } => write!(
                f,
                "{}: named image lists version {version} is not supported; this sfmtool reads \
                 version {NAMED_IMAGE_LISTS_VERSION}",
                path.display()
            ),
            Self::Invalid { path, problems } => {
                for (i, problem) in problems.iter().enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write!(f, "{}: {problem}", path.display())?;
                }
                Ok(())
            }
            Self::InvalidNewLists { path, problems } => {
                for (i, problem) in problems.iter().enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write!(f, "{}: nothing was written: {problem}", path.display())?;
                }
                Ok(())
            }
            Self::ListExists { path, names } => {
                for (i, name) in names.iter().enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write!(
                        f,
                        "{}: the named image list '{name}' is already defined, so nothing \
                         was written",
                        path.display()
                    )?;
                }
                Ok(())
            }
            Self::UnknownList {
                path,
                name,
                defined,
            } => {
                write!(
                    f,
                    "{}: no named image list is called '{name}'; ",
                    path.display()
                )?;
                if defined.is_empty() {
                    write!(f, "the file defines no lists")
                } else {
                    let quoted: Vec<String> = defined.iter().map(|n| format!("'{n}'")).collect();
                    write!(f, "the lists it defines are {}", quoted.join(", "))
                }
            }
            Self::NarrowedEntryWithoutFrameField {
                path,
                list,
                entries,
            } => {
                for (i, (entry, pattern)) in entries.iter().enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write!(
                        f,
                        "{}: list '{list}' entry {} cannot be narrowed by a range: its pattern \
                         '{pattern}' has no frame field (%d or %0Nd)",
                        path.display(),
                        entry + 1
                    )?;
                }
                Ok(())
            }
            Self::DuplicateMatches {
                path,
                list,
                duplicates,
            } => {
                for (i, d) in duplicates.iter().enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write!(
                        f,
                        "{}: list '{list}' matches '{}' by both entry {} and entry {}; a file may \
                         be matched by one entry of a list only",
                        path.display(),
                        d.file,
                        d.first_entry + 1,
                        d.second_entry + 1
                    )?;
                }
                Ok(())
            }
            Self::FrameNumberTooLarge {
                path,
                list,
                entry,
                file,
            } => write!(
                f,
                "{}: list '{list}' entry {} matches '{file}', whose frame number is too large",
                path.display(),
                entry + 1
            ),
            Self::NoMatch { path, list } => write!(
                f,
                "{}: named image list '{list}' matches no image",
                path.display()
            ),
            Self::NarrowedToNothing { path, list, range } => write!(
                f,
                "{}: the range '{range}' left no members of named image list '{list}'",
                path.display()
            ),
            Self::SlicedToNothing {
                path,
                list,
                range,
                slice,
                len,
            } => {
                write!(
                    f,
                    "{}: the slice {slice} of named image list '{list}'",
                    path.display()
                )?;
                if let Some(range) = range {
                    write!(f, " narrowed by the range '{range}'")?;
                }
                let images = if *len == 1 { "image" } else { "images" };
                write!(f, " leaves none of its {len} {images}")
            }
            Self::Cancelled => write!(f, "resolving the named image list was cancelled"),
        }
    }
}

impl std::error::Error for NamedImageListsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// A workspace's named image lists, read from its
/// `.sfm-named-image-lists.json` and checked whole.
#[derive(Debug, Clone)]
pub struct NamedImageLists {
    workspace_dir: PathBuf,
    path: PathBuf,
    lists: Vec<NamedImageList>,
}

impl NamedImageLists {
    /// Read the lists file of the workspace whose root is `workspace_dir`.
    ///
    /// Returns `Ok(None)` when the workspace has no lists file, which is not
    /// the same as a file that defines no lists. The workspace itself is
    /// found by the caller; a `workspace_dir` without a `.sfm-workspace.json`
    /// is an error, since a lists file is only used beside one.
    pub fn read(workspace_dir: &Path) -> Result<Option<Self>, NamedImageListsError> {
        if !workspace_dir.join(WORKSPACE_FILE_NAME).is_file() {
            return Err(NamedImageListsError::NotAWorkspace {
                workspace_dir: workspace_dir.to_path_buf(),
            });
        }
        let path = workspace_dir.join(NAMED_IMAGE_LISTS_FILE_NAME);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(NamedImageListsError::Io {
                    lists_path: path.clone(),
                    path,
                    source,
                })
            }
        };
        Self::parse(workspace_dir, &bytes).map(Some)
    }

    /// Parse and validate the contents of a lists file, as if read from the
    /// workspace whose root is `workspace_dir`, which names the file in every
    /// error and is the root [`Self::resolve`] walks. Checks the whole file
    /// and reports every problem it finds.
    pub fn parse(workspace_dir: &Path, bytes: &[u8]) -> Result<Self, NamedImageListsError> {
        let path = workspace_dir.join(NAMED_IMAGE_LISTS_FILE_NAME);
        // A UTF-8 byte order mark, which some Windows editors write, is not
        // part of the JSON text.
        let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
        let root: Json = match serde_json::from_slice(bytes) {
            Ok(root) => root,
            Err(e) => {
                let text = e.to_string();
                // serde_json appends " at line L column C"; the variant
                // carries the two numbers and says them in its own words.
                let message = match text.rfind(" at line ") {
                    Some(at) => text[..at].to_string(),
                    None => text,
                };
                return Err(NamedImageListsError::InvalidJson {
                    path,
                    line: e.line(),
                    column: e.column(),
                    message,
                });
            }
        };
        let lists = validate(&root, &path)?;
        Ok(Self {
            workspace_dir: workspace_dir.to_path_buf(),
            path,
            lists,
        })
    }

    /// The lists file's path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The workspace root the lists resolve against.
    pub fn workspace_dir(&self) -> &Path {
        &self.workspace_dir
    }

    /// Every list, in the file's order.
    pub fn lists(&self) -> &[NamedImageList] {
        &self.lists
    }

    /// The list called `name`, compared exactly, case included.
    pub fn get(&self, name: &str) -> Option<&NamedImageList> {
        self.lists.iter().find(|list| list.name == name)
    }

    /// The list called `name`, or the error naming the lists there are.
    fn list(&self, name: &str) -> Result<&NamedImageList, NamedImageListsError> {
        self.get(name)
            .ok_or_else(|| NamedImageListsError::UnknownList {
                path: self.path.clone(),
                name: name.to_string(),
                defined: self.lists.iter().map(|list| list.name.clone()).collect(),
            })
    }

    /// Resolve the list called `name` against the files under the workspace
    /// root: its members, in the list's order.
    ///
    /// `narrow`, when given, keeps only the members whose frame number it
    /// also holds; it is an error when an entry of the list has no frame
    /// field. A list that matches no file is an error, as is a file two of its
    /// entries match.
    ///
    /// The walk lists only the directories the patterns' segments lead into,
    /// and every directory below a `**`. It reports the directory entries it
    /// has read through `progress` and stops when `progress` is cancelled.
    pub fn resolve(
        &self,
        name: &str,
        narrow: Option<&RangeExpr>,
        progress: &Progress,
    ) -> Result<Vec<NamedImageListMember>, NamedImageListsError> {
        let list = self.list(name)?;
        self.check_narrowing(list, narrow)?;
        let mut walker = walk::Walker::new(&self.workspace_dir, progress);
        let mut per_entry = Vec::with_capacity(list.entries.len());
        for entry in &list.entries {
            let candidates = walker.candidates(&entry.pattern).map_err(|e| match e {
                walk::WalkError::Io { path, source } => NamedImageListsError::Io {
                    lists_path: self.path.clone(),
                    path,
                    source,
                },
                walk::WalkError::Cancelled => NamedImageListsError::Cancelled,
            })?;
            per_entry.push(candidates);
        }
        self.members(list, narrow, |index| {
            per_entry[index].iter().map(String::as_str)
        })
    }

    /// Resolve the list called `name` against `image_names`, the
    /// workspace-relative, forward-slash paths a reconstruction records,
    /// rather than the filesystem: the names that are members, in the list's
    /// order. The image files need not exist.
    ///
    /// `narrow` and the errors are as for [`Self::resolve`]; a list that
    /// matches none of the names is an error.
    pub fn resolve_in_names<S: AsRef<str>>(
        &self,
        name: &str,
        image_names: &[S],
        narrow: Option<&RangeExpr>,
    ) -> Result<Vec<NamedImageListMember>, NamedImageListsError> {
        let list = self.list(name)?;
        self.check_narrowing(list, narrow)?;
        let names: BTreeSet<&str> = image_names.iter().map(AsRef::as_ref).collect();
        // An entry with no wildcard and no frame field names one image, which
        // is looked up in the set rather than compared with every name.
        let literals: Vec<Option<String>> = list
            .entries
            .iter()
            .map(|entry| pattern_literal(&entry.pattern))
            .collect();
        self.members(list, narrow, |index| -> Box<dyn Iterator<Item = &str>> {
            match &literals[index] {
                Some(literal) => Box::new(names.get(literal.as_str()).copied().into_iter()),
                None => Box::new(names.iter().copied()),
            }
        })
    }

    /// The items `slice` keeps of `items`, in the slice's order: the images a
    /// command takes from the list called `name`, already resolved, narrowed
    /// by `narrow` when one was given, and reduced to what the command uses.
    ///
    /// `name` and `narrow` name the list in the error; `slice` indexes
    /// `items` by position from 0, never by frame number. A slice that keeps
    /// no item is `SlicedToNothing`, and an unknown `name` is `UnknownList`.
    pub fn slice_members<T>(
        &self,
        name: &str,
        narrow: Option<&RangeExpr>,
        slice: &NamedImageListSlice,
        items: Vec<T>,
    ) -> Result<Vec<T>, NamedImageListsError> {
        let list = self.list(name)?;
        let indices = slice.indices(items.len());
        if indices.is_empty() {
            return Err(NamedImageListsError::SlicedToNothing {
                path: self.path.clone(),
                list: list.name.clone(),
                range: narrow.map(ToString::to_string),
                slice: slice.to_string(),
                len: items.len(),
            });
        }
        let mut items: Vec<Option<T>> = items.into_iter().map(Some).collect();
        // A slice's indexes are distinct, so each item is taken once.
        Ok(indices
            .into_iter()
            .filter_map(|index| items[index].take())
            .collect())
    }

    fn check_narrowing(
        &self,
        list: &NamedImageList,
        narrow: Option<&RangeExpr>,
    ) -> Result<(), NamedImageListsError> {
        if narrow.is_none() {
            return Ok(());
        }
        let entries: Vec<(usize, String)> = list
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| !entry.has_frame_field)
            .map(|(i, entry)| (i, entry.pattern.clone()))
            .collect();
        if entries.is_empty() {
            Ok(())
        } else {
            Err(NamedImageListsError::NarrowedEntryWithoutFrameField {
                path: self.path.clone(),
                list: list.name.clone(),
                entries,
            })
        }
    }

    /// The members of `list` among each entry's candidate paths, in the
    /// list's order: entry order, then frame number, then path.
    fn members<'c, I: Iterator<Item = &'c str>>(
        &self,
        list: &NamedImageList,
        narrow: Option<&RangeExpr>,
        candidates: impl Fn(usize) -> I,
    ) -> Result<Vec<NamedImageListMember>, NamedImageListsError> {
        let mut members = Vec::new();
        let mut narrowed_away = false;
        let mut first_entry: BTreeMap<String, usize> = BTreeMap::new();
        let mut duplicates = Vec::new();
        for (index, entry) in list.entries.iter().enumerate() {
            let mut taken: Vec<(Option<i64>, &str)> = Vec::new();
            for candidate in candidates(index) {
                match entry.take(candidate) {
                    Ok(Some(frame)) => taken.push((frame, candidate)),
                    Ok(None) => {}
                    Err(()) => {
                        return Err(NamedImageListsError::FrameNumberTooLarge {
                            path: self.path.clone(),
                            list: list.name.clone(),
                            entry: index,
                            file: candidate.to_string(),
                        })
                    }
                }
            }
            taken.sort();
            for (frame, file) in taken {
                if let Some(&first) = first_entry.get(file) {
                    duplicates.push(DuplicateMatch {
                        file: file.to_string(),
                        first_entry: first,
                        second_entry: index,
                    });
                    continue;
                }
                first_entry.insert(file.to_string(), index);
                if let (Some(narrow), Some(frame)) = (narrow, frame) {
                    if !narrow.contains(frame) {
                        narrowed_away = true;
                        continue;
                    }
                }
                members.push(NamedImageListMember {
                    path: file.to_string(),
                    entry: index,
                    frame,
                });
            }
        }
        if !duplicates.is_empty() {
            return Err(NamedImageListsError::DuplicateMatches {
                path: self.path.clone(),
                list: list.name.clone(),
                duplicates,
            });
        }
        if members.is_empty() && narrowed_away {
            return Err(NamedImageListsError::NarrowedToNothing {
                path: self.path.clone(),
                list: list.name.clone(),
                range: narrow.map(ToString::to_string).unwrap_or_default(),
            });
        }
        if members.is_empty() {
            return Err(NamedImageListsError::NoMatch {
                path: self.path.clone(),
                list: list.name.clone(),
            });
        }
        Ok(members)
    }
}

/// Check the parsed file whole and build its lists, or report every problem.
fn validate(root: &Json, path: &Path) -> Result<Vec<NamedImageList>, NamedImageListsError> {
    let mut problems = Vec::new();
    let Json::Object(members) = root else {
        problems.push(format!(
            "the file holds {}, not an object",
            root.type_name()
        ));
        return Err(NamedImageListsError::Invalid {
            path: path.to_path_buf(),
            problems,
        });
    };

    // The version comes first: under another version nothing else is known.
    let versions: Vec<&Json> = members
        .iter()
        .filter(|(key, _)| key == "version")
        .map(|(_, value)| value)
        .collect();
    match versions.first() {
        None => problems.push("missing key 'version'".to_string()),
        Some(Json::Number(n)) if n.is_i64() || n.is_u64() => {
            if n.as_i64() != Some(NAMED_IMAGE_LISTS_VERSION) {
                return Err(NamedImageListsError::UnsupportedVersion {
                    path: path.to_path_buf(),
                    version: n.to_string(),
                });
            }
        }
        Some(other) => problems.push(format!(
            "'version' is {}, not an integer",
            other.type_name()
        )),
    }

    let mut seen = BTreeSet::new();
    for (key, _) in members {
        if !seen.insert(key.as_str()) {
            problems.push(format!("the top-level key '{key}' appears more than once"));
        }
        if key != "version" && key != "named_image_lists" {
            problems.push(format!(
                "unknown top-level key '{key}'; the keys are 'version' and 'named_image_lists'"
            ));
        }
    }

    let mut lists = Vec::new();
    match members.iter().find(|(key, _)| key == "named_image_lists") {
        None => problems.push("missing key 'named_image_lists'".to_string()),
        Some((_, Json::Object(named))) => {
            let mut names = BTreeSet::new();
            for (name, value) in named {
                if !names.insert(name.as_str()) {
                    problems.push(format!(
                        "named_image_lists: the list '{name}' is defined more than once"
                    ));
                    continue;
                }
                if !is_named_image_list_name(name) {
                    problems.push(format!(
                        "named_image_lists: '{name}' is not a valid list name; {NAME_RULE}"
                    ));
                }
                if let Some(entries) = validate_list(name, value, &mut problems) {
                    lists.push(NamedImageList {
                        name: name.clone(),
                        entries,
                    });
                }
            }
        }
        Some((_, other)) => problems.push(format!(
            "'named_image_lists' is {}, not an object",
            other.type_name()
        )),
    }

    if problems.is_empty() {
        Ok(lists)
    } else {
        Err(NamedImageListsError::Invalid {
            path: path.to_path_buf(),
            problems,
        })
    }
}

/// Check one list's entries, adding each problem to `problems`; the entries
/// when the list has none.
fn validate_list(
    name: &str,
    value: &Json,
    problems: &mut Vec<String>,
) -> Option<Vec<NamedImageListEntry>> {
    let Json::Array(items) = value else {
        problems.push(format!(
            "list '{name}' is {}, not an array of entries",
            value.type_name()
        ));
        return None;
    };
    let before = problems.len();
    let mut entries = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let at = format!("list '{name}' entry {}", index + 1);
        if let Some(entry) = validate_entry(&at, item, problems) {
            entries.push(entry);
        }
    }
    (problems.len() == before).then_some(entries)
}

/// Check one entry, adding each problem to `problems` under the prefix `at`.
fn validate_entry(
    at: &str,
    item: &Json,
    problems: &mut Vec<String>,
) -> Option<NamedImageListEntry> {
    let Json::Object(keys) = item else {
        problems.push(format!("{at} is {}, not an object", item.type_name()));
        return None;
    };
    let before = problems.len();
    let mut seen = BTreeSet::new();
    for (key, _) in keys {
        if !seen.insert(key.as_str()) {
            problems.push(format!("{at}: the key '{key}' appears more than once"));
        }
        if key != "pattern" && key != "range" {
            problems.push(format!(
                "{at}: unknown key '{key}'; an entry's keys are 'pattern' and 'range'"
            ));
        }
    }
    let value_of = |wanted: &str| keys.iter().find(|(key, _)| key == wanted).map(|(_, v)| v);

    let pattern = match value_of("pattern") {
        None => {
            problems.push(format!("{at}: missing key 'pattern'"));
            None
        }
        Some(Json::String(pattern)) => match sfmtool_camrig_format::validate_pattern(pattern) {
            Ok(()) => Some(pattern.clone()),
            Err(reason) => {
                problems.push(format!("{at}: {reason}"));
                None
            }
        },
        Some(other) => {
            problems.push(format!(
                "{at}: 'pattern' is {}, not a string",
                other.type_name()
            ));
            None
        }
    };
    let range = match value_of("range") {
        None => None,
        Some(Json::String(text)) => match RangeExpr::from_str(text) {
            Ok(range) => Some((range, text.clone())),
            Err(e) => {
                problems.push(format!(
                    "{at}: 'range' '{text}' is not a range expression: {e}"
                ));
                None
            }
        },
        Some(other) => {
            problems.push(format!(
                "{at}: 'range' is {}, not a string",
                other.type_name()
            ));
            None
        }
    };
    let has_frame_field = pattern
        .as_deref()
        .is_some_and(|p| count_frame_fields(p) == 1);
    if let (Some(pattern), true) = (&pattern, value_of("range").is_some()) {
        if !has_frame_field {
            problems.push(format!(
                "{at}: has a 'range', but its pattern '{pattern}' has no frame field (%d or \
                 %0Nd) for the range to apply to"
            ));
        }
    }
    if problems.len() != before {
        return None;
    }
    Some(NamedImageListEntry {
        pattern: pattern?,
        range,
        has_frame_field,
    })
}

#[cfg(test)]
mod tests;
