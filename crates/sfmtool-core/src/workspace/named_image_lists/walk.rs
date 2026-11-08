// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The files under a workspace root that an image pattern can match.
//!
//! The walk follows the pattern one path segment at a time and lists only the
//! directories a segment can lead into, so `frames/walk1_%04d.jpg` lists the
//! workspace root and `frames/` and nothing else. A `**` segment is the one
//! place it descends into every directory below. Each segment is compared
//! with the camrig matcher against the names the directory lists, never by
//! asking the filesystem whether a name exists, so a literal segment matches
//! case-sensitively on a case-insensitive filesystem too. A frame field is
//! compared as any run of digits, so the candidates include the files the
//! pattern misses only by the padding of their number, which the caller
//! reports. The caller confirms every candidate against the whole pattern.
//!
//! A directory's listing is kept for the rest of the walk, with a map from
//! each name to its entry, so a list of many entries in one folder
//! (hand-picked photographs, one entry each) reads that folder once, and a
//! segment with no wildcard and no frame field is one lookup in that map
//! rather than a comparison with every name in the folder.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use sfmtool_camrig_format::{pattern_literal, pattern_matches_any_padding};
use sfmtool_progress::{Cancelled, Progress};

/// What a directory entry is, with a symbolic link classified by its target.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    File,
    Dir,
    /// A symbolic link to a directory. A `**` does not descend through one,
    /// so a link back up the tree cannot make the walk endless; a named or
    /// wildcard segment does.
    LinkedDir,
    /// Anything else, such as a broken link.
    Other,
}

struct Item {
    name: String,
    kind: Kind,
}

/// One directory's entries, and each name's index among them.
struct Listing {
    items: Vec<Item>,
    by_name: HashMap<String, usize>,
}

/// Why a walk stopped.
pub(super) enum WalkError {
    /// Listing a directory failed.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The caller's progress asked the walk to stop.
    Cancelled,
}

impl From<Cancelled> for WalkError {
    fn from(_: Cancelled) -> Self {
        WalkError::Cancelled
    }
}

/// A walk of one workspace root, keeping each directory's listing.
pub(super) struct Walker<'a> {
    root: &'a Path,
    progress: &'a Progress<'a>,
    listings: HashMap<String, Rc<Listing>>,
    entries_read: u64,
}

fn join(dir: &str, name: &str) -> String {
    if dir.is_empty() {
        name.to_string()
    } else {
        format!("{dir}/{name}")
    }
}

impl<'a> Walker<'a> {
    pub(super) fn new(root: &'a Path, progress: &'a Progress<'a>) -> Self {
        Self {
            root,
            progress,
            listings: HashMap::new(),
            entries_read: 0,
        }
    }

    /// The listing of the workspace-relative directory `dir` (`""` is the
    /// root), read once per walk. Names that are not UTF-8 are left out, since
    /// no pattern, which is UTF-8, can match them.
    fn list(&mut self, dir: &str) -> Result<Rc<Listing>, WalkError> {
        if let Some(items) = self.listings.get(dir) {
            return Ok(Rc::clone(items));
        }
        self.progress.check_cancel()?;
        let path = self.root.join(dir);
        let io_err = |source| WalkError::Io {
            path: path.clone(),
            source,
        };
        let mut items = Vec::new();
        for entry in std::fs::read_dir(&path).map_err(io_err)? {
            let entry = entry.map_err(io_err)?;
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            let file_type = entry.file_type().map_err(io_err)?;
            let kind = if file_type.is_dir() {
                Kind::Dir
            } else if file_type.is_file() {
                Kind::File
            } else if file_type.is_symlink() {
                match std::fs::metadata(entry.path()) {
                    Ok(meta) if meta.is_dir() => Kind::LinkedDir,
                    Ok(meta) if meta.is_file() => Kind::File,
                    _ => Kind::Other,
                }
            } else {
                Kind::Other
            };
            items.push(Item { name, kind });
        }
        self.entries_read += items.len() as u64;
        self.progress.count(self.entries_read, None, "entries");
        let by_name = items
            .iter()
            .enumerate()
            .map(|(index, item)| (item.name.clone(), index))
            .collect();
        let listing = Rc::new(Listing { items, by_name });
        self.listings.insert(dir.to_string(), Rc::clone(&listing));
        Ok(listing)
    }

    /// Every directory at or below `dir`, `dir` first, not descending
    /// through a link to a directory.
    fn dirs_below(&mut self, dir: &str, out: &mut Vec<String>) -> Result<(), WalkError> {
        out.push(dir.to_string());
        let listing = self.list(dir)?;
        for item in listing.items.iter().filter(|item| item.kind == Kind::Dir) {
            self.dirs_below(&join(dir, &item.name), out)?;
        }
        Ok(())
    }

    /// The workspace-relative paths of the files the walk can reach by
    /// following `pattern`'s segments, sorted. Every file `pattern` matches is
    /// among them; the caller confirms each one against the whole pattern.
    pub(super) fn candidates(&mut self, pattern: &str) -> Result<BTreeSet<String>, WalkError> {
        let segments: Vec<&str> = pattern.split('/').collect();
        let mut dirs: Vec<String> = vec![String::new()];
        let mut files = BTreeSet::new();
        for (i, segment) in segments.iter().enumerate() {
            let last = i + 1 == segments.len();
            let mut next = Vec::new();
            if *segment == "**" {
                let mut below = Vec::new();
                for dir in &dirs {
                    self.dirs_below(dir, &mut below)?;
                }
                if last {
                    for dir in &below {
                        let listing = self.list(dir)?;
                        for item in listing.items.iter().filter(|item| item.kind == Kind::File) {
                            files.insert(join(dir, &item.name));
                        }
                    }
                } else {
                    next = below;
                }
            } else {
                // A segment with no wildcard and no frame field names one
                // entry, found in the listing's map; the map compares names
                // exactly, as the matcher does.
                let literal = pattern_literal(segment);
                for dir in &dirs {
                    let listing = self.list(dir)?;
                    let mut take = |item: &Item| match item.kind {
                        Kind::File if last => {
                            files.insert(join(dir, &item.name));
                        }
                        Kind::Dir | Kind::LinkedDir if !last => {
                            next.push(join(dir, &item.name));
                        }
                        _ => {}
                    };
                    match &literal {
                        Some(name) => {
                            if let Some(&index) = listing.by_name.get(name) {
                                take(&listing.items[index]);
                            }
                        }
                        None => listing
                            .items
                            .iter()
                            .filter(|item| pattern_matches_any_padding(segment, &item.name, false))
                            .for_each(take),
                    }
                }
            }
            // Two `**` segments can reach one directory by two routes.
            next.sort();
            next.dedup();
            dirs = next;
        }
        Ok(files)
    }
}
