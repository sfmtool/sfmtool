// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Lineage: which earlier contents this file's point rows came from.
//!
//! A Point ID names a point by a content hash and a row index in the thing that
//! hash identifies (`specs/formats/sfmr-file-format.md`, "Point ID"). Rewriting
//! a reconstruction gives it a new `content_xxh128`, so an id written down
//! against the earlier content names rows of a file that is no longer the one in
//! hand. The `lineage` field of `metadata.json` is what carries the earlier
//! content forward: one [`LineageEntry`] per ancestor, each holding that
//! ancestor's hash and a [`LineageMap`] from its rows straight into this file's.
//!
//! Two properties make the entry usable on its own:
//!
//! - **Every map is already composed.** An entry's map goes from the ancestor to
//!   *this* file, not to the next ancestor along, so resolving an id is one
//!   lookup and one map application whatever the depth of the chain. A writer
//!   that carries lineage forward composes as it goes.
//! - **It is content, so it is hashed.** `lineage` is a field of
//!   `metadata.json`, which is inside `metadata_xxh128` and so inside
//!   `content_xxh128`. Two files that say different things about where their
//!   rows came from are different files.
//!
//! A file that carries no lineage has an empty list, which is what every file
//! written before version 9 and every reconstruction with no ancestor holds.

use serde::{Deserialize, Serialize};

/// What kind of content a [`LineageEntry::hash`] identifies.
///
/// A base's hash is a `content_xxh128`, so it may also be the hash of a file on
/// disk; a point edit's hash names no file's content and only ever appears here
/// and in an id. A reader that does not know a kind keeps the entry: the hash
/// and the map are what resolution needs, and the kind is what tells a reader
/// whether looking for a file with that hash could ever succeed.
pub const LINEAGE_KIND_BASE: &str = "base";

/// The kind of a [`LineageEntry`] whose hash is a point edit's, not a base's.
///
/// Its rows are the points that edit created, numbered from zero in the order it
/// created them, rather than rows of any whole reconstruction.
pub const LINEAGE_KIND_POINT_EDIT: &str = "point_edit";

/// Where one ancestor's rows are in this file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageEntry {
    /// The ancestor's content hash, 32 lowercase hex digits for a base and the
    /// same shape for a point edit. This is the `{hash}` an id minted against
    /// that ancestor carries, of which an id usually shows the first eight.
    pub hash: String,
    /// [`LINEAGE_KIND_BASE`] or [`LINEAGE_KIND_POINT_EDIT`].
    pub kind: String,
    /// From the ancestor's row indexes to this file's.
    pub map: LineageMap,
}

/// A map from an ancestor's rows to this file's, in whichever of two encodings
/// says it in the fewest numbers.
///
/// Both answer one question -- where did ancestor row `i` go -- and both are
/// exact. The choice between them is a size decision and nothing else, so a
/// reader handles both and a writer picks whichever fits what it has.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "form", rename_all = "snake_case")]
pub enum LineageMap {
    /// Order-preserving: the ancestor's surviving rows appear in this file in
    /// their original order, with rows this file gained interleaved among them.
    ///
    /// `deleted` lists the ancestor rows that are not here, ascending;
    /// `created` lists the rows of *this* file that no ancestor row maps to,
    /// ascending. Those two lists determine the map: walk this file's rows in
    /// order, skip the created ones, and hand out the ancestor's rows in order
    /// with the deleted ones skipped. This is the shape a materialisation
    /// produces, and it costs the size of the edit rather than the size of the
    /// reconstruction.
    ///
    /// `source_rows` is how many rows the ancestor had. It is carried because
    /// the two lists do not imply it: they name only the rows that changed, so
    /// without it the map's domain would have to be guessed from the highest row
    /// it happens to mention, and every unchanged row past that would be
    /// invisible to anything enumerating the map.
    Monotone {
        /// How many rows the ancestor had, which is the map's domain: valid
        /// indexes are `0 .. source_rows`.
        source_rows: u32,
        /// Ancestor rows with no row here, ascending.
        deleted: Vec<u32>,
        /// Rows here with no ancestor row, ascending.
        created: Vec<u32>,
    },
    /// One entry per ancestor row: the row of this file it landed in, or `null`
    /// when it is not here.
    ///
    /// The general encoding, for a step that reorders rows -- which
    /// [`LineageMap::Monotone`] cannot express.
    Dense {
        /// `rows[i]` is where ancestor row `i` is in this file.
        rows: Vec<Option<u32>>,
    },
}

impl LineageMap {
    /// How many rows the ancestor had, which is the map's domain: every index a
    /// caller enumerating the map should ask about is below it.
    ///
    /// Both encodings state it outright, which is what lets a consumer walk a
    /// whole map without knowing which encoding it is in.
    pub fn source_rows(&self) -> u32 {
        match self {
            LineageMap::Monotone { source_rows, .. } => *source_rows,
            LineageMap::Dense { rows } => rows.len() as u32,
        }
    }

    /// Where ancestor row `index` is in this file, or `None` when it is not.
    pub fn forward(&self, index: u32) -> Option<u32> {
        match self {
            LineageMap::Monotone {
                source_rows,
                deleted,
                created,
            } => {
                // Outside the domain is not a row of the ancestor at all, which
                // is the same answer the dense form gives past the end of its
                // list.
                if index >= *source_rows || deleted.binary_search(&index).is_ok() {
                    return None;
                }
                // How many of the ancestor's rows before this one survive, then
                // how far the created rows push that position along.
                let live_before = index as usize - deleted.partition_point(|&d| d < index);
                let mut row = live_before as u32;
                // Each created row at or before the running position displaces
                // it by one; `created` is ascending, so one pass settles it.
                for &c in created {
                    if c <= row {
                        row += 1;
                    } else {
                        break;
                    }
                }
                Some(row)
            }
            LineageMap::Dense { rows } => rows.get(index as usize).copied().flatten(),
        }
    }
}
