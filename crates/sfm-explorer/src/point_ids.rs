// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Point ids over a node's version graph: which id a point is shown under, and
//! which point an id names.
//!
//! See `specs/gui/goto-point.md` and the format spec's Point ID section. An id
//! is a content hash and a row index in the content that hash identifies:
//!
//! ```text
//! pt3d_{hash}_{index}            file form
//! pt3d_{hash}_{index}_n{node}    session form
//! ```
//!
//! A node edits its reconstruction, so the content an index is a row of changes
//! under the id. Two rules keep an id meaning one point anyway, and this module
//! is the two of them:
//!
//! - [`mint`] applies the **earliest rule**. The id shown for a point is its
//!   coordinate not in the value at the cursor but in the earliest content its
//!   identity reaches: the walk goes back through the version graph's maps as
//!   far as the point goes, and names the base, or the point edit, it stops at.
//!   For a point that came out of the file the node was loaded from, that is the
//!   id that file's readers already use, and no edit, undo or save changes it.
//! - [`resolve`] walks the other way. It finds the hash among the node's bases,
//!   its point edits and its bases' recorded lineage, and follows the point from
//!   there to the cursor -- backward out of a branch an undo discarded and then
//!   forward, when the minting version is not an ancestor of the cursor's.
//!
//! Every hash here is computed from a value rather than read off a file, so a
//! node that was never saved -- demo data, a resection -- has ids exactly like
//! one that was.

use sfmtool_core::EditedReconstruction;

use crate::document::VersionSerial;
use crate::scene::SceneNode;

#[cfg(test)]
mod tests;

/// How many hex digits of a hash a displayed id carries.
pub const HASH_PREFIX_LEN: usize = 8;

/// How many hex digits a whole content hash has.
///
/// An XXH128 digest written as lowercase hex, which is what `content_xxh128` is
/// and the only length a hash this module will work from.
pub const HASH_LEN: usize = 32;

/// The first [`HASH_PREFIX_LEN`] hex digits of `edited`'s base content hash.
///
/// Computed from the value and cached on it, so it is the hash a save of that
/// base writes and it costs one pass over the columns per base rather than one
/// per call. `None` only when the value cannot be hashed at all, which is a
/// reconstruction the writer would refuse.
pub fn base_hash_prefix(edited: &EditedReconstruction) -> Option<String> {
    Some(full_base_hash(edited)?[..HASH_PREFIX_LEN].to_string())
}

/// The whole base content hash of `edited`.
///
/// The length is checked **exactly** rather than as a lower bound: a content
/// hash is [`HASH_LEN`] hex digits and nothing else, so a string of any other
/// length is not one, and answering with a prefix of it would put a hash into an
/// id that no file and no other walk will ever match. The empty string a value
/// carrying no stored hash holds is the case this rejects in practice.
fn full_base_hash(edited: &EditedReconstruction) -> Option<&str> {
    let hash = &edited.base_content_hash().ok()?.content_xxh128;
    (hash.len() == HASH_LEN).then_some(hash.as_str())
}

/// The id `node` shows for the point at `index` in the value at its cursor, in
/// the session form.
///
/// `None` when `index` names no live point of that value, or when the earliest
/// content the point reaches can no longer be hashed because the budget released
/// it and every version after it too.
pub fn mint(node: &SceneNode, index: u32) -> Option<String> {
    let (hash, index) = earliest(node, index)?;
    Some(format!(
        "pt3d_{}_{index}_n{}",
        &hash[..HASH_PREFIX_LEN],
        node.id.raw()
    ))
}

/// The earliest content the point at `index` reaches, as `(hash, index in it)`.
///
/// The walk inverts one step at a time. A step that reports the point as not
/// having existed before it is the step that created it, and a created point is
/// named by the edit's own hash and its place among that edit's creations. When
/// the walk instead runs out of graph it has reached the node's first version,
/// and the point is a row of some base along the way -- the oldest one that
/// still holds its value, since a released version has no columns to hash.
fn earliest(node: &SceneNode, index: u32) -> Option<(String, u32)> {
    let history = &node.history;
    // The trail from the cursor backwards, newest first.
    let mut trail = vec![(history.current_version().serial, index)];
    loop {
        let (serial, index) = *trail.last().expect("seeded with the cursor");
        let Some(parent) = history.parent_of(serial) else {
            break;
        };
        let Some(map) = history.map_between(parent, serial) else {
            break;
        };
        match map.inverse(index) {
            Some(before) => trail.push((parent, before)),
            None => {
                // This step is where the point began. It is in no base, so it
                // is named by the edit that made it.
                let created = history.created_by(serial)?;
                let k = created.indexes.iter().position(|&i| i == index)?;
                return Some((created.hash.clone(), k as u32));
            }
        }
    }
    // Oldest first, so the first base that can answer is the earliest one.
    for (serial, index) in trail.iter().rev() {
        let Some(value) = value_at(node, *serial) else {
            continue;
        };
        if (*index as usize) >= value.base_point_count() {
            // An addition of that version's overlay, which only a created-point
            // edit produces and which the loop above has already named.
            continue;
        }
        if let Some(hash) = full_base_hash(value) {
            return Some((hash.to_string(), *index));
        }
    }
    None
}

/// The value of the version `serial`, when the budget has not released it.
fn value_at(node: &SceneNode, serial: VersionSerial) -> Option<&EditedReconstruction> {
    node.history
        .versions()
        .iter()
        .find(|v| v.serial == serial)
        .and_then(|v| v.value.as_ref())
}

/// Where the point that `hash` and `index` name is in the value at `node`'s
/// cursor.
///
/// `hash` is any prefix of a content hash, which is the eight digits a displayed
/// id carries in the ordinary case. The error says which of the two ways of
/// failing happened: the hash names nothing this node has been, or it does and
/// the point it named is not in the value at the cursor -- in which case it
/// names the version the walk stopped at, since that is where the point went.
pub fn resolve(node: &SceneNode, hash: &str, index: u32) -> Result<u32, String> {
    let cursor = node.history.current_version().serial;
    let Some((from, index)) = locate(node, hash, index)? else {
        return Err(format!(
            "{} has never held content with hash {hash}.",
            node.label
        ));
    };
    node.history.follow(from, cursor, index).map_err(|stopped| {
        format!(
            "{}: the point pt3d_{hash}_{index} names is not in {}; it was last there at {stopped}.",
            node.label,
            node.history.current_version().label,
        )
    })
}

/// Whether `hash` names content this node has been, or came from.
///
/// What Go to Point searches the loaded nodes with before it commits to one. It
/// asks about the hash alone, so a node that held the content but not the row is
/// still the node the query belongs to and gets to say what happened to the row,
/// rather than the search silently moving on to another node.
pub fn holds_hash(node: &SceneNode, hash: &str) -> bool {
    let history = &node.history;
    if history
        .all_serials()
        .into_iter()
        .filter_map(|s| history.created_by(s))
        .any(|c| starts_with(&c.hash, hash))
    {
        return true;
    }
    history.versions().iter().any(|version| {
        version.value.as_ref().is_some_and(|value| {
            let stored = &value.base.content_hash.content_xxh128;
            starts_with(full_base_hash(value).unwrap_or_default(), hash)
                || (!stored.is_empty() && starts_with(stored, hash))
                || value
                    .base
                    .metadata
                    .lineage
                    .iter()
                    .any(|entry| starts_with(&entry.hash, hash))
        })
    })
}

/// The version `hash` names and the index `index` is in it, or `None` when this
/// node has never held that content.
///
/// Three places carry a hash a live id can name, and they are searched in the
/// order that costs least: the point edits, whose hashes are held outright; the
/// bases of the versions that still have values; and those bases' recorded
/// lineage, which is how an id minted in an earlier session against content that
/// was materialised away still lands.
///
/// `Err` is the one case where the hash was found and the index was not a row of
/// what it named.
fn locate(
    node: &SceneNode,
    hash: &str,
    index: u32,
) -> Result<Option<(VersionSerial, u32)>, String> {
    let history = &node.history;

    for serial in history.all_serials() {
        let Some(created) = history.created_by(serial) else {
            continue;
        };
        if !starts_with(&created.hash, hash) {
            continue;
        }
        return match created.indexes.get(index as usize) {
            Some(&in_version) => Ok(Some((serial, in_version))),
            None => Err(format!(
                "The edit with hash {hash} created {} points, so index {index} is out of range.",
                created.indexes.len()
            )),
        };
    }

    for version in history.versions() {
        let Some(value) = version.value.as_ref() else {
            continue;
        };
        let stored = &value.base.content_hash.content_xxh128;
        let computed = full_base_hash(value).unwrap_or_default();
        if starts_with(computed, hash) || (!stored.is_empty() && starts_with(stored, hash)) {
            let count = value.base_point_count();
            if (index as usize) >= count {
                return Err(format!(
                    "The content with hash {hash} has {count} points, so index {index} is out of \
                     range."
                ));
            }
            return Ok(Some((version.serial, index)));
        }
        for entry in &value.base.metadata.lineage {
            if !starts_with(&entry.hash, hash) {
                continue;
            }
            return match entry.map.forward(index) {
                Some(row) => Ok(Some((version.serial, row))),
                None => Err(format!(
                    "{} records where the content with hash {hash} went, and row {index} is not \
                     in it.",
                    node.label
                )),
            };
        }
    }

    Ok(None)
}

/// Whether `hash` is a prefix of `full`, ignoring case.
fn starts_with(full: &str, hash: &str) -> bool {
    full.len() >= hash.len() && full[..hash.len()].eq_ignore_ascii_case(hash)
}

/// The node number an id's `_n{node}` suffix names.
///
/// A number rather than a [`crate::scene::ReconId`]: an id from another session
/// can carry a number no node here was ever handed, so a query is matched
/// against the loaded nodes' [`crate::scene::ReconId::raw`] rather than turned
/// into an id of its own.
pub fn parse_node_suffix(text: &str) -> Option<u32> {
    text.strip_prefix('n')?.parse::<u32>().ok()
}
