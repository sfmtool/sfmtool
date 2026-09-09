// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Point ids over a node's version graph: which id a point is shown under, and
//! which point an id names.
//!
//! See `specs/gui/goto-point.md` and the format spec's Point ID section. There
//! is one form, `pt3d_{hash}_{index}`: a content hash and a row index in the
//! content that hash identifies. A point's identity is exactly that pair, and it
//! is the same pair in every node that holds that content, so nothing in an id
//! names a node and which node to show is settled by the selection.
//!
//! A node edits its reconstruction, so the content an index is a row of changes
//! under the id. Two walks keep an id meaning one point anyway, and this module
//! is the two of them:
//!
//! - [`mint`] chooses the content to name. **The version on disk first**: if the
//!   point's identity reaches the version the node was loaded at or last saved
//!   as, and that version is a base, the id is that base's hash and the point's
//!   row in it, so the id resolves straight in the file on disk with nothing to
//!   consult. **Otherwise the earliest rule**: the walk goes back through the
//!   version graph's maps as far as the point's identity reaches and names the
//!   base, or the point edit, it stops at.
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

/// The id `node` shows for the point at `index` in the value at its cursor.
///
/// **The version on disk first.** If the point's identity reaches the version
/// the node was loaded at or last saved as, and it is a row of that version's
/// base, the id is that base's hash and that row. That is the id a reader of the
/// file on disk uses as it stands, with no lineage to consult and no other file
/// to find, which is what someone copying an id out of the viewer almost always
/// wants it for.
///
/// **The earliest content otherwise**, which is the case for a point created
/// since the last save (named by the edit that created it), for a cursor on a
/// branch the disk version is not an ancestor of, such as after an undo past a
/// save, and for a disk version the budget has released. The lineage a save
/// records keeps an earlier id resolving in every file written afterwards, so
/// this is a weaker id than the disk one rather than a broken one.
///
/// `None` when `index` names no live point of the value at the cursor, or when
/// no content the point reaches can still be hashed.
pub fn mint(node: &SceneNode, index: u32) -> Option<String> {
    let Reached { chain, created } = walk_back(node, index)?;

    let disk = node.history.disk_serial();
    let on_disk = chain
        .iter()
        .find(|(serial, _)| *serial == disk)
        .and_then(|(serial, index)| base_named(node, *serial, *index));

    let (hash, index) = on_disk.or_else(|| {
        // The earliest content the identity reaches. A creating edit is
        // before every base on the chain, so it wins when there is one.
        created.or_else(|| {
            chain
                .iter()
                .rev()
                .find_map(|(serial, index)| base_named(node, *serial, *index))
        })
    })?;
    Some(format!("pt3d_{}_{index}", &hash[..HASH_PREFIX_LEN]))
}

/// A content hash and the index the point has in it. What an id is made of, and
/// what both of [`mint`]'s rules produce.
type Named = (String, u32);

/// What walking back from the cursor found.
struct Reached {
    /// `(version, index in it)` newest first, holding every version in which the
    /// point has an index.
    chain: Vec<(VersionSerial, u32)>,
    /// The edit that created the point, when the walk reached one.
    created: Option<Named>,
}

/// The chain of versions the point at `index` reaches, walking back from the
/// cursor, and the edit that created it when the walk found one.
///
/// The walk inverts one step at a time; a step that says it created the point
/// ends the chain and names the creating edit. Both halves are returned because
/// the two rules in [`mint`] want different parts of the same walk.
fn walk_back(node: &SceneNode, index: u32) -> Option<Reached> {
    let history = &node.history;
    let mut chain = vec![(history.current_version().serial, index)];
    loop {
        let (serial, index) = *chain.last().expect("seeded with the cursor");
        // What a step says it created is asked first, and is authoritative. A
        // map may or may not report a created row as having no predecessor --
        // one whose indexes are stable across the edit reports every index as
        // surviving, since none of them stopped resolving -- so the created list
        // is the only place that always knows.
        if let Some(created) = history.created_by(serial) {
            if let Some(k) = created.indexes.iter().position(|&i| i == index) {
                return Some(Reached {
                    chain,
                    created: Some((created.hash.clone(), k as u32)),
                });
            }
        }
        let Some(parent) = history.parent_of(serial) else {
            return Some(Reached {
                chain,
                created: None,
            });
        };
        let Some(map) = history.map_between(parent, serial) else {
            return Some(Reached {
                chain,
                created: None,
            });
        };
        match map.inverse(index) {
            Some(before) => chain.push((parent, before)),
            // The step created the point and said nothing about it, which the
            // check above has already ruled out for a step that names its
            // creations. There is no earlier content to walk to.
            None => {
                return Some(Reached {
                    chain,
                    created: None,
                })
            }
        }
    }
}

/// `(hash, row)` when version `serial` still holds a value and `index` is a row
/// of its base.
///
/// `None` for a version the budget has released, which has no columns to hash,
/// and for an index that is an addition of that version's overlay rather than a
/// row of its base, which only a created-point edit produces and which [`mint`]
/// names through the creating edit instead.
fn base_named(node: &SceneNode, serial: VersionSerial, index: u32) -> Option<(String, u32)> {
    let value = value_at(node, serial)?;
    ((index as usize) < value.base_point_count())
        .then(|| full_base_hash(value).map(|hash| (hash.to_string(), index)))
        .flatten()
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
