// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The track table cache, and the *Find points* operation that fills it.
//!
//! See `specs/gui/mcp-server.md` § "`find_points`" and
//! `specs/core/analysis/track-table.md`. The cache holds one core
//! [`TrackTable`] per node, for the version at its cursor, keyed by
//! `(node, document serial)`. A table's values depend only on the document
//! half of a version, so a bench step, which keeps the document serial, keeps
//! the table, and an edit, an undo or a redo that moves the serial makes it
//! another key. A node holds at most one entry: a table for another serial
//! replaces it, and a close drops it.
//!
//! *Find points* ([`Operation::FIND_POINTS`]) is an asked-for fast-queue task
//! that locks nothing, since it pushes no version. Its job computes a
//! [`TrackTable`] of only the columns the cache does not hold, from the value
//! at the cursor when it was queued, and when it lands the GUI thread merges
//! that table into the cache, runs the query and writes the row. A table for a
//! serial the node has since left is dropped with a failed row.

use sfmtool_core::analysis::track_table::{
    query_points, TrackTable, TrackTableColumn, TrackTableMergeError, TrackTableQueryError,
    TrackTableQueryOptions,
};

use sfmtool_core::progress::Progress;

use crate::background::{Finished, Job, Operation};
use crate::document::VersionSerial;
use crate::scene::ReconId;
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// One cached table: the node and the document serial it was computed for.
struct CachedTable {
    node: ReconId,
    document: VersionSerial,
    table: TrackTable,
}

/// The track table cache: at most one [`TrackTable`] per node, for one
/// document serial.
#[derive(Default)]
pub(crate) struct TrackTableCache {
    entries: Vec<CachedTable>,
}

impl TrackTableCache {
    /// The table held for `node` at `document`, or `None` when the node has
    /// none or holds one for another serial.
    fn get(&self, node: ReconId, document: VersionSerial) -> Option<&TrackTable> {
        self.entries
            .iter()
            .find(|entry| entry.node == node && entry.document == document)
            .map(|entry| &entry.table)
    }

    /// Merge `table`, computed for `node` at `document`, into the cache.
    ///
    /// Into the entry for that serial when there is one, with
    /// [`TrackTable::merge`], which keeps the columns it already holds. A node
    /// with an entry for another serial has it replaced, so the cache never
    /// holds a table for a version the node has left behind once a newer one
    /// lands.
    ///
    /// # Errors
    ///
    /// [`TrackTableMergeError`] when `table`'s rows are not those of the entry
    /// already held for that serial, which leaves the entry as it was.
    fn merge(
        &mut self,
        node: ReconId,
        document: VersionSerial,
        table: TrackTable,
    ) -> Result<(), TrackTableMergeError> {
        match self.entries.iter_mut().find(|entry| entry.node == node) {
            Some(entry) if entry.document == document => entry.table.merge(table),
            Some(entry) => {
                entry.document = document;
                entry.table = table;
                Ok(())
            }
            None => {
                self.entries.push(CachedTable {
                    node,
                    document,
                    table,
                });
                Ok(())
            }
        }
    }

    /// Drop the table held for `node`: what closing the node does.
    pub(crate) fn forget_node(&mut self, node: ReconId) {
        self.entries.retain(|entry| entry.node != node);
    }

    /// Drop every table: what closing every node does.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

/// What a *Find points* job hands back: the columns it computed, for the
/// document serial it read, and the query to run once they are merged.
pub(crate) struct FindPointsRun {
    /// The document serial of the value the job read.
    document: VersionSerial,
    /// The missing columns, for every live point of that value.
    table: TrackTable,
    /// The query the call asked for.
    options: TrackTableQueryOptions,
}

/// What a landed *Find points* found: the matched point indexes, in order,
/// for the document serial whose cached table holds their values.
#[cfg_attr(not(feature = "mcp"), allow(dead_code))]
pub(crate) struct FoundPoints {
    /// The serial the query ran at, which the cache entry is for.
    pub(crate) document: VersionSerial,
    /// The indexes of the points that matched, in the query's order.
    pub(crate) matched: Vec<u32>,
}

/// The row a query writes: how many points matched out of how many.
fn found_sentence(matched: usize, live: usize, label: &str) -> String {
    format!("Found {matched} of {live} points in {label}")
}

impl AppState {
    /// The track table cached for `node` at its cursor, or `None` when there
    /// is none for that version.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn track_table(&self, node: ReconId) -> Option<&TrackTable> {
        let document = self.node(node)?.history.current_version().document_serial;
        self.track_tables.get(node, document)
    }

    /// Queue *Find points* on `node`: compute `columns` for the version at its
    /// cursor, and when it lands merge them into the cache and run `options`
    /// over the cached table. Answers with the task's id.
    ///
    /// Locks nothing and is not refused by a busy node, since it pushes no
    /// version. A busy fast queue makes it wait.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn start_find_points(
        &mut self,
        node: ReconId,
        columns: Vec<TrackTableColumn>,
        options: TrackTableQueryOptions,
    ) -> Result<u64, String> {
        let job = self.find_points_job(node, columns, options)?;
        let label = self.node(node).expect("just read").label.clone();
        self.start_task(Operation::FIND_POINTS, Some(node), label, job)
    }

    /// The work *Find points* does on `node`: `columns` computed for the value
    /// at its cursor now, handed back with that value's document serial and
    /// the query to run once they are merged.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn find_points_job(
        &self,
        node: ReconId,
        columns: Vec<TrackTableColumn>,
        options: TrackTableQueryOptions,
    ) -> Result<Job, String> {
        let scene_node = self
            .node(node)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let document = scene_node.history.current_version().document_serial;
        let edited = scene_node.edited().clone();
        Ok(Box::new(
            move |progress: &Progress<'_>| match TrackTable::compute(&edited, &columns, progress) {
                Ok(table) => Finished::FoundPoints(Box::new(FindPointsRun {
                    document,
                    table,
                    options,
                })),
                Err(_) => Finished::Cancelled,
            },
        ))
    }

    /// Land a *Find points* run on the node at `index`: merge its table into
    /// the cache, run its query, and answer with the row's sentence and what
    /// was found. A run for a serial the node has left is dropped with a
    /// refusal, and the cache is left alone.
    pub(crate) fn land_found_points(
        &mut self,
        index: usize,
        run: FindPointsRun,
    ) -> (Result<String, String>, Option<FoundPoints>) {
        let FindPointsRun {
            document,
            table,
            options,
        } = run;
        let node = &self.scene[index];
        let (id, label) = (node.id, node.label.clone());
        if node.history.current_version().document_serial != document {
            return (
                Err(format!(
                    "Find points of {label} finished, but {label} has changed since it \
                     started, so its answer was dropped; call find_points again."
                )),
                None,
            );
        }
        if let Err(error) = self.track_tables.merge(id, document, table) {
            return (Err(format!("Find points of {label}: {error}")), None);
        }
        let table = self.track_tables.get(id, document).expect("just merged");
        match query_points(self.scene[index].edited(), table, &options) {
            // The call found this bound or sort column cached when it queued
            // the run, so the entry was replaced since: a run for another
            // version landed while that version was at the cursor.
            Err(TrackTableQueryError::ColumnNotComputed(_)) => (
                Err(format!(
                    "Find points of {label} finished, but a run for another version of \
                     {label} replaced the values it needed, so its answer was dropped; \
                     call find_points again."
                )),
                None,
            ),
            Err(error) => (Err(format!("Find points of {label}: {error}")), None),
            Ok(matched) => (
                Ok(found_sentence(
                    matched.len(),
                    table.point_indexes().len(),
                    &label,
                )),
                Some(FoundPoints { document, matched }),
            ),
        }
    }
}
