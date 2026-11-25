// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Filling in the display thumbnail rows that await a mipmap.
//!
//! An open gives an out-of-core photograph whose mipmap is not current the
//! grey placeholder row and puts the photograph on *Build mipmaps*' list
//! ([`crate::display_thumbnails`], [`super::build_mipmaps`]). Each photograph
//! a run of that task answers, made or not, is recorded on its list;
//! [`AppState::drive_thumbnail_fills`], once a frame, takes them, starts one
//! read on the rayon pool for each photograph a node's row awaits
//! ([`filled_row`]: its coarsest mip level file, or a whole decode when its
//! mipmap is still not current), and gives each node whose row awaits a
//! landed photograph a display column with the row filled in. The GUI thread
//! reads no photograph, and nothing here makes a mipmap or queues one.
//!
//! The node's column is the only thing that changes: no version is pushed,
//! the value is untouched, and nothing filled in reaches a save. A row whose
//! node was closed before its read landed is dropped. See
//! `specs/gui/multi-panel-image-browser.md` § "Thumbnail loading".

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ndarray::Array3;

use super::build_mipmaps::lock;
use super::AppState;
use crate::display_thumbnails::filled_row;

#[cfg(test)]
mod tests;

/// Filled rows, each with the photograph it is the row of.
type Rows = Vec<(PathBuf, Arc<Array3<u8>>)>;

/// The rows landed from the reads on the rayon pool.
type Landed = Arc<Mutex<Rows>>;

/// The reads of awaited rows the viewer has started and the rows they have
/// brought back.
#[derive(Default)]
pub(crate) struct ThumbnailFills {
    /// The photographs a read is running for, so one answered again while
    /// its read runs is not read twice.
    in_flight: HashSet<PathBuf>,
    /// The rows the reads brought back that no frame has given a node yet.
    landed: Landed,
}

impl AppState {
    /// Start a read for each photograph *Build mipmaps* has answered whose
    /// row a node awaits, and give each node whose row awaits a photograph
    /// whose read has landed a display column with that row filled in.
    ///
    /// Called once a frame. Returns whether a node's column changed, which
    /// the next frame has to draw. A read calls the viewer's wake as it
    /// lands, so an idle viewer draws its row.
    pub(crate) fn drive_thumbnail_fills(&mut self) -> bool {
        let answered = lock(&self.photograph_requests.mipmaps).take_answered();
        for path in answered {
            let awaited = self.scene.iter().any(|node| {
                node.display_thumbnails
                    .as_ref()
                    .is_some_and(|display| display.awaits(&path))
            });
            if !awaited || !self.thumbnail_fills.in_flight.insert(path.clone()) {
                continue;
            }
            let cache = Arc::clone(&self.photographs);
            let landed = Arc::clone(&self.thumbnail_fills.landed);
            let wake = self.wake.clone();
            self.photograph_requests.spawn(move || {
                let row = filled_row(&path, &cache);
                lock_landed(&landed).push((path, row));
                if let Some(wake) = wake {
                    wake();
                }
            });
        }

        let landed = std::mem::take(&mut *lock_landed(&self.thumbnail_fills.landed));
        if landed.is_empty() {
            return false;
        }
        for (path, _) in &landed {
            self.thumbnail_fills.in_flight.remove(path);
        }
        let mut changed = false;
        for node in &mut self.scene {
            let Some(display) = &node.display_thumbnails else {
                continue;
            };
            if let Some(filled) = display.with_rows(&landed) {
                node.display_thumbnails = Some(Arc::new(filled));
                changed = true;
            }
        }
        changed
    }
}

/// Lock `landed`, ignoring a poison: nothing panics while it is held.
fn lock_landed(landed: &Mutex<Rows>) -> std::sync::MutexGuard<'_, Rows> {
    landed.lock().unwrap_or_else(|e| e.into_inner())
}
