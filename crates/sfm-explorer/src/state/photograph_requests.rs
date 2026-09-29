// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Photographs for the panels, decoded off the GUI thread.
//!
//! A panel draws every frame and must not wait for a JPEG decode, which takes
//! about 40 ms for a 4K photograph. So it asks [`display_photograph`],
//! which only looks in the photograph cache ([`PhotographCache::peek_state`]).
//! On a miss it starts one decode of that file on the rayon pool
//! ([`PhotographCache::get`]) and answers [`DisplayPhotograph::Decoding`]; when
//! the decode ends it asks egui for a repaint, and the next frame's look finds
//! the pyramid, or the remembered failure.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use sfmtool_core::camera::remap::ImageU8Pyramid;
use sfmtool_core::camera::{PeekedPhotograph, PhotographCache};
use sfmtool_core::SfmrReconstruction;

use super::photograph_path;

#[cfg(test)]
mod tests;

/// What a panel has to draw a photograph with on this frame.
#[derive(Clone)]
pub(crate) enum DisplayPhotograph {
    /// Decoded; level 0 is the photograph.
    Decoded(Arc<ImageU8Pyramid>),
    /// Being decoded on a worker, or about to be. A repaint follows the
    /// decode.
    Decoding,
    /// The file could not be read or decoded, or the image index is not in the
    /// table.
    Unreadable,
}

/// The paths a decode has been started for and has not finished, so each is
/// asked for once however many frames look at it meanwhile.
#[derive(Clone, Debug, Default)]
pub(crate) struct PhotographRequests {
    in_flight: Arc<Mutex<HashSet<PathBuf>>>,
}

impl PhotographRequests {
    /// Start decoding `path` into `cache` on the rayon pool unless a decode of
    /// it is already running, and repaint `ctx` when it ends.
    fn request(&self, cache: &Arc<PhotographCache>, path: &Path, ctx: &egui::Context) {
        {
            let mut in_flight = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
            if !in_flight.insert(path.to_path_buf()) {
                return;
            }
        }
        let cache = Arc::clone(cache);
        let in_flight = Arc::clone(&self.in_flight);
        let path = path.to_path_buf();
        let ctx = ctx.clone();
        rayon::spawn(move || {
            cache.get(&path);
            // After the get, so a frame that no longer finds the path in flight
            // finds its entry in the cache.
            in_flight
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&path);
            ctx.request_repaint();
        });
    }

    /// Whether a decode of `path` is running.
    #[cfg(test)]
    pub(crate) fn is_in_flight(&self, path: &Path) -> bool {
        self.in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(path)
    }
}

/// The photograph of image `index` of `recon` as `cache` holds it, asking
/// `requests` to decode it in the background when it holds nothing yet.
///
/// Never reads the file on this thread, except under a cache with a budget of
/// 0: that cache keeps nothing, so a background decode would never be seen
/// and would be asked for again every frame, and the photograph is decoded
/// here instead, as it was before panels stopped decoding.
///
/// A remembered failure is answered from the cache without a stat of the
/// file, so a photograph that appears after it failed shows once something
/// else reads it with `get` or the entry is forgotten.
pub(crate) fn display_photograph(
    cache: &Arc<PhotographCache>,
    requests: &PhotographRequests,
    recon: &SfmrReconstruction,
    index: usize,
    ctx: &egui::Context,
) -> DisplayPhotograph {
    let Some(path) = photograph_path(recon, index) else {
        return DisplayPhotograph::Unreadable;
    };
    match cache.peek_state(&path) {
        PeekedPhotograph::Decoded(pyramid) => DisplayPhotograph::Decoded(pyramid),
        PeekedPhotograph::Unreadable => DisplayPhotograph::Unreadable,
        PeekedPhotograph::NotDecoded if cache.budget_bytes() == 0 => match cache.get(&path) {
            Some(pyramid) => DisplayPhotograph::Decoded(pyramid),
            None => DisplayPhotograph::Unreadable,
        },
        PeekedPhotograph::NotDecoded => {
            requests.request(cache, &path, ctx);
            DisplayPhotograph::Decoding
        }
    }
}
