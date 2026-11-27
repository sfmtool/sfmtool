// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The viewer's notice that a workspace decodes large photographs whole.
//!
//! The first time in a session a read of the photograph cache (an open's
//! render of the patch bitmaps, a step's read, a panel's layout read) finds a
//! photograph with a side over 1536 px in a workspace whose file has no
//! `mipmap_options` ([`PhotographCache::large_photograph_without_mipmap_options`]),
//! [`AppState::give_photograph_tier_notice`], once a frame, writes
//! [`PHOTOGRAPH_TIER_NOTICE`] as one Action Log row of the viewer's, which the
//! 3D view's status line then shows. The read runs off the GUI thread, on
//! the open's worker or the rayon pool; the frame only reads what the cache noted,
//! which takes no lock and reads no file. See `specs/gui/action-log.md` §
//! "What gets logged".
//!
//! [`PhotographCache::large_photograph_without_mipmap_options`]:
//!     sfmtool_core::camera::PhotographCache::large_photograph_without_mipmap_options

use super::AppState;
use crate::action_log::{Actor, Kind};

#[cfg(test)]
mod tests;

/// What the viewer says once a session for a workspace that decodes large
/// photographs whole.
pub(crate) const PHOTOGRAPH_TIER_NOTICE: &str = "This workspace decodes large photographs \
     whole. Run `sfm ws mipmaps --enable` to read them a tile at a time.";

impl AppState {
    /// Write [`PHOTOGRAPH_TIER_NOTICE`] as the viewer's, once a session, as
    /// soon as the photograph cache has found a photograph over 1536 px in
    /// a workspace without `mipmap_options`. Called once a frame.
    pub(crate) fn give_photograph_tier_notice(&mut self) {
        if self.photograph_tier_notice_given
            || self
                .photographs
                .large_photograph_without_mipmap_options()
                .is_none()
        {
            return;
        }
        self.photograph_tier_notice_given = true;
        self.action_log
            .record_as(Actor::Viewer, Kind::Session, PHOTOGRAPH_TIER_NOTICE);
    }
}
