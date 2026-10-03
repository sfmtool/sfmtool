// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The viewed track: the selected point read as an editable track, held off
//! every bench, which Track View draws while no item is focused.
//!
//! See `specs/gui/bench.md` § "Live evaluation". The **viewed point** is the
//! selected point on the selected node while no item is focused on that node
//! ([`AppState::viewed_point`]). Its track is built with core's `create_track`
//! exactly as a put builds a bench track, under the same label (the point's
//! portable ID), so its rows arrive `in` and pinned and the leave-one-out ZNCC
//! is read back from the stored column. It is never written anywhere: it is in
//! no version, no step accepts it, and no bench layer draws it.
//!
//! **Keyed by `(node, point, document serial)`.** A point's content at a given
//! document serial is fixed, so a track built for that key stays right for it.
//! Another point, another node, or a document edit or an undo that moves the
//! document serial is another key. The last [`CACHE_SIZE`] tracks are kept
//! with their evaluations, so clicking back to a point whose evaluation landed
//! draws it at once and starts nothing.
//!
//! **Who asks for it.** Track View's body draws from `&AppState` and cannot
//! build a track, so the dock calls [`AppState::refresh_viewed_track`] before
//! it draws the panel, and the frame calls [`AppState::hide_viewed_track`]
//! before the dock draws. A frame that does not draw Track View therefore
//! leaves no current viewed track, and the cache stays as it was.
//! [`AppState::viewed_track`] answers only while the stored key is still the
//! one the selection and the cursor give, so a selection change or a document
//! edit leaves no current viewed track until the next refresh, which is what
//! cancels a running evaluation of the previous one ([`super::live`]).
//!
//! **The read-only bars** ([`AppState::viewed_thresholds`]) are session state:
//! Track View's threshold boxes judge the viewed track's readings by them and
//! change no verdict. A put of the viewed point carries them onto the new track
//! when they differ from the defaults ([`AppState::put_point_on_bench`]).

use std::sync::Arc;

use sfmtool_core::bench::{self, Bench, CreateTrackOptions, EditableTrack, Thresholds, Verdict};

use super::live::{self, Evaluation};
use crate::document::VersionSerial;
use crate::scene::{PointRef, ReconId};
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// How many viewed tracks are kept.
const CACHE_SIZE: usize = 8;

/// What a viewed track is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ViewedKey {
    node: ReconId,
    point: u32,
    document: VersionSerial,
}

/// The selected point read as an editable track, off every bench.
pub(crate) struct ViewedTrack {
    /// The node the point is on.
    pub(crate) node: ReconId,
    /// The point, as an index of the version it was read from.
    pub(crate) point: u32,
    /// The document half it was read from. A point's content at a given
    /// document serial is fixed, so `(node, point, document)` is the key.
    pub(crate) document: VersionSerial,
    /// The point's portable ID, the label a put would give it on an empty
    /// bench.
    pub(crate) label: String,
    /// The track, with the measurements its last landed evaluation brought.
    pub(crate) track: Arc<EditableTrack>,
    /// Where its evaluation stands.
    pub(crate) evaluation: Evaluation,
}

impl ViewedTrack {
    fn key(&self) -> ViewedKey {
        ViewedKey {
            node: self.node,
            point: self.point,
            document: self.document,
        }
    }
}

/// The viewed-track cache, and which entry is current.
#[derive(Default)]
pub(crate) struct ViewedTracks {
    /// The last few viewed tracks, most recently viewed first, each key once.
    cache: Vec<ViewedTrack>,
    /// The key of the viewed track Track View last asked for, or `None` when
    /// it did not ask on the last frame or there was nothing to view.
    current: Option<ViewedKey>,
    /// Whether the last frame drew Track View showing the viewed track.
    shown: bool,
}

impl ViewedTracks {
    fn get(&self, key: ViewedKey) -> Option<&ViewedTrack> {
        self.cache.iter().find(|viewed| viewed.key() == key)
    }

    /// Drop every viewed track on `id`: what closing the node does.
    pub(crate) fn forget_node(&mut self, id: ReconId) {
        self.cache.retain(|viewed| viewed.node != id);
        self.current = self.current.filter(|key| key.node != id);
    }

    /// Drop every viewed track: what closing every node does.
    pub(crate) fn clear(&mut self) {
        self.cache.clear();
        self.current = None;
    }
}

impl AppState {
    /// The viewed point: the selected point on the selected node, while no
    /// item is focused on that node. It is what Track View draws as the
    /// viewed track, and a put of it carries the read-only bars.
    pub(crate) fn viewed_point(&self) -> Option<PointRef> {
        let id = self.selected_recon?;
        let point = self.selected_point.filter(|point| point.recon == id)?;
        if self.focused_item.is_some_and(|focused| focused.node == id) {
            return None;
        }
        Some(point)
    }

    /// The key the viewed point gives at the node's cursor, or `None` when
    /// there is no viewed point or it is not a live point of that version.
    fn viewed_key(&self) -> Option<ViewedKey> {
        let point = self.viewed_point()?;
        let node = self.node(point.recon)?;
        node.edited().point(point.point)?;
        Some(ViewedKey {
            node: point.recon,
            point: point.point,
            document: node.history.current_version().document_serial,
        })
    }

    /// Make the viewed track for the viewed point the current one, building it
    /// on first ask for its key and taking it from the cache after that.
    ///
    /// Called by the dock before it draws Track View, whichever mode the panel
    /// is in: while an item is focused there is no viewed point, and so no
    /// current viewed track.
    pub(crate) fn refresh_viewed_track(&mut self) {
        self.viewed_tracks.shown = true;
        let Some(key) = self.viewed_key() else {
            self.viewed_tracks.current = None;
            return;
        };
        let cache = &mut self.viewed_tracks.cache;
        match cache.iter().position(|viewed| viewed.key() == key) {
            Some(at) => {
                let viewed = cache.remove(at);
                cache.insert(0, viewed);
            }
            None => {
                let Some(viewed) = self.build_viewed_track(key) else {
                    self.viewed_tracks.current = None;
                    return;
                };
                let cache = &mut self.viewed_tracks.cache;
                cache.insert(0, viewed);
                cache.truncate(CACHE_SIZE);
            }
        }
        self.viewed_tracks.current = Some(key);
    }

    /// Leave no current viewed track, keeping the cache: what the frame does
    /// before the dock draws, so only a frame that draws Track View has one.
    pub(crate) fn hide_viewed_track(&mut self) {
        self.viewed_tracks.current = None;
        self.viewed_tracks.shown = false;
    }

    /// Whether the last frame drew Track View, and so asked for the viewed
    /// track: the wire's `get_point` refreshes it before answering when so,
    /// since a call earlier in the same batch may have moved the selection.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn viewed_track_shown(&self) -> bool {
        self.viewed_tracks.shown
    }

    /// The current viewed track, or `None` when Track View did not ask for one
    /// or the selection or the cursor has moved off it since it did.
    pub(crate) fn viewed_track(&self) -> Option<&ViewedTrack> {
        let key = self.viewed_tracks.current?;
        if self.viewed_key() != Some(key) {
            return None;
        }
        self.viewed_tracks.get(key)
    }

    /// Set the read-only bars Track View's threshold boxes hold while the
    /// panel shows the viewed track. Pushes no version, writes no row, and
    /// changes no verdict of the viewed track.
    pub(crate) fn set_viewed_thresholds(&mut self, bars: Thresholds) {
        self.viewed_thresholds = bars;
    }

    /// The verdict each row of the current viewed track gets from the
    /// read-only bars, `None` where nothing has measured the row: core's
    /// `verdicts_if_unpinned` over a copy of the track carrying those bars,
    /// which is the verdict the bench's own evaluation would give the row once
    /// it is unpinned.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn viewed_verdicts(&self) -> Option<Vec<Option<Verdict>>> {
        let viewed = self.viewed_track()?;
        let mut judged = (*viewed.track).clone();
        judged.thresholds = self.viewed_thresholds.clone();
        Some(bench::verdicts_if_unpinned(&judged))
    }

    /// Build the viewed track for `key` as a put would build it on an empty
    /// bench.
    fn build_viewed_track(&self, key: ViewedKey) -> Option<ViewedTrack> {
        let node = self.node(key.node)?;
        let options = CreateTrackOptions {
            version: node.history.current_version().serial.as_u64(),
            label: Some(crate::scene::point_id(node, key.point as usize)),
        };
        let (bench, report) =
            bench::create_track(&Bench::new(), node.edited(), key.point, &options).ok()?;
        let track = Arc::clone(bench.track(&report.label)?);
        let evaluation = match bench::evaluate_preconditions(&track) {
            Err(why) => Evaluation::Refused(format!("Cannot evaluate {}: {why}", report.label)),
            Ok(()) => Evaluation::Evaluating,
        };
        Some(ViewedTrack {
            node: key.node,
            point: key.point,
            document: key.document,
            label: report.label,
            track,
            evaluation,
        })
    }

    /// The evaluation of the current viewed track, reading the photographs
    /// the way a bench track's evaluation does
    /// ([`AppState::track_photometric_inputs`]).
    pub(super) fn viewed_evaluate_job(&mut self) -> Result<live::EvaluationJob, String> {
        let viewed = self
            .viewed_track()
            .ok_or_else(|| "No point is being viewed.".to_string())?;
        let (node, label) = (viewed.node, viewed.label.clone());
        let track = (*viewed.track).clone();
        let (edited, sources) = self.track_photometric_inputs(node, &track)?;
        Ok(super::evaluate_job(label, edited, track, sources))
    }

    /// Record how an evaluation that read `read` ended, on the cached viewed
    /// track that still holds `read`: its measured track when it brought one,
    /// and its state. A track that has left the cache since takes nothing.
    pub(super) fn install_viewed_evaluation(
        &mut self,
        read: &Arc<EditableTrack>,
        measured: Option<Arc<EditableTrack>>,
        evaluation: Evaluation,
    ) {
        let Some(viewed) = self
            .viewed_tracks
            .cache
            .iter_mut()
            .find(|viewed| Arc::ptr_eq(&viewed.track, read))
        else {
            return;
        };
        if let Some(track) = measured {
            viewed.track = track;
        }
        viewed.evaluation = evaluation;
    }
}

/// The bars in `bars` that differ from the defaults, named as Track View's
/// boxes name them, for the label of a put that carried them: `min ZNCC 80%`,
/// `max shift 4.0 px`. The three ZNCC bars read in percent, the projection
/// error bar in source-image px and the other two in patch-grid px, as the
/// boxes show them.
pub(super) fn bars_phrase(bars: &Thresholds) -> String {
    let defaults = Thresholds::default();
    let percent = |name: &str, value: f64| format!("{name} {:.0}%", 100.0 * value);
    let px = |name: &str, value: f64| format!("{name} {value:.1} px");
    let mut named = Vec::new();
    if bars.min_zncc != defaults.min_zncc {
        named.push(percent("min ZNCC", bars.min_zncc));
    }
    if bars.min_zncc_middle != defaults.min_zncc_middle {
        named.push(percent("min middle ZNCC", bars.min_zncc_middle));
    }
    if bars.max_shift_px != defaults.max_shift_px {
        named.push(px("max shift", bars.max_shift_px));
    }
    if bars.max_zncc_self_similarity_radius != defaults.max_zncc_self_similarity_radius {
        named.push(px(
            "max self-similarity",
            bars.max_zncc_self_similarity_radius,
        ));
    }
    if bars.max_projection_error_px != defaults.max_projection_error_px {
        named.push(px("max projection error", bars.max_projection_error_px));
    }
    if bars.geometry_search_min_relative_zncc != defaults.geometry_search_min_relative_zncc {
        named.push(percent(
            "geometry search min relative ZNCC",
            bars.geometry_search_min_relative_zncc,
        ));
    }
    named.join(", ")
}
