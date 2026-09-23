// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`AppState`]'s reconstruction operations: the methods that make demo data
//! or run a fit, build a [`SceneNode`] out of the answer or move one, and write
//! the sentence the user reads about it. Opening a file is a background task of
//! its own, in [`super::open`].
//!
//! They are a second `impl AppState` rather than a second type, because every
//! one of them ends by editing the scene and the selection that `state.rs`'s
//! accessors describe: demo data that lands a node also selects it, and closing
//! one has to re-point the solo. Splitting them into their own struct would
//! mean handing that struct a `&mut AppState` and gaining nothing but a hop.
//!
//! What the split buys is that `state.rs` reads as what it is: the struct, its
//! defaults, and the small selection and lookup accessors the panels call
//! every frame. Everything here is a *command* — asked for once,
//! synchronously, by a menu item, a dialog or an MCP tool — and every one of
//! them is the only place its own failure text is written.

use crate::action_log::Kind;
use crate::align::{self, AlignOptions};
use crate::resect::{self, ResectFrom};
use crate::scene::{ReconId, SceneNode};
use sfmtool_core::SfmrReconstruction;

use super::AppState;

impl AppState {
    /// Append a node of generated demo data.
    pub fn load_demo(&mut self, num_points: usize) {
        self.append_node(SceneNode::demo(SfmrReconstruction::demo(num_points)));
        self.action_log.record(Kind::File, "Loaded demo data");
    }
    /// Fit `source`'s transform so it lands on top of `target`, and report the
    /// outcome in the status message.
    ///
    /// The fit maps the source's *native* coordinates onto the target's native
    /// coordinates; what the node stores is that composed into the target's
    /// **currently displayed** frame — `source.transform = target.transform ∘
    /// T_fit`, so aligning C→B after B→A chains as expected. The target node is
    /// never touched, and on any failure neither is the source: the transform is
    /// left exactly as it was and only the status line changes.
    ///
    /// The answer is a **reframe**, one version of the source whose value is
    /// untouched, so `Ctrl+Z` steps back out of it and the node does not go
    /// dirty. A source a background task holds is refused, since a version
    /// cannot be pushed onto it.
    ///
    /// The fit runs synchronously. By-cameras is trivially small; by-points is a
    /// bounded RANSAC over the correspondences (see [`crate::align`]).
    pub fn align_node(&mut self, source: ReconId, target: ReconId, options: AlignOptions) {
        if source == target {
            return;
        }
        if let Some(why) = self.busy_refusal(source) {
            self.action_log.fail(Kind::Scene, why);
            return;
        }
        let (Some(si), Some(ti)) = (
            self.scene.iter().position(|n| n.id == source),
            self.scene.iter().position(|n| n.id == target),
        ) else {
            return;
        };
        let (source_label, target_label) =
            (self.scene[si].label.clone(), self.scene[ti].label.clone());
        let fit =
            align::align_reconstructions(self.scene[si].recon(), self.scene[ti].recon(), options);
        match fit {
            Ok(fit) => {
                // `compose` applies the receiver first: the fit takes the source
                // into the target's own coordinates, then the target's transform
                // takes those into world space.
                let next = fit.transform.compose(self.scene[ti].transform());
                let message = align::success_message(&source_label, &target_label, &fit);
                if let Err(why) = self.push_reframe(source, next, message) {
                    self.action_log.fail(Kind::Scene, why);
                }
            }
            Err(reason) => {
                let message = align::failure_message(&source_label, &target_label, &reason);
                self.action_log.fail(Kind::Scene, message);
            }
        }
    }
    /// Run `run` over the correspondence source the menu entry `from` names.
    ///
    /// Here rather than beside the resection itself because this is where the
    /// cache it reads lives: the `.matches` file is put there first by
    /// [`AppState::load_resect_matches`], which the caller runs under its own
    /// failure text, and this only names it. The reconstruction stays the
    /// caller's, since what the resection reads is the version's whole value
    /// rather than the node's base.
    pub(super) fn with_resect_source<T>(
        &self,
        from: ResectFrom,
        run: impl FnOnce(resect::ResectSource<'_>) -> T,
    ) -> T {
        let matches = match from {
            ResectFrom::Observations => None,
            ResectFrom::Matches => self.resect_matches_cache.as_ref().map(|(_, data)| data),
        };
        let kind = match matches {
            Some(data) => resect::ResectSource::Matches(data),
            None => resect::ResectSource::StoredObservations,
        };
        run(kind)
    }
    /// Make sure [`AppState::resect_matches_cache`] holds the `.matches` file
    /// chosen for `source`, reading it if it does not. `Err` carries the reason
    /// for the status line.
    pub(super) fn load_resect_matches(&mut self, source: ReconId) -> Result<(), String> {
        let path = self
            .resect_matches
            .get(&source)
            .cloned()
            .ok_or_else(|| "no .matches file chosen".to_string())?;
        if self
            .resect_matches_cache
            .as_ref()
            .is_some_and(|(cached, _)| *cached == path)
        {
            return Ok(());
        }
        match sfmtool_matches_format::read_matches(&path) {
            Ok(data) => {
                self.resect_matches_cache = Some((path, data));
                Ok(())
            }
            Err(e) => {
                // A path that cannot be read is not a path worth remembering:
                // the next attempt should ask again rather than fail the same
                // way silently.
                self.resect_matches.remove(&source);
                Err(format!("could not read {}: {e}", path.display()))
            }
        }
    }
}
