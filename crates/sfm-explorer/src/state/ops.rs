// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`AppState`]'s reconstruction operations: the methods that run a fit, move
//! a node by the answer, and write the sentence the user reads about it.
//! Opening a file is a background task of its own, in [`super::open`], and so
//! is loading demo data, in [`crate::demo_data`].
//!
//! They are a second `impl AppState` rather than a second type, because every
//! one of them ends by editing the scene that `state.rs`'s accessors describe.
//! Splitting them into their own struct would mean handing that struct a
//! `&mut AppState` and gaining nothing but a hop.
//!
//! What the split buys is that `state.rs` reads as what it is: the struct, its
//! defaults, and the small selection and lookup accessors the panels call
//! every frame. Everything here is a *command* — asked for once,
//! synchronously, by a menu item, a dialog or an MCP tool — and every one of
//! them is the only place its own failure text is written.

use crate::action_log::Kind;
use crate::align::{self, AlignOptions};
use crate::scene::ReconId;
use sfmtool_core::progress::Progress;

use super::AppState;

impl AppState {
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
    pub(crate) fn align_node(&mut self, source: ReconId, target: ReconId, options: AlignOptions) {
        if source == target {
            return;
        }
        if let Some(why) = self.edit_refusal(source) {
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
        // Synchronous, so there is nothing to report progress to or cancel
        // from; pairing the images reads a `sift_files` node's `.sift`
        // metadata on this thread.
        let fit = align::align_reconstructions(
            self.scene[si].recon(),
            self.scene[ti].recon(),
            options,
            &Progress::none(),
        );
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
}
