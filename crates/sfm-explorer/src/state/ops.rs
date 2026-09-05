// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`AppState`]'s reconstruction operations: the methods that read a file off
//! disk, run a fit or a resection, build a [`SceneNode`] out of the answer and
//! write the sentence the user reads about it.
//!
//! They are a second `impl AppState` rather than a second type, because every
//! one of them ends by editing the scene and the selection that `state.rs`'s
//! accessors describe — a resection that lands a derived node also selects it
//! and the image that moved, and a reload has to re-point the solo. Splitting
//! them into their own struct would mean handing that struct a
//! `&mut AppState` and gaining nothing but a hop.
//!
//! What the split buys is that `state.rs` reads as what it is: the struct, its
//! defaults, and the small selection and lookup accessors the panels call
//! every frame. Everything here is a *command* — asked for once,
//! synchronously, by a menu item, a dialog or an MCP tool — and every one of
//! them is the only place its own failure text is written.

use crate::action_log::Kind;
use crate::align::{self, AlignOptions};
use crate::resect::{self, ResectFrom};
use crate::scene::{ImageRef, ReconId, SceneNode};
use sfmtool_core::SfmrReconstruction;

use super::AppState;

impl AppState {
    /// Load a reconstruction from an .sfmr file, **appending** it as a node.
    ///
    /// Opening a path that is already loaded reloads that node in place instead
    /// — the predictable interpretation of "open this again", and it doubles as
    /// a refresh.
    ///
    /// A failure is **returned, not logged**: the File menu records it as
    /// `Failed to load …`, the MCP drain as `open_reconstruction failed: …`,
    /// and one failure that logged itself as well would appear twice, in two
    /// vocabularies. Success is logged here, because there the text is the same
    /// whoever asked.
    pub fn load_file(&mut self, path: &std::path::Path) -> Result<ReconId, String> {
        if let Some(id) = self
            .scene
            .iter()
            .find(|n| n.path.as_deref() == Some(path))
            .map(|n| n.id)
        {
            return self.reload_node(id);
        }
        match SfmrReconstruction::load(path) {
            Ok(recon) => {
                log::info!(
                    "Loaded {} points, {} images from {}",
                    recon.point_count(),
                    recon.image_count(),
                    path.display()
                );
                // Recorded after the append, which is what deduplicates the
                // label: the entry should name the node as the tree does
                // (`global (2)`), not the file stem the node arrived with.
                let id = self.append_node(SceneNode::from_path(path, recon));
                let label = self.label_of(id);
                self.action_log.record(
                    Kind::File,
                    format!("Opened {label} from {}", path.display()),
                );
                Ok(id)
            }
            Err(e) => {
                let msg = format!("Failed to load {}: {}", path.display(), e);
                log::error!("{}", msg);
                Err(msg)
            }
        }
    }
    /// Append a node of generated demo data.
    pub fn load_demo(&mut self, num_points: usize) {
        self.append_node(SceneNode::demo(SfmrReconstruction::demo(num_points)));
        self.action_log.record(Kind::File, "Loaded demo data");
    }
    /// Re-read a node's file from disk, keeping its place in tree order, its
    /// label and its display settings.
    ///
    /// The refreshed node gets a **new** [`ReconId`]. A reload can change every
    /// entity count, so every index-keyed cache entry for the old id is wrong;
    /// a new id makes all of them unreachable rather than merely stale, which
    /// is the same guarantee that makes closing a node safe. Returns the new id,
    /// or the message for a demo node (no file to re-read) or a failed read.
    ///
    /// Like [`AppState::load_file`], a failure is returned rather than logged:
    /// the caller is the one that knows whether it was asked for as a reload or
    /// as an `open_reconstruction` of a path that happened to be loaded.
    pub fn reload_node(&mut self, id: ReconId) -> Result<ReconId, String> {
        let index = self
            .scene
            .iter()
            .position(|n| n.id == id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let path = self.scene[index].path.clone().ok_or_else(|| {
            format!(
                "{} was generated, not loaded from a file, so there is nothing to re-read.",
                self.scene[index].label
            )
        })?;
        let recon = match SfmrReconstruction::load(&path) {
            Ok(recon) => recon,
            Err(e) => {
                let msg = format!("Failed to reload {}: {}", path.display(), e);
                log::error!("{}", msg);
                return Err(msg);
            }
        };
        let mut node = SceneNode::from_path(&path, recon);
        node.label = self.scene[index].label.clone();
        node.copy_display_from(&self.scene[index]);
        let new_id = node.id;
        let was_selected = self.selected_recon == Some(id);
        // A reload mints a fresh id, so the solo — which names an id rather
        // than a position — has to be re-pointed or refreshing the soloed node
        // would silently hide it along with everything else.
        let was_solo = self.solo == Some(id);
        self.scene[index] = node;
        self.forget_recon(id);
        if was_selected || self.selected_recon.is_none() {
            self.selected_recon = Some(new_id);
        }
        if was_solo {
            self.solo = Some(new_id);
        }
        let label = self.label_of(new_id);
        self.action_log
            .record(Kind::File, format!("Reloaded {label}"));
        Ok(new_id)
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
    /// The fit runs synchronously. By-cameras is trivially small; by-points is a
    /// bounded RANSAC over the correspondences (see [`crate::align`]).
    pub fn align_node(&mut self, source: ReconId, target: ReconId, options: AlignOptions) {
        if source == target {
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
            align::align_reconstructions(&self.scene[si].recon, &self.scene[ti].recon, options);
        match fit {
            Ok(fit) => {
                // `compose` applies the receiver first: the fit takes the source
                // into the target's own coordinates, then the target's transform
                // takes those into world space.
                self.scene[si].transform = fit.transform.compose(&self.scene[ti].transform);
                self.transform_epoch += 1;
                let message = align::success_message(&source_label, &target_label, &fit);
                self.action_log.record(Kind::Scene, message);
            }
            Err(reason) => {
                let message = align::failure_message(&source_label, &target_label, &reason);
                self.action_log.fail(Kind::Scene, message);
            }
        }
    }
    /// Re-estimate one image's pose against the rest of its reconstruction, and
    /// show the answer as a new node beside the source.
    ///
    /// The source node is never modified under any outcome. On success the
    /// derived node is named `<source> (resected <image>)`, inherits the
    /// source's current transform (so it lands exactly on top of it), becomes
    /// the selected reconstruction with the resected image selected in it, and
    /// carries the marker that says which image moved. A second resection of the
    /// same image from the same source **replaces** the earlier derived node,
    /// in place, rather than adding a third.
    ///
    /// A refused *estimate* still produces the node — with the stored pose
    /// retained, so the reviewer can see the held-out re-triangulation on its
    /// own — and reports the refusal. A resection that could not be attempted
    /// at all produces no node and only a status line.
    ///
    /// Runs synchronously; see `specs/gui/resect-image.md`, "Performance".
    ///
    /// The outcome is one Action Log entry, and the node arrival and selection
    /// change inside are muted: a resection is one action, and its result — not
    /// its mechanics — is what the log and the status line should carry.
    pub fn resect_image(&mut self, source: ReconId, image: usize, from: ResectFrom) {
        self.action_log.mute();
        let outcome = self.resect_image_inner(source, image, from);
        self.action_log.unmute();
        match outcome {
            Some(Ok(message)) => self.action_log.record(Kind::Scene, message),
            Some(Err(message)) => self.action_log.fail(Kind::Scene, message),
            None => {}
        }
    }
    /// The resection itself. `None` when it could not be attempted at all
    /// (no such node, no such image); otherwise the message the outer method
    /// records, as success or refusal.
    fn resect_image_inner(
        &mut self,
        source: ReconId,
        image: usize,
        from: ResectFrom,
    ) -> Option<Result<String, String>> {
        let index = self.scene.iter().position(|n| n.id == source)?;
        let label = self.scene[index].label.clone();
        let name = self.scene[index]
            .recon
            .images
            .get(image)
            .map(|i| i.name.clone())?;
        let basename = resect::basename(&name).to_string();
        if from == ResectFrom::Matches {
            if let Err(reason) = self.load_resect_matches(source) {
                return Some(Err(resect::failure_message(&basename, &label, &reason)));
            }
        }

        // Both borrows are shared, and the outcome owns its reconstruction — so
        // nothing here is still borrowed when the scene is written below.
        let outcome = {
            let matches = match from {
                ResectFrom::Observations => None,
                ResectFrom::Matches => self.resect_matches_cache.as_ref().map(|(_, data)| data),
            };
            let kind = match matches {
                Some(data) => resect::ResectSource::Matches(data),
                None => resect::ResectSource::StoredObservations,
            };
            // The panel's action is one image, which is the set primitive on a
            // one-element set.
            resect::resect_images(
                &self.scene[index].recon,
                &[image],
                kind,
                &resect::ResectImageOptions::default(),
            )
        };
        let mut resected = match outcome {
            Ok(resected) => resected,
            Err(error) => {
                return Some(Err(resect::failure_message(
                    &basename,
                    &label,
                    &error.to_string(),
                )));
            }
        };
        let report = resected.reports.pop().expect("one target, one report");
        // A refused *estimate* still produces the node, so the message is
        // decided here and carried out past the scene edit below.
        let message = match &report.refusal {
            Some(reason) => Err(resect::failure_message(&basename, &label, reason)),
            None => Ok(resect::success_message(&basename, &label, &report)),
        };

        let derived_label = format!("{label} (resected {basename})");
        let mut node = SceneNode::derived(derived_label.clone(), resected.reconstruction);
        // The derived node lands in the source's *displayed* frame, so it sits
        // exactly on top of it and the two can be compared with every existing
        // affordance.
        node.transform = self.scene[index].transform.clone();
        let new_id = node.id;
        // The derived node's name is its provenance: the same source and image
        // produce the same name, which is how a repeat finds the node it
        // replaces.
        match self
            .scene
            .iter()
            .position(|n| n.path.is_none() && n.label == derived_label)
        {
            // Replaced in place, keeping its position in tree order and its
            // label: this is the same question asked again, not a third answer.
            Some(slot) => {
                let old = self.scene[slot].id;
                node.label = self.scene[slot].label.clone();
                node.copy_display_from(&self.scene[slot]);
                // After the display copy, which brought the *old* derived
                // node's frame with it: the source may have been aligned since.
                node.transform = self.scene[index].transform.clone();
                self.scene[slot] = node;
                self.forget_recon(old);
                if self.solo == Some(old) {
                    self.solo = Some(new_id);
                }
                self.selected_recon = Some(new_id);
            }
            // A first resection arrives like any other node — through the one
            // arrival path, which owns label disambiguation and what a new node
            // does to the selection and the solo.
            None => {
                self.append_node(node);
            }
        }
        self.hovered_image = None;
        self.hovered_point = None;
        // The point of the action is to look at the image that moved, so the
        // point track detail opens on it immediately.
        self.select_image(Some(ImageRef::new(new_id, image)));
        Some(message)
    }
    /// Make sure [`AppState::resect_matches_cache`] holds the `.matches` file
    /// chosen for `source`, reading it if it does not. `Err` carries the reason
    /// for the status line.
    fn load_resect_matches(&mut self, source: ReconId) -> Result<(), String> {
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
        match matches_format::read_matches(&path) {
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
