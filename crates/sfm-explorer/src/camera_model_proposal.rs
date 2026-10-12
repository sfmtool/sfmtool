// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The camera model proposal: one camera of one node fitted to another camera
//! model, or a spline camera refitted to another coefficient count or domain
//! end, shown in the Camera Intrinsics panel before it is applied.
//!
//! See `specs/gui/edits/switch-camera-model.md`. The proposal is viewer state,
//! like the Move Camera lock's pending pose: nothing here is in the version
//! history until **Apply**, which pushes one version holding the switch and the
//! retriangulation of the camera's tracks
//! ([`AppState::apply_camera_model_proposal`]). Every change of a control
//! refits on the GUI thread with
//! [`sfmtool_core::reconstruction::switch_camera_model::fit_camera_switch`],
//! the fit the switch itself makes, so the camera the panel shows is the one
//! Apply installs for the same request.
//!
//! The endings are the lock's: Apply and Cancel; selecting another node, closing
//! the node or an MCP edit tool naming it apply the proposal when its fit
//! succeeded and drop it when the fit was refused; an undo, a redo or a history
//! jump on the node drops it.

use std::sync::Arc;

use sfmtool_core::camera::refit_intrinsics::{
    spline_domain_deg, CameraIntrinsicsRefit, RefitOptions, RefitTarget, ThetaFitSource,
};
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::reconstruction::outermost_keypoint::{outermost_keypoints, KeypointReach};
use sfmtool_core::reconstruction::switch_camera_model::fit_camera_switch;
use sfmtool_core::{EditedReconstruction, RowMap, SfmrReconstruction};

use crate::action_log::Kind;
use crate::document::VersionSerial;
use crate::scene::{CameraRef, ReconId};
use crate::state::edits::SwitchCameraModelRequest;
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// Why an edit of a node other than Apply is refused while a proposal is open
/// on it: the hover text of every greyed menu entry and the refusal of the
/// edits themselves.
pub(crate) const PROPOSAL_OPEN_HINT: &str = "Apply or cancel the camera model proposal first.";

/// Every model a camera can be switched to, in the order the Model list offers
/// them to a fisheye source. A perspective source sees `SFMTOOL_PINHOLE` first.
pub(crate) const TARGET_MODELS: [&str; 14] = [
    "SFMTOOL_FISHEYE",
    "SFMTOOL_PINHOLE",
    "EQUIDISTANT_FISHEYE",
    "OPENCV_FISHEYE",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
    "THIN_PRISM_FISHEYE",
    "RAD_TAN_THIN_PRISM_FISHEYE",
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "FULL_OPENCV",
];

/// How far out the photographs of one camera reach: its outermost keypoint, by
/// incidence angle under the camera's model, among the observations and among
/// the features detected in the images' `.sift` files.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct KeypointExtent {
    /// The outermost observation.
    pub(crate) observed: Option<KeypointReach>,
    /// The outermost detected feature, `None` when no `.sift` file is readable.
    pub(crate) detected: Option<KeypointReach>,
}

impl KeypointExtent {
    /// The camera `camera` of `edited`'s base value, over its observations and,
    /// when `read_sift_files`, its images' `.sift` files.
    pub(crate) fn of(edited: &EditedReconstruction, camera: usize, read_sift_files: bool) -> Self {
        outermost_keypoints(&edited.base, &[camera], read_sift_files)
            .into_iter()
            .next()
            .map(|o| KeypointExtent {
                observed: o.observed,
                detected: o.detected,
            })
            .unwrap_or_default()
    }

    /// The angle the domain row's button sets: the detected keypoint's, or the
    /// observed one's when nothing was detected.
    pub(crate) fn suggested_domain_deg(&self) -> Option<f64> {
        self.detected.or(self.observed).map(|r| r.theta_deg)
    }

    /// The sentence the domain row shows, e.g. `outermost keypoint: 229.7 px,
    /// 101.2° observed; 244.1 px, 107.9° detected`; `None` with no keypoint.
    pub(crate) fn describe(&self) -> Option<String> {
        let part = |r: &KeypointReach, source: &str| {
            format!("{:.1} px, {:.1}° {source}", r.radius_px, r.theta_deg)
        };
        let parts: Vec<String> = [
            self.observed.as_ref().map(|r| part(r, "observed")),
            self.detected.as_ref().map(|r| part(r, "detected")),
        ]
        .into_iter()
        .flatten()
        .collect();
        (!parts.is_empty()).then(|| format!("outermost keypoint: {}", parts.join("; ")))
    }
}

/// What a proposal is fitted from: the target and the fit's choices, each
/// `None` where the default holds.
///
/// Two inputs are equal when they ask the core for the same fit, so a change
/// that leaves them as they were keeps the proposal's generation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProposalInputs {
    /// The target model's registered name.
    pub(crate) camera_model: &'static str,
    /// The spline coefficient count of a spline target, `None` for any other.
    pub(crate) coeff_count: Option<usize>,
    /// The fit's largest angle, `None` for the core's default.
    pub(crate) theta_fit_deg: Option<f64>,
    /// Where a spline target's domain ends, `None` for the default: the
    /// camera's own domain end for its own spline model, the far image corner
    /// otherwise.
    pub(crate) spline_domain_deg: Option<f64>,
}

/// One open proposal: which camera, what it is fitted from, and the fit.
pub(crate) struct CameraModelProposal {
    /// The node the camera belongs to.
    pub(crate) recon: ReconId,
    /// The camera's index in the node's camera table.
    pub(crate) camera: usize,
    /// The node's label, for the sentences that name it.
    pub(crate) label: String,
    /// The document half of the version the proposal was fitted to. Once the
    /// node's value is another, the proposal describes a camera the node no
    /// longer shows. A bench step or a reframe shares the document half, so
    /// it leaves the proposal open.
    pub(crate) opened_on: VersionSerial,
    /// The camera as it was when the proposal opened.
    pub(crate) source: CameraIntrinsics,
    /// The value the fits read: the version's base, or its materialisation
    /// when it carries point edits, made once.
    value: Arc<SfmrReconstruction>,
    /// The materialisation's row map, so a refusal naming a point names it by
    /// the index the version on screen gives it; `None` for the base itself.
    rows: Option<RowMap>,
    /// The models the Model list offers, in order.
    pub(crate) targets: Vec<&'static str>,
    /// How far out the camera's images reach, read once when it opened.
    pub(crate) keypoint_extent: KeypointExtent,
    /// What the current fit was made from.
    pub(crate) inputs: ProposalInputs,
    /// A counter every change of `inputs` advances.
    pub(crate) generation: u64,
    /// The fit, or the core's refusal sentence.
    pub(crate) fit: Result<CameraIntrinsicsRefit, String>,
    /// The fit angle the core takes with none given, and where it came from,
    /// as the last fit that was given none reported it: what Fit to θ shows
    /// and how far it can be raised.
    pub(crate) theta_fit_default: Option<(f64, ThetaFitSource)>,
    /// Where a change of model's spline domain ends with none given, as the
    /// last such fit reported it: what Spline domain shows for an untouched
    /// field.
    pub(crate) spline_domain_default: Option<f64>,
}

impl CameraModelProposal {
    /// Whether `camera_model` is this camera's own spline model, which makes
    /// the proposal a refit of its spline.
    pub(crate) fn is_spline_refit(&self, camera_model: &str) -> bool {
        self.source.model.radial_spline().is_some()
            && self.source.model_name().eq_ignore_ascii_case(camera_model)
    }

    /// Whether `camera_model` takes a spline coefficient count.
    pub(crate) fn has_spline(camera_model: &str) -> bool {
        matches!(camera_model, "SFMTOOL_FISHEYE" | "SFMTOOL_PINHOLE")
    }

    /// The source's spline domain end in degrees, for a spline source.
    pub(crate) fn source_domain_deg(&self) -> Option<f64> {
        spline_domain_deg(&self.source)
    }

    /// The source's spline coefficient count, for a spline source.
    pub(crate) fn source_coeff_count(&self) -> Option<usize> {
        self.source.model.radial_spline().map(|(b, _, _)| b.len())
    }

    /// The inputs the panel starts a target at: the count
    /// `RefitTarget::coeff_count_for` gives (the camera's own for its own
    /// spline model, 8 otherwise), the default fit angle and the default
    /// domain.
    pub(crate) fn default_inputs(&self, camera_model: &'static str) -> ProposalInputs {
        let coeff_count = RefitTarget::from_name(camera_model, None)
            .ok()
            .and_then(|target| target.coeff_count_for(&self.source));
        ProposalInputs {
            camera_model,
            coeff_count,
            theta_fit_deg: None,
            spline_domain_deg: None,
        }
    }

    /// `inputs` as the core is asked for them: a domain equal to the camera's
    /// own in a refit of its spline is the default, so an untouched field keeps
    /// the domain bit for bit.
    fn normalised(&self, mut inputs: ProposalInputs) -> ProposalInputs {
        if self.is_spline_refit(inputs.camera_model) {
            if let (Some(given), Some(own)) = (inputs.spline_domain_deg, self.source_domain_deg()) {
                if (given - own).abs() <= 1e-9 {
                    inputs.spline_domain_deg = None;
                }
            }
        }
        if !Self::has_spline(inputs.camera_model) {
            inputs.spline_domain_deg = None;
        }
        inputs
    }

    /// Change what the proposal is fitted from, refitting when the inputs
    /// changed. Returns whether they did, which is when the generation
    /// advances.
    pub(crate) fn set_inputs(&mut self, inputs: ProposalInputs) -> bool {
        let inputs = self.normalised(inputs);
        if inputs == self.inputs {
            return false;
        }
        self.inputs = inputs;
        self.generation += 1;
        self.fit = self.compute_fit();
        if let Ok(refit) = &self.fit {
            if self.inputs.theta_fit_deg.is_none()
                && refit.theta_fit_source != ThetaFitSource::SplineDomain
            {
                self.theta_fit_default = Some((refit.theta_fit_deg, refit.theta_fit_source));
            }
            if self.inputs.spline_domain_deg.is_none()
                && !self.is_spline_refit(self.inputs.camera_model)
            {
                self.spline_domain_default = refit.spline_domain_deg.or(self.spline_domain_default);
            }
        }
        true
    }

    /// What the Spline domain field shows: the domain given, or the default
    /// (the camera's own in a refit of its spline, the far image corner the
    /// fit placed otherwise).
    pub(crate) fn shown_domain_deg(&self) -> Option<f64> {
        self.inputs.spline_domain_deg.or_else(|| {
            if self.is_spline_refit(self.inputs.camera_model) {
                self.source_domain_deg()
            } else {
                self.spline_domain_default
            }
        })
    }

    /// What the Fit to θ field shows: the angle given, or the default.
    pub(crate) fn shown_theta_fit_deg(&self) -> Option<f64> {
        self.inputs
            .theta_fit_deg
            .or(self.theta_fit_default.map(|(deg, _)| deg))
    }

    /// The fit the current inputs ask for, as the switch would make it.
    fn compute_fit(&self) -> Result<CameraIntrinsicsRefit, String> {
        let target = RefitTarget::from_name(self.inputs.camera_model, self.inputs.coeff_count)
            .map_err(|e| format!("camera {}: {e}", self.camera))?;
        let options = RefitOptions {
            theta_fit_deg: self.inputs.theta_fit_deg,
            spline_domain_deg: self.inputs.spline_domain_deg,
        };
        fit_camera_switch(&self.value, self.camera, &target, &options).map_err(|e| {
            crate::state::edits::switch_refusal_in_version_numbering(e, self.rows.as_ref())
                .to_string()
        })
    }

    /// The request Apply hands the switch.
    ///
    /// A refit of a spline asks as the camera's own model with no model named,
    /// no fit angle unless one was given, the count, and a domain only when it
    /// moved from the camera's own.
    pub(crate) fn request(&self) -> SwitchCameraModelRequest {
        let refit = self.is_spline_refit(self.inputs.camera_model);
        SwitchCameraModelRequest {
            camera: self.camera,
            camera_model: (!refit).then(|| self.inputs.camera_model.to_string()),
            coeff_count: self.inputs.coeff_count,
            spline_domain_deg: self.inputs.spline_domain_deg,
            theta_fit_deg: self.inputs.theta_fit_deg,
        }
    }

    /// The label the Model list gives `camera_model`: the camera's own spline
    /// model reads as the refit.
    pub(crate) fn target_label(&self, camera_model: &str) -> String {
        if self.is_spline_refit(camera_model) {
            format!("{camera_model} (refit spline)")
        } else {
            camera_model.to_string()
        }
    }

    /// The panel's identity of this proposal's camera.
    pub(crate) fn camera_ref(&self) -> CameraRef {
        CameraRef::new(self.recon, self.camera)
    }

    /// The one line under the strip: how far the proposed camera is from the
    /// current one, over what, what was dropped, and where the monotonicity
    /// constraint bound; or the fit's refusal.
    pub(crate) fn fit_line(&self) -> String {
        match &self.fit {
            Ok(refit) => fit_report_line(refit),
            Err(why) => format!("Refused: {why}"),
        }
    }
}

/// `rms 0.130 px, radial 0.013 px, max 0.292 px over θ ≤ 84.5°; aspect fy/fx
/// 0.9978 dropped (single focal); monotone constraint bound at 1 angle,
/// 113.2°`.
pub(crate) fn fit_report_line(refit: &CameraIntrinsicsRefit) -> String {
    let mut line = format!(
        "rms {:.3} px, radial {:.3} px, max {:.3} px over θ ≤ {:.1}°",
        refit.rms_px, refit.radial_rms_px, refit.max_px, refit.theta_fit_deg
    );
    for dropped in &refit.dropped {
        line.push_str(&format!("; {dropped}"));
    }
    line.push_str(&crate::state::edits::monotone_clause(
        &refit.monotone_constraint,
    ));
    line
}

/// The Model list for `source`: every target, the source's own model left out
/// unless it carries a spline, with `SFMTOOL_FISHEYE` first for a fisheye
/// source and `SFMTOOL_PINHOLE` first otherwise.
pub(crate) fn targets_for(source: &CameraIntrinsics) -> Vec<&'static str> {
    let own = source.model_name();
    let keeps_own = source.model.radial_spline().is_some();
    let mut targets: Vec<&'static str> = TARGET_MODELS
        .iter()
        .copied()
        .filter(|name| keeps_own || !name.eq_ignore_ascii_case(own))
        .collect();
    if !source.model.is_fisheye() {
        if let Some(at) = targets.iter().position(|&n| n == "SFMTOOL_PINHOLE") {
            let pinhole = targets.remove(at);
            targets.insert(0, pinhole);
        }
    }
    targets
}

/// Why no proposal can be opened on `camera` of `node` right now, or `None`.
///
/// The three greyed cases every entry carries: the node busy with a background
/// task (`busy`, the sentence the busy node's other edits carry), a proposal
/// already open on another camera of the node, and a node with no posed
/// image; and a camera the table does not have.
pub(crate) fn open_refusal(
    node: &crate::scene::SceneNode,
    camera: usize,
    busy: Option<&str>,
    open: Option<CameraRef>,
) -> Option<String> {
    if let Some(why) = busy {
        return Some(why.to_string());
    }
    if let Some(other) = open.filter(|o| o.recon == node.id && o.index() != camera) {
        return Some(format!(
            "A camera model proposal is open on camera {} of {}. Apply or cancel it first.",
            other.index(),
            node.label
        ));
    }
    let cameras = node.recon().image_table.cameras.len();
    if camera >= cameras {
        return Some(format!(
            "Camera {camera} does not exist; the reconstruction has {cameras} camera(s)."
        ));
    }
    let posed = node.recon().image_table.images.iter().any(|image| {
        image.quaternion_wxyz.coords.iter().all(|c| c.is_finite())
            && image.translation_xyz.iter().all(|c| c.is_finite())
    });
    (!posed).then(|| "No image of this reconstruction carries a pose.".to_string())
}

/// How a proposal ended without Apply, for its Action Log row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dropped {
    /// Cancel, `Escape`, or `cancel_camera_model_proposal`.
    Cancelled,
    /// An undo, a redo or a history jump on the node.
    HistoryMoved,
    /// An implicit ending found the fit refused, so there was nothing to apply.
    FitRefused,
    /// An implicit ending's Apply was refused, its refusal recorded beside.
    ApplyRefused,
    /// The node's version changed under the proposal some other way.
    VersionChanged,
}

impl AppState {
    /// The proposal open on `id`, if any.
    pub(crate) fn camera_model_proposal_on(&self, id: ReconId) -> Option<&CameraModelProposal> {
        self.camera_model_proposal
            .as_ref()
            .filter(|proposal| proposal.recon == id)
    }

    /// Why an edit of `id` other than the proposal's own Apply is refused
    /// while a proposal is open on it, or `None`.
    pub(crate) fn proposal_refusal(&self, id: ReconId) -> Option<String> {
        self.camera_model_proposal_on(id)
            .map(|_| PROPOSAL_OPEN_HINT.to_string())
    }

    /// Why a menu edit of `id` is greyed right now: the node busy, or a
    /// proposal open on it.
    pub(crate) fn edit_refusal(&self, id: ReconId) -> Option<String> {
        self.busy_refusal(id).or_else(|| self.proposal_refusal(id))
    }

    /// Why a proposal cannot be opened on `camera` right now, or `None`: the
    /// greyed state of `Switch Model…`, `Switch Camera Model…` in the Scene
    /// tree and in the Edit menu.
    pub(crate) fn switch_model_refusal(&self, camera: CameraRef) -> Option<String> {
        let Some(node) = self.node(camera.recon) else {
            return Some(crate::state::NOT_LOADED.to_string());
        };
        let busy = self.busy_refusal(camera.recon);
        let open = self.camera_model_proposal.as_ref().map(|p| p.camera_ref());
        open_refusal(node, camera.index(), busy.as_deref(), open)
    }

    /// Open a proposal on `camera`, fitted from `inputs`, or from the panel's
    /// starting inputs when `inputs` is `None`: the camera's own model for a
    /// spline camera, and the Model list's first target otherwise.
    ///
    /// A proposal already open on the same camera is kept, with its generation,
    /// when `inputs` is `None` or asks for the same fit, and refitted
    /// otherwise. One open on another node is ended first by the implicit
    /// rule, since opening this one selects this camera, which is a step away
    /// from that node. The keypoints are read once, here, on the GUI thread.
    /// Writes a `View` row naming the proposal; a refusal is returned for the
    /// caller to record.
    pub(crate) fn open_camera_model_proposal(
        &mut self,
        camera: CameraRef,
        inputs: Option<ProposalInputs>,
    ) -> Result<(), String> {
        if let Some(why) = self.switch_model_refusal(camera) {
            return Err(why);
        }
        if let Some(open) = self.camera_model_proposal.as_mut() {
            if open.camera_ref() == camera {
                if let Some(inputs) = inputs {
                    if open.set_inputs(inputs) {
                        let text = opened_text(open);
                        self.action_log.record(Kind::View, text);
                    }
                }
                self.select_camera(Some(camera));
                return Ok(());
            }
        }
        if self
            .camera_model_proposal
            .as_ref()
            .is_some_and(|open| open.recon != camera.recon)
        {
            self.end_camera_model_proposal_implicitly();
        }
        let node = self
            .node(camera.recon)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let edited = node.history.current();
        let (value, rows) = if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
            (Arc::clone(&edited.base), None)
        } else {
            let (value, map) = edited.materialize();
            (Arc::new(value), Some(map))
        };
        let source = node.recon().image_table.cameras[camera.index()].clone();
        let targets = targets_for(&source);
        let mut proposal = CameraModelProposal {
            recon: camera.recon,
            camera: camera.index(),
            label: node.label.clone(),
            opened_on: node.history.current_version().document_serial,
            source,
            value,
            rows,
            keypoint_extent: KeypointExtent::of(edited, camera.index(), true),
            targets,
            inputs: ProposalInputs {
                camera_model: "",
                coeff_count: None,
                theta_fit_deg: None,
                spline_domain_deg: None,
            },
            generation: 0,
            fit: Err(String::new()),
            theta_fit_default: None,
            spline_domain_default: None,
        };
        let inputs = inputs.unwrap_or_else(|| {
            let start = if proposal.source.model.radial_spline().is_some() {
                proposal
                    .targets
                    .iter()
                    .copied()
                    .find(|t| proposal.is_spline_refit(t))
                    .unwrap_or(proposal.targets[0])
            } else {
                proposal.targets[0]
            };
            proposal.default_inputs(start)
        });
        // A model the wire named that the list leaves out (a camera's own
        // non-spline model) is shown as named.
        if !proposal.targets.contains(&inputs.camera_model) {
            proposal.targets.insert(0, inputs.camera_model);
        }
        proposal.set_inputs(inputs);
        let text = opened_text(&proposal);
        self.camera_model_proposal = Some(proposal);
        self.select_camera(Some(camera));
        self.action_log.record(Kind::View, text);
        Ok(())
    }

    /// Drop the proposal on `id` without applying it, writing one `View` row
    /// that says why. Returns whether there was one.
    pub(crate) fn drop_camera_model_proposal(&mut self, id: ReconId, why: Dropped) -> bool {
        let Some(proposal) = self
            .camera_model_proposal
            .take_if(|proposal| proposal.recon == id)
        else {
            return false;
        };
        let what = format!(
            "camera model proposal for camera {} of {}",
            proposal.camera, proposal.label
        );
        let text = match why {
            Dropped::Cancelled => format!("Cancelled the {what}"),
            Dropped::HistoryMoved => {
                format!("Dropped the {what}: the history moved off the version it was fitted to")
            }
            Dropped::FitRefused => format!("Dropped the {what}: its fit was refused"),
            Dropped::ApplyRefused => format!("Dropped the {what}: its Apply was refused"),
            Dropped::VersionChanged => {
                format!("Dropped the {what}: the reconstruction changed under it")
            }
        };
        self.action_log.record(Kind::View, text);
        true
    }

    /// End the open proposal the way the Move Camera lock ends implicitly:
    /// apply it when its fit succeeded, since an edit has a history and undo
    /// reverses it, and drop it when the fit was refused.
    ///
    /// Returns the node a switch pushed a version on in this call, on the GUI
    /// thread, so the caller can drop the panel caches describing it; an Apply
    /// that started the background task returns `None`, and the frame that
    /// installs its version drops them.
    pub(crate) fn end_camera_model_proposal_implicitly(&mut self) -> Option<ReconId> {
        let proposal = self.camera_model_proposal.as_ref()?;
        let id = proposal.recon;
        if self.node(id).is_none() {
            self.camera_model_proposal = None;
            return None;
        }
        if proposal.fit.is_err() {
            self.drop_camera_model_proposal(id, Dropped::FitRefused);
            return None;
        }
        match self.apply_camera_model_proposal(id) {
            Ok(applied) => {
                // Kept for the frame as well as returned, for the callers that
                // have no panels in reach.
                self.camera_model_switched = applied.pushed();
                applied.pushed()
            }
            // A refused Apply leaves the proposal open and its refusal in the
            // log; an implicit ending drops it then, with its own row, as it
            // drops a proposal whose fit was refused.
            Err(_) => {
                self.drop_camera_model_proposal(id, Dropped::ApplyRefused);
                None
            }
        }
    }

    /// End the open proposal by the implicit rule when it is on `id`.
    pub(crate) fn end_camera_model_proposal_on(&mut self, id: ReconId) -> Option<ReconId> {
        self.camera_model_proposal_on(id)?;
        self.end_camera_model_proposal_implicitly()
    }

    /// Once per frame: drop a proposal whose node has gone or whose value is
    /// no longer the one at the cursor, and end one by the implicit rule when the
    /// selection has moved to another node.
    ///
    /// Returns the node a switch pushed a version on, as
    /// [`Self::end_camera_model_proposal_implicitly`] does, or the one an
    /// implicit ending since the last frame pushed a version on.
    pub(crate) fn tend_camera_model_proposal(&mut self) -> Option<ReconId> {
        if let Some(id) = self.camera_model_switched.take() {
            return Some(id);
        }
        let proposal = self.camera_model_proposal.as_ref()?;
        let id = proposal.recon;
        let Some(node) = self.node(id) else {
            self.camera_model_proposal = None;
            return None;
        };
        if node.history.current_version().document_serial != proposal.opened_on {
            self.drop_camera_model_proposal(id, Dropped::VersionChanged);
            return None;
        }
        match self.selected_recon {
            Some(selected) if selected != id => self.end_camera_model_proposal_implicitly(),
            _ => None,
        }
    }
}

/// Open a proposal on `camera` from one of the window's three entries:
/// `Switch Model…` in the Camera Intrinsics panel, `Switch Camera Model…` on a
/// Scene tree camera row, and `Edit > Switch Camera Model…`.
///
/// A Move Camera lock held on the node is ended first by its own implicit
/// rule, as every editing command ends it. `raise` brings the Camera
/// Intrinsics panel to the front, which the two entries outside it want; it
/// must be called with the dock back in the state. A refusal is recorded in
/// the Action Log. Returns the node a committed lock pushed a version on, for
/// the caller to drop the panel caches describing it.
pub(crate) fn open_in_window(
    viewer: &mut crate::viewer_3d::Viewer3D,
    state: &mut AppState,
    camera: CameraRef,
    raise: bool,
) -> Option<ReconId> {
    let moved = crate::camera_lock::exit_implicitly_for(viewer, state, camera.recon);
    match state.open_camera_model_proposal(camera, None) {
        Ok(()) => {
            if raise {
                state.show_panel(crate::dock::Tab::IntrinsicsDetail);
            }
        }
        Err(why) => state.action_log.fail(
            Kind::Edit,
            format!(
                "Cannot propose a camera model for camera {}: {why}",
                camera.index()
            ),
        ),
    }
    moved
}

/// The `View` row an opened or refitted proposal writes.
fn opened_text(proposal: &CameraModelProposal) -> String {
    let target = proposal.target_label(proposal.inputs.camera_model);
    let fit = match &proposal.fit {
        Ok(refit) => fit_report_line(refit),
        Err(why) => format!("refused: {why}"),
    };
    format!(
        "Proposed {target} for camera {} of {}: {fit}",
        proposal.camera, proposal.label
    )
}
