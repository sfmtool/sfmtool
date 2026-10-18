// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Switch one camera of a node to a camera model fitted to it, as the node's
//! next version.
//!
//! See `specs/gui/edits/switch-camera-model.md`. The mechanism is
//! `sfmtool_core::reconstruction::switch_camera_model` over one camera. A spline
//! camera switched to its own spline model is a refit of its spline to a new
//! coefficient count or domain end. The edit has two forms:
//!
//! - **The direct switch** ([`AppState::switch_camera_model`]), lens only, on
//!   the GUI thread: the MCP tool `switch_camera_model` in its direct form.
//! - **The proposal's Apply** ([`AppState::apply_camera_model_proposal`]): the
//!   switch followed by the retriangulation of the camera's tracks through
//!   `RetriangulateWhich::TheseInBulk`, in one version, run as the `Switch
//!   camera model` background task. On a node whose observations carry no
//!   pixel it is the direct switch, with the reason the tracks were not
//!   retriangulated added to its row.

use std::sync::Arc;
use std::time::Instant;

use sfmtool_core::camera::refit_intrinsics::{
    spline_domain_deg, MonotoneConstraint, RefitError, RefitOptions, RefitTarget,
};
use sfmtool_core::reconstruction::switch_camera_model::{
    switch_camera_model, CameraSwitch, SwitchCameraModelError, SwitchCameraModelOptions,
};
use sfmtool_core::reconstruction::{
    retriangulate_points, RetriangulateError, RetriangulateOptions, RetriangulateWhich,
};
use sfmtool_core::{EditedReconstruction, RowMap, SfmrReconstruction};

use crate::action_log::{version_step_text, Kind};
use crate::background::{Finished, Job, Operation, SwitchedCamera};
use crate::document::PointMap;
use crate::progress::Collector;
use crate::scene::ReconId;
use crate::state::AppState;
use sfmtool_core::progress::Progress;
use sfmtool_core::reconstruction::RetriangulateReport;

use super::{retriangulate_refusal, retriangulate_summary, version_before};

/// What a switch of one camera asks for. Every field but the camera has a
/// default, resolved against the camera as it is when the switch runs.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct SwitchCameraModelRequest {
    /// The camera's index in the node's camera table.
    pub(crate) camera: usize,
    /// The target model's name, or `None` for the camera's own model.
    pub(crate) camera_model: Option<String>,
    /// The spline coefficient count, or `None` for the camera's own count when
    /// the target is its own spline model, and the core default otherwise.
    pub(crate) coeff_count: Option<usize>,
    /// Where a spline target's domain ends, in degrees, or `None` for the
    /// camera's own domain end in a refit of its spline, and the far image
    /// corner otherwise.
    pub(crate) spline_domain_deg: Option<f64>,
    /// The largest incidence angle the fit samples, or `None` for the core's
    /// default. Given, it makes even a refit of a spline an ordinary fit over
    /// that angle.
    pub(crate) theta_fit_deg: Option<f64>,
}

/// The monotonicity constraint's clause of a log row: empty when it did not
/// bind, and otherwise `; monotone constraint bound at 3 angles, 95.2°–118.7°`.
pub(crate) fn monotone_clause(constraint: &MonotoneConstraint) -> String {
    match constraint.range_deg {
        Some([from, to]) if constraint.active => {
            let count = constraint.active_angles;
            let plural = if count == 1 { "" } else { "s" };
            let (from, to) = (format!("{from:.1}"), format!("{to:.1}"));
            let range = if from == to {
                format!("{from}°")
            } else {
                format!("{from}°–{to}°")
            };
            format!("; monotone constraint bound at {count} angle{plural}, {range}")
        }
        _ => String::new(),
    }
}

/// Whether `target` is `camera`'s own spline model, which makes the switch a
/// refit of its spline rather than a change of model.
fn is_spline_refit(camera: &sfmtool_core::CameraIntrinsics, target: &str) -> bool {
    camera.model.radial_spline().is_some() && camera.model_name().eq_ignore_ascii_case(target)
}

/// `error` with any point it names given the index the version on screen
/// calls that point by.
///
/// The switch runs on the materialised value when the version carries an
/// overlay, and materialising renumbers the points past a deleted one, so a
/// refusal naming point 5121 of the materialised value would send the person
/// to a different point. `map` is the materialisation's, `None` when the switch
/// ran on the version's own base.
pub(crate) fn switch_refusal_in_version_numbering(
    mut error: SwitchCameraModelError,
    map: Option<&RowMap>,
) -> SwitchCameraModelError {
    if let (
        Some(map),
        SwitchCameraModelError::Refit {
            error: RefitError::ObservationsPast90 { point, .. },
            ..
        },
    ) = (map, &mut error)
    {
        if let Some(edited) = map.inverse(*point) {
            *point = edited;
        }
    }
    error
}

/// The version label for one switch: what changed about the camera, in one
/// line.
///
/// A refit of a spline reads `Refit spline of camera 0 of kerry: 8 → 10
/// coefficients, domain 150.0° → 108.0°`, naming each of the two only as
/// "kept" where it did not move; a change of model reads `Switched camera 0 of
/// kerry from OPENCV_FISHEYE to SFMTOOL_FISHEYE`.
pub(crate) fn switch_label(label: &str, entry: &CameraSwitch) -> String {
    let (source, target, c) = (&entry.source, &entry.refit.camera, entry.camera);
    match refit_change(source, target) {
        Some(change) => format!("Refit spline of camera {c} of {label}: {change}"),
        None => format!(
            "Switched camera {c} of {label} from {} to {}",
            source.model_name(),
            target.model_name()
        ),
    }
}

/// What a refit of a spline changes, `8 → 10 coefficients, domain 150.0° →
/// 108.0°`, naming each of the two only as "kept" where it did not move; `None`
/// where `target` is not a refit of `source`'s spline but a change of model.
pub(crate) fn refit_change(
    source: &sfmtool_core::CameraIntrinsics,
    target: &sfmtool_core::CameraIntrinsics,
) -> Option<String> {
    if source.model_name() != target.model_name() || source.model.radial_spline().is_none() {
        return None;
    }
    let count = |camera: &sfmtool_core::CameraIntrinsics| {
        camera.model.radial_spline().map_or(0, |(b, _, _)| b.len())
    };
    let (n0, n1) = (count(source), count(target));
    let coeffs = if n0 == n1 {
        format!("{n1} coefficients kept")
    } else {
        format!("{n0} → {n1} coefficients")
    };
    let (d0, d1) = (
        spline_domain_deg(source).unwrap_or(f64::NAN),
        spline_domain_deg(target).unwrap_or(f64::NAN),
    );
    let domain = if (d0 - d1).abs() <= 1e-9 {
        format!("domain {d1:.1}° kept")
    } else {
        format!("domain {d0:.1}° → {d1:.1}°")
    };
    Some(format!("{coeffs}, {domain}"))
}

/// The report half of the Action Log row: how far the new camera is from the
/// old over the fit, where the monotonicity constraint bound, and the median
/// reprojection error of the camera's observations before and after.
pub(crate) fn switch_report_text(entry: &CameraSwitch) -> String {
    let refit = &entry.refit;
    let o = &entry.observations;
    let median = if o.observations > 0 {
        format!(
            "; median error {:.3} → {:.3} px over {} observations",
            o.before.median_px, o.after.median_px, o.observations
        )
    } else {
        String::new()
    };
    format!(
        "fit rms {:.3} px, max {:.3} px over θ ≤ {:.1}° ({}){}{median}",
        refit.rms_px,
        refit.max_px,
        refit.theta_fit_deg,
        refit.theta_fit_source.as_str(),
        monotone_clause(&refit.monotone_constraint),
    )
}

impl AppState {
    /// Switch one camera of `id` to the model `request` names, fitted to it,
    /// and install the answer as the node's next version.
    ///
    /// A bulk edit: the camera table changes and the stored error of every
    /// point the camera's images observe is recomputed, so the next version is
    /// a whole new base; no pose, point, keypoint or track moves, and the row
    /// map read off the two values is the identity. Records its own outcome,
    /// success or refusal, as one Action Log entry, and returns the camera's
    /// entry of the core report for a caller that shows the numbers.
    #[cfg(any(test, feature = "mcp"))]
    pub(crate) fn switch_camera_model(
        &mut self,
        id: ReconId,
        request: &SwitchCameraModelRequest,
    ) -> Result<CameraSwitch, String> {
        self.switch_camera_model_noting(id, request, None)
    }

    /// [`Self::switch_camera_model`], with `note` added to the Action Log row
    /// after the fit's report: the proposal's Apply on a node it cannot
    /// retriangulate says there why the tracks were not moved.
    fn switch_camera_model_noting(
        &mut self,
        id: ReconId,
        request: &SwitchCameraModelRequest,
        note: Option<&str>,
    ) -> Result<CameraSwitch, String> {
        let started = Instant::now();
        let collector = Collector::new(self.action_log.detailed_timing());
        let outcome = self.switch_camera_model_inner(id, request, note, &collector);
        match outcome {
            Ok((message, entry)) => {
                self.action_log
                    .record_done(Kind::Edit, started, message, collector.take());
                Ok(entry)
            }
            Err(message) => {
                self.action_log.fail(Kind::Edit, message.clone());
                Err(message)
            }
        }
    }

    /// What a switch of `id` asks the core for, resolved against the camera as
    /// it is now: the target with its count, and the fit's options. `Err` is
    /// the refusal's whole sentence.
    pub(crate) fn resolve_switch(
        &self,
        id: ReconId,
        request: &SwitchCameraModelRequest,
    ) -> Result<ResolvedSwitch, String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let node = self
            .node(id)
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let label = node.label.clone();
        let c = request.camera;
        let refuse = |why: String| format!("Switch camera model of {label} refused: {why}");
        let cameras = &node.recon().image_table.cameras;
        let Some(camera) = cameras.get(c) else {
            return Err(refuse(format!(
                "camera {c} does not exist; the reconstruction has {} camera(s)",
                cameras.len()
            )));
        };
        let camera_model = request
            .camera_model
            .clone()
            .unwrap_or_else(|| camera.model_name().to_string());
        let refit = is_spline_refit(camera, &camera_model);
        let coeff_count = match (request.coeff_count, refit) {
            (Some(n), _) => Some(n),
            (None, true) => camera.model.radial_spline().map(|(b, _, _)| b.len()),
            (None, false) => None,
        };
        let target = RefitTarget::from_name(&camera_model, coeff_count)
            .map_err(|e| refuse(format!("camera {c}: {e}")))?;
        let options = SwitchCameraModelOptions {
            refit: RefitOptions {
                theta_fit_deg: request.theta_fit_deg,
                spline_domain_deg: request.spline_domain_deg,
            },
            // The edit shows no outermost keypoint (the proposal reads its
            // own when it opens), so it reads no `.sift` file for one.
            outermost_keypoint: false,
        };
        Ok(ResolvedSwitch {
            label,
            camera: c,
            target,
            options,
        })
    }

    /// The edit itself: `Ok` carries the Action Log's sentence and the
    /// camera's report entry, `Err` the refusal's sentence.
    fn switch_camera_model_inner(
        &mut self,
        id: ReconId,
        request: &SwitchCameraModelRequest,
        note: Option<&str>,
        collector: &Collector,
    ) -> Result<(String, CameraSwitch), String> {
        let ResolvedSwitch {
            label,
            camera: c,
            target,
            options,
        } = self.resolve_switch(id, request)?;
        let refuse = |why: String| format!("Switch camera model of {label} refused: {why}");
        let index = self.node_index(id)?;

        // Materialise only when there is an overlay to fold in; an empty one
        // materialises to its own base, which the switch can read directly.
        let edited = self.scene[index].history.current();
        let (materialised, mat_map) =
            if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                (None, None)
            } else {
                let _phase = collector.phase("materialise");
                let (value, map) = edited.materialize();
                (Some(value), Some(map))
            };
        let source: &SfmrReconstruction = match materialised.as_ref() {
            Some(value) => value,
            None => &self.scene[index].history.current().base,
        };

        let (switched, report) = {
            let _phase = collector.phase("switch camera model");
            switch_camera_model(source, &[c], &target, &options).map_err(|e| {
                refuse(switch_refusal_in_version_numbering(e, mat_map.as_ref()).to_string())
            })?
        };
        let mat_map = mat_map.map(PointMap::Rows);
        let entry = report
            .cameras
            .into_iter()
            .next()
            .expect("one camera was named");
        // The switch deletes and creates no points, so this scan is the
        // identity map -- read off the two values rather than asserted.
        let scan = {
            let _phase = collector.phase("row map");
            RowMap::by_scan(source, &switched, None).map_err(|e| refuse(e.to_string()))?
        };
        let mut steps = Vec::new();
        steps.extend(mat_map);
        steps.push(PointMap::Rows(scan));
        let map = PointMap::Chain(steps);

        let text = switch_label(&label, &entry);
        let node = &mut self.scene[index];
        let serial = {
            let _phase = collector.phase("push version");
            node.history.push(
                EditedReconstruction::new(Arc::new(switched)),
                map,
                text.clone(),
            )
        };
        let parent = version_before(node, serial);
        self.follow_selection_forward(id);
        let note = note.map_or_else(String::new, |note| format!("; {note}"));
        let message = version_step_text(
            &format!("{text}: {}{note}", switch_report_text(&entry)),
            parent,
            serial,
        );
        Ok((message, entry))
    }

    /// Apply the proposal open on `id`.
    ///
    /// The proposal ends here. On a node whose observations carry a pixel the
    /// switch and the retriangulation of the camera's tracks start as the
    /// `Switch camera model` background task, which pushes one version when it
    /// finishes; on one whose observations carry none it is the direct switch
    /// alone, on the GUI thread, with `retriangulate_points`' refusal sentence
    /// added to its row. Refused, with the proposal left open: no proposal, a
    /// fit that was refused, a node or a viewer that is busy, and a switch the
    /// core refuses on the GUI thread (a `.sift` file that cannot be read).
    pub(crate) fn apply_camera_model_proposal(
        &mut self,
        id: ReconId,
    ) -> Result<ProposalApplied, String> {
        let label = self
            .node(id)
            .map(|node| node.label.clone())
            .ok_or_else(|| crate::state::NOT_LOADED.to_string())?;
        let Some(proposal) = self
            .camera_model_proposal
            .take_if(|proposal| proposal.recon == id)
        else {
            return Err(format!("No camera model proposal is open on {label}."));
        };
        let request = proposal.request();
        // The job's answer for this generation is the version the task would
        // push, so it is installed here and nothing runs in the background.
        let installs_answer = proposal.current_answer().is_some();
        let refused = match &proposal.fit {
            Err(why) => Some(format!("Switch camera model of {label} refused: {why}")),
            Ok(_) if installs_answer => self.busy_refusal(id),
            Ok(_) => self.busy_refusal(id).or_else(|| self.running_refusal()),
        };
        if let Some(why) = refused {
            self.camera_model_proposal = Some(proposal);
            self.action_log.fail(Kind::Edit, why.clone());
            return Err(why);
        }
        if installs_answer {
            let mut proposal = proposal;
            let Some((_, Ok(answer))) = proposal.triangulation.take() else {
                unreachable!("the answer was found above");
            };
            let entry = self.install_switched_camera(id, *answer);
            self.sync_proposal_field();
            return Ok(ProposalApplied::Pushed(id, Box::new(entry)));
        }
        let node = self.node(id).expect("resolved above");
        match retriangulate_refusal(node, None) {
            None => {
                let started = self.switch_camera_model_job(id, &request).and_then(|job| {
                    self.start_background_task(Operation::SWITCH_CAMERA_MODEL, id, job)
                });
                match started {
                    Ok(()) => {
                        self.sync_proposal_field();
                        Ok(ProposalApplied::Started)
                    }
                    Err(why) => {
                        self.camera_model_proposal = Some(proposal);
                        self.action_log.fail(Kind::Edit, why.clone());
                        Err(why)
                    }
                }
            }
            Some(why) => {
                let note = format!("its tracks were not retriangulated: {why}");
                match self.switch_camera_model_noting(id, &request, Some(&note)) {
                    Ok(entry) => {
                        self.sync_proposal_field();
                        Ok(ProposalApplied::Pushed(id, Box::new(entry)))
                    }
                    Err(why) => {
                        self.camera_model_proposal = Some(proposal);
                        Err(why)
                    }
                }
            }
        }
    }

    /// Push the proposal job's answer as `id`'s next version, on the GUI
    /// thread: the version, label and Action Log row the `Switch camera model`
    /// task would push, since the job ran the task's computation
    /// ([`switch_and_retriangulate`]) over the same value. Returns the switch's
    /// report for the reply.
    fn install_switched_camera(
        &mut self,
        id: ReconId,
        answer: crate::camera_model_proposal::job::ProposalAnswer,
    ) -> CameraSwitch {
        let started = Instant::now();
        let collector = Collector::new(self.action_log.detailed_timing());
        let SwitchedCamera {
            value,
            map,
            version_label,
            text,
            switch,
        } = answer.run;
        let index = self.node_index(id).expect("the proposal's node is loaded");
        let serial = {
            let _phase = collector.phase("push version");
            self.scene[index]
                .history
                .push(EditedReconstruction::new(value), map, version_label)
        };
        let parent = version_before(&self.scene[index], serial);
        self.follow_selection_forward(id);
        self.action_log.record_done(
            Kind::Edit,
            started,
            version_step_text(&text, parent, serial),
            collector.take(),
        );
        switch
    }

    /// The switch and the retriangulation of the switched camera's tracks, as
    /// a function of the [`Progress`] it reports through.
    ///
    /// The closure owns what it reads: a clone of the version's
    /// [`EditedReconstruction`], whose base is the very `Arc` the node goes on
    /// drawing, and the resolved request. The work itself is
    /// [`switch_and_retriangulate`], which the proposal's own job runs too.
    pub(crate) fn switch_camera_model_job(
        &self,
        id: ReconId,
        request: &SwitchCameraModelRequest,
    ) -> Result<Job, String> {
        let resolved = self.resolve_switch(id, request)?;
        let edited = self
            .node(id)
            .expect("resolved above")
            .history
            .current()
            .clone();
        Ok(Box::new(move |progress| {
            let (source, mat_map) =
                if edited.deleted_points.is_empty() && edited.added.points.is_empty() {
                    (Arc::clone(&edited.base), None)
                } else {
                    let (value, map) = edited.materialize();
                    (Arc::new(value), Some(map))
                };
            match switch_and_retriangulate(&source, mat_map, &resolved, progress, |_| {}) {
                Ok((run, _report)) => Finished::SwitchedCamera(Box::new(run)),
                Err(SwitchEnd::Cancelled) => Finished::Cancelled,
                Err(SwitchEnd::Failed(why)) => Finished::Failed(why),
            }
        }))
    }
}

/// How [`switch_and_retriangulate`] ended without an answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SwitchEnd {
    /// Asked to stop, and did.
    Cancelled,
    /// The switch or the retriangulation refused, in the edit's sentence.
    Failed(String),
}

/// The proposal's Apply as one computation: the switch of `resolved`'s camera
/// over `source`, then the retriangulation of the camera's tracks over the
/// switched value through `RetriangulateWhich::TheseInBulk`, with the default
/// options the Retriangulate edits use.
///
/// `source` is the version's base, or its materialisation with `mat_map` its
/// row map. `switched` is called with the switch's report as soon as the
/// switch returns, before the retriangulation starts. The answer is the
/// version Apply pushes, its label, its Action Log sentence, the switch's
/// report and the retriangulation's report, which says what happened to each
/// point in `source`'s rows.
///
/// Run by the `Switch camera model` background task and by the proposal's own
/// job (`crate::camera_model_proposal::job`), so the version the job holds is
/// the version the task would push. The switch takes no `Progress`, so a
/// cancel is read on either side of it; the retriangulation polls the flag
/// between its stages.
pub(crate) fn switch_and_retriangulate(
    source: &SfmrReconstruction,
    mat_map: Option<RowMap>,
    resolved: &ResolvedSwitch,
    progress: &Progress<'_>,
    switched: impl FnOnce(&CameraSwitch),
) -> Result<(SwitchedCamera, RetriangulateReport), SwitchEnd> {
    let ResolvedSwitch {
        label,
        camera: c,
        target,
        options,
    } = resolved;
    let c = *c;
    let refuse =
        |why: String| SwitchEnd::Failed(format!("Switch camera model of {label} refused: {why}"));
    if progress.check_cancel().is_err() {
        return Err(SwitchEnd::Cancelled);
    }
    let [p_switch, p_retriangulate] = progress.split([0.1, 0.9]);
    let (switched_value, entry) = {
        let _phase = p_switch.phase("switch camera model");
        let (value, report) = switch_camera_model(source, &[c], target, options).map_err(|e| {
            refuse(switch_refusal_in_version_numbering(e, mat_map.as_ref()).to_string())
        })?;
        let entry = report
            .cameras
            .into_iter()
            .next()
            .expect("one camera was named");
        (value, entry)
    };
    switched(&entry);
    if progress.check_cancel().is_err() {
        return Err(SwitchEnd::Cancelled);
    }
    // The switch renumbers nothing, so the tracks are named in the switched
    // value's own rows.
    let tracks = tracks_of_camera(&switched_value, c);
    let switched_value = EditedReconstruction::new(Arc::new(switched_value));
    let (next, map, report) = match retriangulate_points(
        &switched_value,
        RetriangulateWhich::TheseInBulk(&tracks),
        &RetriangulateOptions::default(),
        &p_retriangulate,
    ) {
        Ok(solved) => solved,
        Err(RetriangulateError::Cancelled) => return Err(SwitchEnd::Cancelled),
        Err(e) => return Err(refuse(e.to_string())),
    };
    let version_label = format!(
        "{} and retriangulated its tracks",
        switch_label(label, &entry)
    );
    let text = format!(
        "{version_label}: {}; {}",
        switch_report_text(&entry),
        retriangulate_summary(&report)
    );
    let mut steps: Vec<PointMap> = mat_map.map(PointMap::Rows).into_iter().collect();
    steps.push(map);
    Ok((
        SwitchedCamera {
            // `TheseInBulk` writes into a new base with nothing over it, so
            // taking it out of the version is a move.
            value: next.base,
            map: PointMap::Chain(steps),
            version_label,
            text,
            switch: entry,
        },
        report,
    ))
}

/// A switch's request resolved against the camera it names.
pub(crate) struct ResolvedSwitch {
    /// The node's label, for the sentences.
    pub(crate) label: String,
    /// The camera's index in the node's camera table.
    pub(crate) camera: usize,
    /// The model, with its count.
    pub(crate) target: RefitTarget,
    /// The fit's options.
    pub(crate) options: SwitchCameraModelOptions,
}

/// How the proposal's Apply went, when it went.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ProposalApplied {
    /// The `Switch camera model` background task started; its version lands
    /// when it finishes.
    Started,
    /// The switch alone pushed a version of this node on the GUI thread, with
    /// the switch's report for the camera.
    Pushed(ReconId, Box<CameraSwitch>),
}

impl ProposalApplied {
    /// The node a version was pushed on in the call, for the caller to drop
    /// the panel caches describing its geometry.
    pub(crate) fn pushed(&self) -> Option<ReconId> {
        match self {
            ProposalApplied::Started => None,
            ProposalApplied::Pushed(id, _) => Some(*id),
        }
    }
}

/// The proposal's tracks: every point with at least one observation in an
/// image taken through camera `camera`, ascending.
pub(crate) fn tracks_of_camera(recon: &SfmrReconstruction, camera: usize) -> Vec<u32> {
    let table = &recon.image_table;
    let mut seen = vec![false; recon.point_set.points.len()];
    for observation in &recon.point_set.tracks {
        if table.images[observation.image_index as usize].camera_index as usize == camera {
            seen[observation.point_index as usize] = true;
        }
    }
    seen.iter()
        .enumerate()
        .filter(|(_, &s)| s)
        .map(|(point, _)| point as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal naming a point of the materialised value names it by the
    /// index the version on screen gives it: with point 2 deleted, the
    /// materialised value's point 5 is the version's point 6.
    #[test]
    fn a_refusal_names_the_point_by_the_versions_index() {
        let refusal = || SwitchCameraModelError::Refit {
            camera: 1,
            error: RefitError::ObservationsPast90 {
                max_theta_deg: 144.91,
                point: 5,
            },
        };
        let map = RowMap::by_removal(10, &[2]);
        let named = switch_refusal_in_version_numbering(refusal(), Some(&map)).to_string();
        assert!(named.contains("the ray to point 6 is 144.91°"), "{named}");
        let unmapped = switch_refusal_in_version_numbering(refusal(), None).to_string();
        assert!(
            unmapped.contains("the ray to point 5 is 144.91°"),
            "{unmapped}"
        );
    }
}
