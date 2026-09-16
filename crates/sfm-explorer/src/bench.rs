// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench in the viewer: every step on a node's bench, as a version of that
//! node.
//!
//! See `specs/gui/bench.md`. The bench itself and every step over it are
//! `sfmtool_core::bench` -- a value and pure functions, with no history in them (`specs/core/bench/bench.md`).
//! What this module adds is the three
//! things a window needs: the version each step is pushed as, the Action Log
//! row it writes, and, for the two steps that read photographs, the background
//! task they run in.
//!
//! Every method here has the same five moves, which is the shape
//! [`crate::state::edits`] set for a point edit:
//!
//! 1. refuse while the node is busy, and find the node,
//! 2. read the bench at the node's cursor,
//! 3. call the core function, which hands back the next value and a report,
//! 4. push one version -- [`crate::document::History::push_bench`] for every
//!    step but the commit, which states both halves with `push_pair`,
//! 5. record one Action Log row, of kind [`Kind::Bench`] for every step but
//!    the commit, whose row is an `Edit` because it is one.
//!
//! A step that changes nothing pushes no version: setting the verdict an
//! observation already has, or the stage a track is already at, is reported
//! back to the caller and leaves the history alone.

use std::sync::Arc;

use sfmtool_core::bench::{
    self, Bench, BenchItem, ClusterSeed, CreateTrackOptions, EditableTrack, EvaluateOptions,
    Observation, ObservationSeed, Provenance, StageKind, Thresholds, Verdict,
};
use sfmtool_core::EditedReconstruction;

use crate::action_log::Kind;
use crate::background::{Finished, Job, Operation};
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::edits::version_before;
use crate::state::AppState;

#[cfg(test)]
mod tests;

/// Where a seed's position and shape come from.
///
/// One enum for the two steps that seed an observation, because a caller
/// arrives holding one of three things and the step should not care which: a
/// pixel with nothing else, a pixel with a size or a shape read at it, or a
/// `.sift` feature, which carries its own position and its own keypoint frame.
///
/// [`Seed::Pixel`] with no radius is "I have no shape to give you, use the one
/// you have": the viewer's own patch radius for that image when a cluster is
/// being started, and the track's reference shape when an observation is being
/// added to one. That is what a right-click in the Image Detail panel means,
/// and it is why the two cases are one variant rather than two.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Seed {
    /// A pixel someone pointed at, with the patch's half-width in that image's
    /// own pixels where the caller named one.
    Pixel {
        /// Where, in source-image px.
        pixel: [f64; 2],
        /// The patch's half-width in px, or `None` for the shape the step
        /// already has.
        radius_px: Option<f64>,
    },
    /// A pixel with the affine shape read at it: the detector's canonical
    /// keypoint frame mapped onto this image's pixels, which is the convention
    /// [`sfmtool_core::bench::ClusterMeasurement::seed_shape`] states.
    Affine {
        /// Where, in source-image px.
        pixel: [f64; 2],
        /// Keypoint-frame units to this image's pixels.
        shape: [[f64; 2]; 2],
    },
    /// A `.sift` feature of the image, by its index in that file.
    Feature {
        /// The feature's index in its image's `.sift` file.
        feature: u32,
    },
}

/// A [`Seed`] with its `.sift` row read, if it named one.
struct SeededAt {
    /// Where the observation goes, in source-image px.
    pixel: [f64; 2],
    /// The keypoint-frame shape the caller supplied, or `None` for the step's
    /// own.
    shape: Option<[[f64; 2]; 2]>,
    /// The feature it came from, which is what the provenance records.
    feature: Option<u32>,
}

/// What the viewer calls the item a gesture names when the caller named none:
/// the active track of the node's bench.
///
/// A bench panel acts on the active item of the kind it edits, so every method
/// here takes the label explicitly and the panel passes the active one. That
/// keeps the "which track" question in one place -- the panel and the wire --
/// rather than inside each step.
pub(crate) fn active_track_label(bench: &Bench) -> Option<&str> {
    bench.active_label(sfmtool_core::bench::ItemKind::Track)
}

/// Where an observation currently is, in its image's own pixels: the keypoint a
/// track-stage measurement carries, else the cluster stage's refined position
/// or the seed it started from, else nothing.
///
/// One rule in one place, because two panels draw the same answer: the Image
/// Detail bench layer puts its mark there, and a Track Edit row click asks that
/// panel to reveal it. A mark and a row are one observation, so they cannot be
/// allowed to disagree about where it is. The same order the evaluation's own
/// seeding uses -- the measured position wins over the seed.
pub(crate) fn observation_pixel(observation: &Observation) -> Option<[f32; 2]> {
    if let Some(keypoint) = observation.track.as_ref().and_then(|m| m.keypoint) {
        return Some(keypoint);
    }
    let position = observation.cluster.as_ref()?.best_position();
    Some([position[0] as f32, position[1] as f32])
}

impl AppState {
    /// The bench beside `id`, at that node's cursor.
    pub(crate) fn bench(&self, id: ReconId) -> Option<&Arc<Bench>> {
        Some(self.node(id)?.history.current_bench())
    }

    /// The track called `label` on `id`'s bench, at that node's cursor.
    pub(crate) fn bench_track(&self, id: ReconId, label: &str) -> Option<&Arc<EditableTrack>> {
        self.bench(id)?.track(label)
    }

    /// Put the point `point` names on its node's bench as a track-stage
    /// editable track, and make it the active one.
    ///
    /// The label is the point's portable id
    /// ([`crate::point_ids::mint`]), which is what a log row naming the item
    /// has to carry to be worth reading, and which core cannot mint because it
    /// sees one value rather than the node's version graph. Putting on a point
    /// a track already came from activates that track instead of putting a
    /// second one on: the person asked to work on that point, and there it is.
    pub(crate) fn put_point_on_bench(&mut self, point: PointRef) -> Result<String, String> {
        if let Some(why) = self.busy_refusal(point.recon) {
            return Err(why);
        }
        let index = self.node_index(point.recon)?;
        let node = &self.scene[index];
        let serial = node.history.current_version().serial;
        let bench = Arc::clone(node.history.current_bench());

        // Already on it: the origin is the point, followed to this version.
        if let Some(label) = bench
            .entries()
            .iter()
            .find(|entry| match &entry.item {
                BenchItem::Track(track) => self
                    .resolved_origin(node, track)
                    .is_some_and(|origin| origin == point.point),
            })
            .map(|entry| entry.label.clone())
        {
            self.activate_bench_item(point.recon, &label)?;
            return Ok(label);
        }

        let label = crate::scene::point_id(node, point.index());
        let options = CreateTrackOptions {
            version: serial.as_u64(),
            label: Some(label),
        };
        let (next, report) = bench::create_track(&bench, node.edited(), point.point, &options)
            .map_err(|e| format!("Cannot put that point on the bench: {e}"))?;
        let text = format!("Put point {} on the bench as {}", point.point, report.label);
        self.push_bench_step(index, next, text);
        Ok(report.label)
    }

    /// Start a cluster-stage track on `image`'s node from `seed`, and make it
    /// the active track.
    ///
    /// The gesture behind it is the one Create 3D Point uses: the pixel is
    /// where the Image Detail panel's context menu was last opened, and a seed
    /// that names no shape takes the radius that panel offers for a created
    /// point, so a cluster and a created point are started at the same place at
    /// the same size.
    pub(crate) fn start_bench_cluster(
        &mut self,
        image: ImageRef,
        seed: &Seed,
    ) -> Result<String, String> {
        if let Some(why) = self.busy_refusal(image.recon) {
            return Err(why);
        }
        let seeded = self.seeded_at(image, seed)?;
        let shape = match seeded.shape {
            Some(shape) => shape,
            None => ClusterSeed::shape_from_radius_px(f64::from(
                self.create_point_default_radius(image),
            )),
        };
        let index = self.node_index(image.recon)?;
        let node = &self.scene[index];
        let name = node
            .recon()
            .image_table
            .images
            .get(image.index())
            .map(|im| im.name.clone())
            .ok_or_else(|| "That image is no longer in the reconstruction.".to_string())?;
        let stem = crate::resect::basename(&name).rsplit_once('.').map_or_else(
            || crate::resect::basename(&name).to_string(),
            |(s, _)| s.to_string(),
        );
        let seed = ClusterSeed {
            image: image.image,
            image_stem: stem,
            pixel: seeded.pixel,
            shape,
            feature: seeded.feature,
        };
        let bench = Arc::clone(node.history.current_bench());
        let (next, report) = bench::create_cluster(&bench, &seed)
            .map_err(|e| format!("Cannot start a track there: {e}"))?;
        let text = format!("Started {} on the bench", report.label);
        self.push_bench_step(index, next, text);
        Ok(report.label)
    }

    /// Add a candidate observation of the track called `label` in `image`, at
    /// the place `seed` names.
    ///
    /// A seed with no shape of its own is added at the track's own scale, which
    /// is what the core step does with an [`ObservationSeed`] carrying none.
    pub(crate) fn add_bench_observation(
        &mut self,
        label: &str,
        image: ImageRef,
        seed: &Seed,
    ) -> Result<(), String> {
        let seeded = self.seeded_at(image, seed)?;
        let (index, bench, track) = self.bench_step_target(image.recon, label)?;
        let name = self.image_name(image);
        let seed = ObservationSeed {
            image: image.image,
            pixel: seeded.pixel,
            shape: seeded.shape,
            provenance: match seeded.feature {
                Some(feature) => Provenance::Descriptor { feature },
                None => Provenance::Pixel,
            },
        };
        let (next, _report) = bench::add_observation(&track, &seed)
            .map_err(|e| format!("Cannot add that observation: {e}"))?;
        let bench = install(&bench, label, next)?;
        let text = format!("Added {name} to {label}");
        self.push_bench_step(index, bench, text);
        Ok(())
    }

    /// Set the verdict of one observation of the track called `label`, by hand.
    ///
    /// A verdict the observation already carries changes nothing but the pin,
    /// which the core step reports as `changed: false`; no version is pushed
    /// for it, because a history row that says nothing happened is a row that
    /// has to be undone for nothing.
    pub(crate) fn set_bench_verdict(
        &mut self,
        id: ReconId,
        label: &str,
        observation: usize,
        verdict: Verdict,
    ) -> Result<(), String> {
        let (index, bench, track) = self.bench_step_target(id, label)?;
        let image = track
            .observations
            .get(observation)
            .map(|o| o.image as usize)
            .ok_or_else(|| format!("{label} has no observation {observation}."))?;
        let (next, report) = bench::set_verdict(&track, observation, verdict)
            .map_err(|e| format!("Cannot set that verdict: {e}"))?;
        if !report.changed {
            return Ok(());
        }
        let name = self.image_name(ImageRef::new(id, image));
        let bench = install(&bench, label, next)?;
        let text = match verdict {
            Verdict::In => format!("Turned {name} in to {label}"),
            Verdict::Out => format!("Turned {name} out of {label}"),
            Verdict::Candidate => format!("Made {name} a candidate of {label}"),
        };
        self.push_bench_step(index, bench, text);
        Ok(())
    }

    /// Set the track's bars to `thresholds` and paint the proposed verdicts
    /// onto its unpinned observations.
    ///
    /// One version for the two, because they are one gesture: the sliders are
    /// the panel's until the button is pressed, and what the button applies is
    /// the painting those slider positions produce.
    pub(crate) fn apply_bench_thresholds(
        &mut self,
        id: ReconId,
        label: &str,
        thresholds: &Thresholds,
    ) -> Result<(), String> {
        let (index, bench, track) = self.bench_step_target(id, label)?;
        let mut with_bars = (*track).clone();
        with_bars.thresholds = thresholds.clone();
        let (next, report) = bench::apply_thresholds(&with_bars);
        let bench = install(&bench, label, next)?;
        let text = format!(
            "Applied the thresholds to {label}: {} in, {} out, {} pinned, {} unmeasured",
            report.turned_in, report.turned_out, report.pinned, report.unmeasured
        );
        self.push_bench_step(index, bench, text);
        Ok(())
    }

    /// Move the named observations off the track called `label` onto a second
    /// track beside it, and report the label that one took.
    pub(crate) fn split_bench_track(
        &mut self,
        id: ReconId,
        label: &str,
        observations: &[usize],
    ) -> Result<String, String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let index = self.node_index(id)?;
        let node = &self.scene[index];
        let bench = Arc::clone(node.history.current_bench());
        let (next, report) = bench::split(&bench, node.edited(), label, observations)
            .map_err(|e| format!("Cannot split that track: {e}"))?;
        let text = format!(
            "Split {} observations off {label} as {}",
            report.moved, report.label
        );
        self.push_bench_step(index, next, text);
        Ok(report.label)
    }

    /// Make the item called `label` the active one of its kind.
    pub(crate) fn activate_bench_item(&mut self, id: ReconId, label: &str) -> Result<(), String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let index = self.node_index(id)?;
        let bench = Arc::clone(self.scene[index].history.current_bench());
        if active_track_label(&bench) == Some(label) {
            return Ok(());
        }
        let next = bench
            .activate(label)
            .map_err(|e| format!("Cannot activate that item: {e}"))?;
        let text = format!("Made {label} the active track");
        self.push_bench_step(index, next, text);
        Ok(())
    }

    /// Take the item called `label` off the bench.
    ///
    /// No confirmation anywhere that calls this: a discard is a version, and an
    /// undo puts the item back where it was and active as it was.
    pub(crate) fn discard_bench_item(&mut self, id: ReconId, label: &str) -> Result<(), String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let index = self.node_index(id)?;
        let bench = Arc::clone(self.scene[index].history.current_bench());
        let next = bench
            .discard(label)
            .map_err(|e| format!("Cannot discard that item: {e}"))?;
        let text = format!("Discarded {label} from the bench");
        self.push_bench_step(index, next, text);
        Ok(())
    }

    /// Rename the item called `label` to `to`.
    pub(crate) fn rename_bench_item(
        &mut self,
        id: ReconId,
        label: &str,
        to: &str,
    ) -> Result<(), String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let index = self.node_index(id)?;
        let bench = Arc::clone(self.scene[index].history.current_bench());
        if bench.position(label).is_none() {
            return Err(format!("Nothing on the bench is called {label}."));
        }
        // A rename to the label an item already holds is no step: core allows it,
        // and a version for it would have to be undone for nothing.
        if label == to {
            return Ok(());
        }
        let next = bench
            .rename(label, to)
            .map_err(|e| format!("Cannot rename that item: {e}"))?;
        let text = format!("Renamed {label} to {to} on the bench");
        self.push_bench_step(index, next, text);
        Ok(())
    }

    /// Write the track called `label` into the node's reconstruction.
    ///
    /// The one bench step that changes both halves of the version, and the one
    /// whose row is an `Edit`: the map the core commit reports is pushed as it
    /// stands, so the selection follows a replaced point exactly as it follows
    /// an added observation, and an undo restores the pair -- the point gone,
    /// and the track back to the half it had before.
    ///
    /// The track's origin is **followed to the cursor** first. It names the
    /// point by the index it held in the version the track was put on the bench
    /// from, and any number of edits may have moved it since; the version
    /// graph's own walk is what says where it is now, and an origin that names
    /// nothing leaves the commit creating a point rather than replacing one.
    pub(crate) fn commit_bench_track(&mut self, id: ReconId, label: &str) -> Result<(), String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let index = self.node_index(id)?;
        let node = &self.scene[index];
        let bench = Arc::clone(node.history.current_bench());
        let track = bench
            .track(label)
            .ok_or_else(|| format!("Nothing on the bench is called {label}."))?;
        let serial = node.history.current_version().serial;
        let seated = match self.resolved_origin(node, track) {
            Some(point) => track.with_origin(serial.as_u64(), point),
            None => {
                let mut without = (**track).clone();
                without.origin = None;
                without
            }
        };
        let node_label = node.label.clone();

        let (next, report) = bench::commit(node.edited(), &seated)
            .map_err(|e| format!("Cannot commit {label}: {e}"))?;
        // The track stays on the bench, seated on the point it just wrote, so
        // a second commit replaces that rather than putting a second point on
        // one surface. It is seated at the version the commit was computed
        // *from*, by the index the point held there -- the one it replaced, or,
        // for a creation, the index it took, which that version's map carries
        // forward unchanged. The walk from there is what says where the point
        // is at whatever version the cursor later rests on.
        let settled = seated.with_origin(serial.as_u64(), report.replaced.unwrap_or(report.point));
        let bench = install(&bench, label, settled)?;
        let text = report.label(&node_label);
        let created = created_points(&next, &report);
        let node = &mut self.scene[index];
        let serial = node.history.push_pair(
            Some(next),
            Arc::new(bench),
            report.map.clone(),
            text.clone(),
            created,
        );
        let parent = version_before(node, serial);
        self.follow_selection_forward(id);
        self.action_log
            .record(Kind::Edit, format!("{text} ({parent} → {serial})"));
        Ok(())
    }

    /// Measure the track called `label` at the stage it is in, on a worker
    /// thread.
    ///
    /// The photographs the kernels read are decoded **on that worker**: the
    /// file reads and the pyramid builds are seconds of work, and a step that
    /// did them here would hold the frame -- and the wire's reply window --
    /// for the whole of it before the task it defers to had begun. What
    /// crosses to the worker is [`crate::state::edits::ViewSources`] -- a
    /// shared clone of each photograph the node's own cache already holds, and
    /// a path for each one it does not -- with a clone of the value at the
    /// cursor and a clone of the track, so the worker holds no reference into
    /// the scene.
    pub(crate) fn start_bench_evaluate(&mut self, id: ReconId, label: &str) -> Result<(), String> {
        let outcome = match self.bench_evaluate_job(id, label) {
            Ok(job) => self.start_background_task(Operation::BENCH_EVALUATE, id, job),
            Err(message) => Err(message),
        };
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Bench, message.clone());
        }
        outcome
    }

    /// Put the track called `label` into `stage`, on a worker thread.
    ///
    /// Setting the stage a track is already at changes nothing, so it starts no
    /// task and pushes no version.
    pub(crate) fn start_bench_stage(
        &mut self,
        id: ReconId,
        label: &str,
        stage: StageKind,
    ) -> Result<(), String> {
        if self
            .bench_track(id, label)
            .is_some_and(|track| track.stage_kind() == stage)
        {
            return Ok(());
        }
        let outcome = match self.bench_stage_job(id, label, stage) {
            Ok(job) => self.start_background_task(Operation::BENCH_SET_STAGE, id, job),
            Err(message) => Err(message),
        };
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Bench, message.clone());
        }
        outcome
    }

    /// The evaluation itself, as a function of the `Progress` it reports
    /// through.
    fn bench_evaluate_job(&mut self, id: ReconId, label: &str) -> Result<Job, String> {
        let (edited, track, sources) = self.bench_photometric_inputs(id, label)?;
        let label = label.to_string();
        Ok(Box::new(move |progress| {
            let decoded = match sources.decode(progress) {
                Ok(decoded) => decoded,
                Err(e) => return Finished::Failed(format!("Cannot evaluate {label}: {e}")),
            };
            let views = decoded.views();
            match bench::evaluate(
                &track,
                &edited,
                &views,
                &EvaluateOptions::default(),
                progress,
            ) {
                Err(e) => Finished::Failed(format!("Cannot evaluate {label}: {e}")),
                Ok((measured, report)) => Finished::BenchTrack {
                    version_label: format!("Evaluated {label}"),
                    text: format!("Evaluated {label}: {report}"),
                    label,
                    track: Box::new(measured),
                },
            }
        }))
    }

    /// The stage change itself, as a function of the `Progress` it reports
    /// through.
    fn bench_stage_job(
        &mut self,
        id: ReconId,
        label: &str,
        stage: StageKind,
    ) -> Result<Job, String> {
        let (edited, track, sources) = self.bench_photometric_inputs(id, label)?;
        let label = label.to_string();
        Ok(Box::new(move |progress| {
            let decoded = match sources.decode(progress) {
                Ok(decoded) => decoded,
                Err(e) => return Finished::Failed(format!("Cannot set the stage of {label}: {e}")),
            };
            let views = decoded.views();
            match bench::set_stage(
                &track,
                &edited,
                &views,
                stage,
                &EvaluateOptions::default(),
                progress,
            ) {
                Err(e) => Finished::Failed(format!("Cannot set the stage of {label}: {e}")),
                Ok((staged, report)) => {
                    // The stage phrase is written once, here, and what core
                    // adds is the clause that follows it: a report printed
                    // whole would state the stage a second time.
                    let version_label = format!("Set {label} to the {stage} stage");
                    Finished::BenchTrack {
                        text: format!("{version_label}{}", report.detail()),
                        version_label,
                        label,
                        track: Box::new(staged),
                    }
                }
            }
        }))
    }

    /// What a photometric bench step needs: the value at the cursor, the track,
    /// and where one view per image of the node is to come from.
    ///
    /// The images the worker will decode are the ones the track's observations
    /// name. Every other entry becomes a one-pixel placeholder, which no kernel
    /// samples, because the kernels index the view slice by image index.
    fn bench_photometric_inputs(
        &mut self,
        id: ReconId,
        label: &str,
    ) -> Result<
        (
            EditedReconstruction,
            EditableTrack,
            crate::state::edits::ViewSources,
        ),
        String,
    > {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let index = self.node_index(id)?;
        let node = &self.scene[index];
        let track = node
            .history
            .current_bench()
            .track(label)
            .ok_or_else(|| format!("Nothing on the bench is called {label}."))?;
        let track = (**track).clone();
        let edited = node.edited().clone();
        let mut needed: Vec<usize> = track
            .observations
            .iter()
            .map(|o| o.image as usize)
            .collect();
        needed.sort_unstable();
        needed.dedup();
        let sources = self.view_sources_for(id, &needed)?;
        Ok((edited, track, sources))
    }

    /// Where `track`'s origin point sits in the version at `node`'s cursor, or
    /// `None` when it has no origin or the point it named is gone.
    ///
    /// The origin names a version by the serial the viewer minted for it, so
    /// the walk is the node's own version graph -- the same walk a copied point
    /// id takes ([`crate::point_ids`]), and defined across an undo, a redo and
    /// a discarded redo tail alike.
    fn resolved_origin(&self, node: &SceneNode, track: &EditableTrack) -> Option<u32> {
        let origin = track.origin?;
        let from = node
            .history
            .all_serials()
            .into_iter()
            .find(|serial| serial.as_u64() == origin.version)?;
        let to = node.history.current_version().serial;
        let index = node.history.follow(from, to, origin.point).ok()?;
        node.edited().point(index).is_some().then_some(index)
    }

    /// Push one bench step as the node's next version and write its row.
    fn push_bench_step(&mut self, index: usize, bench: Bench, text: String) {
        let node = &mut self.scene[index];
        let serial = node.history.push_bench(Arc::new(bench), text.clone());
        let parent = version_before(node, serial);
        self.action_log
            .record(Kind::Bench, format!("{text} ({parent} → {serial})"));
    }

    /// The node, its bench and the track a step on one item acts on.
    fn bench_step_target(
        &self,
        id: ReconId,
        label: &str,
    ) -> Result<(usize, Arc<Bench>, Arc<EditableTrack>), String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let index = self.node_index(id)?;
        let bench = Arc::clone(self.scene[index].history.current_bench());
        let track = Arc::clone(
            bench
                .track(label)
                .ok_or_else(|| format!("Nothing on the bench is called {label}."))?,
        );
        Ok((index, bench, track))
    }

    /// Where a seed puts an observation, with its `.sift` row read when it
    /// named one.
    ///
    /// The one place a feature index becomes a position and a shape, so the
    /// menu's pixel gesture and a caller holding a feature reach the same two
    /// numbers by the same route.
    fn seeded_at(&mut self, image: ImageRef, seed: &Seed) -> Result<SeededAt, String> {
        match *seed {
            Seed::Pixel { pixel, radius_px } => Ok(SeededAt {
                pixel,
                shape: radius_px.map(ClusterSeed::shape_from_radius_px),
                feature: None,
            }),
            Seed::Affine { pixel, shape } => Ok(SeededAt {
                pixel,
                shape: Some(shape),
                feature: None,
            }),
            Seed::Feature { feature } => {
                let (pixel, shape) = self.sift_feature(image, feature)?;
                Ok(SeededAt {
                    pixel,
                    shape: Some(shape),
                    feature: Some(feature),
                })
            }
        }
    }

    /// Where one `.sift` feature sits and what keypoint frame it carries, read
    /// through the viewer's own feature cache.
    ///
    /// The cache the Image Detail overlay draws its ellipses from
    /// ([`crate::state::ensure_sift_cached`]), so a feature seeded here is the
    /// mark the person is looking at rather than a second reading of the file.
    /// The stored affine is already the cluster stage's own convention -- the
    /// detector's canonical keypoint frame mapped onto this image's pixels --
    /// so it is passed on as it stands.
    fn sift_feature(
        &mut self,
        image: ImageRef,
        feature: u32,
    ) -> Result<([f64; 2], [[f64; 2]; 2]), String> {
        let wanted = feature as usize;
        let name = self.image_name(image);
        // The cache is `&mut` while the reconstruction it reads from is `&`,
        // which is why the read is a free function over the two.
        let AppState {
            scene, sift_cache, ..
        } = self;
        let node = crate::scene::node_by_id(scene, image.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        let cached = crate::state::ensure_sift_cached(
            sift_cache,
            node.recon(),
            image,
            wanted + 1,
            &sfmtool_core::progress::Progress::none(),
        )
        .ok_or_else(|| format!("No .sift file could be read for {name}."))?;
        let pixel = *cached.positions_xy.get(wanted).ok_or_else(|| {
            format!(
                "{name} has {} .sift features; there is no feature {feature}.",
                cached.positions_xy.len()
            )
        })?;
        let shape = cached.affine_shapes[wanted];
        Ok((
            [f64::from(pixel[0]), f64::from(pixel[1])],
            [
                [f64::from(shape[0][0]), f64::from(shape[0][1])],
                [f64::from(shape[1][0]), f64::from(shape[1][1])],
            ],
        ))
    }

    /// Where `id` sits in the scene.
    fn node_index(&self, id: ReconId) -> Result<usize, String> {
        self.scene
            .iter()
            .position(|n| n.id == id)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())
    }
}

/// The bench with `label`'s item replaced by `track`, leaving the order, the
/// label and the activation as they were.
fn install(bench: &Bench, label: &str, track: EditableTrack) -> Result<Bench, String> {
    bench
        .replace(label, BenchItem::Track(Arc::new(track)))
        .map_err(|e| format!("Cannot install that step: {e}"))
}

/// The points a commit created, named by the point edit's own content hash, or
/// `None` for a commit that replaced one.
///
/// A committed track that replaces a point takes an index the value already
/// held, so it is named by the base's hash like any other row; one that creates
/// a point is a row no base has, and is named the way
/// [`AppState::create_point`] names what it makes.
fn created_points(
    next: &EditedReconstruction,
    report: &bench::CommitReport,
) -> Option<crate::document::CreatedPoints> {
    if report.replaced.is_some() {
        return None;
    }
    let record = next.point(report.point)?.to_record();
    next.point_edit_hash(std::slice::from_ref(&record))
        .ok()
        .map(|hash| crate::document::CreatedPoints {
            hash,
            indexes: vec![report.point],
        })
}
