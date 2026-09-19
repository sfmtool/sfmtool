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
//! **A step that changes nothing pushes no version**, and says so instead:
//! setting the verdict an observation already has, dragging a patch to the pixel
//! it already sits under, painting the verdicts the track already carries. The
//! contract is one for every step (`no_effect` below, `specs/gui/bench.md`
//! § "The wire"): the history is left alone, one Action Log row of kind
//! [`Kind::Bench`] records the step's own no-effect sentence, and the caller
//! gets that sentence back beside the version the node still stands at.
//! Deciding *whether* a step had an effect is core's, with a tolerance in the
//! units of the value, because a pixel's round trip through a patch's plane does
//! not return bit for bit.

use std::sync::Arc;

use sfmtool_core::bench::{
    self, Bench, BenchItem, ClusterSeed, CreateTrackOptions, EditableTrack, EvaluateOptions,
    FitOptions, Observation, ObservationSeed, Provenance, SearchOptions, StageKind, Thresholds,
    Verdict,
};
use sfmtool_core::features::kdforest::ImageKeypoints;
use sfmtool_core::EditedReconstruction;

use crate::action_log::Kind;
use crate::background::{Finished, Job, Operation};
use crate::scene::{ImageRef, PointRef, ReconId, SceneNode};
use crate::state::edits::version_before;
use crate::state::AppState;

pub(crate) mod geometry;

#[cfg(test)]
pub(crate) mod tests;

pub(crate) use geometry::PatchEdit;

/// Constellation size a search asks for when nobody names a radius.
///
/// Fifty, which is the size the query's own radius rule is stated at: the share
/// of found images whose warp places the truth within a few pixels falls away
/// sharply above it, because the affine is the first-order approximation of a
/// homography about the patch centre and the term it drops grows with the patch
/// (`specs/core/features/kdf-constellation-query.md`).
const SEARCH_TARGET_FEATURES: usize = 50;

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

/// The point a commit wrote, in the version the commit produced.
///
/// One row is the whole of what a commit adds to the reconstruction, and it is
/// what the commit selects and what the wire reports -- its index and the
/// portable id minted for it. Carried back from the step rather than looked up
/// afterwards, because "the point this commit wrote" is not a question the
/// value can be asked once the version has landed -- a replacement takes a new
/// index and deletes the one it replaced, and a creation takes whatever index
/// the overlay had free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Committed {
    /// The index the written point holds in the new version.
    pub(crate) point: u32,
    /// The index it replaced, now deleted, or `None` where the commit created a
    /// point instead.
    pub(crate) replaced: Option<u32>,
}

/// A [`Seed`] with its `.sift` row read, if it named one, and its pixel brought
/// inside the photograph.
struct SeededAt {
    /// Where the observation goes, in source-image px.
    pixel: [f64; 2],
    /// The keypoint-frame shape the caller supplied, or `None` for the step's
    /// own.
    shape: Option<[[f64; 2]; 2]>,
    /// The feature it came from, which is what the provenance records.
    feature: Option<u32>,
    /// The pixel that was asked for, when it sat off the photograph and was
    /// brought inside it.
    clamped_from: Option<[f64; 2]>,
}

/// What one seeding step put on the bench, and where.
///
/// The label is the handle every later call uses -- the item a cluster was made
/// as, or the item an observation was added to -- and the two pixels are the
/// clamp: where the observation went, and where the caller asked for it when
/// those differ.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Seeded {
    /// The item the step made, or the one it added to.
    pub(crate) label: String,
    /// The pixel the observation was seeded at.
    pub(crate) pixel: [f64; 2],
    /// The pixel that was asked for, when the clamp moved it.
    pub(crate) clamped_from: Option<[f64; 2]>,
}

/// What one patch edit did, as much of it as a reply needs.
///
/// The core reports are five different shapes and a caller outside this module
/// wants three facts out of all of them: whether a version was pushed, the pixel
/// the gesture ended at, and whether that pixel had to be brought inside the
/// photograph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PatchEdited {
    /// Whether the edit had an effect and so pushed a version.
    pub(crate) changed: bool,
    /// The pixel the gesture landed on, for the three edits that name one.
    pub(crate) pixel: Option<[f64; 2]>,
    /// The pixel that was asked for, when the clamp moved it.
    pub(crate) clamped_from: Option<[f64; 2]>,
}

/// The sentence a clamped pixel adds to a step's own, or nothing for a pixel
/// that named a place on the photograph.
///
/// One wording for every step, because a person reading two rows of the Action
/// Log should not have to work out that they say the same thing.
fn clamp_note(clamped_from: Option<[f64; 2]>, landed: [f64; 2]) -> String {
    match clamped_from {
        Some(asked) => format!(
            ", clamped to the photograph from ({:.1}, {:.1}) to ({:.1}, {:.1})",
            asked[0], asked[1], landed[0], landed[1]
        ),
        None => String::new(),
    }
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

/// The reading options a bench step runs with: core's own, with the caller's
/// search radius where one was named.
///
/// The panel's *Search* control and the wire's `search_px` argument both land
/// here, and a fit passes the same value through to the reading it ends with,
/// so every number on screen was measured in one window.
fn evaluate_options(search_px: Option<f64>) -> EvaluateOptions {
    let mut options = EvaluateOptions::default();
    if let Some(search_px) = search_px {
        options.search_px = search_px;
    }
    options
}

/// The default the panel's *Search* control starts at, which is core's own.
pub(crate) fn default_search_px() -> f64 {
    EvaluateOptions::default().search_px
}

/// Where one observation sits, and with what shape.
///
/// One rule in one place, because everything that draws or names a sighting
/// draws the same answer: the Image Detail bench layer puts its mark there, a
/// Track Edit row click asks that panel to reveal it, the Track Edit tile is
/// cut around it, and the wire reports it. A mark, a row, a tile and a reply
/// are one observation, so they cannot be allowed to disagree about where it
/// is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ObservationSite {
    /// Where, in that image's own pixels.
    pub(crate) pixel: [f64; 2],
    /// The affine shape read at it -- keypoint-frame units to this image's
    /// pixels -- where the cluster slot carries one. `None` for an observation
    /// that has only a track-stage keypoint, whose shape is the surfel's.
    pub(crate) shape: Option<[[f64; 2]; 2]>,
}

/// Where an observation currently is: the keypoint a track-stage measurement
/// carries, else the cluster stage's refined position or the seed it started
/// from, else nothing.
///
/// The same order the evaluation's own seeding uses
/// (`sfmtool_core::bench::evaluate`) -- the measured position wins over the
/// seed -- so a fresh candidate, which has only a seed, is drawn and reported
/// where the step that proposed it put it rather than nowhere.
pub(crate) fn observation_site(observation: &Observation) -> Option<ObservationSite> {
    Some(ObservationSite {
        pixel: observation.site()?,
        shape: observation.shape(),
    })
}

/// [`observation_site`]'s pixel, in the panel's own `f32`.
pub(crate) fn observation_pixel(observation: &Observation) -> Option<[f32; 2]> {
    let site = observation_site(observation)?;
    Some([site.pixel[0] as f32, site.pixel[1] as f32])
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
        self.open_default_descriptor_index(point.recon);
        Ok(report.label)
    }

    /// Start a cluster-stage track on `image`'s node from `seed`, and make it
    /// the active track.
    ///
    /// The pixel is where the Image Detail panel's context menu was last
    /// opened, and a seed that names no shape takes the node's own default
    /// patch radius for that image, so a cluster starts at the scale the
    /// reconstruction already works at there.
    pub(crate) fn start_bench_cluster(
        &mut self,
        image: ImageRef,
        seed: &Seed,
    ) -> Result<Seeded, String> {
        if let Some(why) = self.busy_refusal(image.recon) {
            return Err(why);
        }
        let seeded = self.seeded_at(image, seed)?;
        let shape = match seeded.shape {
            Some(shape) => shape,
            None => ClusterSeed::shape_from_radius_px(f64::from(self.default_patch_radius(image))),
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
        let text = format!(
            "Started {} on the bench{}",
            report.label,
            clamp_note(seeded.clamped_from, seeded.pixel)
        );
        self.push_bench_step(index, next, text);
        // Putting something on the bench is the moment a search becomes
        // possible, so it is the moment to look for the index that would serve
        // one. The look is remembered, so the second item costs nothing.
        self.open_default_descriptor_index(image.recon);
        Ok(Seeded {
            label: report.label,
            pixel: seeded.pixel,
            clamped_from: seeded.clamped_from,
        })
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
    ) -> Result<Seeded, String> {
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
        let text = format!(
            "Added {name} to {label}{}",
            clamp_note(seeded.clamped_from, seeded.pixel)
        );
        self.push_bench_step(index, bench, text);
        Ok(Seeded {
            label: label.to_string(),
            pixel: seeded.pixel,
            clamped_from: seeded.clamped_from,
        })
    }

    /// Set the verdict of one observation of the track called `label`, by hand.
    ///
    /// A verdict the observation already carries changes nothing, which the core
    /// step reports as `changed: false`; no version is pushed for it, because a
    /// history row that says nothing happened is a row that has to be undone for
    /// nothing. The row that says so is written instead
    /// (`no_effect`).
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
        let name = self.image_name(ImageRef::new(id, image));
        if !report.changed {
            self.no_effect(format!(
                "Left {name} {verdict} in {label}: no effect, it is {verdict} already"
            ));
            return Ok(());
        }
        let bench = install(&bench, label, next)?;
        let text = match verdict {
            Verdict::In => format!("Turned {name} in to {label}"),
            Verdict::Out => format!("Turned {name} out of {label}"),
            Verdict::Candidate => format!("Made {name} a candidate of {label}"),
        };
        self.push_bench_step(index, bench, text);
        Ok(())
    }

    /// Apply one hand edit of a track's geometry: a sighting placed, the
    /// surfel resized or turned, or one cluster-stage sighting's shape set.
    ///
    /// The one call behind every handle of the Image Detail panel's bench layer
    /// and behind the wire's three patch tools, so a drag and a tool call are
    /// the same version carrying the same sentence
    /// (`specs/gui/multi-panel-image-browser.md` § "The bench layer"). One version per
    /// gesture: a drag pushes nothing until it is released, and a release that
    /// changed nothing -- the sighting put back where it was, the outline the
    /// size it already had -- pushes nothing at all.
    ///
    /// A size is reported in the pixels of the sighting the gesture named,
    /// because a world half-length says nothing to someone looking at a
    /// photograph; the world number is the fallback for a patch that does not
    /// project into that sighting's image.
    pub(crate) fn edit_bench_patch(
        &mut self,
        id: ReconId,
        label: &str,
        edit: &PatchEdit,
    ) -> Result<PatchEdited, String> {
        let (index, bench, track) = self.bench_step_target(id, label)?;
        let (next, report) = geometry::apply(&track, self.scene[index].edited(), edit)
            .map_err(|e| format!("Cannot edit that patch: {e}"))?;
        let edited = PatchEdited {
            changed: report.changed(),
            pixel: report.pixel(),
            clamped_from: report.clamped_from(),
        };
        if !edited.changed {
            let landed = edited.pixel.unwrap_or_default();
            self.no_effect(format!(
                "{}{}",
                report.no_effect_sentence(label),
                clamp_note(edited.clamped_from, landed)
            ));
            return Ok(edited);
        }
        let text = self.patch_edit_label(id, label, &next, &report);
        let bench = install(&bench, label, next)?;
        self.push_bench_step(index, bench, text);
        Ok(edited)
    }

    /// The version label one patch edit takes: what moved, by how much, and in
    /// whose pixels.
    fn patch_edit_label(
        &self,
        id: ReconId,
        label: &str,
        next: &EditableTrack,
        report: &geometry::EditReport,
    ) -> String {
        match report {
            geometry::EditReport::Translated(report) => {
                let centre = report.center;
                format!(
                    "Moved {label} by {:.3} units to ({:.3}, {:.3}, {:.3}){}",
                    report.moved,
                    centre.x,
                    centre.y,
                    centre.z,
                    clamp_note(report.clamped_from, report.pixel)
                )
            }
            geometry::EditReport::Moved(report) => {
                let name = self.image_name(ImageRef::new(id, report.image as usize));
                let moved = report
                    .moved_px
                    .map(|px| format!(" ({px:.1} px)"))
                    .unwrap_or_default();
                format!(
                    "Moved observation {} of {label} to ({:.1}, {:.1}) in {name}{moved}{}",
                    report.observation,
                    report.pixel[0],
                    report.pixel[1],
                    clamp_note(report.clamped_from, report.pixel)
                )
            }
            geometry::EditReport::Resized(report) => {
                let size = self.frame_size_phrase(id, next, report.observation);
                let note = clamp_note(report.clamped_from, report.pixel.unwrap_or_default());
                format!("Resized {label} to {size}{note}")
            }
            geometry::EditReport::Rotated(report) => {
                format!("Rotated {label} by {:.1} degrees", report.degrees)
            }
            geometry::EditReport::Turned { report, degrees } => format!(
                "Rotated observation {} of {label} by {degrees:.1} degrees",
                report.observation
            ),
        }
    }

    /// How large the patch now is, as the phrase the resize sentence ends on.
    ///
    /// In the pixels of the sighting the resize was named at, because a world
    /// half-length says nothing to someone looking at a photograph; in world
    /// units only when the gesture named no sighting or the patch does not
    /// project into its image. `at` is an **observation** index, not an image:
    /// the size a person reads off an outline is the size of the outline drawn
    /// at that sighting, which is the surfel re-anchored on it.
    fn frame_size_phrase(&self, id: ReconId, track: &EditableTrack, at: Option<usize>) -> String {
        let px = at.and_then(|index| {
            let observation = track.observations.get(index)?;
            let image = observation.image as usize;
            let node = self.node(id)?;
            let (camera, pose) = geometry::view_of(&node.edited().base.image_table, image)?;
            let half = match track.stage_kind() {
                StageKind::Track => {
                    let frame = track.track()?.frame.as_ref()?;
                    let anchored = geometry::anchored_frame(frame, &camera, &pose, observation);
                    geometry::half_width_px(&anchored, &camera, &pose)?
                }
                StageKind::Cluster => sfmtool_core::bench::half_width_px(
                    observation.shape()?,
                    track.cluster()?.radius,
                ),
            };
            Some(format!(
                "{half:.1} px in {}",
                self.image_name(ImageRef::new(id, image))
            ))
        });
        px.unwrap_or_else(|| {
            track
                .track()
                .and_then(|payload| payload.frame.as_ref())
                .map(|frame| format!("a half-length of {:.4}", frame.half_extent[0]))
                .unwrap_or_else(|| "its new size".to_string())
        })
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
        // The bars are half of what this step is: moving one and painting nothing
        // is still a change, and leaving both where they are and painting nothing
        // is not.
        if !report.changed && with_bars.thresholds == track.thresholds {
            self.no_effect(format!(
                "Applied the thresholds to {label}: no effect, every verdict is already the \
                 one they propose"
            ));
            return Ok(());
        }
        let bench = install(&bench, label, next)?;
        let text = format!(
            "Applied the thresholds to {label}: {} in, {} out, {} pinned, {} unmeasured",
            report.turned_in, report.turned_out, report.pinned, report.unmeasured
        );
        self.push_bench_step(index, bench, text);
        Ok(())
    }

    /// Put a copy of the track called `label` on the bench beside it, and
    /// report the label the copy took.
    ///
    /// What a second patch over neighbouring ground is started from: the copy
    /// carries the geometry and the judgements and drops only the origin, so a
    /// commit of it creates a point rather than replacing the one the original
    /// came from ([`sfmtool_core::bench::duplicate`]). The copy is the active
    /// track when the step returns, because it is the thing about to be moved.
    pub(crate) fn duplicate_bench_item(
        &mut self,
        id: ReconId,
        label: &str,
    ) -> Result<String, String> {
        if let Some(why) = self.busy_refusal(id) {
            return Err(why);
        }
        let index = self.node_index(id)?;
        let bench = Arc::clone(self.scene[index].history.current_bench());
        let (next, report) = bench::duplicate(&bench, label)
            .map_err(|e| format!("Cannot duplicate that item: {e}"))?;
        let text = format!("Duplicated {label} as {}", report.label);
        self.push_bench_step(index, next, text);
        Ok(report.label)
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
            self.no_effect(format!(
                "Made {label} the active track: no effect, it is active already"
            ));
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
            self.no_effect(format!(
                "Renamed {label} on the bench: no effect, it is called {to} already"
            ));
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
    /// stands, so an index taken before the commit is followed across it like
    /// any other point edit's, and an undo restores the pair -- the point gone,
    /// and the track back to the half it had before.
    ///
    /// The track's origin is **followed to the cursor** first. It names the
    /// point by the index it held in the version the track was put on the bench
    /// from, and any number of edits may have moved it since; the version
    /// graph's own walk is what says where it is now, and an origin that names
    /// nothing leaves the commit creating a point rather than replacing one.
    ///
    /// What comes back is the point it wrote ([`Committed`]), because the whole
    /// of what a commit produces is one row of the reconstruction and a caller
    /// that cannot name it has to go looking for it. The wire reports its index
    /// and its id.
    ///
    /// **The written point becomes the selection**, through
    /// [`AppState::select_point`] like any other, so the viewport puts the
    /// track rays on it and the Point Track Detail panel opens on it --
    /// wherever the selection happened to be, and whether the commit replaced a
    /// point or created one. A commit is a gesture about one point, and the
    /// index it landed at is the one thing the person who asked for it cannot
    /// work out. That replaces the map-following every other edit does here:
    /// the map carries a selection that was already on the origin to the same
    /// row this puts it on, and says nothing about one that was elsewhere.
    pub(crate) fn commit_bench_track(
        &mut self,
        id: ReconId,
        label: &str,
    ) -> Result<Committed, String> {
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
        self.action_log
            .record(Kind::Edit, format!("{text} ({parent} → {serial})"));
        // After the row the edit wrote, because that is the order the two
        // happened in: the point the selection moves to is a row of the version
        // the line above just announced.
        self.select_point(PointRef::new(id, report.point as usize));
        Ok(Committed {
            point: report.point,
            replaced: report.replaced,
        })
    }

    /// Read the track called `label` at the stage it is in, on a worker thread.
    ///
    /// A reading moves nothing: the position, the frame and every keypoint come
    /// back as they were, and what the version carries is what each observation
    /// now says about itself. `search_px` is how far from each observation the
    /// correlation peak is looked for; `None` takes the reading's own default.
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
    ///
    /// **What the track alone decides is decided here**, through
    /// [`sfmtool_core::bench::evaluate_preconditions`], which is the half of
    /// the step's own validation that reads no photograph. So a track with
    /// nothing to register against is refused in the caller's own hand -- a
    /// menu that greys, a status line, a tool error -- rather than starting a
    /// task whose only act is to decode a dozen images and then fail.
    pub(crate) fn start_bench_evaluate(
        &mut self,
        id: ReconId,
        label: &str,
        search_px: Option<f64>,
    ) -> Result<(), String> {
        let outcome = self.begin_bench_evaluate(id, label, search_px);
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Bench, message.clone());
        }
        outcome
    }

    /// The evaluation up to the moment the worker has it, so that everything
    /// this can refuse is refused before a photograph is read.
    fn begin_bench_evaluate(
        &mut self,
        id: ReconId,
        label: &str,
        search_px: Option<f64>,
    ) -> Result<(), String> {
        let (_, _, track) = self.bench_step_target(id, label)?;
        bench::evaluate_preconditions(&track)
            .map_err(|e| format!("Cannot evaluate {label}: {e}"))?;
        let job = self.bench_evaluate_job(id, label, search_px)?;
        self.start_background_task(Operation::BENCH_EVALUATE, id, job)
    }

    /// Fit the track called `label` at the stage it is in, on a worker thread.
    ///
    /// The step that **moves** the track: at the track stage it localizes every
    /// sighting against the surfel, re-triangulates the `in` ones, re-centres
    /// the frame and fuses the consensus, and then reads the result back so the
    /// numbers it leaves behind are the ones *Evaluate* would report.
    ///
    /// **What the track alone decides is decided here**, through
    /// [`sfmtool_core::bench::fit_preconditions`], which carries the two-`in`
    /// rule a reading does not have.
    pub(crate) fn start_bench_fit(
        &mut self,
        id: ReconId,
        label: &str,
        search_px: Option<f64>,
    ) -> Result<(), String> {
        let outcome = self.begin_bench_fit(id, label, search_px);
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Bench, message.clone());
        }
        outcome
    }

    /// The fit up to the moment the worker has it, so that everything this can
    /// refuse is refused before a photograph is read.
    fn begin_bench_fit(
        &mut self,
        id: ReconId,
        label: &str,
        search_px: Option<f64>,
    ) -> Result<(), String> {
        let (_, _, track) = self.bench_step_target(id, label)?;
        bench::fit_preconditions(&track).map_err(|e| format!("Cannot fit {label}: {e}"))?;
        let job = self.bench_fit_job(id, label, search_px)?;
        self.start_background_task(Operation::BENCH_FIT, id, job)
    }

    /// Put the track called `label` into `stage`, on a worker thread.
    ///
    /// Setting the stage a track is already at changes nothing, so it starts no
    /// task and pushes no version.
    ///
    /// **What the track alone decides is decided here**, through
    /// [`sfmtool_core::bench::set_stage_preconditions`]: too few `in`
    /// observations to triangulate from, or a downgrade of a track carrying no
    /// frame or no position. Those are refusals of the gesture, in the
    /// caller's own hand, rather than a task that decodes a dozen photographs
    /// and then fails for a reason that was knowable before it started.
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
            self.no_effect(format!(
                "Set {label} to the {stage} stage: no effect, it is at that stage already"
            ));
            return Ok(());
        }
        let outcome = self.begin_bench_stage(id, label, stage);
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Bench, message.clone());
        }
        outcome
    }

    /// The stage change up to the moment the worker has it, so that everything
    /// this can refuse is refused before a photograph is read.
    fn begin_bench_stage(
        &mut self,
        id: ReconId,
        label: &str,
        stage: StageKind,
    ) -> Result<(), String> {
        let (_, _, track) = self.bench_step_target(id, label)?;
        bench::set_stage_preconditions(&track, stage)
            .map_err(|e| format!("Cannot set the stage of {label}: {e}"))?;
        let job = self.bench_stage_job(id, label, stage)?;
        self.start_background_task(Operation::BENCH_SET_STAGE, id, job)
    }

    /// Search the node's descriptor index from one observation of the track
    /// called `label`, on a worker thread.
    ///
    /// The step that grows a track by more than one sighting at a time: the
    /// keypoints around the observation are looked up in the index, the images
    /// whose hits agree on a warp are found, and each one the track does not
    /// already name becomes a candidate seeded by that warp
    /// (`specs/core/bench/editable-track.md` § "Searching the descriptor
    /// index").
    ///
    /// **What the track, the index and the `.sift` file decide is decided
    /// here**, in the caller's own hand: no index open, an observation with no
    /// place in its photograph, an image whose keypoints cannot be read. The
    /// worker is left with the forest reads, which are the part that can take a
    /// while.
    ///
    /// `radius_px` is the constellation's radius in the searched image's own
    /// pixels; `None` takes the radius that holds about fifty of that image's
    /// keypoints, which is the size the query is worth asking at
    /// (`specs/core/features/kdf-constellation-query.md`).
    pub(crate) fn start_bench_descriptor_search(
        &mut self,
        id: ReconId,
        label: &str,
        observation: usize,
        radius_px: Option<f64>,
        min_inliers: Option<usize>,
    ) -> Result<(), String> {
        let outcome = self.begin_bench_search(id, label, observation, radius_px, min_inliers);
        if let Err(message) = &outcome {
            self.action_log.fail(Kind::Bench, message.clone());
        }
        outcome
    }

    /// The search up to the moment the worker has it.
    fn begin_bench_search(
        &mut self,
        id: ReconId,
        label: &str,
        observation: usize,
        radius_px: Option<f64>,
        min_inliers: Option<usize>,
    ) -> Result<(), String> {
        let job = self.bench_search_job(id, label, observation, radius_px, min_inliers)?;
        self.start_background_task(Operation::BENCH_SEARCH, id, job)
    }

    /// Why the search cannot run from `observation` of `label`, or `None`.
    ///
    /// What greys the table's context-menu entry with a sentence, and what the
    /// step itself asks, so the menu and the step cannot disagree.
    pub(crate) fn bench_search_refusal(
        &self,
        id: ReconId,
        label: &str,
        observation: usize,
    ) -> Option<String> {
        if let Some(why) = self.busy_refusal(id) {
            return Some(why);
        }
        if self.descriptor_index(id).is_none() {
            return Some(
                "No descriptor index is open. Open or build one in the Descriptor index row \
                 above the table."
                    .to_string(),
            );
        }
        let track = self.bench_track(id, label)?;
        let row = track.observations.get(observation)?;
        let node = self.node(id)?;
        let path = node.recon().sift_path_for_image(row.image as usize);
        if !path.is_file() {
            return Some(format!(
                "{} has no readable .sift file, so there are no keypoints to search from.",
                self.image_name(ImageRef::new(id, row.image as usize))
            ));
        }
        None
    }

    /// The search itself, as a function of the `Progress` it reports through.
    ///
    /// The searched image's keypoints are read **here**, through the viewer's
    /// own feature cache, for the reason the draft gives: the viewer already
    /// holds every image's positions and shapes for the overlay, and a second
    /// reading of the same file per gesture would be a second cache. What
    /// crosses to the worker is that keypoint set, a clone of the track, and a
    /// clone of the forest handle -- the forest owns its own query pool and
    /// block cache, so the clone is a handle and not a copy.
    pub(crate) fn bench_search_job(
        &mut self,
        id: ReconId,
        label: &str,
        observation: usize,
        radius_px: Option<f64>,
        min_inliers: Option<usize>,
    ) -> Result<Job, String> {
        if let Some(why) = self.bench_search_refusal(id, label, observation) {
            return Err(why);
        }
        let (_, _, track) = self.bench_step_target(id, label)?;
        let row = track
            .observations
            .get(observation)
            .ok_or_else(|| format!("{label} has no observation {observation}."))?;
        let image = ImageRef::new(id, row.image as usize);
        let forest = Arc::clone(
            &self
                .descriptor_index(id)
                .ok_or_else(|| "No descriptor index is open.".to_string())?
                .forest,
        );
        let keypoints = self.image_keypoints(image)?;
        let radius_px = match radius_px {
            Some(radius) => radius as f32,
            None => self.default_search_radius_px(image, keypoints.len()),
        };
        let options = SearchOptions {
            radius_px,
            min_inliers: min_inliers.unwrap_or(SearchOptions::default().min_inliers),
            ..SearchOptions::default()
        };
        let track = (*track).clone();
        let label = label.to_string();
        Ok(Box::new(move |progress| {
            match bench::search_descriptors(
                &track,
                observation,
                &keypoints,
                &forest,
                &options,
                progress,
            ) {
                Err(sfmtool_core::bench::SearchError::Cancelled) => Finished::Cancelled,
                Err(e) => Finished::Failed(format!("Cannot search {label}: {e}")),
                Ok((grown, report)) => Finished::BenchTrack {
                    // The report is one sentence and it is the whole of what
                    // the step did, so the version wears it as its label and
                    // the row is that sentence under the item's name.
                    version_label: report.to_string(),
                    text: format!("{label}: {report}"),
                    label,
                    track: Box::new(grown),
                },
            }
        }))
    }

    /// Every keypoint of `image`, through the viewer's own feature cache.
    pub(crate) fn image_keypoints(&mut self, image: ImageRef) -> Result<ImageKeypoints, String> {
        let name = self.image_name(image);
        let AppState {
            scene, sift_cache, ..
        } = self;
        let node = crate::scene::node_by_id(scene, image.recon)
            .ok_or_else(|| "That reconstruction is no longer loaded.".to_string())?;
        // The whole file: a constellation is taken from a radius of the image
        // rather than from the first few features, so a prefix would silently
        // search a different patch.
        let cached = crate::state::ensure_sift_cached(
            sift_cache,
            node.recon(),
            image,
            usize::MAX,
            &sfmtool_core::progress::Progress::none(),
        )
        .ok_or_else(|| format!("No .sift file could be read for {name}."))?;
        Ok(ImageKeypoints {
            positions: cached.positions_xy.clone(),
            affine_shapes: cached.affine_shapes.clone(),
        })
    }

    /// The constellation radius an image gets when nobody names one: the radius
    /// that holds about fifty of its keypoints, read off its own sensor size
    /// and its own keypoint count.
    fn default_search_radius_px(&self, image: ImageRef, keypoint_count: usize) -> f32 {
        let dimensions = self.node(image.recon).and_then(|node| {
            let recon = node.recon();
            let entry = recon.image_table.images.get(image.index())?;
            let camera = recon.image_table.cameras.get(entry.camera_index as usize)?;
            Some((camera.width, camera.height))
        });
        match dimensions {
            Some((width, height)) => sfmtool_core::features::kdforest::radius_for_feature_count(
                width,
                height,
                keypoint_count,
                SEARCH_TARGET_FEATURES,
            ),
            None => bench::DEFAULT_RADIUS_PX,
        }
    }

    /// The evaluation itself, as a function of the `Progress` it reports
    /// through.
    ///
    /// Reachable from the crate's tests as well as from the step, so the test
    /// that holds [`Operation::BENCH_EVALUATE`]'s cancellable declaration to
    /// its claim runs the real work rather than a stand-in for it.
    pub(crate) fn bench_evaluate_job(
        &mut self,
        id: ReconId,
        label: &str,
        search_px: Option<f64>,
    ) -> Result<Job, String> {
        let (edited, track, sources) = self.bench_photometric_inputs(id, label)?;
        let label = label.to_string();
        let options = evaluate_options(search_px);
        Ok(Box::new(move |progress| {
            // The decode is seconds of file reads with no poll of its own, so
            // the flag is read on either side of it: a cancel during it is
            // answered the moment it returns rather than after the kernels have
            // run as well.
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }
            let decoded = match sources.decode(progress) {
                Ok(decoded) => decoded,
                Err(e) => return Finished::Failed(format!("Cannot evaluate {label}: {e}")),
            };
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }
            let views = decoded.views();
            match bench::evaluate(&track, &edited, &views, &options, progress) {
                Err(sfmtool_core::bench::EvaluateError::Cancelled) => Finished::Cancelled,
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

    /// The fit itself, as a function of the `Progress` it reports through.
    /// Crate-visible for the reason [`AppState::bench_evaluate_job`] is.
    pub(crate) fn bench_fit_job(
        &mut self,
        id: ReconId,
        label: &str,
        search_px: Option<f64>,
    ) -> Result<Job, String> {
        let (edited, track, sources) = self.bench_photometric_inputs(id, label)?;
        let label = label.to_string();
        let options = FitOptions {
            evaluate: evaluate_options(search_px),
            ..FitOptions::default()
        };
        Ok(Box::new(move |progress| {
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }
            let decoded = match sources.decode(progress) {
                Ok(decoded) => decoded,
                Err(e) => return Finished::Failed(format!("Cannot fit {label}: {e}")),
            };
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }
            let views = decoded.views();
            match bench::fit(&track, &edited, &views, &options, progress) {
                Err(sfmtool_core::bench::FitError::Cancelled) => Finished::Cancelled,
                Err(e) => Finished::Failed(format!("Cannot fit {label}: {e}")),
                Ok((fitted, report)) => Finished::BenchTrack {
                    // The version label carries the classification, because
                    // finite-versus-at-infinity is the fit's real outcome on a
                    // distant track and a history row reading only "Fitted X"
                    // hides the one thing a person scrolling it is looking for.
                    // The counts stay in the Action Log's own text, which is the
                    // whole report.
                    version_label: match &report.classification {
                        Some(call) => format!("Fitted {label}: {call}"),
                        None => format!("Fitted {label}"),
                    },
                    text: format!("Fitted {label}: {report}"),
                    label,
                    track: Box::new(fitted),
                },
            }
        }))
    }

    /// The stage change itself, as a function of the `Progress` it reports
    /// through. Crate-visible for the reason [`AppState::bench_evaluate_job`]
    /// is.
    pub(crate) fn bench_stage_job(
        &mut self,
        id: ReconId,
        label: &str,
        stage: StageKind,
    ) -> Result<Job, String> {
        let (edited, track, sources) = self.bench_photometric_inputs(id, label)?;
        let label = label.to_string();
        Ok(Box::new(move |progress| {
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }
            let decoded = match sources.decode(progress) {
                Ok(decoded) => decoded,
                Err(e) => return Finished::Failed(format!("Cannot set the stage of {label}: {e}")),
            };
            if progress.is_cancelled() {
                return Finished::Cancelled;
            }
            let views = decoded.views();
            match bench::set_stage(
                &track,
                &edited,
                &views,
                stage,
                &FitOptions::default(),
                progress,
            ) {
                Err(sfmtool_core::bench::StageError::Fit(
                    sfmtool_core::bench::FitError::Cancelled,
                )) => Finished::Cancelled,
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

    /// Write the row for a step that had no effect, and push no version.
    ///
    /// The other half of [`Self::push_bench_step`], and the whole of what a
    /// no-effect step does. The row carries no `(v3 -> v4)`, there being no
    /// transition to name, and it is not a failure: nothing was refused, and the
    /// answer to "what happened" is that what was asked for was already so.
    /// `specs/gui/bench.md` section "The wire" states the contract the reply
    /// shares.
    fn no_effect(&mut self, text: String) {
        self.action_log.record(Kind::Bench, text);
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
    /// The pixel a gesture named is brought inside the photograph here
    /// ([`sfmtool_core::bench::clamp_to_photograph`]), which is why the two
    /// seeding steps and the two wire tools behind them agree: a pointer dragged
    /// past the edge of the picture and a call carrying `[-500, -500]` name no
    /// place on it, and the nearest place they do name is the one the step takes.
    /// A `.sift` feature is already a row of that photograph and is never moved.
    fn seeded_at(&mut self, image: ImageRef, seed: &Seed) -> Result<SeededAt, String> {
        let placed = match *seed {
            Seed::Pixel { pixel, radius_px } => SeededAt {
                pixel,
                shape: radius_px.map(ClusterSeed::shape_from_radius_px),
                feature: None,
                clamped_from: None,
            },
            Seed::Affine { pixel, shape } => SeededAt {
                pixel,
                shape: Some(shape),
                feature: None,
                clamped_from: None,
            },
            Seed::Feature { feature } => {
                let (pixel, shape) = self.sift_feature(image, feature)?;
                return Ok(SeededAt {
                    pixel,
                    shape: Some(shape),
                    feature: Some(feature),
                    clamped_from: None,
                });
            }
        };
        let Some(camera) = self.image_camera(image) else {
            return Ok(placed);
        };
        let (clamped_from, pixel) = sfmtool_core::bench::clamp_to_photograph(&camera, placed.pixel);
        Ok(SeededAt {
            pixel,
            clamped_from,
            ..placed
        })
    }

    /// The lens of one camera image, for the clamp and for anything else that
    /// needs the photograph's own extent.
    fn image_camera(&self, image: ImageRef) -> Option<sfmtool_core::camera::CameraIntrinsics> {
        let table = &self.node(image.recon)?.recon().image_table;
        let row = table.images.get(image.index())?;
        table.cameras.get(row.camera_index as usize).cloned()
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
    pub(crate) fn sift_feature(
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
/// a point is a row no base has, and there is nothing but the edit itself for
/// its identity to reach back to.
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
