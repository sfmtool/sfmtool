// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What the **Preview retriangulation** checkbox shows: where each point of the
//! proposal's tracks goes, read off the proposed run's outcome against the
//! point's stored geometry, and the blend `s` that carries the 3D view from the
//! stored positions to those targets and back.
//!
//! See `specs/gui/edits/switch-camera-model.md` § "The preview". The targets
//! are built on the proposal job's thread ([`PreviewTargets::build`]), one row
//! per row of the node's point buffers, so the GUI thread only uploads them.
//! The motion ([`PreviewMotion`]) is stepped once per frame by the shared
//! controller [`crate::motion::step`], and a new answer waits until `s` has run
//! back to 0, since swapping the targets at any other `s` would make the
//! points jump.

use std::sync::Arc;

use nalgebra::Point3;
use sfmtool_core::reconstruction::bundle_adjust::patch_frame_factor;
use sfmtool_core::reconstruction::triangulation::PointVerdict;
use sfmtool_core::reconstruction::{
    GeometryChange, ImageTable, Point3D, RetriangulateOutcome, RetriangulateReport,
};
use sfmtool_core::{RowMap, SfmrReconstruction};

/// The flag of a buffer row that the preview does not move: held, kept, a
/// point no run read, or a deleted point.
pub(crate) const FLAG_STILL: f32 = 0.0;
/// The flag of a row the preview moves to its target.
pub(crate) const FLAG_MOVES: f32 = 1.0;
/// The flag of a row that stays where it stands and takes the failure colour:
/// a finite point the proposed run answers as a direction.
pub(crate) const FLAG_LOSES: f32 = 2.0;

/// The fastest the blend runs, per second.
pub(crate) const BLEND_MAX_SPEED: f64 = 2.0;
/// How fast the blend's speed changes, per second squared.
pub(crate) const BLEND_ACCELERATION: f64 = 10.0;

/// What the preview does with one point of the proposal's tracks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PointPreview {
    /// The value holds the point at its coordinate, so the run never read it
    /// and Apply leaves it alone.
    Held,
    /// Fewer than two of its observations give a usable ray, so Apply keeps
    /// its stored geometry.
    Kept,
    /// Apply writes a position for a finite point, or a direction for one
    /// stored as a direction, and the preview moves it there.
    Moves {
        /// The position Apply writes, or the unit direction for a point at
        /// infinity.
        target: Point3<f64>,
        /// Whether the point is a direction at both ends.
        at_infinity: bool,
        /// What the surfel's size is multiplied by at the target:
        /// `patch_frame_factor` for a finite move, 1 for a direction.
        size_factor: f32,
        /// Whether Apply changes the stored geometry at all.
        changed: bool,
    },
    /// A finite point the run answers as a direction: Apply writes it as a
    /// point at infinity, and the preview leaves it where it stands in the
    /// failure colour.
    Loses(PointVerdict),
    /// A point stored as a direction that the run places at a position. Apply
    /// writes the position, but a direction cannot travel to a place along a
    /// line, so the preview leaves it where it is drawn.
    Placed,
}

impl PointPreview {
    /// The flag its buffer row carries.
    pub(crate) fn flag(&self) -> f32 {
        match self {
            PointPreview::Moves { .. } => FLAG_MOVES,
            PointPreview::Loses(_) => FLAG_LOSES,
            PointPreview::Held | PointPreview::Kept | PointPreview::Placed => FLAG_STILL,
        }
    }
}

/// What the preview does with a point stored as `stored`, whose outcome in the
/// proposed run is `outcome` and whose answer there is `answer`.
///
/// `centroid` is the switched value's camera centroid, taken once, which the
/// surfel's size factor is measured from as the retriangulation measures the
/// frame it rescales.
pub(crate) fn point_preview(
    stored: &Point3D,
    outcome: &RetriangulateOutcome,
    answer: &Point3D,
    centroid: &Point3<f64>,
) -> PointPreview {
    let verdict = match outcome {
        RetriangulateOutcome::Held => return PointPreview::Held,
        RetriangulateOutcome::Kept => return PointPreview::Kept,
        RetriangulateOutcome::Solved { verdict, .. } => *verdict,
    };
    match (stored.is_at_infinity(), answer.is_at_infinity()) {
        (false, true) => PointPreview::Loses(verdict),
        (true, false) => PointPreview::Placed,
        (at_infinity, _) => {
            let size_factor = if at_infinity {
                1.0
            } else {
                let before = ImageTable::placement_scale_from(centroid, &stored.position);
                patch_frame_factor(centroid, Some(before), Some(&answer.position)).unwrap_or(1.0)
            };
            PointPreview::Moves {
                target: answer.position,
                at_infinity,
                size_factor,
                changed: outcome.moved(),
            }
        }
    }
}

/// One point that loses its position under the proposal, for the wire's
/// `lost_points`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LostPoint {
    /// Its index in the version on screen.
    pub(crate) index: u32,
    /// The rule that turned it into a direction.
    pub(crate) verdict: PointVerdict,
}

/// The preview's counts: the banner's second line and the wire's
/// `triangulation` block.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct PreviewCounts {
    /// The proposal's tracks: every point the run was asked about.
    pub(crate) tracks: usize,
    /// How many Apply moves: changed in place, or placed from a direction.
    pub(crate) moves: usize,
    /// How many lose their position, by verdict, largest first.
    pub(crate) lost_by_verdict: Vec<(PointVerdict, usize)>,
    /// How many Apply leaves where they were for too few usable rays.
    pub(crate) kept: usize,
    /// How many the value holds, which the run never read.
    pub(crate) held: usize,
    /// The median distance a finite point that moves travels, in the
    /// reconstruction's units; `None` where none moves.
    pub(crate) median_shift: Option<f64>,
    /// The 90th percentile of the same distances.
    pub(crate) p90_shift: Option<f64>,
    /// Each point that loses its position, by version index, ascending.
    pub(crate) lost_points: Vec<LostPoint>,
}

impl PreviewCounts {
    /// How many points lose their position.
    pub(crate) fn lost(&self) -> usize {
        self.lost_by_verdict.iter().map(|(_, n)| n).sum()
    }
}

/// The preview's targets for one answer: one row per row of the node's base
/// point buffer and of its additions buffer, ready to upload.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreviewTargets {
    /// The proposal generation the answer is for.
    pub(crate) generation: u64,
    /// Per base row: the target in `xyz` (a unit direction for a bearing) and
    /// the flag in `w`.
    pub(crate) base: Vec<[f32; 4]>,
    /// Per base row: the surfel size factor at the target.
    pub(crate) base_size: Vec<f32>,
    /// Per addition row, as [`Self::base`].
    pub(crate) additions: Vec<[f32; 4]>,
    /// Per addition row, as [`Self::base_size`].
    pub(crate) additions_size: Vec<f32>,
    /// Each point the preview moves, by version index, for the track rays the
    /// 3D view hides while `s` is above 0.
    pub(crate) touched: Vec<u32>,
}

impl PreviewTargets {
    /// The targets and counts for the proposed run's `report` and `answer`.
    ///
    /// `stored` is the value the run read, the version's base or its
    /// materialisation, with `rows` its row map (`None` for the base itself),
    /// so a report row is taken back to the index the version gives it: a row
    /// of the base buffer below `base_count`, and of the additions buffer at
    /// or above it. `addition_count` is the additions buffer's row count.
    pub(crate) fn build(
        generation: u64,
        stored: &SfmrReconstruction,
        rows: Option<&RowMap>,
        base_count: usize,
        addition_count: usize,
        report: &RetriangulateReport,
        answer: &SfmrReconstruction,
    ) -> (Self, PreviewCounts) {
        let still = [0.0, 0.0, 0.0, FLAG_STILL];
        let mut targets = PreviewTargets {
            generation,
            base: vec![still; base_count],
            base_size: vec![1.0; base_count],
            additions: vec![still; addition_count],
            additions_size: vec![1.0; addition_count],
            touched: Vec::new(),
        };
        let mut counts = PreviewCounts {
            tracks: report.points.len(),
            ..Default::default()
        };
        let mut shifts = Vec::new();
        let mut lost = std::collections::BTreeMap::<u8, (PointVerdict, usize)>::new();
        let centroid = answer.image_table.camera_centroid();
        for status in &report.points {
            let (Some(stored_point), Some(answer_point)) = (
                stored.point_set.points.get(status.index as usize),
                answer.point_set.points.get(status.new_index as usize),
            ) else {
                continue;
            };
            let preview = point_preview(stored_point, &status.outcome, answer_point, &centroid);
            let version = rows.map_or(Some(status.index), |map| map.inverse(status.index));
            match preview {
                PointPreview::Held => counts.held += 1,
                PointPreview::Kept => counts.kept += 1,
                PointPreview::Placed => counts.moves += 1,
                PointPreview::Moves { changed, .. } => {
                    if changed {
                        counts.moves += 1;
                    }
                    if let RetriangulateOutcome::Solved {
                        change: GeometryChange::Moved { shift: Some(shift) },
                        ..
                    } = status.outcome
                    {
                        shifts.push(shift);
                    }
                }
                PointPreview::Loses(verdict) => {
                    lost.entry(verdict.code()).or_insert((verdict, 0)).1 += 1;
                    if let Some(index) = version {
                        counts.lost_points.push(LostPoint { index, verdict });
                    }
                }
            }
            let Some(index) = version else {
                continue;
            };
            if preview.flag() == FLAG_STILL {
                continue;
            }
            // A row that loses its position stands where it is, so its target
            // is never read.
            let (row, size) = match preview {
                PointPreview::Moves {
                    target,
                    size_factor,
                    ..
                } => (
                    [target.x as f32, target.y as f32, target.z as f32],
                    size_factor,
                ),
                _ => ([0.0; 3], 1.0),
            };
            let row = [row[0], row[1], row[2], preview.flag()];
            let index_usize = index as usize;
            let slot = if index_usize < base_count {
                targets
                    .base
                    .get_mut(index_usize)
                    .zip(targets.base_size.get_mut(index_usize))
            } else {
                let k = index_usize - base_count;
                targets
                    .additions
                    .get_mut(k)
                    .zip(targets.additions_size.get_mut(k))
            };
            if let Some((target, factor)) = slot {
                *target = row;
                *factor = size;
                // A point that loses its position stands where its rays end.
                if preview.flag() == FLAG_MOVES {
                    targets.touched.push(index);
                }
            }
        }
        targets.touched.sort_unstable();
        counts.lost_points.sort_by_key(|p| p.index);
        let mut lost: Vec<(PointVerdict, usize)> = lost.into_values().collect();
        lost.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.code().cmp(&b.0.code())));
        counts.lost_by_verdict = lost;
        shifts.sort_unstable_by(f64::total_cmp);
        counts.median_shift = quantile(&shifts, 0.5);
        counts.p90_shift = quantile(&shifts, 0.9);
        (targets, counts)
    }

    /// Whether the preview moves the point at `index` of the version.
    pub(crate) fn touches(&self, index: u32) -> bool {
        self.touched.binary_search(&index).is_ok()
    }
}

/// The `q` quantile of ascending `sorted`, linearly interpolated; `None` for
/// an empty list.
fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    let last = sorted.len().checked_sub(1)?;
    let at = q * last as f64;
    let (low, high) = (at.floor() as usize, at.ceil() as usize);
    Some(sorted[low] + (sorted[high] - sorted[low]) * (at - low as f64))
}

/// The words a verdict that turns a finite point into a direction is given in
/// the banner, as the Retriangulate edits' Action Log row gives it for several
/// points: "37 behind a camera that sees them".
pub(crate) fn lost_reason(verdict: PointVerdict) -> &'static str {
    crate::state::edits::direction_rule(verdict).map_or(verdict.label(), |(_, words)| words)
}

/// The blend between the stored positions and the targets, and the targets
/// the preview buffers hold.
///
/// `s` runs from 0 (stored) to 1 (target). `drawn` is the answer whose targets
/// the 3D view's preview buffers hold, which can be an older generation's while
/// `s` runs back to 0 to take a new one.
#[derive(Debug, Clone, Default)]
pub(crate) struct PreviewMotion {
    /// The blend.
    pub(crate) s: f64,
    /// Its signed speed, per second.
    pub(crate) velocity: f64,
    /// The targets the preview buffers hold, `None` before the first answer
    /// was shown.
    pub(crate) drawn: Option<Arc<PreviewTargets>>,
}

impl PreviewMotion {
    /// One frame of the blend: `ticked` is whether the preview is asked for,
    /// `answer` the targets for the proposal's current generation once the job
    /// has them, and `animate` whether `s` is stepped (`false` sets it to its
    /// goal at once).
    ///
    /// A new answer is written at once when `s` is 0, or under `animate:
    /// false`; otherwise the goal is set to 0, the points run back on the old
    /// targets, and the new ones are written when `s` reaches 0. Returns
    /// whether the blend is still moving, so the caller keeps frames coming.
    pub(crate) fn step(
        &mut self,
        ticked: bool,
        answer: Option<&Arc<PreviewTargets>>,
        animate: bool,
        dt: f64,
    ) -> bool {
        let answer = answer.filter(|_| ticked);
        // A new answer goes in at once where that makes no point jump.
        if let Some(new) = answer {
            if !self.draws(new) && (self.s == 0.0 || !animate) {
                self.drawn = Some(Arc::clone(new));
            }
        }
        let goal = if answer.is_some_and(|new| self.draws(new)) {
            1.0
        } else {
            0.0
        };
        if animate {
            (self.s, self.velocity) = crate::motion::step(
                self.s,
                self.velocity,
                goal,
                dt,
                BLEND_MAX_SPEED,
                BLEND_ACCELERATION,
            );
        } else {
            (self.s, self.velocity) = (goal, 0.0);
        }
        self.s = self.s.clamp(0.0, 1.0);
        // The run back has ended: the pending targets go in now, and the next
        // frame starts the run up on them.
        if let Some(new) = answer {
            if !self.draws(new) && self.s == 0.0 {
                self.drawn = Some(Arc::clone(new));
                return true;
            }
        }
        self.s != goal || self.velocity != 0.0
    }

    /// Whether the preview buffers hold `targets`' generation.
    fn draws(&self, targets: &PreviewTargets) -> bool {
        self.drawn
            .as_ref()
            .is_some_and(|drawn| drawn.generation == targets.generation)
    }
}

#[cfg(test)]
mod tests;
