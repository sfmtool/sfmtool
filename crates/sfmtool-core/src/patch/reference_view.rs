// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-view measurements that say what each view of a track can
//! contribute to its patch bitmap, and the reference-view rule, which picks
//! the one view whose tile could stand as the bitmap.
//!
//! `specs/core/patch/reference-view.md` is the design. The measurements are
//! taken on each view's `R×R` tile ([`render_view_tile`]): its coverage, the
//! clipped share of the photograph under it ([`clipped_share`]) and the angle
//! the view sees the patch at
//! ([`viewing_angle`](crate::patch::normal_refine::viewing_angle)); and across
//! the views, the ZNCC between each pair over each cell of the ZNCC grid
//! ([`cell_agreement`]). The rule ([`choose_reference_view`]) is a pure
//! function over those readings, the median pairwise ZNCC from
//! [member coherence](crate::patch::member_coherence) and each tile's
//! self-similarity ellipse. Nothing here changes how a patch bitmap is
//! computed; the bench reports the rule's pick beside its other readings.

mod agreement;
mod pair_readings;
mod tile;

#[cfg(test)]
mod tests;

pub(crate) use agreement::finite_middle;
pub use agreement::{
    blur_matched_agreement, cell_agreement, cell_agreement_from_pairs, pair_zncc_grid,
    BlurMatchedAgreement, CellAgreement,
};
pub use pair_readings::{
    blur_matched_pairs, pair_zncc_readings, BlurMatchedPairs, PairReadings, MIN_WINDOWED_SAMPLES,
};
pub use tile::{clipped_share, render_view_tile, ViewTile};

/// The coverage a view's tile needs to be a candidate: the share of its
/// samples that carry image data.
///
/// A tile whose patch crosses the photograph's border holds only part of the
/// patch, and a bitmap taken from it would be missing the rest. `0.99` rather
/// than `1` lets a sample or two at a corner fall off the photograph. Not
/// tuned.
pub const REFERENCE_MIN_COVERAGE: f64 = 0.99;

/// The largest clipped share a candidate's tile may have
/// ([`clipped_share`]).
///
/// A clipped region, a specular highlight or blown-out sky, holds neither
/// texture nor its true colour, so a tile with much of it cannot stand alone as
/// the bitmap. Not tuned.
pub const REFERENCE_MAX_CLIPPED_SHARE: f64 = 0.05;

/// The largest viewing angle a candidate may have, in degrees.
///
/// An oblique tile depends most on the patch model being right: an error in
/// the normal shears it by an amount that grows with `tan θ`. On the review
/// cases it changed no pick, and on the two ground truths removing it made the
/// picks it changes localize the other views 0.12 px worse on average (15
/// tracks). Kept for that reason rather than fitted.
pub const REFERENCE_MAX_VIEWING_ANGLE_DEG: f64 = 65.0;

/// The viewing angle a candidate must stay under in every fallback, in
/// degrees.
///
/// At `90°` the view sees the patch edge on, and past it the view sees the
/// patch's back, so its tile is not a picture of the patch's face at all.
/// Dropping the angle test when no view passes drops only the
/// [`REFERENCE_MAX_VIEWING_ANGLE_DEG`] limit; this one stays. A geometric
/// limit, not tuned.
pub const REFERENCE_FACING_LIMIT_DEG: f64 = 90.0;

/// The largest cell deficit a candidate may have: how far the candidate's
/// agreement with the other views may fall below the track's typical
/// agreement in its worst judged cell ([`CellAgreement::deficit`]).
///
/// The cell check catches a view that agrees with the others over the whole
/// tile but not in one part of it: an occluder, a shadow edge, or parallax
/// within the tile. Values from `0.2` to `0.4` gave the same agreement with the
/// picks on the tuning half of the review cases; `0.3` is the middle.
///
/// The bar is the same on the blur-matched cell deficit. There, with the margin
/// at `0.15`, `0.3` and `0.35` agreed best with the hand picks on the tuning
/// half (18 of 39 exactly, 36 within the lenient bounds), and `0.25` and `0.4`
/// on one track fewer within the lenient bounds. See
/// `specs/core/patch/reference-view.md` § "Blur-matched agreement".
pub const REFERENCE_MAX_CELL_DEFICIT: f64 = 0.3;

/// The typical agreement a cell needs before the cell check judges it.
///
/// A cell where the track's views do not agree, because it holds no texture
/// they share, says nothing about any one view.
pub const REFERENCE_MIN_JUDGED_CELL_ZNCC: f64 = 0.5;

/// The fewest samples with data in both tiles a cell of [`pair_zncc_grid`] is
/// correlated over. Below it a ZNCC is read off a handful of samples.
pub const REFERENCE_MIN_CELL_SAMPLES: usize = 16;

/// How far below the best candidate's median pairwise ZNCC a candidate's may
/// fall and still be considered for its sharpness.
///
/// Sharp views correlate worse with the others, since the detail they carry is
/// missing from the blurrier views, so a narrow margin turns away exactly the
/// sharp views. On the review cases where the earlier `0.05` missed, the view
/// picked by hand sat `0.06` to `0.21` below the best. `0.10` to `0.20` were
/// within one case of each other on the tuning half.
pub const REFERENCE_AGREEMENT_MARGIN: f64 = 0.15;

/// [`REFERENCE_AGREEMENT_MARGIN`] for a rule whose agreement test reads the
/// blur-matched pair ZNCC ([`ReferenceRuleInputs::agreement`]).
///
/// Blur matching takes away part of the penalty a sharp view pays for the
/// detail the blurrier views lack, which is what the plain margin makes room
/// for: it blurs only a tile sharper than its partner along every direction,
/// and leaves most pairs plain. On the tuning half of the hand picks, with the
/// cell bar at `0.3`, margins of `0.15` and `0.18` agreed best (18 of 39
/// exactly, 36 within the lenient bounds), and `0.08` to `0.12` and `0.20` on
/// one track fewer within the lenient bounds; on the held-out half `0.08`
/// agreed on one more track, exactly and within the lenient bounds. See
/// `specs/core/patch/reference-view.md` § "Blur-matched agreement".
pub const REFERENCE_BLUR_MATCHED_AGREEMENT_MARGIN: f64 = 0.15;

/// Which reading of the ZNCC between two views' tiles a test of the rule
/// reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PairZnccReading {
    /// The tiles as rendered.
    #[default]
    Plain,
    /// The tiles blur-matched ([`crate::patch::blur_matched`]): the sharper
    /// one blurred to the other's sharpness before they are correlated.
    BlurMatched,
}

impl PairZnccReading {
    /// The name the wire and the bindings spell it with: `"plain"` or
    /// `"blur_matched"`.
    pub fn name(self) -> &'static str {
        match self {
            PairZnccReading::Plain => "plain",
            PairZnccReading::BlurMatched => "blur_matched",
        }
    }
}

/// Which readings the rule's agreement test and cell check read, and so which
/// thresholds they apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ReferenceRuleInputs {
    /// The pair ZNCC the agreement test reads: the margin is
    /// [`REFERENCE_AGREEMENT_MARGIN`] on the plain reading and
    /// [`REFERENCE_BLUR_MATCHED_AGREEMENT_MARGIN`] on the blur-matched one.
    pub agreement: PairZnccReading,
    /// The pair ZNCC grid the cell check reads. The bar is
    /// [`REFERENCE_MAX_CELL_DEFICIT`] on either reading.
    pub cells: PairZnccReading,
}

impl ReferenceRuleInputs {
    /// Both tests on the plain readings.
    pub const PLAIN: Self = Self {
        agreement: PairZnccReading::Plain,
        cells: PairZnccReading::Plain,
    };

    /// The agreement test's margin.
    pub fn agreement_margin(self) -> f64 {
        match self.agreement {
            PairZnccReading::Plain => REFERENCE_AGREEMENT_MARGIN,
            PairZnccReading::BlurMatched => REFERENCE_BLUR_MATCHED_AGREEMENT_MARGIN,
        }
    }

    /// The cell check's largest cell deficit: [`REFERENCE_MAX_CELL_DEFICIT`],
    /// on either reading.
    pub fn max_cell_deficit(self) -> f64 {
        REFERENCE_MAX_CELL_DEFICIT
    }
}

/// What the rule reads of one view. A reading the view does not have is
/// `None`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ReferenceReadings {
    /// The share of the tile's samples that carry image data.
    pub coverage: Option<f64>,
    /// The share of the photograph's pixels under the tile that are clipped.
    pub clipped_share: Option<f64>,
    /// The viewing angle at the keypoint, in degrees.
    pub viewing_angle_deg: Option<f64>,
    /// The cell deficit ([`CellAgreement::deficit`]), of the reading
    /// [`ReferenceRuleInputs::cells`] names.
    pub cell_deficit: Option<f64>,
    /// The median of the view's pairwise ZNCCs with the other views, of the
    /// reading [`ReferenceRuleInputs::agreement`] names: from member
    /// coherence's matrix for the plain reading.
    pub pair_zncc: Option<f64>,
    /// The self-similarity ellipse's semi-major axis of the view's `R×R`
    /// tile, in grid px: the self-similarity radius.
    pub semi_major: Option<f64>,
    /// Its semi-minor axis, in grid px, which breaks a tie on the semi-major
    /// axis.
    pub semi_minor: Option<f64>,
}

/// One of the rule's tests, in the order it applies them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReferenceTest {
    /// Coverage at least [`REFERENCE_MIN_COVERAGE`].
    Coverage,
    /// Clipped share at most [`REFERENCE_MAX_CLIPPED_SHARE`].
    Clipped,
    /// Viewing angle at most [`REFERENCE_MAX_VIEWING_ANGLE_DEG`], or, under a
    /// fallback that drops that limit, under [`REFERENCE_FACING_LIMIT_DEG`].
    Angle,
    /// Cell deficit at most [`REFERENCE_MAX_CELL_DEFICIT`].
    Cells,
    /// Median pairwise ZNCC within [`REFERENCE_AGREEMENT_MARGIN`] of the best
    /// candidate's, or [`REFERENCE_BLUR_MATCHED_AGREEMENT_MARGIN`] on the
    /// blur-matched reading.
    Agreement,
    /// The smallest self-similarity semi-major axis among the candidates left.
    /// A view that fails only this one passed every other test and lost to a
    /// sharper view, or has no self-similarity reading.
    Sharpness,
}

impl ReferenceTest {
    /// The test's name as the wire and the bindings spell it.
    pub fn name(self) -> &'static str {
        match self {
            ReferenceTest::Coverage => "coverage",
            ReferenceTest::Clipped => "clipped",
            ReferenceTest::Angle => "angle",
            ReferenceTest::Cells => "cells",
            ReferenceTest::Agreement => "agreement",
            ReferenceTest::Sharpness => "sharpness",
        }
    }
}

impl std::fmt::Display for ReferenceTest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Which of the candidate tests the rule kept. When no view passes all of
/// them it drops the angle test, then the cell check, then coverage and the
/// clipped share together, until some view passes what is left. Dropping the
/// angle test drops its [`REFERENCE_MAX_VIEWING_ANGLE_DEG`] limit only: in
/// every fallback a candidate's viewing angle stays under
/// [`REFERENCE_FACING_LIMIT_DEG`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum ReferenceFallback {
    /// Every test applied.
    #[default]
    None,
    /// The angle test was dropped.
    WithoutAngle,
    /// The angle test and the cell check were dropped.
    WithoutAngleOrCells,
    /// Every candidate test was dropped but the facing limit: every view
    /// that faces the patch was a candidate.
    WithoutAny,
}

impl ReferenceFallback {
    /// The fallback's name as the wire and the bindings spell it.
    pub fn name(self) -> &'static str {
        match self {
            ReferenceFallback::None => "none",
            ReferenceFallback::WithoutAngle => "without_angle",
            ReferenceFallback::WithoutAngleOrCells => "without_angle_or_cells",
            ReferenceFallback::WithoutAny => "without_any",
        }
    }

    /// Whether `test` is applied under this fallback. The angle test always
    /// is: at most [`REFERENCE_MAX_VIEWING_ANGLE_DEG`] under
    /// [`ReferenceFallback::None`], and under [`REFERENCE_FACING_LIMIT_DEG`]
    /// under every other.
    pub fn applies(self, test: ReferenceTest) -> bool {
        match test {
            ReferenceTest::Angle => true,
            ReferenceTest::Cells => self <= ReferenceFallback::WithoutAngle,
            ReferenceTest::Coverage | ReferenceTest::Clipped => {
                self <= ReferenceFallback::WithoutAngleOrCells
            }
            ReferenceTest::Agreement | ReferenceTest::Sharpness => true,
        }
    }
}

/// What the rule decided about a track's views.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceChoice {
    /// The view it picks, as an index into the readings, or `None` where no
    /// candidate has a self-similarity reading (or there are no views).
    pub reference: Option<usize>,
    /// Which tests it dropped to find a candidate.
    pub fallback: ReferenceFallback,
    /// Per view: the first test, under the fallback, that turned it away, or
    /// `None` for the view picked.
    pub rejected_by: Vec<Option<ReferenceTest>>,
    /// Which readings the agreement test and the cell check read.
    pub inputs: ReferenceRuleInputs,
}

impl ReferenceChoice {
    /// What the rule decided about view `i`, or `None` for an index past the
    /// views.
    pub fn standing(&self, i: usize) -> Option<ReferenceStanding> {
        Some(ReferenceStanding {
            rejected_by: *self.rejected_by.get(i)?,
            fallback: self.fallback,
            inputs: self.inputs,
        })
    }
}

/// What the rule decided about one view: whether it is the reference view,
/// and which tests the rule dropped for the track it is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceStanding {
    /// The first test, under [`Self::fallback`], that turned the view away, or
    /// `None` for the view the rule picks.
    pub rejected_by: Option<ReferenceTest>,
    /// Which tests the rule dropped to find a candidate among the track's
    /// views.
    pub fallback: ReferenceFallback,
    /// Which readings the agreement test and the cell check read.
    pub inputs: ReferenceRuleInputs,
}

impl ReferenceStanding {
    /// Whether the rule picks this view.
    pub fn is_reference(&self) -> bool {
        self.rejected_by.is_none()
    }
}

/// Pick the reference view of a track from its views' readings.
///
/// 1. **Candidates** pass every one of: coverage at least
///    [`REFERENCE_MIN_COVERAGE`], clipped share at most
///    [`REFERENCE_MAX_CLIPPED_SHARE`], viewing angle at most
///    [`REFERENCE_MAX_VIEWING_ANGLE_DEG`], and cell deficit at most
///    [`REFERENCE_MAX_CELL_DEFICIT`]. When no view passes, the angle test is
///    dropped, then the cell check, then coverage and the clipped share
///    ([`ReferenceFallback`]). Dropping the angle test leaves a limit of
///    [`REFERENCE_FACING_LIMIT_DEG`], so a view that sees the patch edge on or
///    from behind is never a candidate; where every view does, the rule picks
///    nothing.
/// 2. **Agreement**: of the candidates, those whose median pairwise ZNCC is
///    within [`REFERENCE_AGREEMENT_MARGIN`] of the best candidate's. Where no
///    candidate has a pairwise reading, every candidate.
/// 3. **Sharpness**: of those, the one with the smallest self-similarity
///    semi-major axis, a tie going to the smaller semi-minor axis and then to
///    the earlier view.
///
/// A missing coverage or viewing angle fails its test, since the view has to
/// show that it meets it; a missing clipped share or cell deficit passes, since
/// there is nothing there to count against the view. Once the angle test is
/// dropped a missing viewing angle passes, since only an angle shown to be at
/// or past the facing limit turns the view away. A view with no pairwise
/// ZNCC fails the agreement test unless no candidate has one.
///
/// ```
/// use sfmtool_core::patch::reference_view::{
///     choose_reference_view, ReferenceReadings, ReferenceTest,
/// };
///
/// let view = |angle: f64, pair: f64, radius: f64| ReferenceReadings {
///     coverage: Some(1.0),
///     clipped_share: Some(0.0),
///     viewing_angle_deg: Some(angle),
///     cell_deficit: Some(0.0),
///     pair_zncc: Some(pair),
///     semi_major: Some(radius),
///     semi_minor: Some(radius / 2.0),
/// };
/// // The sharpest view grazes the patch, the next sharpest agrees poorly
/// // with the rest, and the third is picked.
/// let choice = choose_reference_view(&[
///     view(80.0, 0.9, 0.3),
///     view(20.0, 0.6, 0.4),
///     view(30.0, 0.9, 0.5),
///     view(10.0, 0.95, 0.9),
/// ]);
/// assert_eq!(choice.reference, Some(2));
/// assert_eq!(choice.rejected_by[0], Some(ReferenceTest::Angle));
/// assert_eq!(choice.rejected_by[1], Some(ReferenceTest::Agreement));
/// assert_eq!(choice.rejected_by[3], Some(ReferenceTest::Sharpness));
/// ```
pub fn choose_reference_view(views: &[ReferenceReadings]) -> ReferenceChoice {
    choose_reference_view_with(views, ReferenceRuleInputs::PLAIN)
}

/// [`choose_reference_view`] on readings whose pair ZNCC and cell deficit are
/// the ones `inputs` names, with the thresholds that go with them: the margin
/// [`ReferenceRuleInputs::agreement_margin`] and the cell bar
/// [`ReferenceRuleInputs::max_cell_deficit`].
pub fn choose_reference_view_with(
    views: &[ReferenceReadings],
    inputs: ReferenceRuleInputs,
) -> ReferenceChoice {
    let margin = inputs.agreement_margin();
    let max_cell_deficit = inputs.max_cell_deficit();
    let failed = |v: &ReferenceReadings, fallback: ReferenceFallback| -> Option<ReferenceTest> {
        let passes = |test: ReferenceTest| match test {
            ReferenceTest::Coverage => v.coverage.is_some_and(|c| c >= REFERENCE_MIN_COVERAGE),
            ReferenceTest::Clipped => v
                .clipped_share
                .is_none_or(|c| c.is_nan() || c <= REFERENCE_MAX_CLIPPED_SHARE),
            ReferenceTest::Angle => match fallback {
                ReferenceFallback::None => v
                    .viewing_angle_deg
                    .is_some_and(|a| a <= REFERENCE_MAX_VIEWING_ANGLE_DEG),
                _ => v
                    .viewing_angle_deg
                    .is_none_or(|a| a.is_nan() || a < REFERENCE_FACING_LIMIT_DEG),
            },
            ReferenceTest::Cells => v
                .cell_deficit
                .is_none_or(|d| d.is_nan() || d <= max_cell_deficit),
            ReferenceTest::Agreement | ReferenceTest::Sharpness => true,
        };
        [
            ReferenceTest::Coverage,
            ReferenceTest::Clipped,
            ReferenceTest::Angle,
            ReferenceTest::Cells,
        ]
        .into_iter()
        .find(|&test| fallback.applies(test) && !passes(test))
    };

    let fallback = [
        ReferenceFallback::None,
        ReferenceFallback::WithoutAngle,
        ReferenceFallback::WithoutAngleOrCells,
        ReferenceFallback::WithoutAny,
    ]
    .into_iter()
    .find(|&fallback| views.iter().any(|v| failed(v, fallback).is_none()))
    .unwrap_or(ReferenceFallback::WithoutAny);

    let mut rejected_by: Vec<Option<ReferenceTest>> =
        views.iter().map(|v| failed(v, fallback)).collect();
    let pair = |v: &ReferenceReadings| v.pair_zncc.filter(|z| z.is_finite());
    let best = (0..views.len())
        .filter(|&i| rejected_by[i].is_none())
        .filter_map(|i| pair(&views[i]))
        .fold(None, |best: Option<f64>, z| {
            Some(best.map_or(z, |b| b.max(z)))
        });
    if let Some(best) = best {
        for (i, v) in views.iter().enumerate() {
            if rejected_by[i].is_none() && !pair(v).is_some_and(|z| z >= best - margin) {
                rejected_by[i] = Some(ReferenceTest::Agreement);
            }
        }
    }

    let radius = |v: &ReferenceReadings| v.semi_major.filter(|r| r.is_finite());
    let tie = |v: &ReferenceReadings| {
        v.semi_minor
            .filter(|r| r.is_finite())
            .unwrap_or(f64::INFINITY)
    };
    let mut reference: Option<usize> = None;
    for (i, v) in views.iter().enumerate() {
        if rejected_by[i].is_some() {
            continue;
        }
        let Some(r) = radius(v) else { continue };
        let better = match reference {
            None => true,
            Some(j) => {
                let rj = radius(&views[j]).expect("only a view with a radius is held");
                r < rj || (r == rj && tie(v) < tie(&views[j]))
            }
        };
        if better {
            reference = Some(i);
        }
    }
    for (i, rejected) in rejected_by.iter_mut().enumerate() {
        if rejected.is_none() && Some(i) != reference {
            *rejected = Some(ReferenceTest::Sharpness);
        }
    }
    ReferenceChoice {
        reference,
        fallback,
        rejected_by,
        inputs,
    }
}
