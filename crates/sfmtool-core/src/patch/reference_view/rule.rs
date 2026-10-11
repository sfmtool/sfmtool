// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The reference-view rule: one cost per view, in grid px, and the view with
//! the lowest cost is the reference.
//!
//! `specs/core/patch/reference-view.md` § "The rule" is the design.

/// What the rule reads of one view. A reading the view does not have is
/// `None`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ReferenceReadings {
    /// The share of the tile's samples that carry image data. `None` for a
    /// view that renders no tile.
    pub coverage: Option<f64>,
    /// The share of the photograph's pixels under the tile that are clipped.
    pub clipped_share: Option<f64>,
    /// The viewing angle at the keypoint, in degrees.
    pub viewing_angle_deg: Option<f64>,
    /// The cell deficit
    /// ([`CellAgreement::deficit`](super::CellAgreement::deficit)).
    pub cell_deficit: Option<f64>,
    /// The median of the view's pairwise ZNCCs with the other views, from
    /// member coherence's matrix.
    pub pair_zncc: Option<f64>,
    /// The self-similarity ellipse's semi-major axis of the view's `R×R`
    /// tile, in grid px: the self-similarity radius.
    pub semi_major: Option<f64>,
    /// Its semi-minor axis, in grid px, which breaks a tie on the cost.
    pub semi_minor: Option<f64>,
}

/// The rule's parameters: the weight of its angle factor, the band and scale
/// of each soft penalty, the coverage floor and the radius it uses where no
/// view has one.
///
/// Each soft penalty is a hinge `1 + max(0, x − band) / scale`, `1` inside
/// its band and `2` at one scale past it, so a scale is the excess at which a
/// view counts as twice as blurry. The defaults were measured against hand
/// picks of the reference view and on two ground truths
/// (`specs/core/patch/reference-view.md` § "How the values were measured"),
/// except [`Self::min_coverage`], which is set by decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceViewOptions {
    /// `k`, the angle factor's weight: a view at viewing angle `θ` counts as
    /// `1 + k · tan θ` times as blurry as its radius says. `0` removes the
    /// angle factor. Default `0.25`.
    pub angle_weight: f64,
    /// How far below the best pair ZNCC a view's may fall before the
    /// agreement penalty starts. Default `0.15`.
    pub agreement_band: f64,
    /// The agreement penalty's scale. Default `0.15`.
    pub agreement_scale: f64,
    /// The cell deficit the cell penalty starts past. Default `0.3`.
    pub cell_deficit_band: f64,
    /// The cell penalty's scale. Default `0.3`.
    pub cell_deficit_scale: f64,
    /// The coverage at or above which the coverage penalty is `1`; the
    /// penalty reads the shortfall below it. Default `0.99`.
    pub coverage_band: f64,
    /// The coverage penalty's scale. Default `0.03`.
    pub coverage_scale: f64,
    /// The clipped share the clipping penalty starts past. Default `0.05`.
    pub clipped_band: f64,
    /// The clipping penalty's scale. Default `0.05`.
    pub clipped_scale: f64,
    /// The hard limit on coverage: a view whose tile has less of its samples
    /// on the photograph is never the reference. Default `0.9`.
    pub min_coverage: f64,
    /// The radius `r`, in grid px, every view takes where no view has a
    /// self-similarity radius. Default `1`.
    pub unread_radius: f64,
}

impl Default for ReferenceViewOptions {
    fn default() -> Self {
        Self {
            angle_weight: 0.25,
            agreement_band: 0.15,
            agreement_scale: 0.15,
            cell_deficit_band: 0.3,
            cell_deficit_scale: 0.3,
            coverage_band: 0.99,
            coverage_scale: 0.03,
            clipped_band: 0.05,
            clipped_scale: 0.05,
            min_coverage: 0.9,
            unread_radius: 1.0,
        }
    }
}

impl ReferenceViewOptions {
    /// The angle factor `1 + k · tan θ` of a view at viewing angle `θ`, in
    /// degrees; a negative angle counts as `0`. `1` at `θ = 0`, and
    /// unbounded as `θ` nears `90°`.
    pub fn angle_factor(&self, viewing_angle_deg: f64) -> f64 {
        1.0 + self.angle_weight * viewing_angle_deg.max(0.0).to_radians().tan()
    }
}

/// One of the terms a view's cost is the product of
/// ([`ReferenceCost::ln_parts`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReferenceTerm {
    /// The self-similarity radius `r`.
    Sharpness,
    /// The angle factor.
    Angle,
    /// The agreement penalty.
    Agreement,
    /// The cell penalty.
    Cells,
    /// The coverage penalty.
    Coverage,
    /// The clipping penalty.
    Clipped,
}

impl ReferenceTerm {
    /// The term's name as the wire and the bindings spell it.
    pub fn name(self) -> &'static str {
        match self {
            ReferenceTerm::Sharpness => "sharpness",
            ReferenceTerm::Angle => "angle",
            ReferenceTerm::Agreement => "agreement",
            ReferenceTerm::Cells => "cells",
            ReferenceTerm::Coverage => "coverage",
            ReferenceTerm::Clipped => "clipped",
        }
    }
}

impl std::fmt::Display for ReferenceTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Why the rule left a view out of the views it picks from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReferenceExclusion {
    /// The view renders no tile: it has no coverage reading.
    NoTile,
    /// Its coverage is under [`ReferenceViewOptions::min_coverage`].
    Coverage,
    /// It sees the patch at a viewing angle of `90°` or more, edge on or from
    /// behind, where no patch is drawn.
    BackFacing,
}

impl ReferenceExclusion {
    /// The exclusion's name as the wire and the bindings spell it.
    pub fn name(self) -> &'static str {
        match self {
            ReferenceExclusion::NoTile => "no_tile",
            ReferenceExclusion::Coverage => "coverage",
            ReferenceExclusion::BackFacing => "back_facing",
        }
    }
}

impl std::fmt::Display for ReferenceExclusion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One view's cost and the terms it is the product of:
/// `cost = radius · angle · agreement · cells · coverage · clipped`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceCost {
    /// `r`, in grid px: the view's self-similarity radius, or the stand-in
    /// for one where it has none.
    pub radius: f64,
    /// Whether [`Self::radius`] is the view's own reading.
    pub radius_read: bool,
    /// The angle factor, `1 + k · tan θ`, `1` or more.
    pub angle: f64,
    /// The agreement penalty, `1` or more.
    pub agreement: f64,
    /// The cell penalty, `1` or more.
    pub cells: f64,
    /// The coverage penalty, `1` or more.
    pub coverage: f64,
    /// The clipping penalty, `1` or more.
    pub clipped: f64,
    /// The cost, in grid px.
    pub cost: f64,
}

impl ReferenceCost {
    /// `ln cost` split into one part per term:
    /// `ln r + ln T + ln A + ln K + ln C + ln P`.
    pub fn ln_parts(&self) -> [(ReferenceTerm, f64); 6] {
        [
            (ReferenceTerm::Sharpness, self.radius.ln()),
            (ReferenceTerm::Angle, self.angle.ln()),
            (ReferenceTerm::Agreement, self.agreement.ln()),
            (ReferenceTerm::Cells, self.cells.ln()),
            (ReferenceTerm::Coverage, self.coverage.ln()),
            (ReferenceTerm::Clipped, self.clipped.ln()),
        ]
    }
}

/// What the rule decided about a track's views.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceChoice {
    /// The view it picks, as an index into the readings: the one with the
    /// lowest cost. `None` where no view is eligible (or there are no views).
    pub reference: Option<usize>,
    /// Per view: its cost, or why it was left out of the views the rule picks
    /// from.
    pub costs: Vec<Result<ReferenceCost, ReferenceExclusion>>,
}

impl ReferenceChoice {
    /// What the rule decided about view `i`, or `None` for an index past the
    /// views.
    pub fn standing(&self, i: usize) -> Option<ReferenceStanding> {
        let cost = *self.costs.get(i)?;
        let pick = self.reference.and_then(|p| self.costs[p].ok());
        let costliest_term = match (cost, pick) {
            (Ok(cost), Some(pick)) if self.reference != Some(i) => costliest_term(&cost, &pick),
            _ => None,
        };
        Some(ReferenceStanding {
            cost,
            reference: self.reference == Some(i),
            pick_cost: pick.map(|p| p.cost),
            costliest_term,
        })
    }
}

/// The term whose part of `ln cost − ln pick.cost` is the largest. A part
/// that is not a number, which only a radius of `0` gives, is passed over;
/// `None` where every part is.
fn costliest_term(cost: &ReferenceCost, pick: &ReferenceCost) -> Option<ReferenceTerm> {
    cost.ln_parts()
        .into_iter()
        .zip(pick.ln_parts())
        .map(|((term, own), (_, picked))| (term, own - picked))
        .filter(|(_, part)| !part.is_nan())
        .fold(
            None,
            |best: Option<(ReferenceTerm, f64)>, (term, part)| match best {
                Some((_, b)) if b >= part => best,
                _ => Some((term, part)),
            },
        )
        .map(|(term, _)| term)
}

/// What the rule decided about one view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceStanding {
    /// Its cost, or why it was left out of the views the rule picks from.
    pub cost: Result<ReferenceCost, ReferenceExclusion>,
    /// Whether the rule picks this view.
    pub reference: bool,
    /// The pick's cost, `None` where the rule picks no view.
    pub pick_cost: Option<f64>,
    /// The term that adds most to this view's cost over the pick's: the one
    /// with the largest part of `ln cost − ln pick_cost` in the split of
    /// [`ReferenceCost::ln_parts`]. `None` for the pick and an excluded view.
    pub costliest_term: Option<ReferenceTerm>,
}

impl ReferenceStanding {
    /// Whether the rule picks this view.
    pub fn is_reference(&self) -> bool {
        self.reference
    }

    /// This view's cost over the pick's; `None` for an excluded view and
    /// where the rule picks none.
    pub fn cost_ratio(&self) -> Option<f64> {
        Some(self.cost.ok()?.cost / self.pick_cost?)
    }
}

/// A hinge `1 + max(0, x − band) / scale`.
fn hinge(x: f64, band: f64, scale: f64) -> f64 {
    1.0 + (x - band).max(0.0) / scale
}

/// The largest of `values`, `None` where it is empty.
fn largest(values: impl Iterator<Item = f64>) -> Option<f64> {
    values.fold(None, |best: Option<f64>, v| {
        Some(best.map_or(v, |b| b.max(v)))
    })
}

/// Pick the reference view of a track from its views' readings.
///
/// Each view the pick is made from gets a cost in grid px,
/// `r · T · A · K · C · P`, and the lowest cost is picked; a tie goes to the
/// smaller semi-minor axis, a missing one counting as infinite, and then to
/// the earlier view. `r` is the self-similarity radius; `T = 1 + k · tan θ`
/// is the angle factor ([`ReferenceViewOptions::angle_factor`]); `A`, `K`,
/// `C` and `P` are hinges on the shortfall of the pair ZNCC below the best
/// one, the cell deficit, the shortfall of the coverage below its band, and
/// the clipped share.
///
/// - **Hard limits.** A view with no coverage reading (no tile), a coverage
///   under [`ReferenceViewOptions::min_coverage`] or a viewing angle of `90°`
///   or more (edge on or from behind) is never picked. The others are
///   **eligible**, and the pick is made from them.
/// - **Missing readings.** A view with no radius takes the largest radius
///   among the views the pick is made from, or
///   [`ReferenceViewOptions::unread_radius`] where none has one; a view with
///   no pair ZNCC the largest shortfall, or `A = 1` where none has one; a view
///   with no angle the largest angle factor, or `T = 1` where none has one; a
///   missing clipped share or cell deficit gives a factor of `1`.
///
/// ```
/// use sfmtool_core::patch::reference_view::{
///     choose_reference_view, ReferenceReadings, ReferenceTerm, ReferenceViewOptions,
/// };
///
/// let view = |angle: f64, radius: f64| ReferenceReadings {
///     coverage: Some(1.0),
///     clipped_share: Some(0.0),
///     viewing_angle_deg: Some(angle),
///     cell_deficit: Some(0.0),
///     pair_zncc: Some(0.9),
///     semi_major: Some(radius),
///     semi_minor: Some(radius / 2.0),
/// };
/// // A view at 20° with a radius of 0.62 grid px beats one at 55° with 0.60.
/// let choice = choose_reference_view(
///     &[view(55.0, 0.60), view(20.0, 0.62)],
///     &ReferenceViewOptions::default(),
/// );
/// assert_eq!(choice.reference, Some(1));
/// let standing = choice.standing(0).unwrap();
/// assert_eq!(standing.costliest_term, Some(ReferenceTerm::Angle));
/// assert!(standing.cost_ratio().unwrap() > 1.15);
/// ```
pub fn choose_reference_view(
    views: &[ReferenceReadings],
    options: &ReferenceViewOptions,
) -> ReferenceChoice {
    let finite = |v: Option<f64>| v.filter(|x| x.is_finite());
    let angle = |v: &ReferenceReadings| v.viewing_angle_deg.filter(|a| !a.is_nan());
    // The hard limits.
    let exclusions: Vec<Option<ReferenceExclusion>> = views
        .iter()
        .map(|v| match v.coverage.filter(|c| !c.is_nan()) {
            None => Some(ReferenceExclusion::NoTile),
            Some(c) if c < options.min_coverage => Some(ReferenceExclusion::Coverage),
            Some(_) if angle(v).is_some_and(|a| a >= 90.0) => Some(ReferenceExclusion::BackFacing),
            Some(_) => None,
        })
        .collect();
    let from: Vec<usize> = (0..views.len())
        .filter(|&i| exclusions[i].is_none())
        .collect();

    // The readings the costs read, with the missing ones filled in.
    let unread_radius = largest(from.iter().filter_map(|&i| finite(views[i].semi_major)))
        .unwrap_or(options.unread_radius);
    let best_pair = largest(from.iter().filter_map(|&i| finite(views[i].pair_zncc)));
    let shortfall = |v: &ReferenceReadings| Some(best_pair? - finite(v.pair_zncc)?);
    let unread_shortfall = largest(from.iter().filter_map(|&i| shortfall(&views[i])));
    let unread_angle = largest(
        from.iter()
            .filter_map(|&i| angle(&views[i]).map(|a| options.angle_factor(a))),
    )
    .unwrap_or(1.0);

    let cost_of = |v: &ReferenceReadings| {
        let own_radius = finite(v.semi_major);
        let radius = own_radius.unwrap_or(unread_radius);
        let angle_factor = angle(v).map_or(unread_angle, |a| options.angle_factor(a));
        let agreement = shortfall(v).or(unread_shortfall).map_or(1.0, |s| {
            hinge(s, options.agreement_band, options.agreement_scale)
        });
        let cells = finite(v.cell_deficit).map_or(1.0, |d| {
            hinge(d, options.cell_deficit_band, options.cell_deficit_scale)
        });
        let coverage = v.coverage.map_or(1.0, |cov| {
            hinge(options.coverage_band - cov, 0.0, options.coverage_scale)
        });
        let clipped = finite(v.clipped_share).map_or(1.0, |s| {
            hinge(s, options.clipped_band, options.clipped_scale)
        });
        ReferenceCost {
            radius,
            radius_read: own_radius.is_some(),
            angle: angle_factor,
            agreement,
            cells,
            coverage,
            clipped,
            cost: radius * angle_factor * agreement * cells * coverage * clipped,
        }
    };
    let costs: Vec<Result<ReferenceCost, ReferenceExclusion>> = views
        .iter()
        .zip(&exclusions)
        .map(|(v, exclusion)| exclusion.map_or_else(|| Ok(cost_of(v)), Err))
        .collect();

    // The lowest cost, a tie going to the smaller semi-minor axis and then to
    // the earlier view.
    let tie = |i: usize| finite(views[i].semi_minor).unwrap_or(f64::INFINITY);
    let mut reference: Option<usize> = None;
    for &i in &from {
        let cost = costs[i].map_or(f64::INFINITY, |c| c.cost);
        let better = match reference {
            None => true,
            Some(j) => {
                let best = costs[j].map_or(f64::INFINITY, |c| c.cost);
                cost < best || (cost == best && tie(i) < tie(j))
            }
        };
        if better {
            reference = Some(i);
        }
    }
    ReferenceChoice { reference, costs }
}
