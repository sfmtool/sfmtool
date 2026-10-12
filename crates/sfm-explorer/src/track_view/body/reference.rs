// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The *Reference* column: which row the track's patch bitmap is rendered
//! from, which row the reference-view rule picks, what the rule decided about
//! each row, the per-view readings it decided on, and the sentence that says
//! why a row is not its pick. The *ZNCC* cell's hover, which shows how the
//! row scores against that bitmap, is in `super::zncc_hover`.
//!
//! The rule picks one `in` row
//! (`sfmtool_core::patch::reference_view::choose_reference_view`). The
//! reference in use is the row the stored bitmap is rendered from
//! (`TrackPayload::reference`, where the track has a bitmap). While that row
//! is pinned every render keeps it, whichever row the rule picks, so the two
//! can differ; unpinning the row, or *Set as reference* on the pick, makes
//! them agree. Each row's ZNCC is its score against the bitmap, plain and
//! blur-matched (`sfmtool_core::patch::stored_bitmap`).

use sfmtool_core::bench::{EditableTrack, Observation, StageKind, TrackMeasurement};
use sfmtool_core::patch::reference_view::{
    tile_max_radius, ReferenceExclusion, ReferenceStanding, ReferenceTerm, ReferenceViewOptions,
};

use crate::bench::live::Evaluation;

/// The *Reference* heading's hover text.
pub(super) const REFERENCE_TIP: &str = "The track's reference, the row its patch bitmap is \
    rendered from, and which row the reference-view rule would pick from the current \
    readings.\n\n\
    The reference reads reference, on a green cell where the rule picks it too and on a red \
    cell where it does not. Then the rule's pick reads pick, on a grey cell: unpinning the \
    reference's row, or Set as reference on the pick, makes the pick the reference. While the \
    reference's row is pinned every render keeps it; a track put on the bench from a point has \
    every row pinned. Where the bitmap is the render of no row (a mean of the views an older \
    file stores), is one for judging only, or there is no bitmap yet, only the pick is marked. \
    Such a bitmap stays only until the first evaluation that renders where the rule picks a \
    row, which makes that pick the reference.\n\n\
    The rule gives each row a cost in grid px and picks the lowest. The cost is the row's \
    self-similarity radius, times an angle factor that grows with the tangent of the viewing \
    angle (1 facing the patch, 1.25 at 45\u{b0}), times four penalties, each 1 inside its \
    band: agreement, for a pair ZNCC (the median of \
    the row's ZNCCs with the other rows that are in) more than 15 points below the best; the \
    ninths, for a ninth of the tile where it agrees with the other rows more than 0.3 below \
    the track's typical agreement there; coverage, for less than 99% of the tile on the \
    photograph; and clipping, for more than 5% of the photograph under the tile clipped to \
    black or white. A row with less than 90% of its tile on the photograph is never picked, \
    and neither is a row that sees the patch at 90\u{b0} or more, edge on or from behind, \
    where no patch is drawn.\n\n\
    The first line is reference or pick, or for any other row its cost over the pick's and \
    the term that adds most to it: less sharp, oblique, agrees less, ninth differs, partial \
    or clipped, as in \u{d7}1.25 oblique; or, for a row the rule left out, partial, no tile \
    or back facing. The second is the viewing angle, the angle between the patch's normal and \
    the direction to the camera, and the pair ZNCC the rule read. Hover a cell for the cost's \
    terms and every reading. An out row is not considered.\n\n\
    Sorting by this column puts the reference first, then the pick, then the rows by their \
    cost over the pick's, and last the rows the rule left out.";

/// How a row's *Reference* cell is marked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReferenceMark {
    /// Neither the reference nor the rule's pick.
    None,
    /// The reference in use, which the rule picks too: a green cell.
    Reference,
    /// The reference in use, which the rule does not pick: a red cell. Its
    /// row's pin holds it.
    ReferenceNotPick,
    /// The reference in use where the rule picks no row: a cell with no fill,
    /// since there is no pick to accept in its place.
    ReferenceWithoutPick,
    /// The rule's pick where it is not the reference in use, or where the
    /// track has none: a grey cell.
    Pick,
}

/// The rows the *Reference* column marks, read off the track once per frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct ReferenceRows {
    /// The reference in use and its image: the row the stored bitmap is
    /// rendered from ([`crate::bench::reference_in_use`]), or `None` where the
    /// track has no bitmap or its bitmap is the render of no row.
    pub(crate) reference: Option<usize>,
    /// The reference's image index, for the hover.
    pub(crate) reference_image: Option<u32>,
    /// The row the reference-view rule picked at the last evaluation
    /// ([`crate::bench::reference_view_pick`]).
    pub(crate) pick: Option<usize>,
    /// The pick's image index, for the hover.
    pub(crate) pick_image: Option<u32>,
    /// Whether the track has a patch bitmap a commit writes: with one and no
    /// reference, the bitmap is the render of no row (a fused mean an older
    /// file stores).
    pub(crate) has_bitmap: bool,
    /// Whether the reference's row is pinned, which holds the reference
    /// whatever the rule picks.
    pub(crate) reference_pinned: bool,
    /// Whether the track's bitmap is one for judging only
    /// (`TrackPayload::bitmap_for_judging`).
    pub(crate) judging: bool,
}

impl ReferenceRows {
    /// The rows `track` marks.
    pub(crate) fn of(track: &EditableTrack) -> Self {
        let image = |i: Option<usize>| i.and_then(|i| track.observations.get(i)).map(|o| o.image);
        let reference = crate::bench::reference_in_use(track);
        let pick = crate::bench::reference_view_pick(track);
        Self {
            reference,
            reference_image: image(reference),
            pick,
            pick_image: image(pick),
            has_bitmap: track
                .track()
                .is_some_and(|p| p.committable_bitmap().is_some()),
            reference_pinned: reference
                .and_then(|r| track.observations.get(r))
                .is_some_and(|o| o.pinned),
            judging: track.track().is_some_and(|p| p.bitmap_for_judging),
        }
    }

    /// How row `index` is marked.
    pub(crate) fn mark(&self, index: usize) -> ReferenceMark {
        if self.reference == Some(index) {
            if self.pick == Some(index) {
                ReferenceMark::Reference
            } else if self.pick.is_none() {
                ReferenceMark::ReferenceWithoutPick
            } else {
                ReferenceMark::ReferenceNotPick
            }
        } else if self.pick == Some(index) {
            ReferenceMark::Pick
        } else {
            ReferenceMark::None
        }
    }
}

/// What one row's *Reference* cell draws.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReferenceCell {
    /// The cell's two lines: `reference`, `pick`, the row's cost over the pick's
    /// and the word for its costliest term, or why the rule left it out; then
    /// the viewing angle and the pair ZNCC; or `-` where
    /// nothing has measured the row.
    pub(crate) text: String,
    /// The hover text with what the mark means, every reading and the reason,
    /// or `None` where the cell is `-`.
    pub(crate) hover: Option<String>,
    /// How the cell is marked: the reference, the rule's pick, or neither.
    pub(crate) mark: ReferenceMark,
}

impl ReferenceCell {
    fn empty() -> Self {
        Self {
            text: "-".to_string(),
            hover: None,
            mark: ReferenceMark::None,
        }
    }
}

/// The *Reference* cell of row `index`, `observation`, at `stage`, as the
/// evaluation's state lets it print: nothing at the cluster stage, which has
/// no patch, or where the track could not be evaluated.
pub(super) fn reference_cell(
    index: usize,
    observation: &Observation,
    stage: StageKind,
    evaluation: &Evaluation,
    rows: &ReferenceRows,
) -> ReferenceCell {
    if stage != StageKind::Track
        || matches!(evaluation, Evaluation::Refused(_) | Evaluation::Failed(_))
    {
        return ReferenceCell::empty();
    }
    let mark = rows.mark(index);
    let Some(m) = observation.track.as_ref() else {
        return ReferenceCell::empty();
    };
    if mark == ReferenceMark::None && m.viewing_angle_deg.is_none() && m.reference_view.is_none() {
        return ReferenceCell::empty();
    }
    let word = match (mark, m.reference_view) {
        (
            ReferenceMark::Reference
            | ReferenceMark::ReferenceNotPick
            | ReferenceMark::ReferenceWithoutPick,
            _,
        ) => "reference".to_string(),
        (ReferenceMark::Pick, _) => "pick".to_string(),
        (ReferenceMark::None, None) => "-".to_string(),
        (ReferenceMark::None, Some(standing)) => standing_word(standing),
    };
    let angle = m
        .viewing_angle_deg
        .map_or_else(|| "-".to_string(), |a| format!("{a:.0}\u{b0}"));
    let second = match m.pair_zncc {
        Some(pair) => format!("{angle}, {:.0}%", 100.0 * pair),
        None => angle,
    };
    let mut hover = mark_sentences(mark, observation.pinned, rows);
    if !hover.is_empty() {
        hover.push_str("\n\n");
    }
    hover.push_str(&reference_hover(m));
    ReferenceCell {
        text: format!("{word}\n{second}"),
        hover: Some(hover),
        mark,
    }
}

/// The lines over a marked cell's readings that say what the mark means and
/// how to change it; empty for an unmarked row.
fn mark_sentences(mark: ReferenceMark, pinned: bool, rows: &ReferenceRows) -> String {
    let image = |i: Option<u32>| i.map_or_else(|| "-".to_string(), |i| i.to_string());
    match mark {
        ReferenceMark::None => String::new(),
        ReferenceMark::Reference => format!(
            "The track's reference: the patch bitmap is this row's render, and the \
             reference-view rule picks this row too.{}",
            if pinned {
                " Its pin holds it."
            } else {
                " Its row is not pinned, so the reference follows the rule's pick at every \
                 render."
            }
        ),
        ReferenceMark::ReferenceWithoutPick => format!(
            "The track's reference: the patch bitmap is this row's render. The \
             reference-view rule picks no row.{}",
            if pinned { " Its pin holds it." } else { "" }
        ),
        ReferenceMark::ReferenceNotPick if !pinned => format!(
            "The track's reference: the patch bitmap is this row's render. Its row is not \
             pinned, so the reference-view rule's pick, the row of image {}, becomes the \
             reference at the next evaluation after a step. Pinning this row keeps it.",
            image(rows.pick_image)
        ),
        ReferenceMark::ReferenceNotPick => {
            let pick = match rows.pick {
                Some(_) => format!(
                    "The reference-view rule picks the row of image {} instead. Unpinning \
                     this row, or Set as reference on that row, makes it the reference.",
                    image(rows.pick_image)
                ),
                None => "The reference-view rule picks no row.".to_string(),
            };
            format!(
                "The track's reference: the patch bitmap is this row's render, held by the \
                 row's pin. {pick}"
            )
        }
        ReferenceMark::Pick => match rows.reference {
            Some(_) if !rows.reference_pinned => format!(
                "The reference-view rule's pick. The track's reference is the row of image {}, \
                 whose row is not pinned, so this row becomes the reference at the next \
                 evaluation after a step.",
                image(rows.reference_image)
            ),
            Some(_) => format!(
                "The reference-view rule's pick. The track's reference is the row of image {}, \
                 held by its pin. Unpinning that row, or Set as reference on this one, makes \
                 this row the reference.",
                image(rows.reference_image)
            ),
            None if rows.judging => "The reference-view rule's pick among the rows that are \
                 in. Fewer than two rows that are in carry a keypoint, and a render from the \
                 rows that are in needs two, so the bitmap is one for judging only: rendered \
                 from the rows with a keypoint, in or out, for the bars to score the rows \
                 against, and never committed. There is no reference to mark."
                .to_string(),
            None if rows.has_bitmap => "The reference-view rule's pick. The patch bitmap is \
                 the render of no row, so there is no \
                 reference to mark. The next evaluation that renders the bitmap renders it \
                 from this row and makes this row the reference. Set as reference on this row \
                 renders the bitmap from it."
                .to_string(),
            None => "The reference-view rule's pick. The track has no patch bitmap yet; the \
                 next render makes one."
                .to_string(),
        },
    }
}

/// The lines of a per-ninth grid, each value in percent, `-` where a ninth
/// has no reading.
pub(super) fn grid_lines(grid: [[f64; 3]; 3]) -> Vec<String> {
    grid.iter()
        .map(|row| {
            row.iter()
                .map(|v| {
                    if v.is_finite() {
                        format!("{:>5.0}", 100.0 * v)
                    } else {
                        format!("{:>5}", "-")
                    }
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .collect()
}

/// The word for the term that adds most to a row's cost over the pick's.
fn term_word(term: ReferenceTerm) -> &'static str {
    match term {
        ReferenceTerm::Sharpness => "less sharp",
        ReferenceTerm::CappedRadius => "capped radius",
        ReferenceTerm::Angle => "oblique",
        ReferenceTerm::Agreement => "agrees less",
        ReferenceTerm::Cells => "ninth differs",
        ReferenceTerm::Coverage => "partial",
        ReferenceTerm::Clipped => "clipped",
    }
}

/// The word for why the rule left a row out.
fn exclusion_word(exclusion: ReferenceExclusion) -> &'static str {
    match exclusion {
        ReferenceExclusion::NoTile => "no tile",
        ReferenceExclusion::Coverage => "partial",
        ReferenceExclusion::BackFacing => "back facing",
    }
}

/// A row's cost over the pick's, `×1.25`, or `×∞` for an infinite one.
fn ratio_text(ratio: f64) -> String {
    if ratio.is_finite() {
        format!("\u{d7}{ratio:.2}")
    } else {
        "\u{d7}\u{221e}".to_string()
    }
}

/// The first line of a cell: `reference`, or the row's cost over the pick's
/// and the word for its costliest term, or the word for why the rule left it
/// out.
fn standing_word(standing: ReferenceStanding) -> String {
    if standing.is_reference() {
        return "reference".to_string();
    }
    match (standing.cost, standing.cost_ratio()) {
        (Err(exclusion), _) => exclusion_word(exclusion).to_string(),
        (Ok(_), Some(ratio)) => match standing.costliest_term {
            Some(term) => format!("{} {}", ratio_text(ratio), term_word(term)),
            None => ratio_text(ratio),
        },
        (Ok(_), None) => "-".to_string(),
    }
}

/// Where row `index`, `observation`, stands for ordering by the column: the
/// reference in use first, then the rule's pick, then the rows by their cost
/// over the pick's, and last the rows the rule left out. `None` for a row
/// with nothing to order by.
pub(super) fn reference_rank(
    index: usize,
    observation: &Observation,
    rows: &ReferenceRows,
) -> Option<f64> {
    if rows.reference == Some(index) {
        return Some(-1.0);
    }
    observation
        .track
        .as_ref()
        .and_then(|m| m.reference_view)
        .map(standing_rank)
}

/// [`reference_rank`] of a row the rule stood somewhere on.
fn standing_rank(standing: ReferenceStanding) -> f64 {
    if standing.is_reference() {
        return 0.0;
    }
    match standing.cost {
        // An infinite ratio sorts after every finite one and before the rows
        // left out.
        Ok(_) => standing
            .cost_ratio()
            .filter(|r| r.is_finite())
            .unwrap_or(f64::MAX),
        Err(_) => f64::INFINITY,
    }
}

/// The phrase for the term that adds most to a row's cost, in a sentence.
fn term_phrase(term: ReferenceTerm) -> &'static str {
    match term {
        ReferenceTerm::Sharpness => "its self-similarity radius",
        ReferenceTerm::CappedRadius => {
            "its self-similarity radius reading at the cap, a lower bound"
        }
        ReferenceTerm::Angle => "its viewing angle",
        ReferenceTerm::Agreement => "its agreement with the other rows",
        ReferenceTerm::Cells => "a ninth of its tile that differs from the other rows",
        ReferenceTerm::Coverage => "the part of its tile off the photograph",
        ReferenceTerm::Clipped => "the clipped part of its photograph",
    }
}

/// The lines that say what the rule decided about a row and why.
fn decision_lines(standing: ReferenceStanding, m: &TrackMeasurement) -> Vec<String> {
    let options = ReferenceViewOptions::default();
    let percent = |v: f64| format!("{:.1}%", 100.0 * v);
    let mut lines = Vec::new();
    match standing.cost {
        Err(exclusion) => lines.push(format!(
            "Left out of the reference view: {}.",
            match exclusion {
                ReferenceExclusion::NoTile => "it renders no tile".to_string(),
                ReferenceExclusion::Coverage => format!(
                    "{} of its tile is on the photograph, under the {:.0}% a reference needs",
                    m.coverage.map_or("-".to_string(), percent),
                    100.0 * options.min_coverage
                ),
                ReferenceExclusion::BackFacing => format!(
                    "it sees the patch at {}, edge on or from behind, where no patch is drawn",
                    m.viewing_angle_deg
                        .map_or("90\u{b0} or more".to_string(), |a| format!("{a:.1}\u{b0}"))
                ),
            }
        )),
        Ok(cost) => {
            if standing.is_reference() {
                lines.push(
                    "The reference view: the rule picks this row's tile, the lowest cost."
                        .to_string(),
                );
            } else {
                let ratio = standing.cost_ratio().map_or("-".to_string(), ratio_text);
                lines.push(match standing.costliest_term {
                    Some(term) => format!(
                        "Not the reference view: its cost is {ratio} the pick's, {} adding \
                         the most.",
                        term_phrase(term)
                    ),
                    None => format!("Not the reference view: its cost is {ratio} the pick's."),
                });
            }
            let finite = |v: f64| {
                if v.is_finite() {
                    format!("{v:.2}")
                } else {
                    "\u{221e}".to_string()
                }
            };
            lines.push(format!(
                "Cost {} grid px: the radius {:.2}{}, then:",
                finite(cost.cost),
                cost.radius,
                if cost.radius_read {
                    ""
                } else {
                    " (not the row's own: the largest any row has, or the default where none \
                     has one)"
                },
            ));
            lines.push(format!(
                "Capped radius \u{d7}{:.2}: {:.2} where the radius reads at the largest radius \
                 searched, {:.0} grid px, and so may be larger; 1 below it.",
                cost.capped_radius,
                options.capped_radius_factor,
                tile_max_radius()
            ));
            lines.push(format!(
                "Angle \u{d7}{}: 1 + {:.2} \u{d7} the tangent of the viewing angle.",
                finite(cost.angle),
                options.angle_weight
            ));
            lines.push(format!(
                "Agreement \u{d7}{:.2}: 1 within {:.0} points of the best pair ZNCC, plus 1 \
                 for every {:.0} points further.",
                cost.agreement,
                100.0 * options.agreement_band,
                100.0 * options.agreement_scale
            ));
            lines.push(format!(
                "Ninths \u{d7}{:.2}: 1 up to a cell deficit of {:.2}, plus 1 for every {:.2} more.",
                cost.cells, options.cell_deficit_band, options.cell_deficit_scale
            ));
            lines.push(format!(
                "Coverage \u{d7}{:.2}: 1 from {:.0}% of the tile, plus 1 for every {:.0}% less.",
                cost.coverage,
                100.0 * options.coverage_band,
                100.0 * options.coverage_scale
            ));
            lines.push(format!(
                "Clipped \u{d7}{:.2}: 1 up to {:.0}% of the photograph, plus 1 for every {:.0}% \
                 more.",
                cost.clipped,
                100.0 * options.clipped_band,
                100.0 * options.clipped_scale
            ));
        }
    }
    lines
}

/// The cell's hover text: the decision and why, then every reading.
pub(super) fn reference_hover(m: &TrackMeasurement) -> String {
    let percent = |v: f64| format!("{:.1}%", 100.0 * v);
    let mut lines: Vec<String> = Vec::new();
    match m.reference_view {
        None => lines.push(
            "Not considered for the reference view: the rule reads only the rows that are in."
                .to_string(),
        ),
        Some(standing) => lines.extend(decision_lines(standing, m)),
    }
    lines.push(String::new());
    if let Some(angle) = m.viewing_angle_deg {
        lines.push(match m.tilt_direction_deg {
            Some(tilt) => format!(
                "Viewing angle {angle:.1}\u{b0}, leaning {tilt:.0}\u{b0} from the patch's u \
                 towards its v."
            ),
            None => format!("Viewing angle {angle:.1}\u{b0}, facing the patch."),
        });
    }
    if let Some(coverage) = m.coverage {
        lines.push(format!("Coverage {} of the tile.", percent(coverage)));
    }
    if let Some(clipped) = m.clipped_share {
        lines.push(format!(
            "Clipped {} of the photograph under the tile.",
            percent(clipped)
        ));
    }
    if let Some(pair) = m.pair_zncc {
        lines.push(format!(
            "Pair ZNCC {:.0}%, the median with the other rows that are in.",
            100.0 * pair
        ));
    }
    if let Some(deficit) = m.cell_deficit {
        lines.push(format!(
            "Cell deficit {deficit:.2}: how far it falls below the track's typical agreement \
             in its worst ninth."
        ));
    }
    if let Some(grid) = m.pair_zncc_grid {
        lines.push("Pair ZNCC per ninth, the median with the other rows that are in:".to_string());
        lines.extend(grid_lines(grid));
    }
    lines.join("\n")
}
